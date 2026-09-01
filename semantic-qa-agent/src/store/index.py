"""Index building and loading -- the bridge from Phase 1 output to Phase 2 search.

Reads `data/processed/chunks.jsonl` (the Phase 1 contract), embeds `embed_text`,
and populates both the dense vector store and the sparse BM25 index.

Note which field feeds which leg, because it matters:

* the **dense** leg embeds `embed_text` -- the heading-enriched form, so a chunk
  carries its parent section into its vector;
* the **sparse** leg indexes `text` + heading -- BM25 matches literal tokens, and
  repeating the document title in every chunk would distort term statistics
  across the corpus.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import Config
from ..embed.embedder import Embedder
from ..ingest.pipeline import load_chunks
from ..retrieve.retriever import Retriever
from ..utils.logging import get_logger
from .base import BM25Index, NumpyStore, build_vector_store


def _embedder_from_config(cfg: Config) -> Embedder:
    ecfg = cfg.extra.get("embedding", {})
    return Embedder(
        model_name=ecfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
        batch_size=int(ecfg.get("batch_size", 64)),
        normalize=bool(ecfg.get("normalize", True)),
        cache_path=cfg.root / ecfg.get("cache_path", ".cache/embeddings.sqlite"),
    )


def build_index(cfg: Config, backend: str | None = None) -> dict:
    """Embed every chunk and build both indexes. Returns a stats dict."""
    log = get_logger()
    records = load_chunks(cfg)
    if not records:
        raise FileNotFoundError(
            "no chunks found -- run `python -m src.cli ingest` first"
        )

    backend = backend or cfg.extra.get("store", {}).get("backend", "chroma")
    embedder = _embedder_from_config(cfg)

    ids = [r["chunk_id"] for r in records]
    embed_texts = [r["embed_text"] for r in records]
    bm25_texts = [
        f"{r.get('section_heading') or ''} {r['text']}".strip() for r in records
    ]

    t0 = time.perf_counter()
    vectors = embedder.encode(embed_texts)
    embed_s = time.perf_counter() - t0
    hits, misses = embedder.cache_stats()
    log.info(
        "embedded %d chunk(s) in %.2fs (dim=%d, cache hits=%d misses=%d)",
        len(ids), embed_s, vectors.shape[1], hits, misses,
    )

    t1 = time.perf_counter()
    store = build_vector_store(backend, cfg)
    store.add(ids, vectors, records)
    if isinstance(store, NumpyStore):
        store.save()
    dense_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    bm25 = BM25Index()
    try:
        bm25.build(ids, bm25_texts)
        bm25.save(cfg.root / "storage" / "bm25_index.pkl")
    except ImportError:
        log.warning("rank_bm25 not installed -- hybrid retrieval will be dense-only")
    sparse_s = time.perf_counter() - t2

    return {
        "chunks": len(ids),
        "dim": int(vectors.shape[1]),
        "backend": type(store).__name__,
        "bm25_backend": bm25.backend,
        "embed_s": round(embed_s, 2),
        "dense_index_s": round(dense_s, 2),
        "sparse_index_s": round(sparse_s, 2),
        "cache_hits": hits,
        "cache_misses": misses,
        "model": embedder.model_name,
        "config_fingerprint": cfg.fingerprint(),
    }


def load_retriever(cfg: Config, backend: str | None = None) -> Retriever:
    """Load an existing index and return a ready-to-query Retriever.

    The dense index is rebuilt in memory when a persisted store is absent; with
    a warm embedding cache that costs milliseconds, so correctness is preferred
    over a stale-index optimisation.
    """
    records = load_chunks(cfg)
    if not records:
        raise FileNotFoundError("no chunks found -- run `python -m src.cli ingest` first")

    backend = backend or cfg.extra.get("store", {}).get("backend", "chroma")
    embedder = _embedder_from_config(cfg)
    chunks_by_id = {r["chunk_id"]: r for r in records}

    store = build_vector_store(backend, cfg)
    if isinstance(store, NumpyStore):
        if not store.load() or store.count() != len(records):
            vectors = embedder.encode([r["embed_text"] for r in records])
            store.add([r["chunk_id"] for r in records], vectors, records)
            store.save()
    elif store.count() != len(records):
        get_logger().info("index is stale (%d indexed vs %d chunks) -- rebuilding",
                          store.count(), len(records))
        vectors = embedder.encode([r["embed_text"] for r in records])
        store.add([r["chunk_id"] for r in records], vectors, records)

    bm25: BM25Index | None = BM25Index()
    chunk_ids = [r["chunk_id"] for r in records]
    loaded = bm25.load(cfg.root / "storage" / "bm25_index.pkl")
    # A pickled index built from a different chunking configuration would
    # reference chunk ids that no longer exist, silently returning nothing for
    # the sparse leg. Verify the ids match rather than trusting the file.
    if loaded and bm25.ids != chunk_ids:
        get_logger().info("bm25 index is stale (%d ids vs %d chunks) -- rebuilding",
                          len(bm25.ids), len(chunk_ids))
        loaded = False
    if not loaded:
        try:
            bm25.build(
                [r["chunk_id"] for r in records],
                [f"{r.get('section_heading') or ''} {r['text']}".strip() for r in records],
            )
        except ImportError:
            bm25 = None

    reranker = None
    rrcfg = cfg.extra.get("rerank", {})
    if rrcfg.get("enabled", True):
        from ..retrieve.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model_name=rrcfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            max_candidates=int(rrcfg.get("max_candidates", 40)),
        )

    rcfg = cfg.extra.get("retrieval", {})
    return Retriever(
        embedder=embedder,
        vector_store=store,
        bm25=bm25,
        chunks_by_id=chunks_by_id,
        dense_top_n=int(rcfg.get("dense_top_n", 25)),
        bm25_top_n=int(rcfg.get("bm25_top_n", 25)),
        rrf_k=int(rcfg.get("rrf_k", 60)),
        final_top_k=int(rcfg.get("final_top_k", 5)),
        reranker=reranker,
    )
