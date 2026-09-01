"""Retrieval -- PRD Sections 8.1, 9.6.

Three retrieval modes behind one interface, so the evaluation harness can run
all of them against identical data and the ablation table compares like with
like:

    dense   -- bi-encoder cosine similarity. Strong on paraphrase, weak on rare
               literal tokens (error codes, SKUs, acronyms).
    sparse  -- Okapi BM25. Exactly the inverse strengths.
    hybrid  -- both, fused with Reciprocal Rank Fusion.

**Why RRF rather than weighted score blending.** Cosine similarity lives in
[-1, 1] and BM25 scores are unbounded and corpus-dependent; any fixed weighting
of the two raw scores is arbitrary and breaks when the corpus changes. RRF uses
*ranks only*:

    score(d) = sum over rankers of  1 / (k + rank_r(d))

It has one parameter (k=60, the value from the original paper), needs no
normalisation, and cannot be destabilised by an outlier score. That robustness
is the reason it is the default here rather than a tuned linear blend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..embed.embedder import Embedder
from ..store.base import BM25Index, VectorStore

Mode = Literal["dense", "sparse", "hybrid"]


@dataclass
class RetrievedChunk:
    """One result, carrying the full provenance of *how* it was retrieved.

    Keeping the per-ranker ranks (not just the final score) is what makes the
    ablation analysis possible: it shows whether a hit came from the dense leg,
    the sparse leg, or both.
    """

    chunk_id: str
    text: str
    metadata: dict
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_rank: int = 0

    @property
    def score(self) -> float:
        """The score that actually determined the final ordering."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score:
            return self.rrf_score
        return self.dense_score if self.dense_score is not None else (self.bm25_score or 0.0)


@dataclass
class RetrievalTrace:
    """Per-query timings and candidate counts -- feeds the latency waterfall."""

    mode: str = "hybrid"
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    embed_ms: float = 0.0
    fuse_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    n_dense: int = 0
    n_bm25: int = 0
    n_fused: int = 0
    n_reranked: int = 0
    reranked: bool = False
    top_score: float = 0.0
    notes: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """Fuse ranked ID lists into a single score map. Ranks are 1-based."""
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class Retriever:
    """Dense / sparse / hybrid retrieval over an indexed corpus."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index | None,
        chunks_by_id: dict[str, dict],
        *,
        dense_top_n: int = 25,
        bm25_top_n: int = 25,
        rrf_k: int = 60,
        final_top_k: int = 5,
        reranker=None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.chunks = chunks_by_id
        self.dense_top_n = dense_top_n
        self.bm25_top_n = bm25_top_n
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.reranker = reranker

    # -- individual legs ----------------------------------------------------
    def _dense(self, query: str, top_n: int, where: dict | None,
               trace: RetrievalTrace) -> list[tuple[str, float]]:
        t0 = time.perf_counter()
        vector = self.embedder.encode_one(query)
        trace.embed_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        hits = self.vector_store.search(vector, top_n, where)
        trace.dense_ms = (time.perf_counter() - t1) * 1000
        trace.n_dense = len(hits)
        return hits

    def _sparse(self, query: str, top_n: int,
                trace: RetrievalTrace) -> list[tuple[str, float]]:
        if self.bm25 is None:
            trace.notes.append("bm25_unavailable")
            return []
        t0 = time.perf_counter()
        hits = self.bm25.search(query, top_n)
        trace.bm25_ms = (time.perf_counter() - t0) * 1000
        trace.n_bm25 = len(hits)
        return hits

    # -- public API ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: Mode = "hybrid",
        where: dict | None = None,
        rerank: bool | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        top_k = top_k or self.final_top_k
        use_rerank = self.reranker is not None if rerank is None else bool(rerank)
        trace = RetrievalTrace(mode=mode)
        started = time.perf_counter()

        dense_hits = (
            self._dense(query, self.dense_top_n, where, trace)
            if mode in ("dense", "hybrid") else []
        )
        sparse_hits = (
            self._sparse(query, self.bm25_top_n, trace)
            if mode in ("sparse", "hybrid") else []
        )

        dense_rank = {cid: i + 1 for i, (cid, _) in enumerate(dense_hits)}
        bm25_rank = {cid: i + 1 for i, (cid, _) in enumerate(sparse_hits)}
        dense_score = dict(dense_hits)
        bm25_score = dict(sparse_hits)

        t_fuse = time.perf_counter()
        if mode == "hybrid":
            fused = reciprocal_rank_fusion(
                [[c for c, _ in dense_hits], [c for c, _ in sparse_hits]], self.rrf_k
            )
            order = sorted(fused, key=lambda c: -fused[c])
        elif mode == "dense":
            fused = {}
            order = [c for c, _ in dense_hits]
        else:
            fused = {}
            order = [c for c, _ in sparse_hits]
        trace.fuse_ms = (time.perf_counter() - t_fuse) * 1000
        trace.n_fused = len(order)

        # Stage 1 produced a wide candidate pool. When re-ranking is on, keep the
        # whole pool for the cross-encoder to reorder; otherwise cut to top_k now.
        pool_size = (
            min(len(order), self.reranker.max_candidates)
            if use_rerank and self.reranker else top_k
        )

        candidates: list[RetrievedChunk] = []
        for rank, chunk_id in enumerate(order[:pool_size], start=1):
            record = self.chunks.get(chunk_id, {})
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=record.get("text", ""),
                    metadata=record,
                    dense_score=dense_score.get(chunk_id),
                    dense_rank=dense_rank.get(chunk_id),
                    bm25_score=bm25_score.get(chunk_id),
                    bm25_rank=bm25_rank.get(chunk_id),
                    rrf_score=fused.get(chunk_id, 0.0),
                    final_rank=rank,
                )
            )

        # -- Stage 2: cross-encoder re-ranking ------------------------------
        if use_rerank and self.reranker and candidates:
            t_rerank = time.perf_counter()
            results = self.reranker.rerank(query, candidates, top_k)
            trace.rerank_ms = (time.perf_counter() - t_rerank) * 1000
            trace.n_reranked = min(len(candidates), self.reranker.max_candidates)
            trace.reranked = self.reranker.available
        else:
            results = candidates[:top_k]

        trace.top_score = results[0].score if results else 0.0
        trace.total_ms = (time.perf_counter() - started) * 1000
        return results, trace

    def retrieve_ids(self, query: str, top_k: int, mode: Mode = "hybrid",
                     rerank: bool | None = None) -> list[str]:
        """Ranked chunk ids only -- the hot path used by the evaluation harness."""
        hits, _ = self.retrieve(query, top_k=top_k, mode=mode, rerank=rerank)
        return [h.chunk_id for h in hits]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0
