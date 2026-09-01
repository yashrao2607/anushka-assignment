"""Vector and sparse stores -- PRD Sections 9.5 and 9.6.

Three stores behind two small interfaces:

* `NumpyStore`  -- exact brute-force cosine search. Always available, exact by
  construction, and for a corpus of this size faster than an approximate index.
  It is also the **ground truth** against which the Chroma backend is checked.
* `ChromaStore` -- persistent ChromaDB with HNSW, used when chromadb is
  installed. This is the backend the PRD commits to and the one that scales.
* `BM25Index`   -- Okapi BM25 lexical search, the sparse leg of hybrid retrieval.

Having two vector backends behind one ABC is not over-engineering: it is the
`VectorStore` extension seam from PRD Section 10.4, and it lets the test suite
assert that the approximate index agrees with exact search.
"""

from __future__ import annotations

import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..utils.logging import get_logger

_TOKEN = re.compile(r"[a-z0-9_]+")

# Function words carry no discriminative signal for BM25 and only inflate
# document length. They are deliberately NOT removed for the dense leg, where
# the transformer uses them for syntax.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "will", "with", "within", "must", "may", "any", "all", "not", "this", "these",
}


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS] if remove_stopwords else tokens


class VectorStore(ABC):
    """Minimal interface every dense backend must satisfy."""

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, top_n: int,
               where: dict | None = None) -> list[tuple[str, float]]: ...

    @abstractmethod
    def count(self) -> int: ...


class NumpyStore(VectorStore):
    """Exact cosine search over an in-memory matrix.

    With L2-normalised vectors, cosine similarity is a single matrix-vector
    product. At 48 chunks this is microseconds; it stays viable to roughly
    100k chunks before an approximate index earns its complexity.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.ids: list[str] = []
        self.metadatas: list[dict] = []
        self.matrix: np.ndarray | None = None

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        self.ids = list(ids)
        self.metadatas = list(metadatas)
        self.matrix = vectors.astype(np.float32)

    def search(self, query: np.ndarray, top_n: int,
               where: dict | None = None) -> list[tuple[str, float]]:
        if self.matrix is None or not len(self.ids):
            return []
        scores = self.matrix @ query.astype(np.float32)

        if where:
            mask = np.array(
                [_matches(md, where) for md in self.metadatas], dtype=bool
            )
            if not mask.any():
                return []
            scores = np.where(mask, scores, -np.inf)

        k = min(top_n, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        # argpartition finds the top-k without sorting the whole array.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def count(self) -> int:
        return len(self.ids)

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump({"ids": self.ids, "meta": self.metadatas, "mat": self.matrix}, fh)

    def load(self) -> bool:
        if not self.path or not self.path.exists():
            return False
        with self.path.open("rb") as fh:
            blob = pickle.load(fh)
        self.ids, self.metadatas, self.matrix = blob["ids"], blob["meta"], blob["mat"]
        return True


class ChromaStore(VectorStore):
    """Persistent ChromaDB collection with an HNSW cosine index."""

    def __init__(self, path: Path, collection: str = "docs_v1",
                 hnsw: dict[str, Any] | None = None) -> None:
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection_name = collection
        settings = hnsw or {}
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={
                "hnsw:space": settings.get("space", "cosine"),
                "hnsw:construction_ef": settings.get("construction_ef", 200),
                "hnsw:M": settings.get("M", 32),
            },
        )

    def add(self, ids: list[str], vectors: np.ndarray, metadatas: list[dict]) -> None:
        # Rebuild rather than upsert, so an ingest with different chunking never
        # leaves orphaned chunks from a previous configuration in the index.
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        for i in range(0, len(ids), 500):  # Chroma prefers bounded batches
            self.collection.add(
                ids=ids[i:i + 500],
                embeddings=vectors[i:i + 500].tolist(),
                metadatas=[_flatten(m) for m in metadatas[i:i + 500]],
                documents=[m.get("text", "") for m in metadatas[i:i + 500]],
            )

    def search(self, query: np.ndarray, top_n: int,
               where: dict | None = None) -> list[tuple[str, float]]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=min(top_n, self.count()),
            where=_to_chroma_where(where) if where else None,
        )
        ids = result["ids"][0]
        # Chroma returns cosine *distance*; convert back to similarity so both
        # backends speak the same units and are directly comparable.
        return [(i, 1.0 - float(d)) for i, d in zip(ids, result["distances"][0])]

    def count(self) -> int:
        return int(self.collection.count())


class BM25Index:
    """Okapi BM25 sparse retrieval -- the lexical leg of hybrid search.

    BM25 is included precisely where dense embeddings are weakest: rare tokens
    with no useful learned representation. A query for `ERR_4092` has no
    meaningful semantic neighbourhood, but it is a perfect lexical match.
    """

    def __init__(self) -> None:
        self.ids: list[str] = []
        self._bm25: Any = None
        self.backend = "unavailable"

    def build(self, ids: list[str], texts: Iterable[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.ids = list(ids)
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.backend = "rank_bm25.BM25Okapi"

    def search(self, query: str, top_n: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        k = min(top_n, len(self.ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        # A zero BM25 score means no query term occurs at all -- that is a
        # non-result, and returning it would silently inflate recall.
        return [(self.ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"ids": self.ids, "bm25": self._bm25}, fh)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        self.ids, self._bm25 = blob["ids"], blob["bm25"]
        self.backend = "rank_bm25.BM25Okapi"
        return True


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _flatten(meta: dict) -> dict:
    """Chroma metadata values must be scalars."""
    return {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in meta.items()
        if v is not None and k != "text"
    }


def _matches(meta: dict, where: dict) -> bool:
    for key, wanted in where.items():
        value = meta.get(key)
        if isinstance(wanted, (list, tuple, set)):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def _to_chroma_where(where: dict) -> dict:
    clauses = [
        {k: {"$in": list(v)}} if isinstance(v, (list, tuple, set)) else {k: {"$eq": v}}
        for k, v in where.items()
    ]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def build_vector_store(backend: str, cfg) -> VectorStore:
    """Factory with an honest fallback.

    If Chroma is requested but unavailable, fall back to exact numpy search and
    say so in the log -- a degraded backend must never be silent, because every
    metric in the report would then describe a different system than the one
    named in the config.
    """
    log = get_logger()
    if backend == "chroma":
        try:
            store_cfg = cfg.extra.get("store", {})
            return ChromaStore(
                cfg.root / store_cfg.get("path", "storage/chroma"),
                store_cfg.get("collection", "docs_v1"),
                store_cfg.get("hnsw"),
            )
        except Exception as exc:
            log.warning("chromadb unavailable (%s) -- falling back to exact numpy search", exc)
    return NumpyStore(cfg.root / "storage" / "numpy_store.pkl")
