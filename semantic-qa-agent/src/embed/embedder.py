"""Embedding layer with a persistent cache -- PRD Section 9.4.

A bi-encoder maps text to a dense vector such that semantically similar texts
land close together. Vectors are **L2-normalised**, which makes cosine
similarity equal to a plain dot product: faster, and numerically better behaved.

**Why the cache is built now, before it is needed.** Phase 3's ablation sweep
re-runs the corpus across ten configurations. Without a cache that is ten full
embedding passes; with one it is a single pass plus nine cache hits. The cache
key is `sha256(text) + model_name`, so changing the model correctly invalidates
it while changing an unrelated setting does not.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.logging import get_logger


class EmbeddingCache:
    """SQLite-backed vector cache. Chosen over pickle so it is concurrent-safe,
    incrementally writable, and inspectable with any SQLite client."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        self.conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(text: str, model: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        out: dict[str, np.ndarray] = {}
        # Chunk the IN clause: SQLite caps host parameters at 999.
        for i in range(0, len(keys), 900):
            batch = keys[i:i + 900]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({placeholders})",
                batch,
            ).fetchall()
            for key, dim, blob in rows:
                out[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
            [(k, int(v.shape[0]), v.astype(np.float32).tobytes()) for k, v in items.items()],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class Embedder:
    """Sentence-Transformers bi-encoder with caching and L2 normalisation."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True,
        cache_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache = EmbeddingCache(cache_path) if cache_path else None
        self._model = None  # loaded lazily: importing torch costs seconds

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            self._model = SentenceTransformer(self.model_name)
            get_logger().info(
                "loaded embedding model %s in %.1fs",
                self.model_name, time.perf_counter() - started,
            )
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
        """Embed `texts`, returning an (n, dim) float32 array in input order."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        keys = [EmbeddingCache.make_key(t, self.model_name) for t in texts]
        cached: dict[str, np.ndarray] = (
            self.cache.get_many(keys) if (self.cache and use_cache) else {}
        )

        todo_idx = [i for i, k in enumerate(keys) if k not in cached]
        if todo_idx:
            fresh = self.model.encode(
                [texts[i] for i in todo_idx],
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            if self.cache and use_cache:
                self.cache.put_many({keys[i]: fresh[j] for j, i in enumerate(todo_idx)})
            for j, i in enumerate(todo_idx):
                cached[keys[i]] = fresh[j]

        vectors = np.vstack([cached[k] for k in keys]).astype(np.float32)
        if self.normalize:
            vectors = l2_normalize(vectors)
        return vectors

    def encode_one(self, text: str, use_cache: bool = True) -> np.ndarray:
        return self.encode([text], use_cache=use_cache)[0]

    def cache_stats(self) -> tuple[int, int]:
        return (self.cache.hits, self.cache.misses) if self.cache else (0, 0)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so that dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector stays zero rather than becoming NaN
    return matrix / norms
