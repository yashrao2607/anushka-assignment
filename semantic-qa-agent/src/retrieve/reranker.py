"""Cross-encoder re-ranker -- PRD Section 9.7, the headline differentiator.

**Why a second stage exists at all.** A bi-encoder must compress a whole chunk
into a single vector *before it has seen the question*. That is what makes it
indexable and fast, and it is also what makes it lossy: the vector must be a
good summary for every possible future query at once.

A cross-encoder makes the opposite trade. It takes the pair

    [CLS] query [SEP] chunk [SEP]

through a transformer that attends across both together, and emits one relevance
logit. Nothing is compressed before the query is known, so it is far more
accurate -- but it is O(n) forward passes per query and cannot be pre-indexed.

Hence two stages: the cheap bi-encoder (plus BM25) casts a wide net for recall,
and the expensive cross-encoder reorders that small candidate set for precision.

Phase 2 measured exactly the gap this is meant to close: Recall@10 was 0.980, so
the right chunk is almost always *in* the pool -- it is simply not at the top.
Phase 2 also showed RRF applies the same fixed arithmetic whether BM25 deserves
trust (an error code) or not (a paraphrase). A cross-encoder reads the actual
query-chunk pair, so it can demote a confidently-wrong lexical match that RRF has
no mechanism to identify.

This runs entirely locally on CPU -- no API calls, no cost.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs jointly and reorders by that score."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_candidates: int = 40,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self._model = None
        self.available = True
        self.last_ms = 0.0

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            started = time.perf_counter()
            self._model = CrossEncoder(self.model_name, max_length=512)
            get_logger().info(
                "loaded cross-encoder %s in %.1fs",
                self.model_name, time.perf_counter() - started,
            )
        return self._model

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Relevance logits for each (query, text) pair."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        pairs = [[query, t] for t in texts]
        started = time.perf_counter()
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        self.last_ms = (time.perf_counter() - started) * 1000
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def rerank(self, query: str, candidates: list, top_k: int) -> list:
        """Reorder `candidates` (RetrievedChunk objects) by cross-encoder score.

        Degrades gracefully: if the model cannot be loaded the original order is
        returned unchanged and the failure is logged, rather than taking the
        whole query path down (PRD principle #6).
        """
        if not candidates:
            return []
        pool = candidates[: self.max_candidates]

        try:
            scores = self.score(query, [c.text for c in pool])
        except Exception as exc:  # pragma: no cover - depends on environment
            get_logger().warning("rerank_unavailable: %s -- passing RRF order through", exc)
            self.available = False
            return candidates[:top_k]

        for candidate, score in zip(pool, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(pool, key=lambda c: c.rerank_score, reverse=True)[:top_k]
        for rank, candidate in enumerate(ranked, start=1):
            candidate.final_rank = rank
        return ranked


def sigmoid(x: float) -> float:
    """Map a raw cross-encoder logit into (0, 1).

    ms-marco cross-encoders emit unbounded logits. A calibrated probability is
    what the refusal gate thresholds on, and what is shown to a user as a
    confidence -- a raw logit of 4.7 means nothing to anyone.
    """
    # Always return a plain Python float: numpy scalars leak into the response
    # dataclass, where `np.False_ is False` is False and json.dumps fails.
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    z = np.exp(x)
    return float(z / (1.0 + z))
