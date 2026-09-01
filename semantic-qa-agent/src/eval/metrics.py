"""Retrieval metrics, implemented from scratch -- PRD Section 13.2.

Written by hand rather than imported deliberately: it demonstrates that the
metrics are understood rather than invoked, it removes a dependency from the
reproducibility path, and every function is unit-tested against a hand-computed
fixture so the numbers in the report can be trusted.

All functions take a ranked list of retrieved ids and a relevance judgement,
and ignore anything beyond position k.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Of the top-k results, what fraction are relevant?

    Divided by k rather than by len(top_k): a query that returns only 2 results
    when k=5 is being *less useful*, and the metric should reflect that.
    """
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    return sum(1 for cid in top if cid in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Of everything relevant that exists, what fraction did we surface in k?"""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary: did at least one relevant item make the top-k?"""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant result; 0 if none in the top-k.

    Models a user who scans from the top and stops at the first useful hit --
    which is why it rewards getting one thing right at position 1 far more than
    getting three things right at positions 8, 9 and 10.
    """
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Discounted Cumulative Gain with exponential gain.

        DCG = sum over i of (2^rel_i - 1) / log2(i + 1)

    The exponential numerator means a grade-3 result is worth far more than
    three grade-1 results; the logarithmic denominator discounts by position.
    """
    total = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        rel = grades.get(cid, 0)
        if rel > 0:
            total += (2 ** rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """DCG normalised by the best achievable DCG for this query.

    Normalisation is what makes nDCG comparable across queries: a query with one
    relevant chunk and a query with six are both scored out of 1.0.
    """
    ideal_order = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(
        (2 ** rel - 1) / math.log2(i + 1)
        for i, rel in enumerate(ideal_order, start=1) if rel > 0
    )
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved, grades, k) / idcg


def average_precision(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Mean of precision@i taken at every rank i that holds a relevant result."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_query(
    retrieved: Sequence[str],
    relevant: set[str],
    grades: Mapping[str, int],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Every metric for a single query, keyed `metric@k`."""
    out: dict[str, float] = {}
    for k in k_values:
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, grades, k)
    max_k = max(k_values)
    out[f"mrr@{max_k}"] = reciprocal_rank(retrieved, relevant, max_k)
    out[f"map@{max_k}"] = average_precision(retrieved, relevant, max_k)
    return out


def aggregate(per_query: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Macro-average over queries: every query counts equally, regardless of how
    many relevant chunks it happens to have."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(mean([q[k] for q in per_query]), 4) for k in keys}
