"""Association costs and optimal assignment.

PRD 9.3: `C = lambda * d_IoU + (1 - lambda) * d_cosine`, solved with the
Hungarian algorithm at a match threshold of 0.8.

Two details that decide whether a tracker works:

* **Cost, not similarity.** Everything here is a *distance* in [0, 1], so the
  fused cost is a convex combination and `lambda` means what the PRD says it
  means. Mixing an IoU similarity with a cosine distance is a sign error that
  produces a tracker which works "sometimes".

* **Gating before assignment.** Hungarian assignment is globally optimal over
  whatever matrix it is handed, which includes pairs that are physically
  impossible. Setting those entries to a large constant *before* solving is
  what stops a globally-optimal-but-absurd match, and it is why appearance
  alone never overrides motion.
"""

from __future__ import annotations

import numpy as np

INF_COST = 1e5


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment, returning matches plus the unmatched rows/columns."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    except ImportError:  # pragma: no cover - scipy ships with the stack
        pairs = _greedy_assignment(cost)

    matches = [(r, c) for r, c in pairs if cost[r, c] <= thresh]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    unmatched_rows = [r for r in range(cost.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost.shape[1]) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols


def _greedy_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Fallback: repeatedly take the globally cheapest remaining pair."""
    pairs: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in order:
        r, c = int(r), int(c)
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    return pairs


def iou_distance(atlbrs: np.ndarray, btlbrs: np.ndarray) -> np.ndarray:
    """1 - IoU between two sets of xyxy boxes."""
    a = np.asarray(atlbrs, dtype=float).reshape(-1, 4)
    b = np.asarray(btlbrs, dtype=float).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.ones((a.shape[0], b.shape[0]), dtype=float)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return 1.0 - iou


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance in [0, 2] between two sets of embeddings.

    Embeddings are L2-normalised at extraction (PRD 9.2), so this is a matrix
    product; the explicit renormalisation guards a caller that forgot.
    """
    a = np.asarray(a, dtype=float).reshape(len(a), -1) if len(a) else np.zeros((0, 1))
    b = np.asarray(b, dtype=float).reshape(len(b), -1) if len(b) else np.zeros((0, 1))
    if a.size == 0 or b.size == 0:
        return np.ones((a.shape[0], b.shape[0]), dtype=float)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - (a @ b.T)


def fuse_cost(
    iou_cost: np.ndarray,
    appearance_cost: np.ndarray | None,
    appearance_weight: float,
    appearance_gate: float = 0.7,
) -> np.ndarray:
    """`C = lambda * d_IoU + (1 - lambda) * d_cosine` with an appearance gate.

    `appearance_weight` is the PRD's lambda: 1.0 is motion-only, 0.0 is
    appearance-only. Appearance distances above `appearance_gate` are treated
    as "no evidence" and fall back to the IoU cost, because a badly-cropped or
    motion-blurred embedding is worse than no embedding at all -- it is
    confidently wrong, and averaging it in drags a good motion match away.
    """
    if appearance_cost is None or appearance_cost.size == 0 or appearance_weight >= 1.0:
        return iou_cost
    app = appearance_cost.copy()
    app[app > appearance_gate] = np.nan
    fused = appearance_weight * iou_cost + (1.0 - appearance_weight) * app
    return np.where(np.isnan(fused), iou_cost, fused)


def gate_cost(
    cost: np.ndarray,
    tracks,
    detections_xyah: np.ndarray,
    kalman,
    gating_threshold: float = 9.4877,
    only_position: bool = False,
) -> np.ndarray:
    """Raise impossible pairs to INF using the Kalman Mahalanobis distance.

    9.4877 is the 0.95 quantile of the chi-square distribution with 4 degrees
    of freedom -- the measurement dimensionality. A pair beyond it is one the
    motion model says cannot happen, and no appearance similarity should be
    allowed to overrule that.
    """
    if cost.size == 0 or not len(tracks):
        return cost
    out = cost.copy()
    for i, track in enumerate(tracks):
        d = kalman.gating_distance(track.mean, track.covariance, detections_xyah)
        if only_position:
            d = d[:2] if d.ndim == 1 else d
        out[i, d > gating_threshold] = INF_COST
    return out


def class_gate(cost: np.ndarray, track_classes, det_classes) -> np.ndarray:
    """Forbid cross-class association.

    PRD 9.4 applies this to the ReID gallery -- "a car can never be
    re-identified as a person, an easy, large accuracy win". The same argument
    holds one stage earlier, during association.
    """
    if cost.size == 0:
        return cost
    t = np.asarray(track_classes).reshape(-1, 1)
    d = np.asarray(det_classes).reshape(1, -1)
    return np.where(t == d, cost, INF_COST)
