"""ReID metrics: Rank-k / CMC / mAP, post-occlusion recovery, cross-camera rate.

PRD 4.4 (M14-M18). Three families, because they answer different questions:

* **Rank-k / CMC / mAP** score the *embedding space* in isolation, on a
  query/gallery protocol. They say whether appearance vectors of the same
  object are closer to each other than to other objects -- independent of any
  tracker.
* **Post-occlusion recovery rate (M17)** scores the *system*: after a genuine
  full occlusion, did the identity come back with its original id? This is the
  number the whole D1 differentiator exists to move, and it cannot be inferred
  from Rank-1.
* **Cross-camera match rate (M18)** is the same question across a camera
  hand-off.

**Why the occlusion events are detectable at all.** The ground truth carries a
per-frame `visibility` column, so a "full occlusion" is defined precisely --
visibility below the ignore threshold for at least `min_gap` consecutive frames,
with the object visible on both sides. That definition is in the data, not in a
human's judgement of what looked occluded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..data.mot import IGNORE_VISIBILITY, MotRow
from .detection_metrics import iou_matrix
from .tracking_metrics import TrackBox

# An occlusion must last at least this many frames to count as a recovery test.
# Shorter gaps are what the Kalman filter and track_buffer already handle; the
# gallery is being scored on the cases that defeat them.
DEFAULT_MIN_GAP = 15          # ~0.5 s at 30 fps
MATCH_IOU = 0.5


# ---------------------------------------------------------------------------
# Query / gallery protocol
# ---------------------------------------------------------------------------


@dataclass
class ReidMetrics:
    rank1: float = 0.0
    rank5: float = 0.0
    map: float = 0.0
    cmc: list[float] = field(default_factory=list)
    n_queries: int = 0
    n_gallery: int = 0
    n_identities: int = 0
    post_occlusion_recovery: float | None = None
    occlusion_events: int = 0
    occlusion_recovered: int = 0
    cross_camera_rate: float | None = None
    cross_camera_events: int = 0
    cross_camera_matched: int = 0

    def headline(self) -> dict[str, float | None]:
        return {
            "M14": self.rank1,
            "M15": self.rank5,
            "M16": self.map,
            "M17": self.post_occlusion_recovery,
            "M18": self.cross_camera_rate,
        }

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def cmc_and_map(
    query_embeddings: Sequence[np.ndarray],
    query_ids: Sequence[int],
    gallery_embeddings: Sequence[np.ndarray],
    gallery_ids: Sequence[int],
    query_cams: Sequence[str] | None = None,
    gallery_cams: Sequence[str] | None = None,
    max_rank: int = 10,
) -> ReidMetrics:
    """Standard ReID evaluation: CMC curve, Rank-k, and mean average precision.

    Following the Market-1501 protocol, a gallery item with the **same identity
    and the same camera** as the query is excluded -- otherwise the easiest
    possible match (the same object, one frame later, same viewpoint) inflates
    Rank-1 towards 1.0 and the metric stops measuring re-identification.
    """
    m = ReidMetrics(
        n_queries=len(query_embeddings),
        n_gallery=len(gallery_embeddings),
        n_identities=len(set(query_ids) | set(gallery_ids)),
    )
    if not len(query_embeddings) or not len(gallery_embeddings):
        return m

    q = np.stack(query_embeddings).astype(np.float32)
    g = np.stack(gallery_embeddings).astype(np.float32)
    q /= np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    g /= np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-12, None)
    distances = 1.0 - (q @ g.T)

    gallery_ids = np.asarray(gallery_ids)
    query_ids = np.asarray(query_ids)
    max_rank = min(max_rank, len(gallery_embeddings))

    cmc_hits = np.zeros(max_rank)
    aps: list[float] = []
    valid = 0

    for i in range(len(q)):
        keep = np.ones(len(g), dtype=bool)
        if query_cams is not None and gallery_cams is not None:
            same_id = gallery_ids == query_ids[i]
            same_cam = np.asarray(gallery_cams) == query_cams[i]
            keep = ~(same_id & same_cam)
        if not keep.any():
            continue

        order = np.argsort(distances[i][keep])
        matches = (gallery_ids[keep][order] == query_ids[i]).astype(int)
        if not matches.any():
            # No correct answer exists in the gallery; scoring this query would
            # penalise the model for the protocol's gap, not for a mistake.
            continue

        valid += 1
        first = int(np.argmax(matches))
        if first < max_rank:
            cmc_hits[first:] += 1

        cum = np.cumsum(matches)
        precision_at_hit = cum[matches == 1] / (np.flatnonzero(matches) + 1)
        aps.append(float(np.mean(precision_at_hit)))

    if valid:
        m.cmc = [round(float(v / valid), 4) for v in cmc_hits]
        m.rank1 = m.cmc[0]
        m.rank5 = m.cmc[min(4, len(m.cmc) - 1)]
        m.map = round(float(np.mean(aps)), 4)
    m.n_queries = valid
    return m


# ---------------------------------------------------------------------------
# M17: post-occlusion recovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcclusionEvent:
    gt_track_id: int
    before_frame: int      # last frame visible before the occlusion
    after_frame: int       # first frame visible again
    gap: int


def find_occlusion_events(
    gt_rows: Sequence[MotRow], min_gap: int = DEFAULT_MIN_GAP
) -> list[OcclusionEvent]:
    """Locate genuine full occlusions from the ground truth's visibility column."""
    by_track: dict[int, list[MotRow]] = defaultdict(list)
    for r in gt_rows:
        by_track[r.track_id].append(r)

    events: list[OcclusionEvent] = []
    for track_id, rows in by_track.items():
        rows.sort(key=lambda r: r.frame)
        visible = [r.frame for r in rows if r.visibility >= IGNORE_VISIBILITY]
        if len(visible) < 2:
            continue
        visible_set = set(visible)
        run_start = None
        for frame in range(min(visible), max(visible) + 1):
            if frame in visible_set:
                if run_start is not None:
                    gap = frame - run_start
                    if gap >= min_gap:
                        events.append(OcclusionEvent(track_id, run_start - 1, frame, gap))
                    run_start = None
            elif run_start is None:
                run_start = frame
    return events


def post_occlusion_recovery(
    gt_rows: Sequence[MotRow],
    predictions: Sequence[TrackBox],
    min_gap: int = DEFAULT_MIN_GAP,
    iou_thr: float = MATCH_IOU,
) -> tuple[float | None, int, int, list[dict[str, Any]]]:
    """M17: fraction of occlusion events where the ORIGINAL id came back.

    For each event, find which predicted id was on the object immediately
    before it disappeared and which is on it when it returns. Recovery means
    those two ids are the same -- which is precisely the failure the PRD opens
    with, measured rather than asserted.
    """
    events = find_occlusion_events(gt_rows, min_gap)
    if not events:
        return None, 0, 0, []

    gt_by_frame: dict[int, list[MotRow]] = defaultdict(list)
    for r in gt_rows:
        gt_by_frame[r.frame].append(r)
    pred_by_frame: dict[int, list[TrackBox]] = defaultdict(list)
    for p in predictions:
        pred_by_frame[p.frame].append(p)

    def id_on_object(frame: int, gt_track_id: int) -> int | None:
        gts = [r for r in gt_by_frame.get(frame, []) if r.track_id == gt_track_id]
        preds = pred_by_frame.get(frame, [])
        if not gts or not preds:
            return None
        m = iou_matrix([p.xyxy for p in preds], [gts[0].xyxy])
        best = int(np.argmax(m[:, 0]))
        return preds[best].track_id if m[best, 0] >= iou_thr else None

    recovered = 0
    detail: list[dict[str, Any]] = []
    scored = 0
    for ev in events:
        before = id_on_object(ev.before_frame, ev.gt_track_id)
        after = id_on_object(ev.after_frame, ev.gt_track_id)
        if before is None or after is None:
            # The object was not detected on one side of the gap, so this is a
            # detection failure, not an identity failure. Counted separately
            # rather than blamed on ReID.
            detail.append({"gt_track": ev.gt_track_id, "gap": ev.gap,
                           "outcome": "not_scorable", "before": before, "after": after})
            continue
        scored += 1
        ok = before == after
        recovered += int(ok)
        detail.append({"gt_track": ev.gt_track_id, "gap": ev.gap,
                       "outcome": "recovered" if ok else "id_switch",
                       "before": before, "after": after})

    rate = round(recovered / scored, 4) if scored else None
    return rate, scored, recovered, detail


# ---------------------------------------------------------------------------
# M18: cross-camera match rate
# ---------------------------------------------------------------------------


def cross_camera_match_rate(
    embeddings_a: Mapping[int, np.ndarray],
    embeddings_b: Mapping[int, np.ndarray],
    identity_map: Mapping[int, int],
    threshold: float,
    class_of: Mapping[int, int] | None = None,
) -> tuple[float | None, int, int, list[dict[str, Any]]]:
    """M18: fraction of camera-A identities correctly matched in camera B.

    `identity_map` gives the ground-truth correspondence `a_track_id ->
    b_track_id`. A match counts only when the nearest camera-B embedding is the
    right one *and* is inside `threshold` -- a nearest neighbour beyond the
    calibrated threshold would be rejected at runtime, so counting it here
    would report a capability the system does not have.
    """
    if not embeddings_a or not embeddings_b:
        return None, 0, 0, []

    b_ids = list(embeddings_b)
    b_mat = np.stack([embeddings_b[i] for i in b_ids]).astype(np.float32)
    b_mat /= np.clip(np.linalg.norm(b_mat, axis=1, keepdims=True), 1e-12, None)

    matched = 0
    scored = 0
    detail: list[dict[str, Any]] = []
    for a_id, emb in embeddings_a.items():
        truth = identity_map.get(a_id)
        if truth is None or truth not in embeddings_b:
            continue
        scored += 1
        q = np.asarray(emb, dtype=np.float32).reshape(-1)
        q /= max(1e-12, float(np.linalg.norm(q)))
        distances = 1.0 - (b_mat @ q)
        if class_of is not None:
            for j, bid in enumerate(b_ids):
                if class_of.get(bid) != class_of.get(a_id):
                    distances[j] = np.inf
        best = int(np.argmin(distances))
        pred = b_ids[best]
        ok = pred == truth and distances[best] < threshold
        matched += int(ok)
        detail.append({
            "cam_a_track": a_id, "expected_cam_b_track": truth, "predicted": pred,
            "distance": round(float(distances[best]), 4), "matched": bool(ok),
        })

    rate = round(matched / scored, 4) if scored else None
    return rate, scored, matched, detail
