"""The three trackers, in ablation order (PRD 13.2 rows B1-B5, US-3.3).

    IouTracker   -- greedy IoU on raw detections. No Kalman, no second stage,
                    no appearance. This is the speed floor and the honest
                    baseline the other two must beat.
    ByteTracker  -- Kalman motion + BYTE's two-stage association.
    BotSortTracker -- ByteTrack + camera-motion compensation + appearance
                    embeddings fused into the association cost.

They share `BaseTracker`'s lifecycle and emit identical `Track` records, so
EXP-3 compares them **on identical detections** -- the only comparison that
means anything.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..eval.detection_metrics import PredBox
from .base import BaseTracker, KalmanTrack, Track, TrackState, split_by_confidence
from .gmc import GMC
from .kalman import xyxy_to_xyah
from .matching import (
    INF_COST,
    class_gate,
    cosine_distance,
    fuse_cost,
    iou_distance,
    linear_assignment,
)


# ---------------------------------------------------------------------------
# B1: the baseline
# ---------------------------------------------------------------------------


class IouTracker(BaseTracker):
    """Greedy IoU association with no motion model.

    Deliberately naive. It has no way to predict where an occluded object went,
    so every occlusion costs an identity -- which is precisely the number the
    other two trackers have to improve on.
    """

    name = "iou"

    def update(self, detections: Sequence[PredBox], frame=None) -> list[KalmanTrack]:
        self.frame_id += 1
        dets = [d for d in detections if d.conf >= self.t.track_high_thresh]

        cost = iou_distance([t.xyxy for t in self.tracked], [d.xyxy for d in dets])
        cost = class_gate(cost, [t.cls_id for t in self.tracked], [d.cls_id for d in dets])
        matches, u_tracks, u_dets = linear_assignment(cost, 1.0 - 0.3)

        for ti, di in matches:
            track, det = self.tracked[ti], dets[di]
            track.xyxy = det.xyxy
            track.conf = det.conf
            track.hits += 1
            track.frames_since_update = 0
            track.last_seen_frame = self.frame_id
            track.state = TrackState.TRACKED
            track.history.append(track.centroid)

        still: list[KalmanTrack] = [self.tracked[ti] for ti, _ in matches]
        for ti in u_tracks:
            track = self.tracked[ti]
            track.frames_since_update += 1
            # No coasting: the baseline drops a track the moment it is missed.
            if track.frames_since_update <= 1:
                track.state = TrackState.LOST
                self.removed.append(track)

        for di in u_dets:
            track = self._birth(dets[di])
            track.state = TrackState.TRACKED
            still.append(track)

        self.tracked = still
        return self.active_tracks()


# ---------------------------------------------------------------------------
# B3: ByteTrack
# ---------------------------------------------------------------------------


class ByteTracker(BaseTracker):
    """Kalman motion model plus BYTE two-stage association."""

    name = "bytetrack"
    use_gmc = False
    use_appearance = False

    def __init__(self, cfg, camera_id: str = "cam0"):
        super().__init__(cfg, camera_id)
        self.gmc = GMC(cfg.tracking.gmc_method) if self.use_gmc else None

    # -- cost hook, overridden by BoT-SORT ---------------------------------
    def _cost(self, tracks: Sequence[KalmanTrack], dets: Sequence[PredBox]) -> np.ndarray:
        cost = iou_distance([t.xyxy for t in tracks], [d.xyxy for d in dets])
        return class_gate(cost, [t.cls_id for t in tracks], [d.cls_id for d in dets])

    def _post_match(self, track: KalmanTrack, det: PredBox) -> None:
        return None

    def update(self, detections: Sequence[PredBox], frame=None) -> list[KalmanTrack]:
        self.frame_id += 1
        hi, lo = split_by_confidence(detections, self.t.track_high_thresh, self.t.track_low_thresh)

        pool: list[KalmanTrack] = self.tracked + self.lost
        for track in pool:
            track.predict()

        if self.gmc is not None and frame is not None:
            warp = self.gmc.apply(frame)
            if not np.allclose(warp, np.eye(2, 3), atol=1e-6):
                for track in pool:
                    track.apply_warp(warp)

        # -- stage 1: high-confidence detections ---------------------------
        cost = self._cost(pool, hi)
        matches, u_tracks, u_hi = linear_assignment(cost, self.t.match_thresh)
        activated: list[KalmanTrack] = []
        for ti, di in matches:
            track = pool[ti]
            track.update_from(hi[di], self.frame_id)
            self._post_match(track, hi[di])
            track.state = (
                TrackState.TRACKED if track.hits >= self.t.min_hits else TrackState.NEW
            )
            activated.append(track)

        remaining = [pool[i] for i in u_tracks]

        # -- stage 2: rescue with LOW-confidence detections -----------------
        # Only tracks that were confidently alive last frame are eligible.
        # Letting a long-lost track re-attach to a weak detection is how
        # ByteTrack's rescue stage turns into an ID-switch generator.
        rescue_pool = [t for t in remaining if t.state == TrackState.TRACKED]
        cost2 = iou_distance([t.xyxy for t in rescue_pool], [d.xyxy for d in lo])
        cost2 = class_gate(cost2, [t.cls_id for t in rescue_pool], [d.cls_id for d in lo])
        matches2, u_tracks2, _ = linear_assignment(cost2, 0.5)
        for ti, di in matches2:
            track = rescue_pool[ti]
            track.update_from(lo[di], self.frame_id)
            track.state = TrackState.TRACKED
            activated.append(track)

        unmatched = [rescue_pool[i] for i in u_tracks2] + [
            t for t in remaining if t.state != TrackState.TRACKED
        ]
        for track in unmatched:
            if track.state != TrackState.LOST:
                track.state = TrackState.LOST
                track.frames_since_update = 1
            if track not in self.lost:
                self.lost.append(track)

        # -- birth ----------------------------------------------------------
        for di in u_hi:
            det = hi[di]
            if det.conf >= self.t.new_track_thresh:
                track = self._birth(det)
                self._post_match(track, det)
                activated.append(track)

        self.tracked = [t for t in activated if t.state in (TrackState.TRACKED, TrackState.NEW)]
        self.lost = [t for t in self.lost if t not in self.tracked]
        self._age_and_retire()
        return self.active_tracks()


# ---------------------------------------------------------------------------
# B4/B5: BoT-SORT
# ---------------------------------------------------------------------------


class BotSortTracker(ByteTracker):
    """ByteTrack + camera-motion compensation + appearance-fused cost.

    PRD 8.2 on why this is the default: "ByteTrack's low-confidence second
    association stage is excellent, but BoT-SORT adds appearance embeddings +
    camera-motion compensation -- the two things that make identity survive
    occlusion and camera movement."

    `with_reid=False` gives the B4 row (GMC only); `True` gives B5.
    """

    name = "botsort"
    use_gmc = True

    def __init__(self, cfg, camera_id: str = "cam0", with_reid: bool = True, extractor=None):
        self.use_appearance = bool(with_reid)
        super().__init__(cfg, camera_id)
        self.extractor = extractor
        self._frame_embeddings: dict[int, np.ndarray] = {}

    def set_embeddings(self, embeddings: dict[int, np.ndarray]) -> None:
        """Embeddings for this frame's detections, keyed by index into `detections`."""
        self._frame_embeddings = embeddings or {}

    def _embedding_for(self, det_index: int) -> np.ndarray | None:
        return self._frame_embeddings.get(det_index)

    def _cost(self, tracks: Sequence[KalmanTrack], dets: Sequence[PredBox]) -> np.ndarray:
        iou_cost = iou_distance([t.xyxy for t in tracks], [d.xyxy for d in dets])
        iou_cost = class_gate(iou_cost, [t.cls_id for t in tracks], [d.cls_id for d in dets])
        if not self.use_appearance or not self._frame_embeddings or not len(tracks):
            return iou_cost

        track_emb = [t.smoothed_embedding for t in tracks]
        det_emb = [self._embedding_for(i) for i in range(len(dets))]
        if not any(e is not None for e in track_emb) or not any(e is not None for e in det_emb):
            return iou_cost

        dim = next(e.shape[-1] for e in track_emb + det_emb if e is not None)
        # A missing embedding is neutral, not similar: an unusable crop
        # (too small, too blurred) must not fabricate appearance evidence.
        t_mat = np.stack([e if e is not None else np.zeros(dim) for e in track_emb])
        d_mat = np.stack([e if e is not None else np.zeros(dim) for e in det_emb])
        app = cosine_distance(t_mat, d_mat)
        for i, e in enumerate(track_emb):
            if e is None:
                app[i, :] = np.nan
        for j, e in enumerate(det_emb):
            if e is None:
                app[:, j] = np.nan

        fused = fuse_cost(iou_cost, app, self.t.appearance_weight)
        return np.where(iou_cost >= INF_COST, INF_COST, fused)

    def _post_match(self, track: KalmanTrack, det: PredBox) -> None:
        """Exponential-moving-average embedding smoothing (PRD 9.2).

        "Each track keeps an exponential moving average of its embeddings
        rather than only the latest -- a single blurred frame then cannot
        poison the track's identity."
        """
        if not self.use_appearance:
            return
        try:
            idx = self._current_dets.index(det)
        except (AttributeError, ValueError):
            return
        emb = self._embedding_for(idx)
        if emb is None:
            return
        alpha = self.cfg.reid.embedding_ema
        if track.smoothed_embedding is None:
            track.smoothed_embedding = emb.copy()
        else:
            blended = alpha * track.smoothed_embedding + (1.0 - alpha) * emb
            track.smoothed_embedding = blended / max(1e-12, float(np.linalg.norm(blended)))

    def update(self, detections: Sequence[PredBox], frame=None) -> list[KalmanTrack]:
        self._current_dets = list(detections)
        return super().update(detections, frame)


TRACKERS = {"iou": IouTracker, "bytetrack": ByteTracker, "botsort": BotSortTracker}


def build_tracker(cfg, tracker_type: str | None = None, camera_id: str = "cam0", **kw):
    name = (tracker_type or cfg.tracking.tracker_type).lower()
    if name not in TRACKERS:
        raise ValueError(f"unknown tracker {name!r}; choose from {sorted(TRACKERS)}")
    if name == "botsort":
        return BotSortTracker(cfg, camera_id=camera_id, **kw)
    return TRACKERS[name](cfg, camera_id=camera_id)
