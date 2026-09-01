"""Track records, lifecycle, and the tracker interface.

PRD 9.3 track lifecycle:

    NEW --(min_hits=3 consecutive matches)--> TRACKED
    TRACKED --(unmatched)--> LOST
    LOST --(track_buffer frames unmatched)--> REMOVED

`min_hits` suppresses flicker from spurious detections; `track_buffer` is what
lets an identity survive an occlusion, and is tuned empirically in EXP-9
because too short loses identities through occlusion while too long causes ID
reuse on genuinely departed objects.

Every tracker in this package returns the same `Track` objects, so the metrics
harness, the renderer and the ReID gallery are written once and the three
trackers are compared on identical plumbing (US-3.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..eval.detection_metrics import PredBox
from .kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah


class TrackState:
    NEW = "new"
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class Track:
    """One object identity. Field names mirror PRD 11.1."""

    track_id: int
    cls_id: int
    xyxy: tuple[float, float, float, float]
    conf: float
    state: str = TrackState.NEW
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    smoothed_embedding: np.ndarray | None = None
    history: list[tuple[float, float]] = field(default_factory=list)
    hits: int = 0
    age: int = 0
    frames_since_update: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    camera_id: str = "cam0"
    reid_restored: bool = False

    @property
    def is_active(self) -> bool:
        return self.state == TrackState.TRACKED

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_row(self, frame: int, class_names: Sequence[str]) -> dict[str, Any]:
        """One row of `results.csv` (PRD 11.2)."""
        x1, y1, x2, y2 = self.xyxy
        return {
            "frame": frame,
            "track_id": self.track_id,
            "class": class_names[self.cls_id] if 0 <= self.cls_id < len(class_names) else self.cls_id,
            "conf": round(self.conf, 4),
            "x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2),
            "reid_restored": int(self.reid_restored),
            "camera_id": self.camera_id,
        }


class KalmanTrack(Track):
    """A `Track` whose position is maintained by a Kalman filter."""

    _kf = KalmanFilterXYAH()

    def initiate(self) -> None:
        self.mean, self.covariance = self._kf.initiate(xyxy_to_xyah(self.xyxy))
        self.history.append(self.centroid)

    def predict(self) -> None:
        if self.mean is None:
            self.initiate()
            return
        mean = self.mean.copy()
        # A track that is not currently observed has no reliable height
        # velocity; letting it run makes a lost box inflate or collapse over a
        # long occlusion until it can never re-match.
        if self.state != TrackState.TRACKED:
            mean[7] = 0.0
        self.mean, self.covariance = self._kf.predict(mean, self.covariance)
        self.xyxy = xyah_to_xyxy(self.mean)

    def update_from(self, det: PredBox, frame: int) -> None:
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, xyxy_to_xyah(det.xyxy)
        )
        self.xyxy = xyah_to_xyxy(self.mean)
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.hits += 1
        self.frames_since_update = 0
        self.last_seen_frame = frame
        self.history.append(self.centroid)
        if len(self.history) > 120:
            del self.history[:-120]

    def apply_warp(self, warp: np.ndarray) -> None:
        """Camera-motion compensation: warp the predicted state (PRD 9.3)."""
        if self.mean is None:
            return
        from .gmc import apply_warp_to_xyxy

        self.xyxy = apply_warp_to_xyxy(self.xyxy, warp)
        xyah = xyxy_to_xyah(self.xyxy)
        self.mean[:4] = xyah
        # Rotate the velocity too, or the filter spends the next few frames
        # unlearning a motion that was the camera's, not the object's.
        r = warp[:2, :2]
        self.mean[4:6] = r @ self.mean[4:6]


class BaseTracker:
    """Shared lifecycle bookkeeping. Subclasses implement `associate`."""

    name = "base"

    def __init__(self, cfg, camera_id: str = "cam0"):
        self.cfg = cfg
        self.t = cfg.tracking
        self.camera_id = camera_id
        self.frame_id = 0
        self._next_id = 1
        self.tracked: list[KalmanTrack] = []
        self.lost: list[KalmanTrack] = []
        self.removed: list[KalmanTrack] = []
        self.id_switch_hint = 0

    def reset(self) -> None:
        self.frame_id = 0
        self._next_id = 1
        self.tracked.clear()
        self.lost.clear()
        self.removed.clear()

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _birth(self, det: PredBox) -> KalmanTrack:
        track = KalmanTrack(
            track_id=self.new_id(),
            cls_id=det.cls_id,
            xyxy=det.xyxy,
            conf=det.conf,
            state=TrackState.NEW,
            hits=1,
            first_seen_frame=self.frame_id,
            last_seen_frame=self.frame_id,
            camera_id=self.camera_id,
        )
        track.initiate()
        # min_hits == 1 means "display immediately"; anything higher holds the
        # track back until it has proved itself, which is what suppresses
        # single-frame detector noise from becoming a permanent identity.
        if self.t.min_hits <= 1:
            track.state = TrackState.TRACKED
        return track

    def _age_and_retire(self) -> None:
        for track in list(self.lost):
            track.age += 1
            track.frames_since_update += 1
            if track.frames_since_update > self.t.track_buffer:
                track.state = TrackState.REMOVED
                self.lost.remove(track)
                self.removed.append(track)

    def update(self, detections: Sequence[PredBox], frame=None) -> list[KalmanTrack]:
        raise NotImplementedError

    def active_tracks(self) -> list[KalmanTrack]:
        return [t for t in self.tracked if t.is_active]


def split_by_confidence(
    detections: Iterable[PredBox], high: float, low: float
) -> tuple[list[PredBox], list[PredBox]]:
    """ByteTrack's two-tier split.

    PRD 9.3: "match high-confidence detections first, then use the remaining
    *low*-confidence detections to rescue tracks that would otherwise die. This
    is the single most effective trick against occlusion-driven fragmentation,
    because a partially occluded object is precisely a low-confidence detection."
    """
    hi: list[PredBox] = []
    lo: list[PredBox] = []
    for d in detections:
        if d.conf >= high:
            hi.append(d)
        elif d.conf >= low:
            lo.append(d)
    return hi, lo
