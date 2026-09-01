"""ReID gallery and re-association -- **the D1 differentiator**.

PRD 9.4. This is the layer that separates true ReID from frame-to-frame
tracking, and the PRD is explicit about why it is a separate component rather
than a tracker setting:

> "an internal tracker's appearance memory is short-lived. An explicit gallery
>  is what enables *long* occlusion recovery and *cross-camera* matching -- the
>  capabilities the brief names."

The mechanism, and the four guards that keep it from doing more harm than good:

1. **Class gating.** A car is never re-identified as a person. Free accuracy:
   it removes the largest and most embarrassing category of false match.
2. **A calibrated threshold.** `tau_reid` is swept over annotated occlusion
   events and chosen to maximise recovery F1 (`calibrate.py`), never guessed.
   A too-loose threshold merges two different objects into one identity, which
   is worse than issuing a new id.
3. **TTL expiry.** Entries older than `gallery_ttl` frames are dropped, which
   bounds memory and, more importantly, prevents an object that genuinely left
   the scene ten seconds ago from stealing a new arrival's identity.
4. **EMA-smoothed embeddings.** Each entry stores a running average, so one
   motion-blurred frame cannot poison an identity.

Cross-camera matching (M18) is the *same* mechanism with a `camera_id` field
and a wider temporal window -- which is exactly why the design generalises from
occlusion recovery to camera hand-off with no new machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..config import Config
from ..utils.logging import get_logger

log = get_logger("reid.gallery")


@dataclass
class GalleryEntry:
    track_id: int
    cls_id: int
    embedding: np.ndarray          # L2-normalised, EMA-smoothed
    last_seen_frame: int
    camera_id: str = "cam0"
    hits: int = 1
    first_seen_frame: int = 0

    def update(self, embedding: np.ndarray, frame: int, alpha: float) -> None:
        blended = alpha * self.embedding + (1.0 - alpha) * embedding
        norm = float(np.linalg.norm(blended))
        self.embedding = blended / norm if norm > 1e-12 else blended
        self.last_seen_frame = frame
        self.hits += 1


@dataclass
class MatchOutcome:
    matched: bool
    track_id: int | None = None
    distance: float = 1.0
    camera_id: str | None = None
    reason: str = ""


@dataclass
class GalleryStats:
    queries: int = 0
    restorations: int = 0
    rejected_distance: int = 0
    rejected_class: int = 0
    expired: int = 0
    cross_camera_matches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ReidGallery:
    """Holds embeddings for LOST tracks and, in cross-camera mode, other views."""

    def __init__(self, cfg: Config, threshold: float | None = None):
        g = cfg.reid.gallery
        self.threshold = float(threshold if threshold is not None else g.threshold)
        self.ttl = int(g.ttl_frames)
        self.class_gated = bool(g.class_gated)
        self.alpha = float(cfg.reid.embedding_ema)
        self.entries: dict[tuple[str, int], GalleryEntry] = {}
        self.stats = GalleryStats()

    # -- maintenance -------------------------------------------------------
    def add(self, track_id: int, cls_id: int, embedding: np.ndarray, frame: int,
            camera_id: str = "cam0") -> None:
        if embedding is None:
            return
        key = (camera_id, track_id)
        entry = self.entries.get(key)
        if entry is None:
            self.entries[key] = GalleryEntry(
                track_id=track_id, cls_id=cls_id, embedding=np.asarray(embedding, dtype=np.float32),
                last_seen_frame=frame, camera_id=camera_id, first_seen_frame=frame,
            )
        else:
            entry.update(np.asarray(embedding, dtype=np.float32), frame, self.alpha)
            entry.cls_id = cls_id

    def expire(self, frame: int) -> int:
        """Drop entries older than the TTL. Returns how many were removed."""
        stale = [k for k, e in self.entries.items() if frame - e.last_seen_frame > self.ttl]
        for k in stale:
            del self.entries[k]
        self.stats.expired += len(stale)
        return len(stale)

    def remove(self, track_id: int, camera_id: str = "cam0") -> None:
        self.entries.pop((camera_id, track_id), None)

    # -- query -------------------------------------------------------------
    def query(
        self,
        embedding: np.ndarray | None,
        cls_id: int,
        frame: int,
        camera_id: str = "cam0",
        exclude: Iterable[int] = (),
        cross_camera: bool = False,
    ) -> MatchOutcome:
        """Find the closest compatible gallery identity, or report why not."""
        self.stats.queries += 1
        if embedding is None or not self.entries:
            return MatchOutcome(False, reason="no_embedding" if embedding is None else "empty_gallery")

        exclude = set(exclude)
        candidates = [
            e for e in self.entries.values()
            if e.track_id not in exclude
            and (cross_camera or e.camera_id == camera_id)
            and frame - e.last_seen_frame <= self.ttl
        ]
        if not candidates:
            return MatchOutcome(False, reason="no_candidates")

        if self.class_gated:
            gated = [e for e in candidates if e.cls_id == cls_id]
            if len(gated) < len(candidates):
                self.stats.rejected_class += len(candidates) - len(gated)
            candidates = gated
        if not candidates:
            return MatchOutcome(False, reason="class_gated")

        mat = np.stack([e.embedding for e in candidates])
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        q = q / max(1e-12, float(np.linalg.norm(q)))
        distances = 1.0 - (mat @ q)
        best = int(np.argmin(distances))
        d = float(distances[best])

        if d >= self.threshold:
            self.stats.rejected_distance += 1
            return MatchOutcome(False, distance=d, reason="above_threshold")

        entry = candidates[best]
        self.stats.restorations += 1
        if entry.camera_id != camera_id:
            self.stats.cross_camera_matches += 1
        return MatchOutcome(
            True, track_id=entry.track_id, distance=d,
            camera_id=entry.camera_id, reason="restored",
        )

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self) -> dict[str, Any]:
        return {
            "tau_reid": self.threshold,
            "gallery_ttl_frames": self.ttl,
            "class_gated": self.class_gated,
            "embedding_ema": self.alpha,
            "entries": len(self.entries),
            **self.stats.as_dict(),
        }


def rank_gallery(
    query: np.ndarray,
    gallery_embeddings: Sequence[np.ndarray],
    gallery_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rank a gallery by cosine distance. Returns (sorted ids, sorted distances).

    Used by the Rank-k / CMC protocol in `eval/reid_metrics.py` rather than by
    the live pipeline, which needs only the single best match.
    """
    if not len(gallery_embeddings):
        return np.array([], dtype=int), np.array([])
    mat = np.stack(gallery_embeddings)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    q = q / max(1e-12, float(np.linalg.norm(q)))
    distances = 1.0 - (mat @ q)
    order = np.argsort(distances)
    return np.asarray(gallery_ids)[order], distances[order]
