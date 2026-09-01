"""Cross-camera re-identification (M18) -- the camera hand-off.

PRD US-4.2 and 9.4. The key design claim being demonstrated here is that
**cross-camera matching needs no new machinery**: it is the same gallery, the
same cosine distance, the same class gate and the same calibrated threshold as
post-occlusion recovery -- with a `camera_id` field and a wider temporal
window. If the design were right, camera hand-off should fall out of it; this
module is the test of that claim.

Each camera is tracked independently, producing per-identity embeddings
(EMA-smoothed over that identity's lifetime, so a single frame does not decide
a hand-off). Camera A's identities are then queried against camera B's gallery.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..data.mot import IGNORE_VISIBILITY, read_mot
from ..eval.reid_metrics import cross_camera_match_rate
from ..utils.logging import get_logger

log = get_logger("pipeline.cross_camera")


@dataclass
class CameraTracks:
    camera_id: str
    video: str
    embeddings: dict[int, np.ndarray] = field(default_factory=dict)   # gt track id -> embedding
    classes: dict[int, int] = field(default_factory=dict)
    frames_seen: dict[int, int] = field(default_factory=dict)


@dataclass
class CrossCameraResult:
    cam_a: str = ""
    cam_b: str = ""
    match_rate: float | None = None
    scored: int = 0
    matched: int = 0
    threshold: float = 0.35
    backend: str = ""
    detail: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cam_a": self.cam_a, "cam_b": self.cam_b,
            "M18_cross_camera_match_rate": self.match_rate,
            "identities_scored": self.scored, "identities_matched": self.matched,
            "tau_reid": self.threshold, "reid_backend": self.backend,
        }


def build_camera_gallery(
    cfg: Config,
    video: Path,
    camera_id: str,
    extractor,
    stride: int = 15,
    max_samples_per_id: int = 12,
) -> CameraTracks:
    """Embed each ground-truth identity across a video into one smoothed vector.

    Sampling every `stride` frames rather than every frame is deliberate: 450
    near-identical crops of the same object add almost no information to the
    average while costing 450 forward passes. The EMA over ~12 well-spread
    samples is what makes the descriptor robust to a single bad viewpoint.
    """
    gt_file = cfg.root / "data" / "gt" / f"{video.stem}_gt.txt"
    if not gt_file.exists():
        raise FileNotFoundError(f"no ground truth for {video.name}: {gt_file}")

    rows = read_mot(gt_file)
    by_frame: dict[int, list] = defaultdict(list)
    for r in rows:
        if r.visibility >= IGNORE_VISIBILITY:
            by_frame[r.frame].append(r)

    out = CameraTracks(camera_id=camera_id, video=video.name)
    alpha = cfg.reid.embedding_ema
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")

    try:
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % stride:
                continue
            visible = [
                r for r in by_frame.get(frame_no, [])
                if out.frames_seen.get(r.track_id, 0) < max_samples_per_id
            ]
            if not visible:
                continue
            feats = extractor.extract(frame, [r.xyxy for r in visible])
            for idx, emb in feats.items():
                r = visible[idx]
                prev = out.embeddings.get(r.track_id)
                if prev is None:
                    out.embeddings[r.track_id] = np.asarray(emb, dtype=np.float32)
                else:
                    blended = alpha * prev + (1 - alpha) * emb
                    norm = float(np.linalg.norm(blended))
                    out.embeddings[r.track_id] = blended / norm if norm > 1e-12 else blended
                out.classes[r.track_id] = r.cls_id
                out.frames_seen[r.track_id] = out.frames_seen.get(r.track_id, 0) + 1
    finally:
        cap.release()
    return out


def run_cross_camera(
    cfg: Config,
    cam_a: Path,
    cam_b: Path,
    extractor=None,
    threshold: float | None = None,
    identity_map: dict[int, int] | None = None,
) -> CrossCameraResult:
    """Match every identity in camera A against camera B's gallery."""
    if extractor is None:
        from ..reid.extractor import ReidExtractor

        extractor = ReidExtractor(cfg)

    tau = float(threshold if threshold is not None else cfg.reid.gallery.threshold)
    a = build_camera_gallery(cfg, cam_a, "camA", extractor)
    b = build_camera_gallery(cfg, cam_b, "camB", extractor)

    # The two views of one scene share ground-truth track ids by construction,
    # so the correspondence is the identity map unless one is supplied. Stated
    # explicitly rather than assumed, because on real two-camera footage this
    # mapping is annotation work and the evaluation depends entirely on it.
    mapping = identity_map or {tid: tid for tid in a.embeddings if tid in b.embeddings}

    rate, scored, matched, detail = cross_camera_match_rate(
        a.embeddings, b.embeddings, mapping, tau,
        class_of={**a.classes, **b.classes},
    )
    log.info(
        f"cross-camera {cam_a.name} -> {cam_b.name}: {matched}/{scored} identities matched "
        f"(rate {rate}) at tau {tau}, backend {extractor.describe().get('reid_backend')}",
        extra={"event": "cross_camera", "rate": rate, "scored": scored},
    )
    return CrossCameraResult(
        cam_a=cam_a.name, cam_b=cam_b.name, match_rate=rate, scored=scored,
        matched=matched, threshold=tau,
        backend=str(extractor.describe().get("reid_backend", "")),
        detail=detail,
    )


def find_camera_pairs(cfg: Config) -> list[tuple[Path, Path]]:
    """Locate two-view scenes by the `<scene>_<camera>.mp4` naming convention."""
    videos = sorted(cfg.path("raw_videos_dir").glob("*.mp4"))
    by_scene: dict[str, list[Path]] = defaultdict(list)
    for v in videos:
        scene = v.stem.split("_")[0] if "_" in v.stem else v.stem
        by_scene[scene].append(v)
    return [(v[0], v[1]) for v in by_scene.values() if len(v) >= 2]
