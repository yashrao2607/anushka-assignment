"""Frame sampler -- raw video in, attributed frames + manifest out.

PRD 7.1: "sample at ~2 fps from source video, never every frame. Consecutive
frames are ~99% redundant; annotating them wastes labelling budget and
inflates apparent dataset size without adding information."

Two decisions worth stating:

1. **Sampling is by frame index, not by wall-clock seek.** `cap.set(POS_FRAMES)`
   is unreliable on many codecs (it seeks to the nearest keyframe), which would
   silently produce duplicate or missing frames. Reading sequentially and
   keeping every Nth frame is slower but exact, and this runs once.

2. **A frame dropped for blur is logged, not silently discarded** (Principle 6,
   "fail visibly"). The count of dropped frames is reported, so a video that is
   90% unusable is discovered now rather than after it has skewed a metric.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from ..config import Config
from ..utils.logging import get_logger
from .attributes import compute_attributes
from .manifest import ManifestRow

log = get_logger("data.sampler")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")


@dataclass
class VideoSampleResult:
    source_video: str
    scene_id: str
    fps: float
    total_frames: int
    kept: int
    dropped_blur: int
    unreadable: int
    rows: list[ManifestRow]

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("rows")
        return d


def scene_id_for(video_path: Path) -> str:
    """A scene identifier derived from the filename stem.

    Convention: `<scene>_<camera>_<take>.mp4` -> scene = the first token.
    Frames from the same physical scene must never be split apart (PRD 7.3),
    and two takes of the same junction from two angles are the *same* scene
    even though they are different files -- which is exactly the case the
    cross-camera clip creates.
    """
    stem = video_path.stem
    return stem.split("_")[0] if "_" in stem else stem


def find_videos(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
    )


def _image_id(video_stem: str, frame_no: int) -> str:
    return f"{video_stem}_f{frame_no:06d}"


def sample_video(video_path: Path, cfg: Config, force: bool = False) -> VideoSampleResult:
    """Sample one video to disk at ~`sampling.target_fps`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # A malformed or variable-frame-rate file reports fps 0 or something absurd.
    # Assume 30 and say so, rather than dividing by zero three lines later.
    if not (1.0 <= src_fps <= 240.0):
        log.warning(
            "implausible source fps; assuming 30",
            extra={"event": "fps_fallback", "video": video_path.name, "reported_fps": src_fps},
        )
        src_fps = 30.0

    stride = max(1, round(src_fps / cfg.sampling.target_fps))
    out_dir = cfg.path("frames_dir") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    kept = dropped_blur = unreadable = 0
    prev_gray = None
    frame_no = -1
    scene = scene_id_for(video_path)

    while kept < cfg.sampling.max_frames_per_video:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % stride:
            continue
        if frame is None or frame.size == 0:
            unreadable += 1
            continue

        attrs, gray = compute_attributes(frame, cfg.attributes, prev_gray)
        prev_gray = gray

        if attrs.blur_score < cfg.sampling.min_blur_score:
            dropped_blur += 1
            log.debug(
                "frame dropped: unusably blurred",
                extra={
                    "event": "frame_dropped_blur",
                    "video": video_path.name,
                    "frame_no": frame_no,
                    "blur_score": attrs.blur_score,
                },
            )
            continue

        image_id = _image_id(video_path.stem, frame_no)
        image_path = out_dir / f"{image_id}{cfg.sampling.image_format}"
        if force or not image_path.exists():
            cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), cfg.sampling.jpeg_quality],
            )

        h, w = frame.shape[:2]
        rows.append(
            ManifestRow(
                image_id=image_id,
                source_video=video_path.name,
                scene_id=scene,
                frame_no=frame_no,
                timestamp_s=round(frame_no / src_fps, 3),
                width=w,
                height=h,
                lighting=attrs.lighting,
                blur_score=attrs.blur_score,
                blur_level=attrs.blur_level,
                brightness=attrs.brightness,
                contrast=attrs.contrast,
                motion_score=attrs.motion_score,
                image_path=str(image_path.relative_to(cfg.root).as_posix()),
                label_path=str(
                    (cfg.path("labels_dir") / video_path.stem / f"{image_id}.txt")
                    .relative_to(cfg.root)
                    .as_posix()
                ),
            )
        )
        kept += 1

    cap.release()
    return VideoSampleResult(
        source_video=video_path.name,
        scene_id=scene,
        fps=src_fps,
        total_frames=total,
        kept=kept,
        dropped_blur=dropped_blur,
        unreadable=unreadable,
        rows=rows,
    )


def dataset_hash(rows: Iterable[ManifestRow]) -> str:
    """Stable hash over image ids + split assignment.

    Recorded with every run (PRD 9.7) so a metric can be tied to the exact
    dataset version that produced it -- the difference between "mAP went up"
    and "mAP went up *and* the test set did not change".
    """
    h = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.image_id):
        h.update(f"{row.image_id}|{row.split}|{row.n_objects}".encode())
    return h.hexdigest()[:12]
