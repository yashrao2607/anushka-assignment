"""Run detection + tracking over a video and emit MOT-compatible results.

PRD 11.2: the per-frame output is MOT-challenge compatible "so standard
evaluation tooling (TrackEval) works without a converter". The same records
feed the tracking metrics harness, the renderer, and (Phase 3) the ReID
gallery.

The tracker sees **every** frame, while the dataset was sampled at 2 fps.
That is not an inconsistency: a tracker's whole job is temporal association, so
starving it of 14 out of every 15 frames would measure a different system than
the one that ships. Detection metrics are computed on the sampled, annotated
frames; tracking metrics are computed on the full-rate sequence against the
full-rate ground truth.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..data.mot import MotRow, read_mot
from ..eval.detection_metrics import PredBox
from ..eval.tracking_metrics import TrackBox
from ..tracking.trackers import build_tracker
from ..utils.logging import get_logger
from ..utils.viz import VideoWriter, colour_for, draw_box, draw_hud

log = get_logger("pipeline.track")

RESULT_COLUMNS = (
    "frame", "timestamp", "track_id", "class", "conf",
    "x1", "y1", "x2", "y2", "reid_restored", "camera_id",
)


@dataclass
class TrackVideoResult:
    video: str
    tracker: str
    frames: int = 0
    detections: int = 0
    tracks_created: int = 0
    unique_by_class: dict[str, int] = field(default_factory=dict)
    predictions: list[TrackBox] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)
    extractor: dict[str, Any] = field(default_factory=dict)
    gallery: dict[str, Any] = field(default_factory=dict)
    restorations: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def fps(self) -> float:
        total = self.timing.get("total_ms", 0.0)
        return round(self.frames / (total / 1000.0), 2) if total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "tracker": self.tracker,
            "frames": self.frames,
            "detections": self.detections,
            "unique_tracks": self.tracks_created,
            "unique_by_class": self.unique_by_class,
            "id_restorations": len(self.restorations),
            "fps": self.fps,
            **{k: round(v, 2) for k, v in self.timing.items()},
        }


def track_video(
    cfg: Config,
    video_path: str | Path,
    tracker_type: str | None = None,
    weights: str | Path | None = None,
    save_video: Path | None = None,
    save_csv: Path | None = None,
    max_frames: int | None = None,
    with_reid: bool = False,
    with_gallery: bool = False,
    camera_id: str = "cam0",
    show_trail: bool = True,
    extractor=None,
    gallery=None,
) -> TrackVideoResult:
    from ..models.detector import Detector

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = Detector(cfg, weights)
    name = (tracker_type or cfg.tracking.tracker_type).lower()

    if with_reid and name != "botsort":
        # Only BoT-SORT consumes appearance. Silently ignoring the flag would
        # produce a "with ReID" ablation row that had no ReID in it.
        raise ValueError(f"--reid requires the botsort tracker, not {name!r}")

    if with_reid and extractor is None:
        from ..reid.extractor import ReidExtractor

        extractor = ReidExtractor(cfg)

    tracker = (
        build_tracker(cfg, name, camera_id=camera_id, with_reid=with_reid,
                      with_gallery=with_gallery, gallery=gallery)
        if name == "botsort"
        else build_tracker(cfg, name, camera_id=camera_id)
    )

    label = name + ("+reid" if with_reid else "") + ("+gallery" if with_gallery else "")
    result = TrackVideoResult(video=video_path.name, tracker=label)
    result.extractor = extractor.describe() if extractor is not None else {}
    writer = VideoWriter(save_video, fps=src_fps) if save_video else None
    seen_ids: set[int] = set()
    class_ids: dict[str, set[int]] = {}
    t_det = t_trk = t_emb = 0.0
    t0 = time.perf_counter()
    frame_no = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if max_frames and frame_no > max_frames:
                break
            # frame_skip > 1 is the CPU-fallback path (PRD M20/NFR-2).
            if cfg.detection.frame_skip > 1 and (frame_no - 1) % cfg.detection.frame_skip:
                continue

            td = time.perf_counter()
            dets: Sequence[PredBox] = detector.predict(
                frame, image_id=f"{video_path.stem}_f{frame_no:06d}"
            )
            t_det += time.perf_counter() - td
            result.detections += len(dets)

            if extractor is not None and dets:
                te = time.perf_counter()
                # PRD 9.2: every crop in the frame goes through ONE forward
                # pass. Per-crop calls are the single biggest reason a naive
                # ReID pipeline runs at a fraction of its achievable FPS.
                tracker.set_embeddings(extractor.extract(frame, [d.xyxy for d in dets]))
                t_emb += time.perf_counter() - te

            tt = time.perf_counter()
            tracks = tracker.update(dets, frame=frame)
            t_trk += time.perf_counter() - tt

            for track in tracks:
                seen_ids.add(track.track_id)
                cname = cfg.dataset.classes[track.cls_id]
                class_ids.setdefault(cname, set()).add(track.track_id)
                result.predictions.append(
                    TrackBox(
                        frame=frame_no,
                        track_id=track.track_id,
                        xyxy=tuple(float(v) for v in track.xyxy),
                        cls_id=track.cls_id,
                    )
                )
                row = track.to_row(frame_no, cfg.dataset.classes)
                row["timestamp"] = round(frame_no / src_fps, 3)
                result.rows.append({k: row.get(k, "") for k in RESULT_COLUMNS})

            if writer is not None:
                writer.write(_render(frame, tracks, cfg, frame_no, seen_ids, class_ids, show_trail))

            result.frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    total_ms = (time.perf_counter() - t0) * 1000.0
    result.timing = {
        "total_ms": total_ms,
        "detect_ms": t_det * 1000.0,
        "embed_ms": t_emb * 1000.0,
        "track_ms": t_trk * 1000.0,
        "detect_ms_per_frame": (t_det * 1000.0 / result.frames) if result.frames else 0.0,
        "track_ms_per_frame": (t_trk * 1000.0 / result.frames) if result.frames else 0.0,
        "embed_ms_per_frame": (t_emb * 1000.0 / result.frames) if result.frames else 0.0,
    }
    if getattr(tracker, "gallery", None) is not None:
        result.gallery = tracker.gallery.describe()
        result.restorations = list(getattr(tracker, "restored_ids", []))
    result.tracks_created = len(seen_ids)
    result.unique_by_class = {k: len(v) for k, v in sorted(class_ids.items())}

    if save_csv:
        write_results_csv(save_csv, result.rows)
    return result


def _render(frame, tracks, cfg, frame_no, seen_ids, class_ids, show_trail):
    out = frame.copy()
    for track in tracks:
        cname = cfg.dataset.classes[track.cls_id]
        colour = colour_for(track.track_id)
        label = f"{cname} {track.conf:.2f} #{track.track_id}"
        if getattr(track, "reid_restored", False):
            label += " R"          # identity restored from the gallery
        draw_box(out, track.xyxy, label, colour)
        if show_trail and len(track.history) > 1:
            pts = [(int(x), int(y)) for x, y in track.history[-30:]]
            for i in range(1, len(pts)):
                cv2.line(out, pts[i - 1], pts[i], colour, 2)
    hud = [
        f"frame {frame_no}   active {len(tracks)}   unique {len(seen_ids)}",
        "  ".join(f"{k}:{len(v)}" for k, v in sorted(class_ids.items())) or "no objects yet",
    ]
    return draw_hud(out, hud)


def write_results_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RESULT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_gt_tracks(gt_file: Path) -> list[TrackBox]:
    """MOT ground truth -> `TrackBox`, carrying the ignore flag through."""
    return [
        TrackBox(
            frame=r.frame, track_id=r.track_id, xyxy=r.xyxy,
            cls_id=r.cls_id, ignore=r.ignore,
        )
        for r in read_mot(gt_file)
    ]
