"""Live inference demo -- webcam / RTSP / file, one `--source` flag (US-5.1).

PRD EPIC-5. The overlay carries box, class, confidence, track id, a motion
trail, live FPS and unique-object counters (US-5.2); `--save` writes an
annotated MP4 plus the per-frame CSV (US-5.3); `--blur-faces` applies the
privacy blur that PRD §17 makes the default for shared output.

This is separate from `track_video.py` because the two have genuinely different
contracts. `track_video` is the *evaluation* path: it reads a file, must see
every frame in order, and must never drop one, because the tracking metrics
score the sequence on disk. `demo` is the *real-time* path: it reads from a
threaded queue, drops stale frames on a live source, and prefers the newest
frame to a complete-but-lagging one (PRD §9.5).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import Config
from ..eval.tracking_metrics import TrackBox
from ..tracking.trackers import build_tracker
from ..utils.logging import get_logger
from ..utils.viz import VideoWriter, colour_for, draw_box, draw_hud
from .analytics import count_line_crossings, summarise_tracks
from .reader import ThreadedFrameReader
from .track_video import RESULT_COLUMNS, write_results_csv

log = get_logger("pipeline.demo")

# Haar cascades are crude face detectors, which is exactly right here: the goal
# is to *destroy* facial detail, so a few false positives cost nothing and a
# heavyweight detector would burn the latency budget the demo is measured on.
_FACE_CASCADE = None


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(path)) if path.exists() else False
    return _FACE_CASCADE or None


def blur_regions(frame: np.ndarray, tracks, class_names, person_only: bool = True) -> np.ndarray:
    """Privacy blur (PRD §17): faces inside person boxes, plus plate-height bands.

    Applied to the *upper third* of a person box and the lower third of a
    vehicle box even when no face or plate is detected. A blur that only fires
    when a detector succeeds is not a privacy guarantee -- it is a privacy
    guarantee that fails exactly when the detector does.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    cascade = _face_cascade()

    for track in tracks:
        name = class_names[track.cls_id] if 0 <= track.cls_id < len(class_names) else ""
        x1, y1, x2, y2 = (int(round(v)) for v in track.xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue

        if name == "person":
            ry2 = y1 + max(8, (y2 - y1) // 3)
            region = out[y1:ry2, x1:x2]
        elif not person_only and name in ("car", "bus", "truck", "motorcycle"):
            ry1 = y2 - max(8, (y2 - y1) // 3)
            region = out[ry1:y2, x1:x2]
        else:
            continue

        if region.size:
            k = max(9, (min(region.shape[:2]) // 2) * 2 + 1)
            out[y1:y1 + region.shape[0], x1:x1 + region.shape[1]] = cv2.GaussianBlur(
                region, (k, k), 0)

    if cascade is not None:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        for (fx, fy, fw, fh) in cascade.detectMultiScale(gray, 1.2, 5, minSize=(24, 24)):
            region = out[fy:fy + fh, fx:fx + fw]
            if region.size:
                k = max(9, (min(region.shape[:2]) // 2) * 2 + 1)
                out[fy:fy + fh, fx:fx + fw] = cv2.GaussianBlur(region, (k, k), 0)
    return out


def run_demo(
    cfg: Config,
    source: str,
    tracker_type: str | None = None,
    weights: str | None = None,
    with_reid: bool = False,
    with_gallery: bool = False,
    save: bool = False,
    show: bool = False,
    max_frames: int | None = None,
    blur_faces: bool = False,
    line: tuple[float, float, float, float] | None = None,
    camera_id: str = "cam0",
) -> dict[str, Any]:
    from ..models.detector import Detector

    detector = Detector(cfg, weights)
    name = (tracker_type or cfg.tracking.tracker_type).lower()
    extractor = None
    if with_reid:
        from ..reid.extractor import ReidExtractor

        extractor = ReidExtractor(cfg)
    tracker = (
        build_tracker(cfg, name, camera_id=camera_id, with_reid=with_reid,
                      with_gallery=with_gallery)
        if name == "botsort" else build_tracker(cfg, name, camera_id=camera_id)
    )

    stem = Path(str(source)).stem if not str(source).isdigit() else f"webcam{source}"
    out_dir = cfg.path("reports_dir") / "demo"
    writer = VideoWriter(out_dir / f"{stem}_demo.mp4", fps=30.0) if save else None

    predictions: list[TrackBox] = []
    csv_rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    class_ids: dict[str, set[int]] = {}
    latencies: list[float] = []
    frame_no = 0
    t_start = time.perf_counter()
    fps_display = 0.0

    reader = ThreadedFrameReader(source).start()
    log.info(
        f"demo source={source} live={reader.is_live} tracker={name} "
        f"reid={with_reid} gallery={with_gallery}",
        extra={"event": "demo_start", "source": str(source)},
    )

    try:
        for frame in reader.frames():
            frame_no += 1
            if max_frames and frame_no > max_frames:
                break
            t0 = time.perf_counter()

            dets = detector.predict(frame, image_id=f"{stem}_f{frame_no:06d}")
            if extractor is not None and dets:
                tracker.set_embeddings(extractor.extract(frame, [d.xyxy for d in dets]))
            tracks = tracker.update(dets, frame=frame)

            for track in tracks:
                seen_ids.add(track.track_id)
                cname = cfg.dataset.classes[track.cls_id]
                class_ids.setdefault(cname, set()).add(track.track_id)
                predictions.append(TrackBox(frame_no, track.track_id,
                                            tuple(float(v) for v in track.xyxy), track.cls_id))
                row = track.to_row(frame_no, cfg.dataset.classes)
                row["timestamp"] = round(frame_no / max(reader.stats.source_fps, 1e-6), 3)
                csv_rows.append({k: row.get(k, "") for k in RESULT_COLUMNS})

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            # Smoothed so the badge is readable rather than flickering.
            fps_display = 0.9 * fps_display + 0.1 * (1000.0 / max(elapsed_ms, 1e-6))

            if writer is not None or show:
                canvas = blur_regions(frame, tracks, cfg.dataset.classes,
                                      person_only=False) if blur_faces else frame.copy()
                for track in tracks:
                    colour = colour_for(track.track_id)
                    label = (f"{cfg.dataset.classes[track.cls_id]} {track.conf:.2f} "
                             f"#{track.track_id}" + (" R" if track.reid_restored else ""))
                    draw_box(canvas, track.xyxy, label, colour)
                    if len(track.history) > 1:
                        pts = [(int(x), int(y)) for x, y in track.history[-30:]]
                        for i in range(1, len(pts)):
                            cv2.line(canvas, pts[i - 1], pts[i], colour, 2)
                if line:
                    cv2.line(canvas, (int(line[0]), int(line[1])),
                             (int(line[2]), int(line[3])), (0, 220, 255), 2)
                draw_hud(canvas, [
                    f"FPS {fps_display:5.1f}   frame {frame_no}   active {len(tracks)}",
                    f"unique {len(seen_ids)}   " +
                    "  ".join(f"{k}:{len(v)}" for k, v in sorted(class_ids.items())),
                    ("privacy blur ON" if blur_faces else "") +
                    ("   ReID gallery ON" if with_gallery else ""),
                ])
                if writer is not None:
                    writer.write(canvas)
                if show:
                    cv2.imshow("cv-detection-reid", canvas)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        reader.stop()
        if writer is not None:
            writer.close()
        if show:
            cv2.destroyAllWindows()

    total_s = time.perf_counter() - t_start
    analytics = summarise_tracks(predictions, cfg.dataset.classes,
                                 fps=reader.stats.source_fps or 30.0)
    out: dict[str, Any] = {
        "source": str(source),
        "live_source": reader.is_live,
        "tracker": name + ("+reid" if with_reid else "") + ("+gallery" if with_gallery else ""),
        "frames": frame_no,
        "wall_fps": round(frame_no / total_s, 2) if total_s else 0.0,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "unique_objects": analytics.unique_total,
        "unique_by_class": analytics.unique_by_class,
        "mean_dwell_s_by_class": analytics.mean_dwell_by_class,
        "frames_dropped": reader.stats.frames_dropped,
        "stream_reconnects": reader.stats.reconnects,
        "privacy_blur": blur_faces,
    }
    if line:
        out["line_crossings"] = count_line_crossings(predictions, line, cfg.dataset.classes)
    if save:
        csv_path = out_dir / f"{stem}_results.csv"
        write_results_csv(csv_path, csv_rows)
        out["annotated_video"] = str((out_dir / f"{stem}_demo.mp4").relative_to(cfg.root))
        out["results_csv"] = str(csv_path.relative_to(cfg.root))
    return out
