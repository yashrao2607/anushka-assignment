"""Detection evaluation runner -- one command, full report.

PRD US-6.1 and 13.3. Runs the detector over a split, scores it with the harness
in `detection_metrics.py`, and emits the same numbers as markdown, JSON and a
console table -- including the **difficulty slices** that are the D4
differentiator.

The slices are read from the manifest attributes captured at sampling time
(7.6). That is the whole reason the sampler computes them: an averaged mAP
hides exactly the failures that matter operationally, and the attributes cannot
be reconstructed after the fact.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import Config
from ..data.manifest import ManifestRow, read_manifest
from ..data.validate_labels import parse_label_file
from ..utils.logging import get_logger
from .detection_metrics import GTBox, PredBox, DetectionMetrics, evaluate_detections

log = get_logger("eval.runner")


def load_ground_truth(rows: Sequence[ManifestRow], cfg: Config) -> list[GTBox]:
    """Read YOLO label files back into absolute-pixel `GTBox` records."""
    out: list[GTBox] = []
    n_classes = len(cfg.dataset.classes)
    for row in rows:
        path = cfg.root / row.label_path
        if not path.exists() or row.width <= 0 or row.height <= 0:
            continue
        boxes, _ = parse_label_file(path, n_classes, row.image_id)
        for b in boxes:
            out.append(GTBox(row.image_id, b.cls_id, b.to_xyxy(row.width, row.height)))
    return out


def build_slices(rows: Sequence[ManifestRow], cfg: Config) -> dict[str, list[str]]:
    """PRD 13.3 slice definitions, as image-id sets."""
    s: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        s[f"lighting:{r.lighting}"].append(r.image_id)
        s[f"blur:{r.blur_level}"].append(r.image_id)
        if r.n_objects > cfg.attributes.crowded_object_count:
            s["crowded:>15 objects"].append(r.image_id)
        else:
            s["crowded:<=15 objects"].append(r.image_id)
        # `motion_score` is the frame-to-frame difference between *sampled*
        # frames, so a high value means the scene or the camera moved a lot
        # between samples -- the dashcam case where GMC earns its place.
        s["motion:high" if r.motion_score >= 25.0 else "motion:low"].append(r.image_id)
    # Drop degenerate slices; a slice with one image is noise, not evidence.
    return {k: v for k, v in sorted(s.items()) if len(v) >= 3}


@dataclass
class EvalRun:
    split: str
    overall: DetectionMetrics
    slices: dict[str, DetectionMetrics] = field(default_factory=dict)
    detector: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    n_rows: int = 0

    def headline(self) -> dict[str, float]:
        o = self.overall
        return {
            "M1": o.map50,
            "M2": o.map50_95,
            "M3": o.precision,
            "M4": o.recall,
            "M5": o.mean_iou,
            "M7": o.size_ap50.get("small", 0.0),
        }


def run_detection_eval(
    cfg: Config,
    split: str = "test",
    weights: str | Path | None = None,
    limit: int | None = None,
    with_slices: bool = True,
) -> EvalRun:
    from ..models.detector import Detector

    rows = [r for r in read_manifest(cfg.path("manifest")) if r.split == split]
    if not rows:
        raise ValueError(
            f"no frames assigned to split '{split}' -- run `python -m src.cli split` first"
        )
    if limit:
        rows = rows[:limit]

    detector = Detector(cfg, weights)
    log.info(
        f"evaluating split '{split}' on {len(rows)} frames with {detector.weights} "
        f"on {detector.device.device}",
        extra={"event": "eval_start", "split": split, "n": len(rows)},
    )

    preds: list[PredBox] = []
    for i, row in enumerate(rows, start=1):
        image_path = cfg.root / row.image_path
        if not image_path.exists():
            log.warning(f"missing image {image_path}", extra={"event": "missing_image"})
            continue
        preds.extend(detector.predict(str(image_path), row.image_id, conf=0.001))
        if i % 50 == 0:
            log.info(f"  {i}/{len(rows)} frames", extra={"event": "eval_progress", "done": i})

    gts = load_ground_truth(rows, cfg)
    ev = cfg.eval
    kwargs = dict(
        class_names=cfg.dataset.classes,
        iou_thresholds=ev.iou_thresholds,
        primary_iou=ev.primary_iou,
        operating_conf=ev.operating_conf,
        recall_points=ev.recall_points,
        small_area_max=ev.small_area_max,
        medium_area_max=ev.medium_area_max,
    )
    overall = evaluate_detections(preds, gts, image_ids=[r.image_id for r in rows], **kwargs)

    slices: dict[str, DetectionMetrics] = {}
    if with_slices:
        for name, ids in build_slices(rows, cfg).items():
            slices[name] = evaluate_detections(preds, gts, image_ids=ids, **kwargs)

    return EvalRun(
        split=split,
        overall=overall,
        slices=slices,
        detector=detector.describe(),
        timing=detector.stats.summary(),
        n_rows=len(rows),
    )


# Evaluation runs the detector at conf=0.001 so the precision-recall curve has
# a full low-confidence tail -- AP is the area under that curve, and truncating
# it at the operating threshold understates AP while leaving mAP looking
# plausibly high. P/R/mean-IoU are then reported at `eval.operating_conf`
# inside the harness, which is the number an operator actually experiences.
