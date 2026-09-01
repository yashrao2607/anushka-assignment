"""The failure gallery -- a required deliverable (PRD 13.5).

> "≥ 20 saved failure images/clips in `reports/failures/`, each annotated with
>  the predicted and ground-truth boxes and a root cause from a fixed taxonomy.
>  Each entry names the remediation. **A frequency table of these causes is what
>  turns the next iteration into a plan instead of a guess** -- and honest
>  failure reporting is a strength in review, not a weakness."

The taxonomy is fixed and the assignment is **automatic**, derived from the
geometry of each failure rather than from eyeballing. That matters for the
frequency table: hand-labelled causes drift with the labeller's mood and the
resulting counts cannot be compared across iterations.

Each cause carries its remediation, so the table reads as a work plan.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..data.manifest import ManifestRow
from ..utils.logging import get_logger
from .detection_metrics import GTBox, PredBox, iou_matrix, iou_xyxy

log = get_logger("eval.failures")

# PRD 13.5 taxonomy -> the remediation each one implies.
REMEDIATION = {
    "missed_small_object": "higher imgsz / tiled (SAHI) inference; scale augmentation (EXP-2, R5)",
    "low_light_miss": "night footage + low-light gamma augmentation (EXP-5, R10)",
    "motion_blur_miss": "motion-blur augmentation; shorter exposure source footage (EXP-5)",
    "occlusion_miss": "lower conf threshold; two-stage association already mitigates (US-3.2)",
    "duplicate_box_nms": "tune NMS IoU; class-wise NMS is already on (EXP-10)",
    "class_confusion": "targeted collection for the confused pair; class-weighted loss (7.4)",
    "localisation_drift": "more box-loss weight; check annotation tightness (label protocol 7.2)",
    "false_positive_background": "hard-negative mining round (7.5.1, EXP-6)",
    "id_switch_occlusion": "raise track_buffer; enable the ReID gallery (EXP-9, 9.4)",
    "id_switch_crossing": "raise appearance weight lambda; stronger ReID backbone (EXP-3/4)",
    "track_fragmentation": "raise track_buffer; lower track_low_thresh (EXP-9)",
    "reid_false_match": "tighten tau_reid; class gating is already on (9.4, R7)",
    "label_error": "fix the annotation and re-run; update the annotation guide (US-1.3)",
}


@dataclass
class FailureCase:
    image_id: str
    cause: str
    detail: str
    image_path: str = ""
    saved_path: str = ""
    lighting: str = ""
    blur_level: str = ""
    iou: float = 0.0
    conf: float = 0.0
    cls_gt: str = ""
    cls_pred: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FailureReport:
    cases: list[FailureCase] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)
    saved: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cases": len(self.cases),
            "saved_images": self.saved,
            "counts": dict(self.counts.most_common()),
            "remediation": {c: REMEDIATION.get(c, "") for c in self.counts},
        }


def _classify_miss(gt: GTBox, row: ManifestRow, cfg: Config, best_iou: float) -> tuple[str, str]:
    """Assign a root cause to a missed ground-truth box, most specific first."""
    if 0.1 <= best_iou < cfg.eval.primary_iou:
        return "localisation_drift", f"box found but IoU only {best_iou:.2f}"
    if gt.area < cfg.eval.small_area_max:
        return "missed_small_object", f"area {gt.area:.0f} px^2 < {cfg.eval.small_area_max:.0f}"
    if row.lighting == "night":
        return "low_light_miss", f"brightness {row.brightness:.0f}"
    if row.blur_level == "blurred":
        return "motion_blur_miss", f"normalised sharpness {row.blur_score:.1f}"
    if row.n_objects > cfg.attributes.crowded_object_count:
        return "occlusion_miss", f"{row.n_objects} objects in frame"
    return "occlusion_miss", "no overlapping prediction in a clear frame"


def collect_detection_failures(
    preds: Sequence[PredBox],
    gts: Sequence[GTBox],
    rows: Sequence[ManifestRow],
    cfg: Config,
    conf_thr: float | None = None,
    max_cases: int = 60,
) -> FailureReport:
    """Diagnose every miss, hallucination and confusion at the operating point."""
    conf_thr = cfg.eval.operating_conf if conf_thr is None else conf_thr
    by_image_rows = {r.image_id: r for r in rows}
    names = cfg.dataset.classes
    report = FailureReport()

    preds_by_image: dict[str, list[PredBox]] = {}
    for p in preds:
        if p.conf >= conf_thr:
            preds_by_image.setdefault(p.image_id, []).append(p)
    gts_by_image: dict[str, list[GTBox]] = {}
    for g in gts:
        if not g.ignore:
            gts_by_image.setdefault(g.image_id, []).append(g)

    for image_id in sorted(set(preds_by_image) | set(gts_by_image)):
        row = by_image_rows.get(image_id)
        if row is None:
            continue
        ip = sorted(preds_by_image.get(image_id, []), key=lambda p: -p.conf)
        ig = gts_by_image.get(image_id, [])
        mat = iou_matrix([p.xyxy for p in ip], [g.xyxy for g in ig])

        matched_g: set[int] = set()
        matched_p: set[int] = set()
        for pi, pred in enumerate(ip):
            best_gi, best = -1, cfg.eval.primary_iou
            for gi in range(len(ig)):
                if gi in matched_g:
                    continue
                if mat[pi, gi] >= best:
                    best_gi, best = gi, float(mat[pi, gi])
            if best_gi < 0:
                continue
            matched_g.add(best_gi)
            matched_p.add(pi)
            if pred.cls_id != ig[best_gi].cls_id:
                report.cases.append(FailureCase(
                    image_id, "class_confusion",
                    f"predicted {names[pred.cls_id]} on a {names[ig[best_gi].cls_id]}",
                    row.image_path, lighting=row.lighting, blur_level=row.blur_level,
                    iou=round(best, 3), conf=round(pred.conf, 3),
                    cls_gt=names[ig[best_gi].cls_id], cls_pred=names[pred.cls_id]))

        for gi, gt in enumerate(ig):
            if gi in matched_g:
                continue
            best_iou = float(mat[:, gi].max()) if mat.size else 0.0
            cause, detail = _classify_miss(gt, row, cfg, best_iou)
            report.cases.append(FailureCase(
                image_id, cause, detail, row.image_path, lighting=row.lighting,
                blur_level=row.blur_level, iou=round(best_iou, 3),
                cls_gt=names[gt.cls_id]))

        for pi, pred in enumerate(ip):
            if pi in matched_p:
                continue
            # A high-IoU overlap with an already-claimed GT is a duplicate box
            # (weak NMS); anything else is a background hallucination.
            overlaps_used = any(
                gi in matched_g and mat[pi, gi] >= 0.4 for gi in range(len(ig))
            )
            cause = "duplicate_box_nms" if overlaps_used else "false_positive_background"
            report.cases.append(FailureCase(
                image_id, cause,
                f"conf {pred.conf:.2f} {names[pred.cls_id]} with no unclaimed ground truth",
                row.image_path, lighting=row.lighting, blur_level=row.blur_level,
                conf=round(pred.conf, 3), cls_pred=names[pred.cls_id]))

    report.counts = Counter(c.cause for c in report.cases)
    # Keep a spread across causes rather than the first N of one cause, so the
    # gallery a reviewer opens is representative of the failure profile.
    report.cases = _diversify(report.cases, max_cases)
    return report


def _diversify(cases: list[FailureCase], limit: int) -> list[FailureCase]:
    by_cause: dict[str, list[FailureCase]] = {}
    for c in cases:
        by_cause.setdefault(c.cause, []).append(c)
    out: list[FailureCase] = []
    i = 0
    while len(out) < limit and any(i < len(v) for v in by_cause.values()):
        for bucket in by_cause.values():
            if i < len(bucket) and len(out) < limit:
                out.append(bucket[i])
        i += 1
    return out


def render_failure_images(
    cases: Sequence[FailureCase],
    gts: Sequence[GTBox],
    preds: Sequence[PredBox],
    cfg: Config,
    out_dir: Path,
    conf_thr: float | None = None,
) -> int:
    """Save each case annotated with ground truth (green) and prediction (red)."""
    import cv2

    from ..utils.viz import draw_box

    conf_thr = cfg.eval.operating_conf if conf_thr is None else conf_thr
    out_dir.mkdir(parents=True, exist_ok=True)
    names = cfg.dataset.classes
    saved = 0

    gt_by_image: dict[str, list[GTBox]] = {}
    for g in gts:
        gt_by_image.setdefault(g.image_id, []).append(g)
    pred_by_image: dict[str, list[PredBox]] = {}
    for p in preds:
        if p.conf >= conf_thr:
            pred_by_image.setdefault(p.image_id, []).append(p)

    for i, case in enumerate(cases):
        path = cfg.root / case.image_path
        img = cv2.imread(str(path)) if path.exists() else None
        if img is None:
            continue
        for g in gt_by_image.get(case.image_id, []):
            draw_box(img, g.xyxy, f"GT {names[g.cls_id]}", (0, 200, 0))
        for p in pred_by_image.get(case.image_id, []):
            draw_box(img, p.xyxy, f"P {names[p.cls_id]} {p.conf:.2f}", (0, 0, 235))
        cv2.putText(img, f"{case.cause}: {case.detail}", (8, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        dest = out_dir / f"{i:03d}_{case.cause}_{case.image_id}.jpg"
        cv2.imwrite(str(dest), img)
        case.saved_path = str(dest.relative_to(cfg.root).as_posix())
        saved += 1
    return saved
