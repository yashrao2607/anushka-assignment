"""Detection metrics -- IoU, AP, mAP@0.5, mAP@0.5:0.95, P/R, size slices.

PRD 4.2 (M1-M7) and 13.1. This is the *measuring instrument*, and it is built
in Phase 1, before the detector is fine-tuned in Phase 2, so that every later
improvement is provable rather than felt.

Implemented from the definition rather than imported, for three reasons:
  1. `tests/test_metrics.py` checks it against hand-computed worked examples,
     which is only meaningful if the code under test is the code being read.
  2. Ultralytics' validator only scores its own dataloader; the B0 zero-shot
     baseline scores a *COCO-pretrained* model against *our* class space, and
     the difficulty slices (13.3) need arbitrary per-frame subsetting.
  3. "Measure the right thing" (Principle 2) is hard to audit through a wrapper.

Conventions, stated because they are exactly where two mAP implementations
usually disagree:
  * Boxes are absolute-pixel `(x1, y1, x2, y2)`, x2/y2 exclusive.
  * Matching is greedy by descending confidence -- the COCO protocol. A
    detection takes the highest-IoU unmatched ground truth of its own class.
  * A second detection on an already-matched object is a **false positive**,
    not a duplicate-suppressed no-op. That is what makes weak NMS visible.
  * AP is the 101-point interpolated area under the precision-recall curve
    (COCO), not the legacy 11-point VOC form. The two differ by ~1-2 points and
    mixing them is a classic source of "our mAP went up" illusions.
  * `ignore` ground truth (annotation guide rule 2: objects occluded > 70%)
    absorbs a matching detection without scoring it either way.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GTBox:
    image_id: str
    cls_id: int
    xyxy: tuple[float, float, float, float]
    ignore: bool = False

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class PredBox:
    image_id: str
    cls_id: int
    xyxy: tuple[float, float, float, float]
    conf: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two absolute-pixel boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(preds: Sequence[Sequence[float]], gts: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorised IoU between P predictions and G ground truths -> (P, G)."""
    if not len(preds) or not len(gts):
        return np.zeros((len(preds), len(gts)), dtype=float)
    p = np.asarray(preds, dtype=float)
    g = np.asarray(gts, dtype=float)
    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_p = np.clip(p[:, 2] - p[:, 0], 0, None) * np.clip(p[:, 3] - p[:, 1], 0, None)
    area_g = np.clip(g[:, 2] - g[:, 0], 0, None) * np.clip(g[:, 3] - g[:, 1], 0, None)
    union = area_p[:, None] + area_g[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


# ---------------------------------------------------------------------------
# Matching + AP
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Per-prediction outcome at one IoU threshold, ordered by descending conf."""

    tp: np.ndarray            # 1 where the prediction is a true positive
    fp: np.ndarray            # 1 where it is a false positive
    conf: np.ndarray
    matched_iou: np.ndarray   # IoU of the matched GT, 0 for FPs
    n_gt: int                 # non-ignore ground truths available to match


def match_predictions(
    preds: Sequence[PredBox], gts: Sequence[GTBox], iou_thr: float
) -> MatchResult:
    """Greedy confidence-ordered matching within one class.

    Preconditions: `preds` and `gts` are already filtered to a single class.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i].conf)
    by_image: dict[str, list[int]] = defaultdict(list)
    for gi, gt in enumerate(gts):
        by_image[gt.image_id].append(gi)

    n_gt = sum(1 for g in gts if not g.ignore)
    used = [False] * len(gts)
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    matched_iou = np.zeros(len(preds))
    conf = np.zeros(len(preds))

    for rank, pi in enumerate(order):
        pred = preds[pi]
        conf[rank] = pred.conf
        candidates = by_image.get(pred.image_id, ())

        best_gi, best_iou = -1, iou_thr
        # Real ground truth first: a detection should only be absorbed by an
        # ignore region when it has nothing legitimate to match.
        for gi in candidates:
            if gts[gi].ignore or used[gi]:
                continue
            v = iou_xyxy(pred.xyxy, gts[gi].xyxy)
            if v >= best_iou:
                best_gi, best_iou = gi, v

        if best_gi >= 0:
            used[best_gi] = True
            tp[rank] = 1.0
            matched_iou[rank] = best_iou
            continue

        absorbed = any(
            gts[gi].ignore and iou_xyxy(pred.xyxy, gts[gi].xyxy) >= iou_thr
            for gi in candidates
        )
        if not absorbed:
            fp[rank] = 1.0

    return MatchResult(tp=tp, fp=fp, conf=conf, matched_iou=matched_iou, n_gt=n_gt)


def precision_recall_curve(match: MatchResult) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative precision and recall along the confidence-sorted predictions."""
    if match.n_gt == 0:
        return np.array([]), np.array([])
    tp_cum = np.cumsum(match.tp)
    fp_cum = np.cumsum(match.fp)
    denom = tp_cum + fp_cum
    precision = np.divide(tp_cum, denom, out=np.zeros_like(tp_cum), where=denom > 0)
    recall = tp_cum / match.n_gt
    return precision, recall


def average_precision(
    precision: np.ndarray, recall: np.ndarray, recall_points: int = 101
) -> float:
    """101-point interpolated AP (COCO).

    The precision envelope is made monotonically non-increasing first, which
    removes the sawtooth caused by a lucky detection deep in the ranking; AP is
    then the mean of that envelope sampled at evenly spaced recall levels.
    """
    if precision.size == 0 or recall.size == 0:
        return 0.0
    mono = precision.copy()
    for i in range(mono.size - 1, 0, -1):
        mono[i - 1] = max(mono[i - 1], mono[i])
    levels = np.linspace(0.0, 1.0, recall_points)
    idx = np.searchsorted(recall, levels, side="left")
    sampled = np.where(idx < mono.size, mono[np.clip(idx, 0, mono.size - 1)], 0.0)
    return float(np.mean(sampled))


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------


@dataclass
class DetectionMetrics:
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0        # M3, at the operating confidence
    recall: float = 0.0           # M4
    f1: float = 0.0
    mean_iou: float = 0.0         # M5, over matched boxes
    per_class_ap50: dict[str, float] = field(default_factory=dict)   # M6
    per_class_ap50_95: dict[str, float] = field(default_factory=dict)
    size_ap50: dict[str, float] = field(default_factory=dict)        # M7 (small)
    n_images: int = 0
    n_gt: int = 0
    n_pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: dict[str, int] = field(default_factory=dict)
    pr_curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "pr_curves"}
        return d


def _band(area: float, small_max: float, medium_max: float) -> str:
    if area < small_max:
        return "small"
    if area < medium_max:
        return "medium"
    return "large"


def evaluate_detections(
    preds: Sequence[PredBox],
    gts: Sequence[GTBox],
    class_names: Sequence[str],
    iou_thresholds: Sequence[float] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10)),
    primary_iou: float = 0.5,
    operating_conf: float = 0.25,
    recall_points: int = 101,
    small_area_max: float = 1024.0,
    medium_area_max: float = 9216.0,
    image_ids: Iterable[str] | None = None,
) -> DetectionMetrics:
    """Full detection evaluation over one split or one difficulty slice."""
    images = set(image_ids) if image_ids is not None else {
        *(p.image_id for p in preds), *(g.image_id for g in gts)
    }
    preds = [p for p in preds if p.image_id in images]
    gts = [g for g in gts if g.image_id in images]

    m = DetectionMetrics(
        n_images=len(images),
        n_gt=sum(1 for g in gts if not g.ignore),
        n_pred=len(preds),
    )

    preds_by_cls: dict[int, list[PredBox]] = defaultdict(list)
    gts_by_cls: dict[int, list[GTBox]] = defaultdict(list)
    for p in preds:
        preds_by_cls[p.cls_id].append(p)
    for g in gts:
        gts_by_cls[g.cls_id].append(g)

    ap_by_thr: dict[float, list[float]] = {t: [] for t in iou_thresholds}

    for cls_id, name in enumerate(class_names):
        cls_gts = gts_by_cls.get(cls_id, [])
        cls_preds = preds_by_cls.get(cls_id, [])
        support = sum(1 for g in cls_gts if not g.ignore)
        m.support[name] = support
        # A class with no ground truth in this split contributes no AP. Scoring
        # it as 0 would silently drag mAP down for a class the split simply
        # never contained -- the split warning in the splitter is where that
        # gets surfaced, not here.
        if support == 0:
            continue

        for thr in iou_thresholds:
            match = match_predictions(cls_preds, cls_gts, thr)
            precision, recall = precision_recall_curve(match)
            ap = average_precision(precision, recall, recall_points)
            ap_by_thr[thr].append(ap)
            if thr == primary_iou:
                m.per_class_ap50[name] = round(ap, 4)
                if precision.size:
                    m.pr_curves[name] = {
                        "precision": [round(float(v), 4) for v in precision],
                        "recall": [round(float(v), 4) for v in recall],
                        "conf": [round(float(v), 4) for v in match.conf],
                    }

        per_thr = [
            average_precision(*precision_recall_curve(match_predictions(cls_preds, cls_gts, t)),
                              recall_points)
            for t in iou_thresholds
        ]
        m.per_class_ap50_95[name] = round(float(np.mean(per_thr)), 4)

    m.map50 = round(float(np.mean(ap_by_thr[primary_iou])) if ap_by_thr[primary_iou] else 0.0, 4)
    all_aps = [ap for t in iou_thresholds for ap in ap_by_thr[t]]
    m.map50_95 = round(float(np.mean(all_aps)) if all_aps else 0.0, 4)

    # -- M3/M4/M5 at the stated operating point -----------------------------
    tp = fp = 0
    ious: list[float] = []
    confused: dict[str, Counter] = {n: Counter() for n in class_names}
    for cls_id, name in enumerate(class_names):
        cls_preds = [p for p in preds_by_cls.get(cls_id, []) if p.conf >= operating_conf]
        cls_gts = gts_by_cls.get(cls_id, [])
        match = match_predictions(cls_preds, cls_gts, primary_iou)
        tp += int(match.tp.sum())
        fp += int(match.fp.sum())
        ious.extend(float(v) for v in match.matched_iou[match.tp > 0])
        confused[name]["__matched__"] = int(match.tp.sum())

    m.tp, m.fp = tp, fp
    m.fn = max(0, m.n_gt - tp)
    m.precision = round(tp / (tp + fp), 4) if (tp + fp) else 0.0
    m.recall = round(tp / m.n_gt, 4) if m.n_gt else 0.0
    m.f1 = (
        round(2 * m.precision * m.recall / (m.precision + m.recall), 4)
        if (m.precision + m.recall)
        else 0.0
    )
    m.mean_iou = round(float(np.mean(ious)), 4) if ious else 0.0

    # -- M7: AP by object-size band -----------------------------------------
    # COCO's areaRng protocol: ground truth outside the band becomes `ignore`
    # (so a detection landing on it is neither rewarded nor punished) and
    # predictions outside the band are excluded from the ranking.
    for band, lo, hi in (
        ("small", 0.0, small_area_max),
        ("medium", small_area_max, medium_area_max),
        ("large", medium_area_max, float("inf")),
    ):
        band_aps: list[float] = []
        for cls_id, name in enumerate(class_names):
            cls_gts = gts_by_cls.get(cls_id, [])
            in_band = [
                GTBox(g.image_id, g.cls_id, g.xyxy, ignore=g.ignore or not (lo <= g.area < hi))
                for g in cls_gts
            ]
            if not any(not g.ignore for g in in_band):
                continue
            cls_preds = [p for p in preds_by_cls.get(cls_id, []) if lo <= p.area < hi]
            match = match_predictions(cls_preds, in_band, primary_iou)
            band_aps.append(average_precision(*precision_recall_curve(match), recall_points))
        m.size_ap50[band] = round(float(np.mean(band_aps)), 4) if band_aps else 0.0

    m.confusion = build_confusion(preds, gts, class_names, primary_iou, operating_conf)
    return m


def build_confusion(
    preds: Sequence[PredBox],
    gts: Sequence[GTBox],
    class_names: Sequence[str],
    iou_thr: float,
    conf_thr: float,
) -> dict[str, dict[str, int]]:
    """Class-confusion matrix at one operating point, with a background row/column.

    Matching here is deliberately **class-agnostic** -- that is the only way a
    `truck` predicted where a `bus` stands shows up as a confusion rather than
    as one false positive plus one unrelated miss. `background` columns are
    hallucinations; `background` rows are misses.
    """
    labels = list(class_names) + ["background"]
    matrix = {gt: {pred: 0 for pred in labels} for gt in labels}

    by_image_p: dict[str, list[PredBox]] = defaultdict(list)
    by_image_g: dict[str, list[GTBox]] = defaultdict(list)
    for p in preds:
        if p.conf >= conf_thr:
            by_image_p[p.image_id].append(p)
    for g in gts:
        if not g.ignore:
            by_image_g[g.image_id].append(g)

    for image_id in set(by_image_p) | set(by_image_g):
        ip = sorted(by_image_p.get(image_id, []), key=lambda p: -p.conf)
        ig = by_image_g.get(image_id, [])
        mat = iou_matrix([p.xyxy for p in ip], [g.xyxy for g in ig])
        used_g: set[int] = set()
        for pi, pred in enumerate(ip):
            best_gi, best = -1, iou_thr
            for gi in range(len(ig)):
                if gi in used_g:
                    continue
                if mat[pi, gi] >= best:
                    best_gi, best = gi, mat[pi, gi]
            if best_gi >= 0:
                used_g.add(best_gi)
                matrix[class_names[ig[best_gi].cls_id]][class_names[pred.cls_id]] += 1
            else:
                matrix["background"][class_names[pred.cls_id]] += 1
        for gi, gt in enumerate(ig):
            if gi not in used_g:
                matrix[class_names[gt.cls_id]]["background"] += 1

    return matrix
