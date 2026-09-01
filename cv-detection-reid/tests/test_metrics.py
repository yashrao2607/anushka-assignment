"""Detection metrics against hand-computed worked examples.

PRD 18, Unit level: "IoU computation against hand-computed boxes; NMS
behaviour on overlapping boxes; AP calculation against a worked example."

Every expected value below is derived by hand in the comment above it. That is
the point: a metrics harness validated against another implementation only
proves the two agree, whereas these prove the implementation matches the
definition the report claims to be using.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.detection_metrics import (
    GTBox,
    PredBox,
    average_precision,
    build_confusion,
    evaluate_detections,
    iou_matrix,
    iou_xyxy,
    match_predictions,
    precision_recall_curve,
)

CLASSES = ("person", "car")


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges_is_zero():
    # Sharing an edge is zero overlap area, not a sliver of one.
    assert iou_xyxy((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap_hand_computed():
    # A = (0,0,10,10) area 100. B = (5,0,15,10) area 100.
    # intersection = 5 wide x 10 tall = 50. union = 100 + 100 - 50 = 150.
    # IoU = 50/150 = 1/3.
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_contained_box_hand_computed():
    # inner 5x5 = 25 inside outer 10x10 = 100. intersection 25, union 100.
    assert iou_xyxy((0, 0, 10, 10), (2, 2, 7, 7)) == pytest.approx(0.25)


def test_iou_matrix_matches_scalar():
    preds = [(0, 0, 10, 10), (5, 0, 15, 10)]
    gts = [(0, 0, 10, 10), (20, 20, 30, 30)]
    m = iou_matrix(preds, gts)
    assert m.shape == (2, 2)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            assert m[i, j] == pytest.approx(iou_xyxy(p, g))


def test_iou_matrix_handles_empty_inputs():
    assert iou_matrix([], [(0, 0, 1, 1)]).shape == (0, 1)
    assert iou_matrix([(0, 0, 1, 1)], []).shape == (1, 0)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_duplicate_detection_on_one_object_is_a_false_positive():
    """Weak NMS must be visible, not silently forgiven."""
    gts = [GTBox("f1", 0, (0, 0, 10, 10))]
    preds = [
        PredBox("f1", 0, (0, 0, 10, 10), 0.9),
        PredBox("f1", 0, (1, 1, 11, 11), 0.8),   # IoU ~0.68 with the same GT
    ]
    m = match_predictions(preds, gts, 0.5)
    assert m.tp.tolist() == [1.0, 0.0]
    assert m.fp.tolist() == [0.0, 1.0]


def test_matching_is_greedy_by_confidence_not_by_iou():
    """The COCO protocol: the most confident detection claims the GT first."""
    gts = [GTBox("f1", 0, (0, 0, 10, 10))]
    preds = [
        PredBox("f1", 0, (2, 2, 12, 12), 0.95),   # lower IoU, higher conf -> wins
        PredBox("f1", 0, (0, 0, 10, 10), 0.60),   # perfect IoU, lower conf -> FP
    ]
    m = match_predictions(preds, gts, 0.5)
    assert m.tp.tolist() == [1.0, 0.0]


def test_detection_below_iou_threshold_is_a_false_positive():
    gts = [GTBox("f1", 0, (0, 0, 10, 10))]
    preds = [PredBox("f1", 0, (7, 0, 17, 10), 0.9)]     # IoU = 3/17 ~ 0.176
    m = match_predictions(preds, gts, 0.5)
    assert m.fp.tolist() == [1.0]


def test_ignore_region_absorbs_a_detection_without_scoring_it():
    """Annotation guide rule 2: > 70% occluded objects score neither way."""
    gts = [GTBox("f1", 0, (0, 0, 10, 10), ignore=True)]
    preds = [PredBox("f1", 0, (0, 0, 10, 10), 0.9)]
    m = match_predictions(preds, gts, 0.5)
    assert m.n_gt == 0
    assert m.tp.sum() == 0 and m.fp.sum() == 0


def test_real_gt_is_preferred_over_an_ignore_region():
    gts = [
        GTBox("f1", 0, (0, 0, 10, 10), ignore=True),
        GTBox("f1", 0, (0, 0, 10, 10), ignore=False),
    ]
    preds = [PredBox("f1", 0, (0, 0, 10, 10), 0.9)]
    m = match_predictions(preds, gts, 0.5)
    assert m.tp.tolist() == [1.0]


def test_detections_do_not_match_across_images():
    gts = [GTBox("f1", 0, (0, 0, 10, 10))]
    preds = [PredBox("f2", 0, (0, 0, 10, 10), 0.9)]
    m = match_predictions(preds, gts, 0.5)
    assert m.fp.tolist() == [1.0]


# ---------------------------------------------------------------------------
# AP
# ---------------------------------------------------------------------------


def test_perfect_detector_scores_ap_one():
    gts = [GTBox(f"f{i}", 0, (0, 0, 10, 10)) for i in range(4)]
    preds = [PredBox(f"f{i}", 0, (0, 0, 10, 10), 0.9) for i in range(4)]
    m = match_predictions(preds, gts, 0.5)
    p, r = precision_recall_curve(m)
    assert average_precision(p, r) == pytest.approx(1.0)


def test_ap_worked_example_two_of_four_recalled():
    """Two GT, ranking TP, FP, TP.

    cumulative TP = 1, 1, 2 ; FP = 0, 1, 1
    precision     = 1/1, 1/2, 2/3
    recall        = 0.5, 0.5, 1.0
    Monotone envelope: [1.0, 2/3, 2/3].
    101-point sampling: recall levels <= 0.5 (52 of them, 0.00..0.50) take the
    first index with recall >= level -> precision 1.0; levels 0.51..1.00
    (49 of them) take the index with recall 1.0 -> precision 2/3.
    AP = (52 * 1.0 + 49 * 2/3) / 101.
    """
    gts = [GTBox("f1", 0, (0, 0, 10, 10)), GTBox("f2", 0, (0, 0, 10, 10))]
    preds = [
        PredBox("f1", 0, (0, 0, 10, 10), 0.9),      # TP
        PredBox("f1", 0, (50, 50, 60, 60), 0.8),    # FP
        PredBox("f2", 0, (0, 0, 10, 10), 0.7),      # TP
    ]
    m = match_predictions(preds, gts, 0.5)
    p, r = precision_recall_curve(m)
    expected = (52 * 1.0 + 49 * (2 / 3)) / 101
    assert average_precision(p, r) == pytest.approx(expected, abs=1e-6)


def test_ap_is_zero_when_nothing_is_detected():
    gts = [GTBox("f1", 0, (0, 0, 10, 10))]
    m = match_predictions([], gts, 0.5)
    p, r = precision_recall_curve(m)
    assert average_precision(p, r) == 0.0


def test_ap_uses_101_points_not_11():
    """The two conventions genuinely differ; the report claims 101."""
    gts = [GTBox("f1", 0, (0, 0, 10, 10)), GTBox("f2", 0, (0, 0, 10, 10))]
    preds = [
        PredBox("f1", 0, (0, 0, 10, 10), 0.9),
        PredBox("f1", 0, (50, 50, 60, 60), 0.8),
        PredBox("f2", 0, (0, 0, 10, 10), 0.7),
    ]
    m = match_predictions(preds, gts, 0.5)
    p, r = precision_recall_curve(m)
    ap101 = average_precision(p, r, 101)
    ap11 = average_precision(p, r, 11)
    assert ap101 != pytest.approx(ap11)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def test_evaluate_detections_perfect_case():
    gts = [GTBox("f1", 0, (0, 0, 40, 40)), GTBox("f1", 1, (100, 100, 140, 140))]
    preds = [
        PredBox("f1", 0, (0, 0, 40, 40), 0.99),
        PredBox("f1", 1, (100, 100, 140, 140), 0.99),
    ]
    m = evaluate_detections(preds, gts, CLASSES)
    assert m.map50 == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.mean_iou == pytest.approx(1.0)
    assert m.fn == 0 and m.fp == 0


def test_class_with_no_ground_truth_does_not_drag_map_down():
    """A split that never contained a class must not score it as zero."""
    gts = [GTBox("f1", 0, (0, 0, 40, 40))]
    preds = [PredBox("f1", 0, (0, 0, 40, 40), 0.99)]
    m = evaluate_detections(preds, gts, CLASSES)
    assert m.map50 == pytest.approx(1.0)
    assert m.support["car"] == 0
    assert "car" not in m.per_class_ap50


def test_empty_size_band_reports_none_not_zero():
    """"No small objects here" and "missed every small object" are opposites."""
    gts = [GTBox("f1", 0, (0, 0, 200, 200))]        # area 40000 -> large
    preds = [PredBox("f1", 0, (0, 0, 200, 200), 0.9)]
    m = evaluate_detections(preds, gts, CLASSES)
    assert m.size_ap50["small"] is None
    assert m.size_ap50["large"] == pytest.approx(1.0)


def test_size_bands_split_on_area_thresholds():
    # 30x30 = 900 < 1024 -> small; 50x50 = 2500 -> medium; 200x200 -> large.
    gts = [
        GTBox("f1", 0, (0, 0, 30, 30)),
        GTBox("f2", 0, (0, 0, 50, 50)),
        GTBox("f3", 0, (0, 0, 200, 200)),
    ]
    preds = [
        PredBox("f1", 0, (0, 0, 30, 30), 0.9),
        PredBox("f2", 0, (0, 0, 50, 50), 0.9),
        PredBox("f3", 0, (0, 0, 200, 200), 0.9),
    ]
    m = evaluate_detections(preds, gts, CLASSES)
    for band in ("small", "medium", "large"):
        assert m.size_ap50[band] == pytest.approx(1.0), band


def test_slicing_by_image_id_changes_the_score():
    """The difficulty slices (PRD 13.3) depend on this subsetting working."""
    gts = [GTBox("easy", 0, (0, 0, 40, 40)), GTBox("hard", 0, (0, 0, 40, 40))]
    preds = [PredBox("easy", 0, (0, 0, 40, 40), 0.9)]   # `hard` is missed
    assert evaluate_detections(preds, gts, CLASSES, image_ids=["easy"]).map50 == pytest.approx(1.0)
    assert evaluate_detections(preds, gts, CLASSES, image_ids=["hard"]).map50 == pytest.approx(0.0)


def test_confusion_matrix_records_class_confusion_not_two_unrelated_errors():
    gts = [GTBox("f1", 0, (0, 0, 40, 40))]                     # person
    preds = [PredBox("f1", 1, (0, 0, 40, 40), 0.9)]            # predicted car
    cm = build_confusion(preds, gts, CLASSES, 0.5, 0.25)
    assert cm["person"]["car"] == 1
    assert cm["person"]["background"] == 0
    assert cm["background"]["car"] == 0


def test_confusion_matrix_records_hallucinations_and_misses():
    gts = [GTBox("f1", 0, (0, 0, 40, 40))]
    preds = [PredBox("f1", 1, (500, 500, 540, 540), 0.9)]      # nowhere near
    cm = build_confusion(preds, gts, CLASSES, 0.5, 0.25)
    assert cm["background"]["car"] == 1
    assert cm["person"]["background"] == 1


def test_low_confidence_predictions_excluded_from_operating_point_but_not_from_ap():
    """AP integrates the whole ranking; P/R report one operating point."""
    gts = [GTBox("f1", 0, (0, 0, 40, 40)), GTBox("f2", 0, (0, 0, 40, 40))]
    preds = [
        PredBox("f1", 0, (0, 0, 40, 40), 0.90),
        PredBox("f2", 0, (0, 0, 40, 40), 0.10),   # below the 0.25 operating conf
    ]
    m = evaluate_detections(preds, gts, CLASSES, operating_conf=0.25)
    assert m.recall == pytest.approx(0.5)          # only the confident one counts
    assert m.map50 > 0.5                            # but AP sees both
