"""Tracking metrics against hand-computed sequences.

PRD 4.3 / 13.1. Each case is a tiny sequence whose MOTA, IDF1 and ID-switch
count can be worked out on paper, because these are exactly the metrics that
are easy to implement *plausibly* and wrong -- an ID-switch counter that never
fires and a tracker that never switches look identical in a report.
"""

from __future__ import annotations

import pytest

from src.eval.tracking_metrics import (
    TrackBox,
    clear_mot,
    evaluate_tracking,
    hota_score,
    idf1_score,
)


def box(frame, tid, x, cls_id=0, ignore=False):
    """A 20x20 box at (x, 0), so consecutive x values give clean IoU steps."""
    return TrackBox(frame=frame, track_id=tid, xyxy=(float(x), 0.0, float(x + 20), 20.0),
                    cls_id=cls_id, ignore=ignore)


def perfect_sequence(n=5, tid=1):
    return [box(f, tid, f * 30) for f in range(1, n + 1)]


# ---------------------------------------------------------------------------
# CLEAR MOT
# ---------------------------------------------------------------------------


def test_perfect_tracking_scores_mota_one_with_no_switches():
    gt = perfect_sequence()
    pred = perfect_sequence()
    m = clear_mot(gt, pred)
    assert m["mota"] == pytest.approx(1.0)
    assert m["idsw"] == 0
    assert m["fp"] == 0 and m["fn"] == 0
    assert m["tp"] == 5


def test_missed_frames_are_false_negatives():
    """5 GT, 3 matched -> FN = 2, MOTA = 1 - 2/5 = 0.6."""
    gt = perfect_sequence()
    pred = [b for b in perfect_sequence() if b.frame <= 3]
    m = clear_mot(gt, pred)
    assert m["fn"] == 2
    assert m["mota"] == pytest.approx(0.6)


def test_spurious_tracks_are_false_positives():
    """5 GT, 5 TP plus 5 ghosts -> FP = 5, MOTA = 1 - 5/5 = 0.0."""
    gt = perfect_sequence()
    pred = perfect_sequence() + [box(f, 99, 500) for f in range(1, 6)]
    m = clear_mot(gt, pred)
    assert m["fp"] == 5
    assert m["mota"] == pytest.approx(0.0)


def test_mota_can_go_negative_when_errors_exceed_ground_truth():
    """Not a bug: more errors than objects is a real, reportable state."""
    gt = perfect_sequence(2)
    pred = [box(f, 90 + f, 500 + f) for f in range(1, 8)]
    m = clear_mot(gt, pred)
    assert m["mota"] < 0


def test_id_switch_is_counted_when_the_predicted_id_changes_mid_track():
    """One GT object; the tracker relabels it at frame 4 -> exactly one switch."""
    gt = perfect_sequence(6)
    pred = [box(f, 1 if f <= 3 else 2, f * 30) for f in range(1, 7)]
    m = clear_mot(gt, pred)
    assert m["idsw"] == 1
    # 6 TP, 0 FP, 0 FN, 1 switch -> MOTA = 1 - 1/6.
    assert m["mota"] == pytest.approx(1 - 1 / 6, abs=1e-4)


def test_id_switch_survives_a_gap_in_the_prediction():
    """Losing an object and re-acquiring it under a NEW id is still a switch.

    This is the occlusion case the whole ReID layer exists to prevent, so the
    metric must charge for it rather than quietly forgiving the gap.
    """
    gt = perfect_sequence(6)
    pred = [box(f, 1, f * 30) for f in (1, 2, 3)] + [box(f, 2, f * 30) for f in (5, 6)]
    m = clear_mot(gt, pred)
    assert m["idsw"] == 1
    assert m["fn"] == 1                       # frame 4 was never predicted


def test_two_objects_swapping_ids_costs_two_switches():
    gt = [box(f, 1, 0) for f in range(1, 5)] + [box(f, 2, 200) for f in range(1, 5)]
    pred = (
        [box(f, 10, 0) for f in (1, 2)] + [box(f, 20, 0) for f in (3, 4)]
        + [box(f, 20, 200) for f in (1, 2)] + [box(f, 10, 200) for f in (3, 4)]
    )
    m = clear_mot(gt, pred)
    assert m["idsw"] == 2


def test_sticky_matching_does_not_invent_switches_between_equal_candidates():
    """Two identical overlapping predictions must not make the match flap."""
    gt = perfect_sequence(4)
    pred = [b for f in range(1, 5) for b in (box(f, 1, f * 30), box(f, 2, f * 30 + 1))]
    m = clear_mot(gt, pred)
    assert m["idsw"] == 0


def test_ignore_ground_truth_absorbs_predictions_without_penalty():
    gt = [box(f, 1, 0, ignore=True) for f in range(1, 5)]
    pred = [box(f, 7, 0) for f in range(1, 5)]
    m = clear_mot(gt, pred)
    assert m["fp"] == 0
    assert m["n_gt"] == 0


def test_mostly_tracked_and_mostly_lost_classification():
    # obj 1 tracked in 5/5 frames -> MT ; obj 2 tracked in 1/5 -> ML.
    gt = [box(f, 1, 0) for f in range(1, 6)] + [box(f, 2, 300) for f in range(1, 6)]
    pred = [box(f, 1, 0) for f in range(1, 6)] + [box(1, 2, 300)]
    m = clear_mot(gt, pred)
    assert m["mt"] == 1
    assert m["ml"] == 1
    assert m["mt_ratio"] == pytest.approx(0.5)


def test_motp_reports_mean_matched_iou():
    # 20-wide boxes offset by 5: intersection 15x20=300, union 500 -> IoU 0.6.
    # (An offset of 10 gives IoU 1/3, below the 0.5 match threshold, so it is
    # correctly not a match at all -- which is itself worth pinning down.)
    gt = [box(1, 1, 0)]
    pred = [box(1, 1, 5)]
    m = clear_mot(gt, pred)
    assert m["motp"] == pytest.approx(0.6, abs=1e-3)
    assert clear_mot([box(1, 1, 0)], [box(1, 1, 10)])["tp"] == 0


# ---------------------------------------------------------------------------
# IDF1
# ---------------------------------------------------------------------------


def test_idf1_is_one_for_a_perfect_tracker():
    assert idf1_score(perfect_sequence(), perfect_sequence())["idf1"] == pytest.approx(1.0)


def test_idf1_punishes_a_mid_sequence_switch_more_than_mota_does():
    """The reason both are reported: they disagree, and the disagreement is the point.

    A single relabel costs MOTA one error out of six. IDF1 matches whole
    trajectories one-to-one, so the second half of the object's life is
    unattributable and the score drops far further.
    """
    gt = perfect_sequence(6)
    pred = [box(f, 1 if f <= 3 else 2, f * 30) for f in range(1, 7)]
    mota = clear_mot(gt, pred)["mota"]
    idf1 = idf1_score(gt, pred)["idf1"]
    assert mota == pytest.approx(1 - 1 / 6, abs=1e-4)
    assert idf1 < mota
    # 3 of 6 attributable: IDTP=3, IDFP=3, IDFN=3 -> 2*3/(6+3+3) = 0.5
    assert idf1 == pytest.approx(0.5)


def test_idf1_is_zero_when_nothing_overlaps():
    gt = perfect_sequence()
    pred = [box(f, 9, 900) for f in range(1, 6)]
    assert idf1_score(gt, pred)["idf1"] == 0.0


def test_idf1_handles_empty_predictions():
    out = idf1_score(perfect_sequence(), [])
    assert out["idf1"] == 0.0
    assert out["idfn"] == 5


# ---------------------------------------------------------------------------
# HOTA
# ---------------------------------------------------------------------------


def test_hota_is_one_for_a_perfect_tracker():
    out = hota_score(perfect_sequence(), perfect_sequence())
    assert out["hota"] == pytest.approx(1.0)
    assert out["deta"] == pytest.approx(1.0)
    assert out["assa"] == pytest.approx(1.0)


def test_hota_separates_detection_quality_from_association_quality():
    """Perfect detection with broken association: DetA high, AssA low."""
    gt = perfect_sequence(6)
    pred = [box(f, f, f * 30) for f in range(1, 7)]      # a new id every frame
    out = hota_score(gt, pred)
    assert out["deta"] == pytest.approx(1.0)
    assert out["assa"] < 0.3
    assert out["hota"] < out["deta"]


def test_hota_is_bounded_by_the_geometric_mean_relationship():
    gt = perfect_sequence(6)
    pred = [box(f, 1 if f <= 3 else 2, f * 30) for f in range(1, 7)]
    out = hota_score(gt, pred)
    assert 0.0 <= out["hota"] <= 1.0
    assert out["hota"] <= max(out["deta"], out["assa"])


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def test_evaluate_tracking_reports_every_prd_metric():
    m = evaluate_tracking(perfect_sequence(10), perfect_sequence(10))
    head = m.headline()
    assert set(head) == {"M8", "M9", "M10", "M11", "M12", "M12b", "M13"}
    assert head["M8"] == pytest.approx(1.0)
    assert head["M9"] == pytest.approx(1.0)
    assert head["M10"] == pytest.approx(1.0)
    assert head["M11"] == 0.0


def test_idsw_per_1k_frames_normalises_by_sequence_length():
    """M11 is a rate, not a count: a longer clip must not look worse."""
    gt = perfect_sequence(100)
    pred = [box(f, 1 if f <= 50 else 2, f * 30) for f in range(1, 101)]
    m = evaluate_tracking(gt, pred)
    assert m.idsw == 1
    assert m.idsw_per_1k == pytest.approx(10.0)
