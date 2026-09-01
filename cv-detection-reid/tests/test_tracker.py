"""Tracker unit tests.

PRD 18, Unit level: "Kalman predict/update on a synthetic constant-velocity
trajectory; Hungarian assignment on a known cost matrix; cosine distance and
gallery matching."

The trackers are driven with **synthetic detections** rather than a detector,
which is the only way to attribute a failure. Given perfect detections, any id
churn is the tracker's fault and nothing else's.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config
from src.eval.detection_metrics import PredBox
from src.tracking.gmc import GMC, apply_warp_to_xyxy
from src.tracking.kalman import KalmanFilterXYAH, xyah_to_xyxy, xyxy_to_xyah
from src.tracking.matching import (
    INF_COST,
    class_gate,
    cosine_distance,
    fuse_cost,
    iou_distance,
    linear_assignment,
)
from src.tracking.trackers import build_tracker


@pytest.fixture
def cfg():
    return load_config()


def det(x, y, cls_id=0, conf=0.9, w=40, h=80):
    return PredBox("f", cls_id, (float(x), float(y), float(x + w), float(y + h)), conf)


# ---------------------------------------------------------------------------
# Kalman
# ---------------------------------------------------------------------------


def test_xyxy_xyah_round_trip():
    box = (100.0, 50.0, 140.0, 130.0)
    assert xyah_to_xyxy(xyxy_to_xyah(box)) == pytest.approx(box, abs=1e-6)


def test_kalman_predicts_constant_velocity_after_learning_it():
    """Feed a straight line; the filter must extrapolate it, not lag behind."""
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(xyxy_to_xyah((0.0, 0.0, 40.0, 80.0)))
    for step in range(1, 15):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, xyxy_to_xyah((step * 10.0, 0.0, step * 10.0 + 40, 80.0)))

    predicted, _ = kf.predict(mean, cov)
    x1, _, x2, _ = xyah_to_xyxy(predicted)
    # After 14 consistent steps of +10 px the next centre should be near 155.
    assert (x1 + x2) / 2 == pytest.approx(155.0, abs=6.0)


def test_kalman_uncertainty_grows_while_coasting():
    """An unobserved track must become less certain, or gating stops working."""
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(xyxy_to_xyah((0.0, 0.0, 40.0, 80.0)))
    before = float(np.trace(cov))
    for _ in range(5):
        mean, cov = kf.predict(mean, cov)
    assert float(np.trace(cov)) > before


def test_kalman_update_reduces_uncertainty():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(xyxy_to_xyah((0.0, 0.0, 40.0, 80.0)))
    mean, cov = kf.predict(mean, cov)
    after_predict = float(np.trace(cov))
    mean, cov = kf.update(mean, cov, xyxy_to_xyah((2.0, 0.0, 42.0, 80.0)))
    assert float(np.trace(cov)) < after_predict


def test_gating_distance_is_small_for_the_true_measurement():
    kf = KalmanFilterXYAH()
    mean, cov = kf.initiate(xyxy_to_xyah((0.0, 0.0, 40.0, 80.0)))
    close = xyxy_to_xyah((2.0, 0.0, 42.0, 80.0))
    far = xyxy_to_xyah((600.0, 400.0, 640.0, 480.0))
    d = kf.gating_distance(mean, cov, np.stack([close, far]))
    assert d[0] < d[1]


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_hungarian_finds_the_optimal_assignment_on_a_known_matrix():
    # The greedy choice (0,0)=1 forces (1,1)=4 for a total of 5.
    # The optimal assignment is (0,1)=2 + (1,0)=2 = 4.
    cost = np.array([[1.0, 2.0], [2.0, 4.0]])
    matches, _, _ = linear_assignment(cost, thresh=10.0)
    assert sum(cost[r, c] for r, c in matches) == pytest.approx(4.0)


def test_assignment_respects_the_threshold():
    cost = np.array([[0.2, 0.9], [0.9, 0.95]])
    matches, u_rows, u_cols = linear_assignment(cost, thresh=0.5)
    assert matches == [(0, 0)]
    assert u_rows == [1] and u_cols == [1]


def test_iou_distance_is_one_minus_iou():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])          # IoU = 1/3
    assert iou_distance(a, b)[0, 0] == pytest.approx(1 - 1 / 3)


def test_cosine_distance_of_identical_vectors_is_zero():
    v = np.array([[1.0, 0.0, 0.0]])
    assert cosine_distance(v, v)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_of_orthogonal_vectors_is_one():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert cosine_distance(a, b)[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_class_gate_blocks_cross_class_association():
    cost = np.array([[0.0, 0.0]])
    gated = class_gate(cost, [0], [0, 1])
    assert gated[0, 0] == 0.0
    assert gated[0, 1] == INF_COST


def test_fuse_cost_is_a_convex_combination():
    iou = np.array([[0.4]])
    app = np.array([[0.2]])
    assert fuse_cost(iou, app, 0.5)[0, 0] == pytest.approx(0.3)
    assert fuse_cost(iou, app, 1.0)[0, 0] == pytest.approx(0.4)   # motion only


def test_fuse_cost_ignores_appearance_beyond_the_gate():
    """A confidently-wrong embedding must not drag a good motion match away."""
    iou = np.array([[0.1]])
    app = np.array([[0.95]])          # above the 0.7 gate -> "no evidence"
    assert fuse_cost(iou, app, 0.5, appearance_gate=0.7)[0, 0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# GMC
# ---------------------------------------------------------------------------


def test_gmc_none_returns_identity():
    g = GMC("none")
    assert np.allclose(g.apply(np.zeros((64, 64, 3), np.uint8)), np.eye(2, 3))


def test_gmc_first_frame_returns_identity():
    g = GMC("sparseOptFlow")
    frame = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    assert np.allclose(g.apply(frame), np.eye(2, 3))


def test_apply_warp_translates_a_box():
    warp = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    assert apply_warp_to_xyxy((0, 0, 10, 10), warp) == pytest.approx((5.0, -3.0, 15.0, 7.0))


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["iou", "bytetrack", "botsort"])
def test_tracker_keeps_one_id_for_a_smoothly_moving_object(cfg, name):
    tracker = build_tracker(cfg, name)
    ids = set()
    for step in range(20):
        for track in tracker.update([det(step * 5, 100)], frame=None):
            ids.add(track.track_id)
    assert len(ids) == 1, f"{name} churned ids on a clean trajectory: {ids}"


@pytest.mark.parametrize("name", ["bytetrack", "botsort"])
def test_kalman_trackers_survive_a_short_gap(cfg, name):
    """The whole point of track_buffer: a brief miss must not cost the identity."""
    tracker = build_tracker(cfg, name)
    ids = set()
    for step in range(30):
        occluded = 12 <= step < 18            # 6 frames with no detection
        dets = [] if occluded else [det(step * 5, 100)]
        for track in tracker.update(dets, frame=None):
            ids.add(track.track_id)
    assert len(ids) == 1, f"{name} lost the identity across a 6-frame gap: {ids}"


def test_iou_baseline_loses_the_identity_across_a_gap(cfg):
    """The baseline is meant to fail here -- that is the number BoT-SORT beats."""
    tracker = build_tracker(cfg, "iou")
    ids = set()
    for step in range(30):
        dets = [] if 12 <= step < 18 else [det(step * 5, 100)]
        for track in tracker.update(dets, frame=None):
            ids.add(track.track_id)
    assert len(ids) > 1


def test_tracker_does_not_associate_across_classes(cfg):
    """A person leaving and a bus arriving at the same place are two objects."""
    tracker = build_tracker(cfg, "botsort")
    ids_by_class: dict[int, set[int]] = {}
    for step in range(12):
        cls_id = 0 if step < 6 else 3
        for track in tracker.update([det(100, 100, cls_id=cls_id)], frame=None):
            ids_by_class.setdefault(track.cls_id, set()).add(track.track_id)
    assert len(ids_by_class) == 2
    assert not (ids_by_class[0] & ids_by_class[3])


def test_two_separated_objects_get_two_ids(cfg):
    tracker = build_tracker(cfg, "botsort")
    ids = set()
    for step in range(15):
        for track in tracker.update([det(step * 4, 60), det(step * 4 + 400, 300)], frame=None):
            ids.add(track.track_id)
    assert len(ids) == 2


def test_low_confidence_detection_does_not_create_a_track(cfg):
    """new_track_thresh exists so detector noise never becomes an identity."""
    tracker = build_tracker(cfg, "botsort")
    ids = set()
    for _ in range(10):
        for track in tracker.update([det(100, 100, conf=0.2)], frame=None):
            ids.add(track.track_id)
    assert not ids


def test_gallery_restores_the_original_id_after_a_long_occlusion(cfg):
    """The D1 differentiator, in miniature.

    An object disappears for longer than the tracker can coast, then returns
    with the same appearance. Without the gallery this is a new identity;
    with it, the original id comes back flagged `reid_restored`.
    """
    tracker = build_tracker(cfg, "botsort", with_reid=True, with_gallery=True)
    signature = np.zeros(32, dtype=np.float32)
    signature[0] = 1.0

    seen: list[int] = []
    restored = False
    for step in range(120):
        gap = 30 <= step < 100            # longer than track_buffer (30)
        dets = [] if gap else [det(100 + (step % 5), 100)]
        if dets:
            tracker.set_embeddings({0: signature})
        for track in tracker.update(dets, frame=None):
            seen.append(track.track_id)
            restored |= track.reid_restored

    assert seen, "the object was never tracked at all"
    assert restored, "the gallery never restored an identity"
    assert len(set(seen)) == 1, f"identity was not preserved: {set(seen)}"
