"""Config, data-format and ReID-gallery tests.

PRD 18: config validation; label-format validation; MOT compatibility; cosine
distance and gallery matching; and the fault-injection cases (zero detections,
empty gallery, corrupt input) that NFR-7 requires not to crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config, ConfigError, load_config
from src.data.manifest import COLUMNS, ManifestRow, read_manifest, write_manifest
from src.data.mot import MotRow, frame_index, read_mot, to_yolo_lines, write_mot
from src.data.validate_labels import parse_label_file
from src.pipeline.analytics import count_line_crossings, summarise_tracks, uoca
from src.eval.tracking_metrics import TrackBox


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_loads_and_validates():
    cfg = load_config()
    assert cfg.dataset.classes
    assert abs(sum(cfg.splits.ratios.values()) - 1.0) < 1e-9


def test_fingerprint_is_stable_and_sensitive():
    a = load_config()
    b = load_config()
    assert a.fingerprint() == b.fingerprint()
    changed = Config(**{**a.__dict__, "seed": a.seed + 1})
    assert changed.fingerprint() != a.fingerprint()


def test_unknown_config_key_is_rejected(tmp_path):
    """A typo'd knob that is silently ignored is how numbers become irreproducible."""
    path = tmp_path / "configs" / "bad.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("detection:\n  confidence: 0.3\n", encoding="utf-8")   # not `conf`
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


@pytest.mark.parametrize("body,match", [
    ("splits:\n  train: 0.8\n  val: 0.3\n  test: 0.1\n", "sum to 1.0"),
    ("detection:\n  conf: 1.5\n", "conf"),
    ("detection:\n  imgsz: 100\n", "multiple of 32"),
    ("eval:\n  primary_iou: 0.42\n", "primary_iou"),
    ("tracking:\n  tracker_type: deepsort\n", "tracker_type"),
    ("attributes:\n  lighting_night_below: 200\n", "lighting"),
])
def test_invalid_config_values_fail_loudly(tmp_path, body, match):
    path = tmp_path / "configs" / "bad.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_config(path)


def test_config_override_reaches_the_typed_object(tmp_path):
    cfg = load_config(overrides={"detection.conf": 0.4})
    assert cfg.detection.conf == 0.4


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trips_with_correct_types(tmp_path):
    rows = [ManifestRow(
        image_id="s01_f000030", source_video="s01.mp4", scene_id="s01",
        frame_no=30, timestamp_s=1.0, width=960, height=540,
        blur_score=61.5, n_objects=3, classes="bus;person",
    )]
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    back = read_manifest(path)[0]
    assert back.frame_no == 30 and isinstance(back.frame_no, int)
    assert back.blur_score == pytest.approx(61.5)
    assert back.classes == "bus;person"
    assert back.width == 960


def test_manifest_missing_column_is_an_error(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("image_id,frame_no\nx,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        read_manifest(path)


def test_manifest_schema_matches_the_dataclass():
    assert set(COLUMNS) <= set(ManifestRow.__dataclass_fields__)


# ---------------------------------------------------------------------------
# MOT <-> YOLO
# ---------------------------------------------------------------------------


def test_mot_frames_are_one_indexed():
    assert frame_index(0) == 1


def test_mot_round_trip(tmp_path):
    rows = [MotRow(1, 7, 10.0, 20.0, 30.0, 40.0, 1.0, 2, 0.9)]
    path = tmp_path / "gt.txt"
    write_mot(path, rows)
    back = read_mot(path)[0]
    assert back.track_id == 7 and back.cls_id == 2
    assert back.xyxy == pytest.approx((10.0, 20.0, 40.0, 60.0))


def test_low_visibility_rows_are_ignored():
    assert MotRow(1, 1, 0, 0, 10, 10, visibility=0.1).ignore
    assert not MotRow(1, 1, 0, 0, 10, 10, visibility=0.9).ignore


def test_to_yolo_lines_normalises_and_centres():
    rows = [MotRow(1, 1, 100.0, 50.0, 200.0, 100.0, cls_id=3)]
    line = to_yolo_lines(rows, 1000, 500)[0].split()
    assert int(line[0]) == 3
    assert float(line[1]) == pytest.approx(0.2)      # (100+200/2)/1000
    assert float(line[2]) == pytest.approx(0.2)      # (50+100/2)/500
    assert float(line[3]) == pytest.approx(0.2)      # 200/1000
    assert float(line[4]) == pytest.approx(0.2)      # 100/500


def test_to_yolo_lines_clips_a_truncated_box_to_the_frame():
    """Annotation guide rule 4: edge-truncated objects ARE labelled, clipped."""
    rows = [MotRow(1, 1, -50.0, 0.0, 100.0, 100.0)]
    line = to_yolo_lines(rows, 200, 200)[0].split()
    assert 0.0 <= float(line[1]) <= 1.0
    assert float(line[3]) == pytest.approx(50 / 200)   # only 50 px are visible


def test_to_yolo_lines_drops_ignored_rows_by_default():
    rows = [MotRow(1, 1, 0.0, 0.0, 50.0, 50.0, visibility=0.05)]
    assert to_yolo_lines(rows, 200, 200) == []
    assert len(to_yolo_lines(rows, 200, 200, include_ignored=True)) == 1


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("line,code", [
    ("0 0.5 0.5 0.2 0.2 0.98", "bad_field_count"),      # a stray confidence column
    ("9 0.5 0.5 0.2 0.2", "bad_class_id"),
    ("0 320 240 100 80", "out_of_range"),               # pixels, not normalised
    ("0 0.5 0.5 0.0 0.2", "degenerate_box"),
    ("0 0.95 0.5 0.5 0.2", "box_out_of_frame"),
    ("0 0.5 x 0.2 0.2", "unparseable"),
])
def test_label_errors_get_specific_codes(tmp_path, line, code):
    path = tmp_path / "a.txt"
    path.write_text(line + "\n", encoding="utf-8")
    boxes, issues = parse_label_file(path, n_classes=6)
    assert not boxes
    assert issues and issues[0].code == code


def test_valid_label_parses():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.txt"
        path.write_text("2 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        boxes, issues = parse_label_file(path, n_classes=6)
    assert not issues and len(boxes) == 1
    assert boxes[0].to_xyxy(1000, 500) == pytest.approx((400.0, 150.0, 600.0, 350.0))


def test_empty_label_file_is_a_hard_negative_not_an_error(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("", encoding="utf-8")
    boxes, issues = parse_label_file(path, n_classes=6)
    assert not boxes and not issues


# ---------------------------------------------------------------------------
# ReID gallery
# ---------------------------------------------------------------------------


@pytest.fixture
def gallery():
    from src.reid.gallery import ReidGallery

    return ReidGallery(load_config(), threshold=0.3)


def vec(*values):
    v = np.zeros(16, dtype=np.float32)
    for i, x in enumerate(values):
        v[i] = x
    return v / max(1e-12, float(np.linalg.norm(v)))


def test_gallery_restores_a_close_match(gallery):
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=10)
    out = gallery.query(vec(1, 0.05), cls_id=0, frame=40)
    assert out.matched and out.track_id == 7


def test_gallery_rejects_a_distant_match(gallery):
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=10)
    out = gallery.query(vec(0, 1), cls_id=0, frame=40)
    assert not out.matched and out.reason == "above_threshold"


def test_gallery_never_matches_across_classes(gallery):
    """PRD 9.4: a car is never re-identified as a person."""
    gallery.add(7, cls_id=1, embedding=vec(1, 0), frame=10)
    out = gallery.query(vec(1, 0), cls_id=0, frame=40)
    assert not out.matched and out.reason == "class_gated"


def test_gallery_entries_expire_after_the_ttl(gallery):
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=10)
    gallery.expire(10 + gallery.ttl + 1)
    assert len(gallery) == 0
    assert not gallery.query(vec(1, 0), cls_id=0, frame=10 + gallery.ttl + 1).matched


def test_gallery_excludes_currently_active_ids(gallery):
    """An id already on screen must not also be restored onto another box."""
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=10)
    assert not gallery.query(vec(1, 0), cls_id=0, frame=20, exclude=[7]).matched


def test_gallery_ema_moves_the_stored_embedding_towards_new_evidence(gallery):
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=10)
    before = gallery.entries[("cam0", 7)].embedding.copy()
    gallery.add(7, cls_id=0, embedding=vec(0, 1), frame=20)
    after = gallery.entries[("cam0", 7)].embedding
    assert after[1] > before[1]
    assert float(np.linalg.norm(after)) == pytest.approx(1.0, abs=1e-5)


def test_empty_gallery_and_missing_embedding_do_not_crash(gallery):
    """NFR-7 fault injection: the empty cases must be states, not exceptions."""
    assert not gallery.query(vec(1, 0), cls_id=0, frame=1).matched
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=1)
    assert not gallery.query(None, cls_id=0, frame=2).matched


def test_cross_camera_query_is_opt_in(gallery):
    gallery.add(7, cls_id=0, embedding=vec(1, 0), frame=10, camera_id="camA")
    assert not gallery.query(vec(1, 0), cls_id=0, frame=20, camera_id="camB").matched
    out = gallery.query(vec(1, 0), cls_id=0, frame=20, camera_id="camB", cross_camera=True)
    assert out.matched and out.track_id == 7


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


def tb(frame, tid, x, cls_id=0):
    return TrackBox(frame, tid, (float(x), 0.0, float(x + 20), 40.0), cls_id)


def test_short_flickers_are_not_counted_as_unique_objects():
    """The identity layer exists to stop detector noise becoming an object."""
    preds = [tb(f, 1, f * 5) for f in range(1, 20)] + [tb(5, 99, 500)]
    out = summarise_tracks(preds, ("person", "car"), min_track_frames=3)
    assert out.unique_total == 1


def test_dwell_time_uses_the_frame_span():
    preds = [tb(f, 1, f * 5) for f in range(1, 31)]      # 30 frames at 30 fps
    out = summarise_tracks(preds, ("person",), fps=30.0)
    assert out.dwell_seconds[1] == pytest.approx(1.0)


def test_uoca_is_absolute_percentage_error():
    assert uoca(40, 40) == 0.0
    assert uoca(44, 40) == pytest.approx(10.0)
    assert uoca(36, 40) == pytest.approx(10.0)


def test_line_crossing_counts_direction_and_nets_out_a_return():
    down = [tb(f, 1, 0) for f in range(1, 4)]
    down = [TrackBox(f, 1, (0.0, float(f * 40), 20.0, float(f * 40 + 40)), 0) for f in range(1, 6)]
    counts = count_line_crossings(down, (0.0, 100.0, 200.0, 100.0), ("person",))
    assert counts["in"] + counts["out"] == 1
