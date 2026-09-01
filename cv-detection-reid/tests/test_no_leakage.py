"""The leakage gate. PRD US-1.2, Principle 4, Risk R3 (rated **Critical**).

> "an automated test asserts zero video overlap between splits and fails CI if
>  violated"

If this test ever fails on the committed manifest, every metric produced since
the leak was introduced is void and the experiments must be re-run. It is
therefore the one test that is allowed to be blunt.

The guard is checked in three ways:
  1. against the real committed manifest, if one exists;
  2. against a synthetic leaked manifest, proving the assertion actually fires
     (a guard that has never been seen to fail is not a guard);
  3. against the `random` split mode, proving *why* frame-level splitting is
     forbidden -- it leaks by construction.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.manifest import ManifestRow, read_manifest
from src.data.splitter import (
    LeakageError,
    assert_no_leakage,
    group_by_scene,
    split_dataset,
)


def make_rows(n_scenes=6, frames_per_scene=10):
    """A synthetic manifest: `sceneNN` with two camera files each."""
    rows = []
    for s in range(n_scenes):
        scene = f"scene{s:02d}"
        for cam in ("camA", "camB"):
            for f in range(frames_per_scene):
                rows.append(
                    ManifestRow(
                        image_id=f"{scene}_{cam}_f{f:04d}",
                        source_video=f"{scene}_{cam}.mp4",
                        scene_id=scene,
                        frame_no=f * 15,
                        timestamp_s=f * 0.5,
                        width=960, height=540,
                        lighting=("day", "dusk", "night")[s % 3],
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# The guard fires
# ---------------------------------------------------------------------------


def test_leakage_is_detected_when_a_scene_spans_two_splits():
    rows = make_rows()
    for row in rows:
        row.split = "train"
    rows[0].split = "test"          # one frame of scene00 leaks into test
    with pytest.raises(LeakageError):
        assert_no_leakage(rows)


def test_leakage_is_detected_when_a_video_spans_two_splits():
    rows = make_rows()
    for row in rows:
        row.split = "train" if row.source_video.endswith("camA.mp4") else "val"
    # Every scene now spans train and val through its two camera files.
    with pytest.raises(LeakageError):
        assert_no_leakage(rows)


def test_random_frame_splitting_leaks_by_construction():
    """This is exactly the fatal flaw PRD Principle 4 names."""
    cfg = load_config()
    cfg = type(cfg)(**{**cfg.__dict__, "splits": type(cfg.splits)(0.7, 0.15, 0.15, "random")})
    rows = make_rows()
    split_dataset(rows, cfg)
    with pytest.raises(LeakageError):
        assert_no_leakage(rows)


# ---------------------------------------------------------------------------
# The real splitter does not leak
# ---------------------------------------------------------------------------


def test_scene_level_split_produces_no_leakage():
    cfg = load_config()
    rows = make_rows()
    split_dataset(rows, cfg)
    assert_no_leakage(rows)          # must not raise


def test_scene_level_split_keeps_both_camera_views_together():
    """The cross-camera clip (M18) is two files of ONE scene; they must not split."""
    cfg = load_config()
    rows = make_rows()
    split_dataset(rows, cfg)
    for scene, scene_rows in group_by_scene(rows).items():
        assert len({r.split for r in scene_rows}) == 1, scene


def test_split_ratios_are_close_to_target():
    cfg = load_config()
    rows = make_rows(n_scenes=20, frames_per_scene=10)
    report = split_dataset(rows, cfg)
    for name, target in cfg.splits.ratios.items():
        assert abs(report.realised[name] - target) < 0.12, (name, report.realised)


def test_every_split_is_non_empty_when_scenes_allow_it():
    cfg = load_config()
    rows = make_rows(n_scenes=4, frames_per_scene=5)
    report = split_dataset(rows, cfg)
    assert all(report.counts[s] > 0 for s in ("train", "val", "test")), report.counts


def test_split_is_deterministic_for_a_fixed_seed():
    """PRD NFR-10: two runs of the same config must agree."""
    cfg = load_config()
    a, b = make_rows(), make_rows()
    split_dataset(a, cfg)
    split_dataset(b, cfg)
    assert [r.split for r in a] == [r.split for r in b]


def test_split_report_warns_when_a_class_is_missing_from_a_split():
    cfg = load_config()
    rows = make_rows()
    for row in rows:
        row.classes = "person"
    rows[0].classes = "bicycle"      # a class that lives in exactly one scene
    report = split_dataset(rows, cfg)
    assert any("bicycle" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# The committed dataset
# ---------------------------------------------------------------------------


def test_committed_manifest_has_no_leakage():
    """The CI gate proper. Skips only when no dataset has been built yet."""
    cfg = load_config()
    path = cfg.path("manifest")
    if not path.exists():
        pytest.skip("no manifest built yet -- run `python -m src.cli sample`")
    rows = read_manifest(path)
    if not any(r.split in ("train", "val", "test") for r in rows):
        pytest.skip("manifest not split yet -- run `python -m src.cli split`")
    assert_no_leakage(rows)


def test_committed_manifest_test_split_is_never_empty():
    cfg = load_config()
    path = cfg.path("manifest")
    if not path.exists():
        pytest.skip("no manifest built yet")
    rows = read_manifest(path)
    counts = {s: sum(1 for r in rows if r.split == s) for s in ("train", "val", "test")}
    if sum(counts.values()) == 0:
        pytest.skip("manifest not split yet")
    assert counts["test"] > 0, counts
    assert counts["val"] > 0, counts
