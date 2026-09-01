"""Scene-level train/val/test splitting.

PRD Principle 4 and Risk R3, which rates a random frame split as **Critical**:

> "Splits are by video/scene, never by random frame -- adjacent frames are
>  near-duplicates and random splitting silently inflates every metric. This
>  single decision is the most common fatal flaw in video-CV projects."

The unit of assignment here is the **scene**, not the video, which is stricter
than the PRD's minimum. Two files can be two camera angles on the same
junction at the same minute; assigning them to different splits would leak the
test set into training just as surely as splitting by frame would. The
cross-camera clip required by M18 is precisely that case, so scene-level
grouping is not hypothetical caution.

`mode: random` exists only so `tests/test_no_leakage.py` can demonstrate that
the guard actually fires. It never produces a reported number.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..config import Config
from ..utils.logging import get_logger
from .manifest import ManifestRow

log = get_logger("data.splitter")

SPLITS = ("train", "val", "test")


class LeakageError(AssertionError):
    """Raised when a scene or video appears in more than one split."""


@dataclass
class SplitReport:
    assignment: dict[str, str]                      # source_video -> split
    scene_assignment: dict[str, str]                # scene_id     -> split
    counts: dict[str, int] = field(default_factory=dict)          # split -> n frames
    realised: dict[str, float] = field(default_factory=dict)      # split -> fraction
    target: dict[str, float] = field(default_factory=dict)
    scenes: dict[str, list[str]] = field(default_factory=dict)    # split -> scene ids
    class_counts: dict[str, Counter] = field(default_factory=dict)  # split -> class -> n
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts,
            "realised": self.realised,
            "target": self.target,
            "scenes": self.scenes,
            "class_counts": {k: dict(v) for k, v in self.class_counts.items()},
            "warnings": self.warnings,
        }


def group_by_scene(rows: Sequence[ManifestRow]) -> dict[str, list[ManifestRow]]:
    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        groups[row.scene_id].append(row)
    return dict(groups)


def assign_scenes(
    groups: dict[str, list[ManifestRow]],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, str]:
    """Greedy largest-scene-first assignment to the split with the biggest deficit.

    Whole scenes are indivisible, so exact ratios are generally unreachable.
    Placing the largest scene first minimises the worst-case imbalance -- the
    same argument as longest-processing-time-first bin packing. Ties break on a
    seeded shuffle so the result is reproducible (NFR-10) without being an
    artefact of dict ordering.
    """
    total = sum(len(v) for v in groups.values())
    if total == 0:
        return {}

    targets = {s: ratios[s] * total for s in SPLITS}
    current = {s: 0 for s in SPLITS}
    assignment: dict[str, str] = {}

    scene_ids = list(groups)
    random.Random(seed).shuffle(scene_ids)
    # Largest first; the shuffle above only decides ties.
    scene_ids.sort(key=lambda s: len(groups[s]), reverse=True)

    # Guarantee every split is non-empty when there are enough scenes to do so,
    # otherwise a 3-scene dataset can hand every frame to train and leave the
    # test split empty -- which reads as "no leakage" while being useless.
    if len(scene_ids) >= len(SPLITS):
        for split, scene in zip(sorted(SPLITS, key=lambda s: ratios[s]), reversed(scene_ids[-len(SPLITS):])):
            assignment[scene] = split
            current[split] += len(groups[scene])

    for scene in scene_ids:
        if scene in assignment:
            continue
        deficit = {s: targets[s] - current[s] for s in SPLITS}
        split = max(SPLITS, key=lambda s: (deficit[s], -ratios[s]))
        assignment[scene] = split
        current[split] += len(groups[scene])

    return assignment


def split_dataset(rows: Sequence[ManifestRow], cfg: Config) -> SplitReport:
    """Assign every frame to a split and return an auditable report."""
    if cfg.splits.mode == "random":
        log.warning(
            "splits.mode='random' produces LEAKED splits; for the leakage test only",
            extra={"event": "random_split_requested"},
        )
        rng = random.Random(cfg.seed)
        per_row: dict[str, str] = {}
        for row in rows:
            r = rng.random()
            if r < cfg.splits.train:
                per_row[row.image_id] = "train"
            elif r < cfg.splits.train + cfg.splits.val:
                per_row[row.image_id] = "val"
            else:
                per_row[row.image_id] = "test"
        for row in rows:
            row.split = per_row[row.image_id]
        return _summarise(rows, cfg, {}, {})

    groups = group_by_scene(rows)
    scene_assignment = assign_scenes(groups, cfg.splits.ratios, cfg.seed)
    for scene, scene_rows in groups.items():
        for row in scene_rows:
            row.split = scene_assignment[scene]

    video_assignment = {row.source_video: row.split for row in rows}
    return _summarise(rows, cfg, video_assignment, scene_assignment)


def _summarise(
    rows: Sequence[ManifestRow],
    cfg: Config,
    video_assignment: dict[str, str],
    scene_assignment: dict[str, str],
) -> SplitReport:
    counts = Counter(row.split for row in rows)
    total = sum(counts.values()) or 1
    scenes: dict[str, list[str]] = {s: [] for s in SPLITS}
    for scene, split in sorted(scene_assignment.items()):
        scenes[split].append(scene)

    class_counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for row in rows:
        if not row.classes:
            continue
        for name in row.classes.split(";"):
            if name:
                class_counts[row.split][name] += 1

    warnings: list[str] = []
    for split in SPLITS:
        if counts.get(split, 0) == 0:
            warnings.append(f"split '{split}' is empty -- not enough distinct scenes")
    # PRD 7.3: "Class distribution across splits is reported so a split is not
    # accidentally missing a class."
    for name in cfg.dataset.classes:
        missing = [s for s in SPLITS if counts.get(s, 0) and class_counts[s][name] == 0]
        if missing and any(class_counts[s][name] for s in SPLITS):
            warnings.append(f"class '{name}' is absent from split(s): {', '.join(missing)}")

    for w in warnings:
        log.warning(w, extra={"event": "split_warning"})

    return SplitReport(
        assignment=video_assignment,
        scene_assignment=scene_assignment,
        counts={s: counts.get(s, 0) for s in SPLITS},
        realised={s: round(counts.get(s, 0) / total, 4) for s in SPLITS},
        target={s: cfg.splits.ratios[s] for s in SPLITS},
        scenes=scenes,
        class_counts=class_counts,
        warnings=warnings,
    )


def assert_no_leakage(rows: Iterable[ManifestRow]) -> None:
    """Fail loudly if any scene or video spans more than one split.

    This is the assertion `tests/test_no_leakage.py` runs in CI. Per R3, if it
    ever fires, every number produced before the fix is void.
    """
    scene_splits: dict[str, set[str]] = defaultdict(set)
    video_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scene_splits[row.scene_id].add(row.split)
        video_splits[row.source_video].add(row.split)

    bad_scenes = {k: sorted(v) for k, v in scene_splits.items() if len(v) > 1}
    bad_videos = {k: sorted(v) for k, v in video_splits.items() if len(v) > 1}
    if bad_scenes or bad_videos:
        raise LeakageError(
            "DATA LEAKAGE: the same source appears in multiple splits.\n"
            f"  scenes: {bad_scenes}\n"
            f"  videos: {bad_videos}\n"
            "Per PRD R3 every metric computed on these splits is void."
        )


def materialise_splits(rows: Sequence[ManifestRow], cfg: Config, copy: bool = False) -> dict[str, int]:
    """Write `data/splits/{train,val,test}/{images,labels}` listings.

    By default this writes *listing files* (`train.txt` etc., the format
    Ultralytics accepts directly) rather than copying pixels. Copying 3,000
    JPEGs three times over is wasted disk and, worse, creates a second copy of
    the truth that can drift from the manifest.
    """
    import shutil

    out_root = cfg.path("splits_dir")
    written: dict[str, int] = {}
    for split in SPLITS:
        split_rows = [r for r in rows if r.split == split]
        listing = out_root / f"{split}.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text(
            "\n".join(str((cfg.root / r.image_path).as_posix()) for r in split_rows) + "\n",
            encoding="utf-8",
        )
        written[split] = len(split_rows)

        if copy:
            for sub in ("images", "labels"):
                (out_root / split / sub).mkdir(parents=True, exist_ok=True)
            for r in split_rows:
                src_img = cfg.root / r.image_path
                if src_img.exists():
                    shutil.copy2(src_img, out_root / split / "images" / src_img.name)
                src_lbl = cfg.root / r.label_path
                if src_lbl.exists():
                    shutil.copy2(src_lbl, out_root / split / "labels" / src_lbl.name)
    return written
