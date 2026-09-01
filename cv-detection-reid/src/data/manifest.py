"""The dataset manifest -- one row per sampled frame.

PRD 7.6 fixes the schema:
    image_id, source_video, frame_no, split, timestamp, lighting, weather,
    occlusion_level, n_objects, classes, blur_score

Everything downstream reads this file: the splitter writes `split` into it, the
label validator reads `n_objects`/`classes` back out, and the difficulty-sliced
evaluation (13.3) joins predictions to it on `image_id`. It is therefore the
one artefact whose schema must not drift, so it is defined once, here.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

# Column order is part of the contract; readers may be positional.
COLUMNS = (
    "image_id",
    "source_video",
    "scene_id",
    "frame_no",
    "timestamp_s",
    "split",
    "width",
    "height",
    "lighting",
    "weather",
    "occlusion_level",
    "blur_score",
    "blur_level",
    "brightness",
    "contrast",
    "motion_score",
    "n_objects",
    "classes",
    "image_path",
    "label_path",
)


@dataclass
class ManifestRow:
    image_id: str
    source_video: str
    scene_id: str
    frame_no: int
    timestamp_s: float
    split: str = "unassigned"
    width: int = 0
    height: int = 0
    lighting: str = "unknown"
    weather: str = "unknown"
    occlusion_level: str = "unknown"
    blur_score: float = 0.0
    blur_level: str = "unknown"
    brightness: float = 0.0
    contrast: float = 0.0
    motion_score: float = 0.0
    n_objects: int = 0
    classes: str = ""          # ";"-joined class names present in the frame
    image_path: str = ""
    label_path: str = ""

    def as_row(self) -> dict[str, Any]:
        return {k: asdict(self)[k] for k in COLUMNS}


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())
            n += 1
    return n


def read_manifest(path: Path) -> list[ManifestRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found: {path} -- run `python -m src.cli sample` first"
        )
    typed = {f.name: f.type for f in fields(ManifestRow)}
    out: list[ManifestRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest {path} is missing column(s): {sorted(missing)}")
        for raw in reader:
            kwargs: dict[str, Any] = {}
            for key, value in raw.items():
                if key not in typed:
                    continue
                t = typed[key]
                if t is int:
                    kwargs[key] = int(float(value)) if value else 0
                elif t is float:
                    kwargs[key] = float(value) if value else 0.0
                else:
                    kwargs[key] = value
            out.append(ManifestRow(**kwargs))
    return out


def class_list(names: Iterable[str]) -> str:
    """Deterministic, sorted, de-duplicated encoding of a frame's classes."""
    return ";".join(sorted(set(n for n in names if n)))


def update_splits(path: Path, assignment: dict[str, str]) -> int:
    """Rewrite only the `split` column, keyed by source_video.

    Kept separate from `write_manifest` so the splitter never has to
    re-derive attributes -- re-computing them would risk a manifest whose
    difficulty labels no longer match the frames they describe.
    """
    rows = read_manifest(path)
    changed = 0
    for row in rows:
        target = assignment.get(row.source_video)
        if target and row.split != target:
            row.split = target
            changed += 1
    write_manifest(path, rows)
    return changed
