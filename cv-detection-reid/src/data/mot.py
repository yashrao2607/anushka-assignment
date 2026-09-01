"""MOT-Challenge ground truth: read, write, and convert to YOLO labels.

PRD 11.2 requires the per-frame output to be "MOT-challenge compatible, so
standard evaluation tooling (TrackEval) works without a converter". The same
format is the natural container for *ground truth* with identities, which
Phase 2.4 needs for MOTA/IDF1/HOTA. Adopting it in Phase 1 means the tracking
metrics harness has its input format already settled.

Format, one comma-separated row per object per frame:

    frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility

Two conventions that bite if left implicit:
  * `frame` is **1-indexed** (MOTChallenge). Our sampler indexes frames from 0
    as OpenCV reads them, so every lookup crosses that boundary exactly once,
    in `frame_index()` below, and nowhere else.
  * `class` is **our** class id from `dataset.classes`, not a MOT class id. Our
    label space is the one every other component speaks.

`visibility` in [0, 1] drives the ignore rule: annotation guide rule 2 says an
object occluded more than 70% is labelled `ignore` -- excluded from the loss
and from the metrics -- rather than mislabelled or dropped.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# Below this visibility an object is `ignore` (annotation guide rule 2).
IGNORE_VISIBILITY = 0.30


@dataclass(frozen=True)
class MotRow:
    frame: int          # 1-indexed
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: float = 1.0
    cls_id: int = 0
    visibility: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.left + self.width, self.top + self.height)

    @property
    def ignore(self) -> bool:
        return self.visibility < IGNORE_VISIBILITY


def frame_index(frame_no_zero_based: int) -> int:
    """Our 0-indexed frame number -> the MOT 1-indexed frame number."""
    return frame_no_zero_based + 1


def write_mot(path: Path, rows: Iterable[MotRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for r in sorted(rows, key=lambda r: (r.frame, r.track_id)):
            writer.writerow([
                r.frame, r.track_id,
                round(r.left, 2), round(r.top, 2), round(r.width, 2), round(r.height, 2),
                round(r.conf, 4), r.cls_id, round(r.visibility, 3),
            ])
            n += 1
    return n


def read_mot(path: Path) -> list[MotRow]:
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")
    out: list[MotRow] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for line_no, parts in enumerate(csv.reader(fh), start=1):
            parts = [p for p in parts if p.strip() != ""]
            if not parts or parts[0].lstrip().startswith("#"):
                continue
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no}: expected >= 6 fields, got {len(parts)}")
            vals = [float(p) for p in parts[:9]]
            while len(vals) < 9:
                vals.append(1.0 if len(vals) == 6 else (0.0 if len(vals) == 7 else 1.0))
            out.append(
                MotRow(
                    frame=int(vals[0]), track_id=int(vals[1]),
                    left=vals[2], top=vals[3], width=vals[4], height=vals[5],
                    conf=vals[6], cls_id=int(vals[7]), visibility=vals[8],
                )
            )
    return out


def group_by_frame(rows: Iterable[MotRow]) -> dict[int, list[MotRow]]:
    groups: dict[int, list[MotRow]] = defaultdict(list)
    for r in rows:
        groups[r.frame].append(r)
    return dict(groups)


def to_yolo_lines(
    rows: Sequence[MotRow], width: int, height: int, include_ignored: bool = False
) -> list[str]:
    """Convert one frame's MOT rows to normalised YOLO `cls cx cy w h` lines.

    Boxes are clipped to the frame first: annotation guide rule 4 keeps
    truncated objects at the frame edge, and a synthetic or tracker-derived box
    can legitimately extend past the border. Clipping preserves the visible
    extent, which is exactly what rule 1 asks to be labelled.
    """
    lines: list[str] = []
    for r in rows:
        if r.ignore and not include_ignored:
            continue
        x1 = max(0.0, min(r.left, width))
        y1 = max(0.0, min(r.top, height))
        x2 = max(0.0, min(r.left + r.width, width))
        y2 = max(0.0, min(r.top + r.height, height))
        w, h = x2 - x1, y2 - y1
        if w <= 1.0 or h <= 1.0:
            continue  # nothing visible left after clipping
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        lines.append(f"{r.cls_id} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    return lines
