"""YOLO label validation.

PRD 18, "Data" level: label-format validation (normalised coords in [0,1],
valid class ids), corrupt-image detection, class-distribution report.

A label file is `class_id cx cy w h`, all box values normalised to [0, 1].
Nearly every silent training failure at this stage is one of five things, so
each gets its own error code rather than a generic "bad label":

    bad_field_count   -- a stray column, usually a confidence score left in by
                         a model-assisted pre-labelling export (PRD R1)
    bad_class_id      -- a class id outside the configured class list, which is
                         what happens when someone reorders `dataset.classes`
    out_of_range      -- pixel coordinates exported instead of normalised ones
    degenerate_box    -- zero or negative width/height
    box_out_of_frame  -- centre plus half-extent falls outside the image

The counts feed the dataset card, and every finding is logged as a structured
event so the failure gallery in Phase 3.4 can cite `label_error` cases.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import Config
from ..utils.logging import get_logger
from .manifest import ManifestRow, class_list

log = get_logger("data.validate")

# A box may exceed the frame by this much before it is flagged. Truncated
# objects at the frame edge ARE labelled (annotation guide rule 4), and an
# annotation tool's rounding can push a clipped edge a hair past 1.0.
EDGE_TOLERANCE = 1e-3


@dataclass
class LabelIssue:
    image_id: str
    label_path: str
    line_no: int
    code: str
    detail: str


@dataclass
class ValidationReport:
    files_checked: int = 0
    files_missing: int = 0
    files_empty: int = 0          # a legitimate hard negative, not an error
    boxes_total: int = 0
    boxes_valid: int = 0
    class_counts: Counter = field(default_factory=Counter)
    issues: list[LabelIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "files_checked": self.files_checked,
            "files_missing": self.files_missing,
            "files_empty": self.files_empty,
            "boxes_total": self.boxes_total,
            "boxes_valid": self.boxes_valid,
            "class_counts": dict(self.class_counts),
            "issue_counts": dict(Counter(i.code for i in self.issues)),
            "n_issues": len(self.issues),
        }


@dataclass
class ParsedBox:
    cls_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        x1 = (self.cx - self.w / 2) * width
        y1 = (self.cy - self.h / 2) * height
        x2 = (self.cx + self.w / 2) * width
        y2 = (self.cy + self.h / 2) * height
        return x1, y1, x2, y2


def parse_label_file(
    path: Path, n_classes: int, image_id: str = ""
) -> tuple[list[ParsedBox], list[LabelIssue]]:
    """Parse one YOLO label file, returning valid boxes and per-line issues."""
    boxes: list[ParsedBox] = []
    issues: list[LabelIssue] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                LabelIssue(image_id, str(path), line_no, "bad_field_count",
                           f"expected 5 fields, got {len(parts)}: {line[:60]!r}")
            )
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append(
                LabelIssue(image_id, str(path), line_no, "unparseable",
                           f"non-numeric field in {line[:60]!r}")
            )
            continue

        if not 0 <= cls_id < n_classes:
            issues.append(
                LabelIssue(image_id, str(path), line_no, "bad_class_id",
                           f"class id {cls_id} outside [0, {n_classes - 1}]")
            )
            continue
        if any(not (0.0 - EDGE_TOLERANCE <= v <= 1.0 + EDGE_TOLERANCE) for v in (cx, cy, w, h)):
            issues.append(
                LabelIssue(image_id, str(path), line_no, "out_of_range",
                           f"({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f}) not normalised to [0,1] "
                           "-- pixel coordinates exported by mistake?")
            )
            continue
        if w <= 0 or h <= 0:
            issues.append(
                LabelIssue(image_id, str(path), line_no, "degenerate_box",
                           f"w={w:.4f}, h={h:.4f}")
            )
            continue
        if (cx - w / 2 < -EDGE_TOLERANCE or cy - h / 2 < -EDGE_TOLERANCE
                or cx + w / 2 > 1.0 + EDGE_TOLERANCE or cy + h / 2 > 1.0 + EDGE_TOLERANCE):
            issues.append(
                LabelIssue(image_id, str(path), line_no, "box_out_of_frame",
                           f"centre {cx:.3f},{cy:.3f} with extent {w:.3f}x{h:.3f} leaves the frame")
            )
            continue

        boxes.append(ParsedBox(cls_id, cx, cy, w, h))
    return boxes, issues


def validate_and_enrich(rows: Sequence[ManifestRow], cfg: Config) -> ValidationReport:
    """Validate every label file and write `n_objects` / `classes` back to the rows.

    Enrichment happens here rather than in the sampler because object counts
    only exist once labels do -- and `n_objects` is what makes the "crowded"
    slice (PRD 13.3) and the class-distribution check in the splitter possible.
    """
    report = ValidationReport()
    names = cfg.dataset.classes

    for row in rows:
        label_path = cfg.root / row.label_path
        if not label_path.exists():
            report.files_missing += 1
            row.n_objects = 0
            row.classes = ""
            continue

        report.files_checked += 1
        boxes, issues = parse_label_file(label_path, len(names), row.image_id)
        report.issues.extend(issues)
        report.boxes_total += len(boxes) + len(issues)
        report.boxes_valid += len(boxes)
        if not boxes and not issues:
            report.files_empty += 1

        row.n_objects = len(boxes)
        row.classes = class_list(names[b.cls_id] for b in boxes)
        for b in boxes:
            report.class_counts[names[b.cls_id]] += 1

        if row.n_objects > cfg.attributes.crowded_object_count:
            row.occlusion_level = "heavy" if row.occlusion_level == "unknown" else row.occlusion_level

    for issue in report.issues[:50]:
        log.warning(
            f"label issue [{issue.code}] {issue.label_path}:{issue.line_no} -- {issue.detail}",
            extra={"event": "label_issue", "code": issue.code, "image_id": issue.image_id},
        )
    if len(report.issues) > 50:
        log.warning(
            f"... and {len(report.issues) - 50} further label issues (see reports/)",
            extra={"event": "label_issue_overflow", "n": len(report.issues)},
        )
    return report


def check_images_readable(rows: Iterable[ManifestRow], cfg: Config) -> list[str]:
    """Corrupt-image detection (PRD 18, Data level). Returns offending image ids."""
    import cv2

    bad: list[str] = []
    for row in rows:
        path = cfg.root / row.image_path
        if not path.exists():
            bad.append(row.image_id)
            continue
        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            bad.append(row.image_id)
    return bad
