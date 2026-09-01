"""Markdown + JSON report rendering.

PRD US-6.1: "One command produces the full metrics report." One renderer feeds
both the console and `reports/`, so a number a reviewer reads on screen is
byte-identical to the number committed to the repository.

Every table carries its PRD metric id (M1, M8, ...) and its target, and states
PASS/FAIL against it. A metrics report that does not say whether it met its own
acceptance criteria makes the reviewer do arithmetic the author should have
done.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.logging import render_table

# PRD 4.2-4.5. (metric id, human name, target, direction) where direction is
# "ge" for at-least and "le" for at-most.
DETECTION_TARGETS: tuple[tuple[str, str, float, str], ...] = (
    ("M1", "mAP@0.5", 0.75, "ge"),
    ("M2", "mAP@0.5:0.95", 0.50, "ge"),
    ("M3", "Precision @ conf 0.25", 0.80, "ge"),
    ("M4", "Recall @ conf 0.25", 0.75, "ge"),
    ("M5", "Mean IoU (matched)", 0.78, "ge"),
    ("M7", "Small-object AP", 0.35, "ge"),
)

TRACKING_TARGETS: tuple[tuple[str, str, float, str], ...] = (
    ("M8", "MOTA", 0.65, "ge"),
    ("M9", "IDF1", 0.70, "ge"),
    ("M10", "HOTA", 0.55, "ge"),
    ("M11", "ID switches / 1k frames", 15.0, "le"),
    ("M12", "MT ratio", 0.60, "ge"),
    ("M12b", "ML ratio", 0.15, "le"),
    ("M13", "Fragmentation (avg)", 2.0, "le"),
)


def verdict(value: float, target: float, direction: str) -> str:
    ok = value >= target if direction == "ge" else value <= target
    return "PASS" if ok else "FAIL"


def _fmt(v: Any, places: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{places}f}"
    return str(v)


def targets_table(values: Mapping[str, float], targets: Sequence[tuple[str, str, float, str]],
                  markdown: bool = True) -> str:
    rows = []
    for mid, name, target, direction in targets:
        if mid not in values:
            continue
        v = values[mid]
        arrow = "≥" if direction == "ge" else "≤"
        rows.append([mid, name, _fmt(v), f"{arrow} {target}", verdict(v, target, direction)])
    return render_table(["ID", "Metric", "Measured", "Target", "Verdict"], rows, markdown=markdown)


def per_class_table(ap50: Mapping[str, float], ap5095: Mapping[str, float],
                    support: Mapping[str, int], markdown: bool = True) -> str:
    rows = []
    for name in sorted(set(ap50) | set(support)):
        n = support.get(name, 0)
        rows.append([
            name,
            n,
            _fmt(ap50.get(name, 0.0)) if n else "n/a",
            _fmt(ap5095.get(name, 0.0)) if n else "n/a",
            # PRD M6: every class individually >= 0.60
            ("PASS" if ap50.get(name, 0.0) >= 0.60 else "FAIL") if n else "no GT",
        ])
    return render_table(["Class", "GT boxes", "AP@0.5", "AP@0.5:0.95", "M6 (>= 0.60)"], rows,
                        markdown=markdown)


def slice_table(slices: Mapping[str, Mapping[str, Any]], markdown: bool = True) -> str:
    """PRD 13.3, the difficulty-sliced results -- differentiator D4."""
    rows = []
    for name, m in slices.items():
        rows.append([
            name,
            m.get("n_images", 0),
            m.get("n_gt", 0),
            _fmt(m.get("map50", 0.0)),
            _fmt(m.get("recall", 0.0)),
            _fmt(m.get("precision", 0.0)),
        ])
    return render_table(["Slice", "Images", "GT boxes", "mAP@0.5", "Recall", "Precision"], rows,
                        markdown=markdown)


def confusion_table(matrix: Mapping[str, Mapping[str, int]], markdown: bool = True) -> str:
    labels = list(matrix.keys())
    headers = ["GT \\ Pred"] + labels
    rows = [[gt] + [matrix[gt].get(p, 0) for p in labels] for gt in labels]
    return render_table(headers, rows, markdown=markdown)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return path


def write_markdown(path: Path, title: str, sections: Sequence[tuple[str, str]],
                   provenance: Mapping[str, Any] | None = None) -> Path:
    """Write a report with a provenance block.

    PRD 9.7: every artefact records the config, commit and dataset version that
    produced it, so any number can be traced back to the exact run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [f"# {title}", "", f"*Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}*", ""]
    if provenance:
        out.append("## Provenance")
        out.append("")
        out.append(render_table(["Key", "Value"], [[k, v] for k, v in provenance.items()], markdown=True))
        out.append("")
    for heading, body in sections:
        out.append(f"## {heading}")
        out.append("")
        out.append(body)
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def git_sha(root: Path) -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
