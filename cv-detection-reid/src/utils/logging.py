"""Console + JSONL logging, and a small markdown/console table renderer.

Two sinks by design:
  * console -- human-readable progress for the operator.
  * JSONL   -- machine-readable events for the failure analysis in Phase 3.4.
               Warnings such as `frame_dropped_blur` or `label_out_of_range`
               are the raw material for the PRD's failure gallery, so they
               must be queryable, not printed once and lost.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonlHandler(logging.Handler):
    """Writes one JSON object per line, including structured `extra` fields."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            for key, value in record.__dict__.items():
                if key not in _RESERVED:
                    try:
                        json.dumps(value)
                    except (TypeError, ValueError):
                        value = repr(value)
                    payload[key] = value
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:  # never let logging kill a long run
            self.handleError(record)


def setup_logging(log_path: Path | None = None, verbose: bool = False) -> logging.Logger:
    root = logging.getLogger("cvdr")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname).1s %(message)s"))
    root.addHandler(console)

    if log_path is not None:
        root.addHandler(JsonlHandler(log_path))
    return root


def get_logger(name: str = "cvdr") -> logging.Logger:
    return logging.getLogger(name if name.startswith("cvdr") else f"cvdr.{name}")


# ---------------------------------------------------------------------------
# Table rendering -- the same function feeds the console and the markdown
# reports, so a number a reviewer reads on screen is byte-identical to the one
# committed to reports/.
# ---------------------------------------------------------------------------

def render_table(headers: Sequence[str], rows: Iterable[Sequence[Any]], markdown: bool = False) -> str:
    rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        padded = [str(c).ljust(widths[i]) for i, c in enumerate(cells)]
        return ("| " + " | ".join(padded) + " |") if markdown else ("  " + "  ".join(padded))

    out = [line(headers)]
    if markdown:
        out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    else:
        out.append("  " + "  ".join("-" * w for w in widths))
    out.extend(line(r) for r in rows)
    return "\n".join(out)
