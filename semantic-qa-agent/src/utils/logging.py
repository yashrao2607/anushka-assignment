"""Console + JSONL logging.

Two sinks by design:
  * console  -- human-readable progress for the operator.
  * JSONL    -- machine-readable events for the failure analysis in Phase 3.
                Warnings such as `unparsed_file` are the raw material for the
                PRD's failure-analysis deliverable, so they must be queryable,
                not just printed and lost.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlHandler(logging.Handler):
    """Writes one JSON object per line, including any structured `extra` fields."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message", "asctime", "taskName",
    }

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
                if key not in self._RESERVED and not key.startswith("_"):
                    try:
                        json.dumps(value)
                        payload[key] = value
                    except (TypeError, ValueError):
                        payload[key] = str(value)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:  # logging must never take the pipeline down
            self.handleError(record)


_CONFIGURED = False


def setup_logging(log_dir: Path, verbose: bool = False) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("sqa")
    if _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(console)

    jsonl = JsonlHandler(Path(log_dir) / "ingest.jsonl")
    jsonl.setLevel(logging.DEBUG)
    logger.addHandler(jsonl)

    _CONFIGURED = True
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("sqa")


def render_table(rows: list[tuple[str, Any]], title: str = "") -> str:
    """Small dependency-free table renderer for the run summary."""
    if not rows:
        return ""
    width_l = max(len(str(k)) for k, _ in rows)
    width_r = max(len(str(v)) for _, v in rows)
    bar = "+" + "-" * (width_l + 2) + "+" + "-" * (width_r + 2) + "+"
    out = []
    if title:
        out.append(title)
    out.append(bar)
    for key, value in rows:
        out.append(f"| {str(key).ljust(width_l)} | {str(value).rjust(width_r)} |")
    out.append(bar)
    return "\n".join(out)
