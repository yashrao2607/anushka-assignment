"""Core data models shared across the ingestion pipeline.

These mirror Section 10 of the PRD. The important property is *provenance*:
every Chunk can be traced back to an exact file, page and character span, which
is what makes the Phase 3 citation feature possible at all. If provenance is not
captured here, no amount of later work can recover it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Page:
    """One page (PDF) or one logical block (DOCX/MD/TXT/HTML/CSV)."""

    page_no: int
    text: str
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class RawDocument:
    """A parsed but not-yet-chunked source document."""

    doc_id: str
    source_path: str
    doc_title: str
    doc_type: str
    pages: list[Page] = field(default_factory=list)
    bytes_size: int = 0
    load_error: str | None = None
    scanned_pages: list[int] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.pages)


@dataclass
class Chunk:
    """A retrievable unit of a document, with full provenance.

    `text`       -- what a user is shown and what a citation quotes.
    `embed_text` -- what is actually embedded; may carry a heading prefix so the
                    chunk is not semantically orphaned from its parent section.
    """

    chunk_id: str
    doc_id: str
    text: str
    embed_text: str
    source_path: str
    doc_title: str
    doc_type: str
    page_no: int | None
    section_heading: str | None
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    content_sha256: str
    ingested_at: str
    config_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestStats:
    """Summary of one ingestion run -- printed and written to the manifest."""

    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    pages: int = 0
    chunks: int = 0
    tokens: int = 0
    chars: int = 0
    chunks_dropped: int = 0
    duplicates_skipped: int = 0
    scanned_pages: int = 0
    duration_s: float = 0.0

    def as_rows(self) -> list[tuple[str, Any]]:
        return [
            ("Files found", self.files_seen),
            ("Files ingested", self.files_ingested),
            ("Files skipped (unsupported)", self.files_skipped),
            ("Files failed (parse error)", self.files_failed),
            ("Pages / blocks parsed", self.pages),
            ("Pages with no text layer", self.scanned_pages),
            ("Chunks produced", self.chunks),
            ("Chunks dropped (too small / noisy)", self.chunks_dropped),
            ("Duplicate chunks skipped", self.duplicates_skipped),
            ("Total tokens (approx.)", self.tokens),
            ("Total characters", self.chars),
            ("Duration (s)", round(self.duration_s, 2)),
        ]
