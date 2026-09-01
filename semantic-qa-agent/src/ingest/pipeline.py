"""Ingestion pipeline orchestrator -- PRD Section 8.2, stages [1]-[3].

Ties loading, cleaning and chunking into one idempotent, observable run:

    discover -> load -> detect boilerplate -> clean -> chunk -> dedupe -> write

**Idempotency** (PRD US-1.2) is enforced by a content SHA-256 per chunk. Running
ingestion twice on the same corpus produces the same index and zero duplicates,
so a re-run is always safe -- which matters because during development it will
be run dozens of times.

Outputs:
  data/processed/chunks.jsonl   -- one Chunk per line, ready for Phase 2
  data/manifest.csv             -- auditable per-document record
  reports/unparsed.csv          -- every file that failed, with the reason
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from ..config import Config
from ..models import Chunk, IngestStats, RawDocument
from ..utils.logging import get_logger
from .chunker import TOKENIZER, chunk_page
from .cleaner import clean_text, find_boilerplate_lines
from .loaders import discover_files, load_document

MANIFEST_FIELDS = [
    "doc_id", "filename", "source_path", "doc_type", "pages", "chars",
    "chunks", "tokens", "scanned_pages", "status", "error", "ingested_at",
]


def _clean_document(doc: RawDocument, cfg: Config) -> list[tuple[int, str]]:
    """Clean every page of a document; returns [(page_no, cleaned_text)]."""
    page_texts = [p.text for p in doc.pages]
    banned = (
        find_boilerplate_lines(page_texts, cfg.cleaning.boilerplate_page_ratio)
        if cfg.cleaning.strip_boilerplate
        else set()
    )
    return [
        (page.page_no, clean_text(page.text, cfg.cleaning, banned))
        for page in doc.pages
    ]


def ingest_document(
    doc: RawDocument, cfg: Config, seen_hashes: set[str]
) -> tuple[list[Chunk], int, int]:
    """Clean + chunk one document. Returns (chunks, dropped, duplicates)."""
    chunks: list[Chunk] = []
    dropped = duplicates = 0
    next_index = 0
    fingerprint = cfg.fingerprint()

    for page_no, cleaned in _clean_document(doc, cfg):
        page_chunks, page_dropped = chunk_page(
            text=cleaned,
            doc_id=doc.doc_id,
            doc_title=doc.doc_title,
            doc_type=doc.doc_type,
            source_path=doc.source_path,
            page_no=page_no,
            start_index=next_index,
            cfg=cfg.chunking,
            fingerprint=fingerprint,
        )
        dropped += page_dropped
        for chunk in page_chunks:
            if chunk.content_sha256 in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(chunk.content_sha256)
            chunks.append(chunk)
        next_index += len(page_chunks)

    return chunks, dropped, duplicates


def run_ingestion(
    cfg: Config, input_dir: Path | None = None, force: bool = False
) -> tuple[list[Chunk], IngestStats]:
    """Execute a full ingestion run over `input_dir`."""
    log = get_logger()
    started = time.perf_counter()
    root = Path(input_dir) if input_dir else cfg.path("raw_dir")
    stats = IngestStats()

    supported, unsupported = discover_files(root, cfg)
    stats.files_seen = len(supported) + len(unsupported)
    stats.files_skipped = len(unsupported)
    for path in unsupported:
        log.warning("unsupported file type, skipping: %s", path.name,
                    extra={"event": "unsupported_file", "file": str(path)})

    log.info("discovered %d supported file(s) under %s", len(supported), root)
    log.info("tokenizer: %s | config fingerprint: %s", TOKENIZER, cfg.fingerprint())

    all_chunks: list[Chunk] = []
    seen_hashes: set[str] = set()
    manifest_rows: list[dict[str, object]] = []
    unparsed_rows: list[dict[str, str]] = []

    for path in supported:
        doc = load_document(path, cfg, root)
        if doc.load_error:
            stats.files_failed += 1
            unparsed_rows.append(
                {"file": str(path), "doc_type": doc.doc_type, "reason": doc.load_error}
            )
            manifest_rows.append({
                "doc_id": doc.doc_id, "filename": path.name,
                "source_path": str(path), "doc_type": doc.doc_type,
                "pages": doc.n_pages, "chars": doc.char_count, "chunks": 0,
                "tokens": 0, "scanned_pages": len(doc.scanned_pages),
                "status": "failed", "error": doc.load_error, "ingested_at": "",
            })
            continue

        chunks, dropped, duplicates = ingest_document(doc, cfg, seen_hashes)
        all_chunks.extend(chunks)

        stats.files_ingested += 1
        stats.pages += doc.n_pages
        stats.chars += doc.char_count
        stats.chunks += len(chunks)
        stats.chunks_dropped += dropped
        stats.duplicates_skipped += duplicates
        stats.scanned_pages += len(doc.scanned_pages)
        doc_tokens = sum(c.token_count for c in chunks)
        stats.tokens += doc_tokens

        manifest_rows.append({
            "doc_id": doc.doc_id, "filename": path.name, "source_path": str(path),
            "doc_type": doc.doc_type, "pages": doc.n_pages, "chars": doc.char_count,
            "chunks": len(chunks), "tokens": doc_tokens,
            "scanned_pages": len(doc.scanned_pages), "status": "ok", "error": "",
            "ingested_at": chunks[0].ingested_at if chunks else "",
        })
        log.info("  %-34s %3d page(s) -> %4d chunk(s)", path.name, doc.n_pages, len(chunks))

    stats.duration_s = time.perf_counter() - started

    _write_chunks(cfg, all_chunks)
    _write_manifest(cfg, manifest_rows)
    _write_unparsed(cfg, unparsed_rows)
    return all_chunks, stats


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #

def _write_chunks(cfg: Config, chunks: list[Chunk]) -> Path:
    out_dir = cfg.path("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    get_logger().info("wrote %d chunk(s) -> %s", len(chunks), path)
    return path


def _write_manifest(cfg: Config, rows: list[dict[str, object]]) -> Path:
    path = cfg.root / "data" / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    get_logger().info("wrote manifest -> %s", path)
    return path


def _write_unparsed(cfg: Config, rows: list[dict[str, str]]) -> Path | None:
    if not rows:
        return None
    path = cfg.path("reports_dir") / "unparsed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "doc_type", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    get_logger().warning("%d file(s) could not be parsed -> %s", len(rows), path)
    return path


def load_chunks(cfg: Config) -> list[dict]:
    """Read back the chunk set produced by a previous run (used by Phase 2)."""
    path = cfg.path("processed_dir") / "chunks.jsonl"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
