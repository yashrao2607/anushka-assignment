"""Chunking engine -- PRD Section 9.3.

Recursive-character splitting: try to split on the largest natural boundary
first (paragraph), and only fall back to finer separators (line, sentence,
clause, word) when a piece is still too large. The result respects the
document's own structure instead of cutting blindly every N characters.

Three decisions here are what separate this from a naive splitter:

1. **Overlap.** Consecutive chunks share `overlap_tokens`, so an answer that
   straddles a boundary is not lost by *both* chunks.
2. **Heading enrichment.** Each chunk's `embed_text` is prefixed with its
   document title and nearest section heading. A chunk reading "This does not
   apply to contractors." is meaningless alone; "[HR Policy > Casual Leave]
   This does not apply to contractors." is retrievable.
3. **True character spans.** Every chunk records where it came from in the
   cleaned page text, so a citation can highlight the exact source region.

Token counting uses tiktoken when available and a calibrated word-based
approximation otherwise. The approximation is deliberate: it keeps Phase 1
runnable with zero heavy dependencies, and chunk boundaries are not so
sensitive that a few percent of drift changes retrieval behaviour.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..config import ChunkingConfig
from ..models import Chunk, sha256_text, utc_now

# Markdown ATX headings, plus DOCX headings which the loader normalises to the
# same form, plus common ALL-CAPS / numbered policy-document headings.
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.{1,120})$")
_NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][^.\n]{2,110})$")
_CAPS_HEADING = re.compile(r"^\s*([A-Z][A-Z0-9 &/,'\-]{3,80})\s*$")

_WORD = re.compile(r"\w+")

try:  # optional, better accuracy when present
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))

    TOKENIZER = "tiktoken/cl100k_base"
except Exception:  # pragma: no cover - depends on environment
    def count_tokens(text: str) -> int:
        """Approximate GPT-style token count.

        Calibrated on English prose: ~1.33 tokens per whitespace word, plus a
        small allowance for punctuation that tokenises separately.
        """
        if not text:
            return 0
        words = len(_WORD.findall(text))
        punct = sum(1 for ch in text if ch in ".,;:!?()[]{}\"'/-")
        return max(1, int(words * 1.33) + punct // 4)

    TOKENIZER = "approx/word-based"


def tokens_to_chars(tokens: int) -> int:
    """Convert a token budget to an approximate character budget (~4 chars/token)."""
    return max(1, tokens * 4)


def detect_heading(line: str) -> str | None:
    """Return a heading's text if `line` looks like a section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    m = _MD_HEADING.match(stripped)
    if m:
        return m.group(2).strip()
    m = _NUMBERED_HEADING.match(stripped)
    if m:
        return f"{m.group(1)} {m.group(2)}".strip()
    m = _CAPS_HEADING.match(stripped)
    if m and len(stripped.split()) <= 12:
        return m.group(1).strip().title()
    return None


def build_heading_map(text: str) -> list[tuple[int, str]]:
    """Map character offsets to the section heading in force at that offset."""
    headings: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        heading = detect_heading(line)
        if heading:
            headings.append((offset, heading))
        offset += len(line) + 1
    return headings


def heading_at(headings: list[tuple[int, str]], pos: int) -> str | None:
    current: str | None = None
    for offset, heading in headings:
        if offset <= pos:
            current = heading
        else:
            break
    return current


def _split_recursive(text: str, separators: Iterable[str], max_chars: int) -> list[str]:
    """Split `text` into pieces of at most `max_chars`, preferring early separators."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    seps = list(separators)
    for i, sep in enumerate(seps):
        if sep not in text:
            continue
        parts = text.split(sep)
        pieces: list[str] = []
        buffer = ""
        for part in parts:
            candidate = part if not buffer else buffer + sep + part
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                pieces.append(buffer)
            if len(part) > max_chars:
                # This single part still exceeds the budget: recurse with the
                # remaining, finer separators.
                pieces.extend(_split_recursive(part, seps[i + 1:], max_chars))
                buffer = ""
            else:
                buffer = part
        if buffer:
            pieces.append(buffer)
        if pieces:
            return [p for p in (piece.strip() for piece in pieces) if p]

    # No separator helped (e.g. one enormous unbroken token) -- hard-cut.
    return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars)]


def _tail_overlap(piece: str, overlap_chars: int) -> str:
    """Take the trailing overlap text, snapped to a word boundary.

    A raw character slice produces fragments like "te. Delays in", which starts
    mid-word. Those broken tokens are then embedded, adding noise to the vector
    for no benefit. Snapping forward to the next whitespace costs a few
    characters of overlap and keeps every carried token intact.
    """
    if overlap_chars <= 0 or not piece:
        return ""
    tail = piece[-overlap_chars:]
    if len(piece) > overlap_chars:
        space = tail.find(" ")
        # Only snap when a boundary exists reasonably early; otherwise the
        # overlap would shrink to almost nothing on long unbroken spans.
        if 0 <= space < len(tail) // 2:
            tail = tail[space + 1:]
    return tail.strip()


def _is_noise(text: str, cfg: ChunkingConfig) -> bool:
    """Reject fragments too small or too garbled to be worth indexing."""
    if len(text) < cfg.min_chunk_chars:
        return True
    alnum = sum(1 for ch in text if ch.isalnum())
    non_alnum_ratio = 1.0 - (alnum / len(text))
    return non_alnum_ratio > cfg.max_non_alnum_ratio


def chunk_page(
    *,
    text: str,
    doc_id: str,
    doc_title: str,
    doc_type: str,
    source_path: str,
    page_no: int | None,
    start_index: int,
    cfg: ChunkingConfig,
    fingerprint: str,
) -> tuple[list[Chunk], int]:
    """Chunk one cleaned page. Returns (chunks, n_dropped)."""
    if not text.strip():
        return [], 0

    headings = build_heading_map(text)
    max_chars = tokens_to_chars(cfg.chunk_size_tokens)
    overlap_chars = tokens_to_chars(cfg.overlap_tokens)
    pieces = _split_recursive(text, cfg.separators, max_chars)

    chunks: list[Chunk] = []
    dropped = 0
    cursor = 0
    index = start_index
    carry = ""  # overlap text carried from the previous chunk

    for piece in pieces:
        # Locate the piece in the source text to record a true character span.
        found = text.find(piece, cursor)
        if found == -1:
            found = text.find(piece)
        char_start = found if found != -1 else cursor
        char_end = char_start + len(piece)
        cursor = max(cursor, char_end)

        body = f"{carry} {piece}".strip() if carry else piece
        if _is_noise(body, cfg):
            dropped += 1
            carry = ""
            continue

        section = heading_at(headings, char_start)
        if cfg.prefix_with_heading:
            prefix = f"[{doc_title}" + (f" > {section}" if section else "") + "] "
            embed_text = prefix + body
        else:
            embed_text = body

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}__p{page_no or 1}__c{index}",
                doc_id=doc_id,
                text=body,
                embed_text=embed_text,
                source_path=source_path,
                doc_title=doc_title,
                doc_type=doc_type,
                page_no=page_no,
                section_heading=section,
                chunk_index=index,
                char_start=char_start,
                char_end=char_end,
                token_count=count_tokens(body),
                content_sha256=sha256_text(body),
                ingested_at=utc_now(),
                config_fingerprint=fingerprint,
            )
        )
        index += 1
        carry = _tail_overlap(piece, overlap_chars)

    return chunks, dropped
