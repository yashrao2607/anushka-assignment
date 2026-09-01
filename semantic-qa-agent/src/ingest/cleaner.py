"""Text cleaning -- PRD Section 9.2.

Ordered transforms, each targeting a specific, observed corruption in real
document text. Order matters: unicode normalisation must run before ligature
repair, and de-hyphenation must run before whitespace collapsing (otherwise the
newline that marks the hyphenation is already gone).

Why this matters for retrieval: an embedding model sees "infor- mation" and
"information" as different tokens. Every uncleaned artefact is a small,
permanent loss of retrieval quality that no downstream component can recover.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from ..config import CleaningConfig

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}

# "infor-\nmation" -> "information"; requires a lowercase letter either side so
# that genuine hyphenated compounds at a line break ("state-\nof-the-art") and
# numeric ranges are not silently merged.
_HYPHEN_BREAK = re.compile(r"([a-z])-\s*\n\s*([a-z])")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t ]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_PAGE_NUMBER_LINE = re.compile(
    r"^\s*(?:page\s+)?[-–—\[\(]?\s*\d{1,4}\s*(?:/\s*\d{1,4})?\s*[-–—\]\)]?\s*$",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def fix_ligatures(text: str) -> str:
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    return text


def dedupe_hyphenation(text: str) -> str:
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def strip_page_numbers(text: str) -> str:
    kept = [ln for ln in text.split("\n") if not _PAGE_NUMBER_LINE.match(ln)]
    return "\n".join(kept)


def find_boilerplate_lines(pages: list[str], ratio: float) -> set[str]:
    """Identify running headers/footers.

    A short line that appears on more than `ratio` of pages is structural
    furniture, not content. Requiring >= 3 pages avoids the degenerate case
    where a 2-page document's every line looks like boilerplate.
    """
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        seen_on_this_page = {
            ln.strip() for ln in page.split("\n") if 0 < len(ln.strip()) <= 120
        }
        counts.update(seen_on_this_page)
    threshold = max(2, int(len(pages) * ratio))
    return {line for line, n in counts.items() if n >= threshold}


def strip_lines(text: str, banned: set[str]) -> str:
    if not banned:
        return text
    kept = [ln for ln in text.split("\n") if ln.strip() not in banned]
    return "\n".join(kept)


def clean_text(text: str, cfg: CleaningConfig, banned: set[str] | None = None) -> str:
    """Apply the full ordered cleaning pipeline to one page/block of text."""
    if not text:
        return ""
    if cfg.normalize_unicode:
        text = normalize_unicode(text)
    if cfg.fix_ligatures:
        text = fix_ligatures(text)
    if cfg.dedupe_hyphenation:
        text = dedupe_hyphenation(text)
    if cfg.strip_boilerplate and banned:
        text = strip_lines(text, banned)
    if cfg.strip_page_numbers:
        text = strip_page_numbers(text)
    if cfg.collapse_whitespace:
        text = collapse_whitespace(text)
    return text
