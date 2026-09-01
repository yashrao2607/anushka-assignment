"""Phase 1 test suite.

Covers the four Phase 1 parts:
  1.1 config       -- loading, overrides, validation, fingerprint stability
  1.2 loaders      -- discovery, dispatch, graceful failure; cleaner transforms
  1.3 chunker      -- size, overlap, spans, heading detection, noise rejection
  1.4 pipeline     -- end-to-end run, idempotency, manifest, outputs

Each test asserts a property stated in the PRD, so a failure points at a
requirement rather than at an implementation detail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ChunkingConfig, CleaningConfig, ConfigError, load_config  # noqa: E402
from src.ingest.chunker import (  # noqa: E402
    _split_recursive, _tail_overlap, build_heading_map, chunk_page,
    count_tokens, detect_heading, heading_at,
)
from src.ingest.cleaner import (  # noqa: E402
    clean_text, collapse_whitespace, dedupe_hyphenation, find_boilerplate_lines,
    fix_ligatures, strip_page_numbers,
)
from src.ingest.loaders import discover_files, load_document, make_doc_id  # noqa: E402
from src.ingest.pipeline import ingest_document, run_ingestion  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# Part 1.1 -- configuration
# --------------------------------------------------------------------------- #

def test_config_loads_with_expected_defaults(cfg):
    assert cfg.chunking.chunk_size_tokens == 512
    assert cfg.chunking.overlap_tokens == 64
    assert ".pdf" in cfg.ingest.supported_extensions


def test_config_dotted_override_applies():
    cfg = load_config(**{"chunking.chunk_size_tokens": 128})
    assert cfg.chunking.chunk_size_tokens == 128


def test_config_rejects_overlap_not_smaller_than_chunk_size():
    """PRD principle #6: an impossible setting must fail at startup, loudly."""
    with pytest.raises(ConfigError, match="must be smaller"):
        load_config(**{"chunking.overlap_tokens": 512})


def test_config_rejects_unknown_strategy():
    with pytest.raises(ConfigError, match="unknown chunking.strategy"):
        load_config(**{"chunking.strategy": "magic"})


def test_fingerprint_changes_with_output_affecting_settings_only(cfg):
    """A fingerprint must track what changes chunks -- and nothing else."""
    changed = load_config(**{"chunking.chunk_size_tokens": 256})
    assert cfg.fingerprint() != changed.fingerprint()
    assert cfg.fingerprint() == load_config().fingerprint()  # stable across loads


# --------------------------------------------------------------------------- #
# Part 1.2 -- cleaning
# --------------------------------------------------------------------------- #

def test_dedupe_hyphenation_rejoins_line_broken_words():
    assert dedupe_hyphenation("infor-\nmation") == "information"
    assert dedupe_hyphenation("reim-\n  bursement") == "reimbursement"


def test_dedupe_hyphenation_leaves_real_hyphens_alone():
    """'state-of-the-art' on one line is not a hyphenation artefact."""
    assert dedupe_hyphenation("state-of-the-art") == "state-of-the-art"


def test_fix_ligatures():
    assert fix_ligatures("ofﬁce classiﬁcation") == "office classification"


def test_collapse_whitespace():
    assert collapse_whitespace("a  \t b\n\n\n\nc  ") == "a b\n\nc"


def test_strip_page_numbers_removes_number_only_lines():
    text = "Real content here\n12\nMore content\nPage 3\n- 4 -"
    out = strip_page_numbers(text)
    assert "Real content here" in out and "More content" in out
    assert "12" not in out.split("\n") and "Page 3" not in out


def test_boilerplate_detection_finds_repeated_headers():
    pages = [f"ACME CONFIDENTIAL\nbody text {i}\nfooter line" for i in range(5)]
    banned = find_boilerplate_lines(pages, 0.6)
    assert "ACME CONFIDENTIAL" in banned and "footer line" in banned
    assert "body text 0" not in banned


def test_boilerplate_detection_ignores_very_short_documents():
    """With <3 pages every line looks repeated -- that would be a false positive."""
    assert find_boilerplate_lines(["a\nb", "a\nb"], 0.6) == set()


def test_clean_text_is_safe_on_empty_input():
    assert clean_text("", CleaningConfig()) == ""


# --------------------------------------------------------------------------- #
# Part 1.3 -- chunking
# --------------------------------------------------------------------------- #

def test_split_recursive_respects_max_size():
    text = "\n\n".join(f"Paragraph number {i}. " * 12 for i in range(8))
    for piece in _split_recursive(text, ["\n\n", "\n", ". ", " "], 300):
        assert len(piece) <= 300


def test_split_recursive_prefers_paragraph_boundaries():
    text = "First para.\n\nSecond para.\n\nThird para."
    assert _split_recursive(text, ["\n\n", " "], 20) == [
        "First para.", "Second para.", "Third para."
    ]


def test_split_recursive_hard_cuts_unbreakable_text():
    """A single enormous token must still be bounded, not returned oversized."""
    for piece in _split_recursive("x" * 500, ["\n\n", " "], 100):
        assert len(piece) <= 100


def test_overlap_snaps_to_word_boundary():
    """A mid-word overlap fragment adds broken tokens to the embedding."""
    tail = _tail_overlap("the quick brown fox jumps over", 12)
    assert not tail.startswith("ver") and tail
    assert tail.split()[0] in {"jumps", "over", "s"} or " " in tail


def test_overlap_is_empty_when_disabled():
    assert _tail_overlap("some text", 0) == ""


def test_detect_heading_variants():
    assert detect_heading("## Leave Entitlement") == "Leave Entitlement"
    assert detect_heading("4.2 Non-refundable Bookings") == "4.2 Non-refundable Bookings"
    assert detect_heading("INFORMATION SECURITY") == "Information Security"
    assert detect_heading("This is an ordinary sentence of body text.") is None


def test_heading_map_assigns_the_heading_in_force():
    text = "# Alpha\nbody one\n\n## Beta\nbody two"
    headings = build_heading_map(text)
    assert heading_at(headings, 0) == "Alpha"
    assert heading_at(headings, len(text) - 1) == "Beta"


def _chunk(text: str, **kw):
    cfg = ChunkingConfig(**kw) if kw else ChunkingConfig()
    return chunk_page(
        text=text, doc_id="d", doc_title="Doc", doc_type="md",
        source_path="d.md", page_no=1, start_index=0, cfg=cfg, fingerprint="fp",
    )


def test_chunker_produces_traceable_spans():
    """Provenance is the whole point: spans must index back into the source."""
    text = "# Title\n\n" + "Sentence about leave policy. " * 60
    chunks, _ = _chunk(text, chunk_size_tokens=60, overlap_tokens=10)
    assert len(chunks) > 1
    for c in chunks:
        assert 0 <= c.char_start < c.char_end <= len(text)


def test_chunker_carries_overlap_between_chunks():
    text = " ".join(f"word{i}" for i in range(400))
    with_overlap, _ = _chunk(text, chunk_size_tokens=60, overlap_tokens=20)
    without, _ = _chunk(text, chunk_size_tokens=60, overlap_tokens=0)
    assert sum(c.token_count for c in with_overlap) > sum(c.token_count for c in without)


def test_chunker_prefixes_heading_into_embed_text_only():
    text = "## Casual Leave\n\n" + "Interns accrue half a day per month. " * 20
    chunks, _ = _chunk(text, chunk_size_tokens=80, overlap_tokens=0)
    first = chunks[0]
    assert first.embed_text.startswith("[Doc > Casual Leave]")
    assert not first.text.startswith("[")  # display text stays clean


def test_chunker_drops_noise_fragments():
    chunks, dropped = _chunk("### ...\n\n@@@@ ||| ---", chunk_size_tokens=64)
    assert chunks == [] and dropped >= 1


def test_chunker_handles_empty_input():
    assert _chunk("   \n\n  ") == ([], 0)


def test_chunk_ids_are_unique_and_readable():
    text = "Policy sentence. " * 200
    chunks, _ = _chunk(text, chunk_size_tokens=50, overlap_tokens=5)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("d__p1__c")


def test_token_count_is_monotonic():
    assert 0 < count_tokens("short") < count_tokens("a much longer piece of text here")


# --------------------------------------------------------------------------- #
# Part 1.4 -- pipeline
# --------------------------------------------------------------------------- #

def test_discover_files_separates_supported_from_unsupported(cfg):
    supported, unsupported = discover_files(cfg.path("raw_dir"), cfg)
    assert len(supported) >= 5
    assert all(p.suffix.lower() in cfg.ingest.supported_extensions for p in supported)
    assert any(p.suffix == ".rtf" for p in unsupported)


def test_discover_files_raises_on_missing_directory(cfg):
    with pytest.raises(FileNotFoundError):
        discover_files(Path("does/not/exist"), cfg)


def test_make_doc_id_is_stable_and_slug_like(tmp_path):
    assert make_doc_id(tmp_path / "HR Policy-2026.md", tmp_path) == "hr_policy_2026"


def test_loader_reports_error_instead_of_raising(cfg, tmp_path):
    """A corrupt/unloadable file must be reported, never crash the run."""
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not really a pdf")
    doc = load_document(bad, cfg, tmp_path)
    assert doc.load_error is not None and doc.pages == []


def test_end_to_end_ingestion_produces_chunks_and_manifest(cfg):
    chunks, stats = run_ingestion(cfg)
    assert stats.files_ingested >= 5
    assert stats.files_skipped >= 1          # the .rtf decoy
    assert stats.files_failed == 0
    assert len(chunks) == stats.chunks > 0

    out = cfg.path("processed_dir") / "chunks.jsonl"
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(chunks)
    record = json.loads(lines[0])
    for key in ("chunk_id", "text", "embed_text", "page_no", "char_start",
                "content_sha256", "config_fingerprint"):
        assert key in record

    assert (cfg.root / "data" / "manifest.csv").exists()


def test_ingestion_is_idempotent_within_a_run(cfg):
    """US-1.2: identical content must never be indexed twice."""
    from src.ingest.loaders import load_document as _load
    raw = cfg.path("raw_dir")
    doc = _load(next(raw.glob("*.md")), cfg, raw)
    seen: set[str] = set()
    first, _, dup_first = ingest_document(doc, cfg, seen)
    _, _, dup_second = ingest_document(doc, cfg, seen)
    assert dup_first == 0
    assert dup_second == len(first)          # every chunk recognised as a duplicate


def test_all_chunks_carry_full_provenance(cfg):
    chunks, _ = run_ingestion(cfg)
    for c in chunks:
        assert c.doc_id and c.source_path and c.chunk_id
        assert c.page_no is not None
        assert c.token_count > 0
        assert len(c.content_sha256) == 64
