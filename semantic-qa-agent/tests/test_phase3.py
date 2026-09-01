"""Phase 3 test suite.

  3.1 re-ranking  -- reordering, graceful degradation, sigmoid calibration
  3.2 generation  -- citation parsing/validation, context budgeting, the gate
  3.2 calibration -- threshold sweep arithmetic against hand-computed values
  3.3 client      -- cache keying, budget enforcement (no network in tests)

Every test here runs **offline**. Not one makes an API call: the Groq client is
exercised through its cache and its budget guard, so the suite is free to run,
deterministic, and safe on a free-tier key.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.eval.calibrate import sweep  # noqa: E402
from src.eval.judge import _parse_judgement, stratified_sample  # noqa: E402
from src.generate.groq_client import GroqClient, GroqError  # noqa: E402
from src.generate.prompts import (  # noqa: E402
    REFUSAL_TEXT, build_context, extract_citations, is_refusal,
    normalize_citation_markers, validate_citations,
)
from src.retrieve.reranker import CrossEncoderReranker, sigmoid  # noqa: E402


@dataclass
class FakeChunk:
    """Stand-in for RetrievedChunk -- keeps these tests free of model loading."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    rerank_score: float | None = None
    final_rank: int = 0
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


# --------------------------------------------------------------------------- #
# Part 3.1 -- re-ranking
# --------------------------------------------------------------------------- #

def test_sigmoid_maps_logits_into_unit_interval():
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < sigmoid(-8.0) < 0.01
    assert 0.99 < sigmoid(8.0) < 1.0


def test_sigmoid_is_numerically_stable_at_extremes():
    """The naive 1/(1+exp(-x)) overflows for large negative x. Cross-encoders
    routinely emit logits near -11, so this is a real input, not a corner case."""
    for x in (-800.0, 800.0):
        value = sigmoid(x)
        assert 0.0 <= value <= 1.0


def test_sigmoid_is_monotonic():
    values = [sigmoid(x) for x in (-10, -5, -1, 0, 1, 5, 10)]
    assert values == sorted(values)


def test_reranker_reorders_by_score_not_input_order():
    class StubReranker(CrossEncoderReranker):
        def score(self, query, texts):
            import numpy as np
            # Deliberately reverse: the highest score goes to the LAST input, so
            # a test that passes cannot be explained by the original ordering.
            return np.array([float(i) for i in range(len(texts))], dtype=float)

    reranker = StubReranker()
    candidates = [FakeChunk(f"c{i}", f"text {i}") for i in range(4)]
    ranked = reranker.rerank("q", candidates, top_k=2)
    assert [c.chunk_id for c in ranked] == ["c3", "c2"]
    assert [c.final_rank for c in ranked] == [1, 2]


def test_reranker_degrades_gracefully_when_the_model_fails():
    """PRD principle #6: a missing optional component must not take the query
    path down -- it falls back to the stage-1 order and marks itself unavailable."""
    class BrokenReranker(CrossEncoderReranker):
        def score(self, query, texts):
            raise RuntimeError("model weights missing")

    reranker = BrokenReranker()
    candidates = [FakeChunk(f"c{i}", f"t{i}") for i in range(5)]
    ranked = reranker.rerank("q", candidates, top_k=3)
    assert [c.chunk_id for c in ranked] == ["c0", "c1", "c2"]
    assert reranker.available is False


def test_reranker_respects_the_candidate_cap():
    class CountingReranker(CrossEncoderReranker):
        seen = 0

        def score(self, query, texts):
            import numpy as np
            CountingReranker.seen = len(texts)
            return np.zeros(len(texts))

    reranker = CountingReranker(max_candidates=5)
    reranker.rerank("q", [FakeChunk(f"c{i}", "t") for i in range(20)], top_k=3)
    assert CountingReranker.seen == 5


def test_reranker_handles_empty_candidates():
    assert CrossEncoderReranker().rerank("q", [], top_k=5) == []


# --------------------------------------------------------------------------- #
# Part 3.2 -- citations
# --------------------------------------------------------------------------- #

def test_extract_citations_in_order_of_first_appearance():
    assert extract_citations("Claim A [2]. Claim B [1]. Claim C [2].") == [2, 1]


def test_extract_citations_accepts_cjk_brackets():
    """A REGRESSION TEST for a real bug.

    gpt-oss returns CJK lenticular brackets rather than the ASCII brackets the
    prompt asks for. Parsing only `[1]` silently dropped every citation from
    answers that were in fact perfectly grounded -- which looked like a total
    failure of the citation feature when nothing was actually wrong.
    """
    assert extract_citations("grounded claim【1】and another【2】") == [1, 2]


def test_extract_citations_accepts_suffixed_file_citations():
    """Second observed variant: 【1†L1-L3】, OpenAI file-citation style with a
    line range. Found while running the demo, after the CJK fix had landed."""
    assert extract_citations("hardware failure【1†L1-L3】【2†L4-L7】") == [1, 2]


def test_extract_citations_ignores_parenthesised_numbers_in_prose():
    """`(1)` is ordinary prose, not a citation. Accepting it would invent
    references the model never made."""
    assert extract_citations("resolved within (1) business day") == []


def test_normalize_citation_markers_produces_ascii():
    assert normalize_citation_markers("see【3】here") == "see[3]here"
    assert normalize_citation_markers("x【2†L4-L7】y") == "x[2]y"


def test_validate_citations_accepts_valid_markers():
    cleaned, valid, invalid = validate_citations("Answer [1] and [2].", n_sources=3)
    assert valid == [1, 2] and invalid == []
    assert "[1]" in cleaned


def test_validate_citations_strips_dangling_markers():
    """A `[7]` when only 3 sources were supplied is a hallucinated citation --
    worse than none, because it *looks* verifiable."""
    cleaned, valid, invalid = validate_citations("Real [1]. Fake [7].", n_sources=3)
    assert valid == [1] and invalid == [7]
    assert "[7]" not in cleaned and "[1]" in cleaned


def test_validate_citations_normalizes_cjk_then_validates():
    cleaned, valid, invalid = validate_citations("Claim【2】here.", n_sources=3)
    assert valid == [2] and invalid == []
    assert "[2]" in cleaned


def test_is_refusal_detects_the_contracted_sentence():
    assert is_refusal(REFUSAL_TEXT)
    assert is_refusal("I could not find information about this in the documents.")
    assert not is_refusal("Interns receive 6 casual leave days per year [1].")


# --------------------------------------------------------------------------- #
# Part 3.2 -- context assembly
# --------------------------------------------------------------------------- #

def _chunks(n: int, size: int = 100) -> list[FakeChunk]:
    return [
        FakeChunk(f"c{i}", "word " * size,
                  {"doc_title": f"Doc {i}", "page_no": 1, "section_heading": "S"})
        for i in range(n)
    ]


def test_build_context_numbers_blocks_from_one():
    context, used = build_context(_chunks(3), max_tokens=3000)
    assert "[1]" in context and "[2]" in context and "[3]" in context
    assert len(used) == 3


def test_build_context_includes_provenance_in_each_block():
    context, _ = build_context(_chunks(1), max_tokens=3000)
    assert "source:" in context and "Doc 0" in context and "p.1" in context


def test_build_context_truncates_to_the_token_budget():
    """Chunks arrive in score order, so truncation drops the least relevant
    first -- and must never return more than the budget allows."""
    _, used = build_context(_chunks(40, size=200), max_tokens=500)
    assert 0 < len(used) < 40


def test_build_context_always_keeps_at_least_one_chunk():
    """Even an over-budget single chunk must be passed, or the model would be
    asked to answer from nothing."""
    _, used = build_context(_chunks(3, size=5000), max_tokens=10)
    assert len(used) == 1


def test_build_context_handles_no_chunks():
    context, used = build_context([], max_tokens=3000)
    assert context == "" and used == []


# --------------------------------------------------------------------------- #
# Part 3.2 -- the refusal gate
# --------------------------------------------------------------------------- #

class StubAnswerer:
    """Exercises gate logic alone, with no models and no network."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    gate = None


def _gate(hits, threshold):
    from src.generate.answerer import Answerer

    stub = Answerer.__new__(Answerer)
    stub.threshold = threshold
    return Answerer.gate(stub, hits)


def test_gate_refuses_when_nothing_is_retrieved():
    assert _gate([], 0.02) == (True, 0.0)


def test_gate_refuses_below_the_threshold():
    hit = FakeChunk("c1", "t", rerank_score=-8.0)   # sigmoid(-8) ~ 0.00034
    refuse, confidence = _gate([hit], 0.02)
    assert refuse is True and confidence < 0.02


def test_gate_answers_above_the_threshold():
    hit = FakeChunk("c1", "t", rerank_score=2.0)    # sigmoid(2) ~ 0.88
    refuse, confidence = _gate([hit], 0.02)
    assert refuse is False and confidence > 0.8


def test_gate_falls_back_to_retrieval_score_without_reranking():
    hit = FakeChunk("c1", "t", rerank_score=None)
    hit.rrf_score = 0.5
    refuse, confidence = _gate([hit], 0.02)
    assert refuse is False and confidence == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Part 3.2 -- threshold calibration arithmetic
# --------------------------------------------------------------------------- #

def test_sweep_counts_are_hand_verifiable():
    """answerable = [0.9, 0.8, 0.1]; unanswerable = [0.05, 0.5]
       At tau = 0.5:  TP = 2 (0.9, 0.8)   FN = 1 (0.1)
                      FP = 1 (0.5)        TN = 1 (0.05)
       precision = 2/3, recall = 2/3, F1 = 2/3"""
    row = next(r for r in sweep([0.9, 0.8, 0.1], [0.05, 0.5]) if r["threshold"] == 0.5)
    assert (row["tp"], row["fn"], row["fp"], row["tn"]) == (2, 1, 1, 1)
    assert row["f1"] == pytest.approx(2 / 3, abs=1e-3)


def test_sweep_refusal_correctness_rises_with_threshold():
    rows = sweep([0.9, 0.8, 0.3], [0.2, 0.4])
    low = next(r for r in rows if r["threshold"] == 0.1)
    high = next(r for r in rows if r["threshold"] == 0.5)
    assert high["refusal_correctness"] >= low["refusal_correctness"]
    assert high["answer_correctness"] <= low["answer_correctness"]


def test_sweep_handles_perfect_separation():
    """With no overlap there must exist a threshold scoring 1.0 on both sides."""
    rows = sweep([0.9, 0.95], [0.01, 0.02])
    assert any(r["balanced"] == 1.0 for r in rows)


def test_sweep_covers_the_full_threshold_range():
    rows = sweep([0.5], [0.1])
    assert rows[0]["threshold"] == 0.01 and rows[-1]["threshold"] == 0.89


# --------------------------------------------------------------------------- #
# Part 3.4 -- judge helpers
# --------------------------------------------------------------------------- #

def test_parse_judgement_reads_clean_json():
    verdict = _parse_judgement('{"faithful": 1, "relevant": 1, "unsupported_claim": ""}')
    assert verdict["faithful"] == 1 and verdict["relevant"] == 1


def test_parse_judgement_survives_surrounding_prose():
    verdict = _parse_judgement('Sure! {"faithful": 0, "relevant": 1} Hope that helps.')
    assert verdict["faithful"] == 0 and verdict["relevant"] == 1


def test_parse_judgement_flags_unparseable_output_rather_than_crashing():
    """This is how the broken-judge bug was caught: a parse failure is recorded
    as `unparseable` instead of silently scoring 0 with no explanation."""
    assert _parse_judgement("I think it looks fine")["unsupported_claim"] == "unparseable"


def test_stratified_sample_spans_categories():
    @dataclass
    class Q:
        qid: str
        category: str

    questions = (
        [Q(f"a{i}", "direct") for i in range(10)]
        + [Q(f"b{i}", "paraphrase") for i in range(10)]
        + [Q(f"c{i}", "numeric") for i in range(2)]
    )
    picked = stratified_sample(questions, 6)
    assert len(picked) == 6
    assert len({q.category for q in picked}) == 3


# --------------------------------------------------------------------------- #
# Part 3.3 -- Groq client (offline: cache and budget only)
# --------------------------------------------------------------------------- #

def test_client_returns_a_cached_response_without_network(tmp_path):
    client = GroqClient(root=tmp_path, cache_path=tmp_path / "c.json")
    key = client._key("sys", "usr", client.model)
    client._cache[key] = "cached answer"
    assert client.complete("sys", "usr") == "cached answer"
    assert client.usage.calls_live == 0 and client.usage.calls_cached == 1


def test_cache_key_separates_models_and_params(tmp_path):
    """A judge run in JSON mode must not collide with a plain generation on the
    same prompt, or one would silently serve the other's answer."""
    client = GroqClient(root=tmp_path, cache_path=tmp_path / "c.json")
    base = client._key("s", "u", "model-a")
    assert base != client._key("s", "u", "model-b")
    assert base != client._key("s", "u", "model-a", json_mode=True)
    assert base != client._key("s", "u", "model-a", reasoning_effort="low")


def test_client_enforces_its_call_budget(tmp_path):
    """Guards the free tier: exhausting the budget raises a clear error rather
    than quietly consuming the daily quota."""
    client = GroqClient(root=tmp_path, cache_path=tmp_path / "c.json", max_calls=0)
    client.api_key = "test-key-not-used"
    with pytest.raises(GroqError, match="budget exhausted"):
        client.complete("sys", "usr")


def test_client_reports_a_missing_key_actionably(tmp_path):
    client = GroqClient(root=tmp_path, cache_path=tmp_path / "c.json")
    client.api_key = ""
    with pytest.raises(GroqError, match="GROQ_API_KEY"):
        client.complete("sys", "usr")


def test_client_survives_a_corrupt_cache_file(tmp_path):
    (tmp_path / "c.json").write_text("{not json", encoding="utf-8")
    client = GroqClient(root=tmp_path, cache_path=tmp_path / "c.json")
    assert client._cache == {}


def test_env_loader_reads_dotenv(tmp_path, monkeypatch):
    from src.generate.groq_client import load_env

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / ".env").write_text('GROQ_API_KEY="abc123"\n# comment\n', encoding="utf-8")
    load_env(tmp_path)
    import os

    assert os.environ["GROQ_API_KEY"] == "abc123"


# --------------------------------------------------------------------------- #
# Reports produced by Phase 3
# --------------------------------------------------------------------------- #

def test_calibration_report_exists_and_is_well_formed():
    cfg = load_config()
    path = cfg.path("reports_dir") / "calibration.json"
    if not path.exists():
        pytest.skip("run `python -m src.cli calibrate` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert 0.0 < payload["recommended_threshold"] < 1.0
    assert payload["at_recommended"]["refusal_correctness"] >= 0.9
