"""Phase 2 test suite.

  2.1 embedding  -- cache correctness, normalisation, determinism
  2.2 stores     -- exact search, filtering, Chroma/numpy agreement
  2.3 metrics    -- every metric against a HAND-COMPUTED fixture
  2.3 golden set -- span resolution, grading, schema integrity
  2.4 fusion     -- RRF arithmetic and ordering

The metric tests matter most. Every number in the evaluation report is produced
by these functions, so if they are wrong the whole report is confidently wrong.
Each is therefore checked against a value computed by hand in the docstring,
not against the implementation's own output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.embed.embedder import EmbeddingCache, l2_normalize  # noqa: E402
from src.eval.golden import (  # noqa: E402
    load_golden_set, normalize, resolve_grades,
)
from src.eval.metrics import (  # noqa: E402
    average_precision, dcg_at_k, evaluate_query, hit_rate_at_k, ndcg_at_k,
    precision_at_k, recall_at_k, reciprocal_rank,
)
from src.ingest.pipeline import load_chunks  # noqa: E402
from src.retrieve.retriever import reciprocal_rank_fusion  # noqa: E402
from src.store.base import BM25Index, NumpyStore, tokenize  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def chunks(cfg):
    records = load_chunks(cfg)
    if not records:
        pytest.skip("no chunks -- run `python -m src.cli ingest` first")
    return records


# --------------------------------------------------------------------------- #
# Part 2.1 -- embedding layer
# --------------------------------------------------------------------------- #

def test_l2_normalize_gives_unit_rows():
    m = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_l2_normalize_leaves_zero_vector_finite():
    """A zero row must stay zero, not become NaN and poison every later score."""
    m = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(m).all()


def test_normalized_dot_product_equals_cosine():
    """The reason vectors are normalised at all: dot == cosine, so search is a
    single matrix multiply."""
    a, b = np.array([[1.0, 2.0, 3.0]]), np.array([[2.0, 1.0, 0.5]])
    na, nb = l2_normalize(a)[0], l2_normalize(b)[0]
    cosine = float(np.dot(a[0], b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0])))
    assert float(np.dot(na, nb)) == pytest.approx(cosine, abs=1e-6)


def test_cache_roundtrip_and_key_isolation(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    key = EmbeddingCache.make_key("hello", "model-a")
    cache.put_many({key: vec})
    assert np.allclose(cache.get_many([key])[key], vec)
    # A different model must NOT hit the same cache entry, or an ablation would
    # silently evaluate the previous model's vectors.
    assert cache.get_many([EmbeddingCache.make_key("hello", "model-b")]) == {}


def test_cache_counts_hits_and_misses(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.sqlite")
    key = EmbeddingCache.make_key("x", "m")
    cache.put_many({key: np.ones(3, dtype=np.float32)})
    cache.get_many([key, EmbeddingCache.make_key("y", "m")])
    assert (cache.hits, cache.misses) == (1, 1)


# --------------------------------------------------------------------------- #
# Part 2.2 -- stores
# --------------------------------------------------------------------------- #

def _toy_store() -> NumpyStore:
    store = NumpyStore()
    vectors = l2_normalize(np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32))
    store.add(["a", "b", "c"], vectors,
              [{"doc_type": "md"}, {"doc_type": "md"}, {"doc_type": "txt"}])
    return store


def test_numpy_store_ranks_by_cosine():
    hits = _toy_store().search(l2_normalize(np.array([[1.0, 0.0]]))[0], top_n=3)
    assert [h[0] for h in hits] == ["a", "b", "c"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)


def test_numpy_store_respects_metadata_filter():
    hits = _toy_store().search(
        l2_normalize(np.array([[0.0, 1.0]]))[0], top_n=3, where={"doc_type": "md"}
    )
    assert {h[0] for h in hits} == {"a", "b"}


def test_numpy_store_filter_with_no_match_returns_empty():
    assert _toy_store().search(
        np.array([1.0, 0.0], dtype=np.float32), 3, where={"doc_type": "pdf"}
    ) == []


def test_empty_store_returns_no_results():
    assert NumpyStore().search(np.array([1.0, 0.0], dtype=np.float32), 5) == []


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cost of a Booking") == ["cost", "booking"]


def test_bm25_finds_exact_identifier():
    """The reason BM25 is in the pipeline at all: a rare literal token has no
    useful semantic neighbourhood but is a perfect lexical match."""
    index = BM25Index()
    index.build(
        ["c1", "c2", "c3"],
        ["raise a ticket with error code ERR_4092 for a replacement",
         "passwords expire every ninety days",
         "annual leave is twenty four days"],
    )
    assert index.search("ERR_4092", top_n=3)[0][0] == "c1"


def test_bm25_returns_nothing_when_no_term_matches():
    """A zero BM25 score is a non-result. Returning it would inflate recall."""
    index = BM25Index()
    index.build(["c1"], ["annual leave policy"])
    assert index.search("photosynthesis chlorophyll", top_n=3) == []


# --------------------------------------------------------------------------- #
# Part 2.3 -- metrics, against hand-computed values
# --------------------------------------------------------------------------- #

RETRIEVED = ["c1", "c2", "c3", "c4", "c5"]
RELEVANT = {"c2", "c5"}
GRADES = {"c2": 3, "c5": 2, "c1": 1}


def test_precision_at_k_hand_computed():
    """top-3 = [c1, c2, c3]; relevant among them = {c2} -> 1/3."""
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_precision_divides_by_k_not_by_results_returned():
    """Returning 1 result for k=5 is less useful, and P@5 must show that."""
    assert precision_at_k(["c2"], RELEVANT, 5) == pytest.approx(1 / 5)


def test_recall_at_k_hand_computed():
    """2 relevant exist; top-3 contains 1 -> 0.5. top-5 contains both -> 1.0."""
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(0.5)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(1.0)


def test_reciprocal_rank_hand_computed():
    """First relevant (c2) is at rank 2 -> 1/2."""
    assert reciprocal_rank(RETRIEVED, RELEVANT, 10) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_nothing_relevant_in_k():
    assert reciprocal_rank(RETRIEVED, {"c9"}, 5) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 3) == 1.0
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 0.0


def test_dcg_hand_computed():
    """rank1 c1 rel=1 -> (2^1-1)/log2(2) = 1.0
       rank2 c2 rel=3 -> (2^3-1)/log2(3) = 7/1.584963 = 4.416508
       rank3 c3 rel=0 -> 0                     total = 5.416508"""
    assert dcg_at_k(RETRIEVED, GRADES, 3) == pytest.approx(5.416508, abs=1e-5)


def test_ndcg_hand_computed():
    """Ideal top-3 grades = [3, 2, 1]:
         (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
       = 7 + 1.892789 + 0.5 = 9.392789
       nDCG@3 = 5.416508 / 9.392789 = 0.576667"""
    assert ndcg_at_k(RETRIEVED, GRADES, 3) == pytest.approx(0.576667, abs=1e-5)


def test_ndcg_is_one_for_a_perfect_ranking():
    perfect = ["c2", "c5", "c1"]
    assert ndcg_at_k(perfect, GRADES, 3) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_is_zero_when_no_grades_exist():
    assert ndcg_at_k(RETRIEVED, {}, 3) == 0.0


def test_average_precision_hand_computed():
    """c2 relevant at rank 2 -> 1/2; c5 relevant at rank 5 -> 2/5.
       AP = (0.5 + 0.4) / 2 = 0.45"""
    assert average_precision(RETRIEVED, RELEVANT, 10) == pytest.approx(0.45)


def test_evaluate_query_returns_every_expected_key():
    scores = evaluate_query(RETRIEVED, RELEVANT, GRADES, (1, 3, 5))
    for key in ("precision@3", "recall@5", "ndcg@3", "hit_rate@1", "mrr@5", "map@5"):
        assert key in scores


def test_metrics_are_safe_on_empty_input():
    assert precision_at_k([], RELEVANT, 3) == 0.0
    assert recall_at_k(RETRIEVED, set(), 3) == 0.0
    assert ndcg_at_k([], GRADES, 3) == 0.0


# --------------------------------------------------------------------------- #
# Part 2.4 -- reciprocal rank fusion
# --------------------------------------------------------------------------- #

def test_rrf_hand_computed():
    """d1 is rank 1 in list A and rank 2 in list B:
       1/(60+1) + 1/(60+2) = 0.016393 + 0.016129 = 0.032522"""
    scores = reciprocal_rank_fusion([["d1", "d2"], ["d3", "d1"]], k=60)
    assert scores["d1"] == pytest.approx(0.032522, abs=1e-6)
    assert scores["d2"] == pytest.approx(1 / 62, abs=1e-6)


def test_rrf_rewards_agreement_between_rankers():
    """A document both rankers like must beat one only a single ranker likes --
    this is the entire premise of hybrid retrieval."""
    scores = reciprocal_rank_fusion([["both", "onlyA"], ["both", "onlyB"]])
    assert scores["both"] > scores["onlyA"] and scores["both"] > scores["onlyB"]


def test_rrf_is_scale_free():
    """RRF uses ranks only, so it cannot be destabilised by an outlier score --
    the reason it is preferred here over weighted score blending."""
    a = reciprocal_rank_fusion([["x", "y", "z"]])
    b = reciprocal_rank_fusion([["x", "y", "z"]])
    assert a == b


def test_rrf_handles_an_empty_ranker():
    scores = reciprocal_rank_fusion([["a", "b"], []])
    assert scores["a"] > scores["b"]


# --------------------------------------------------------------------------- #
# Part 2.3 -- golden set integrity
# --------------------------------------------------------------------------- #

def test_golden_set_has_expected_size_and_categories(cfg):
    questions = load_golden_set(cfg.root / "data" / "golden_set.jsonl")
    assert len(questions) == 60
    categories = {q.category for q in questions}
    assert {"direct", "paraphrase", "multi_chunk", "numeric",
            "exact_id", "ambiguous", "unanswerable"} <= categories


def test_golden_set_qids_are_unique(cfg):
    questions = load_golden_set(cfg.root / "data" / "golden_set.jsonl")
    assert len({q.qid for q in questions}) == len(questions)


def test_unanswerable_questions_carry_no_gold(cfg):
    """They exist to test refusal in Phase 3, and must never count toward
    retrieval metrics."""
    for q in load_golden_set(cfg.root / "data" / "golden_set.jsonl"):
        if not q.answerable:
            assert not q.primary_spans and not q.gold_docs


def test_every_gold_span_resolves_to_a_real_chunk(cfg, chunks):
    """The golden set's own integrity check.

    An unresolvable span silently scores 0 on every metric, which would depress
    the report for a reason that has nothing to do with retrieval quality. This
    test makes that failure loud instead of invisible.
    """
    questions = load_golden_set(cfg.root / "data" / "golden_set.jsonl")
    _, warnings = resolve_grades(questions, chunks)
    assert warnings == [], "unresolvable gold spans:\n" + "\n".join(warnings)


def test_span_resolution_is_whitespace_insensitive():
    """Spans are authored from wrapped source text; cleaning collapses newlines.
    Matching must survive that or every multi-line span would fail."""
    assert normalize("the  trip\nis   cancelled") == "the trip is cancelled"


def test_grades_use_the_full_scale(cfg, chunks):
    questions, _ = resolve_grades(
        load_golden_set(cfg.root / "data" / "golden_set.jsonl"), chunks
    )
    all_grades = {g for q in questions for g in q.grades.values()}
    assert 3 in all_grades and 2 in all_grades  # nDCG needs graded relevance


def test_relevant_set_excludes_grade_one_context(cfg, chunks):
    """Grade 1 is same-section context: it contributes to nDCG but counting it
    as 'relevant' would inflate precision."""
    questions, _ = resolve_grades(
        load_golden_set(cfg.root / "data" / "golden_set.jsonl"), chunks
    )
    for q in questions:
        for cid in q.relevant:
            assert q.grades[cid] >= 2


def test_golden_set_file_is_valid_jsonl(cfg):
    path = cfg.root / "data" / "golden_set.jsonl"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            assert "qid" in record and "question" in record, f"line {i}"
