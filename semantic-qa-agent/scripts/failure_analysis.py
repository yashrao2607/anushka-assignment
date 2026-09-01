"""Failure analysis -- PRD Section 13.5.

Takes every golden question the best configuration still gets wrong, diagnoses a
root cause from a fixed taxonomy, and writes `reports/failure_analysis.md`.

The taxonomy matters more than the count. "12 failures" is not actionable;
"9 of 12 failures are `gold_too_narrow`, 2 are `embedding_semantic_gap`" points
at exactly what to fix next -- and, importantly, distinguishes defects in the
*system* from defects in the *golden set*, which is a distinction most
evaluations never make and which changes what you should do about them.

Runs entirely offline: no API calls.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.eval.golden import load_golden_set, resolve_grades
from src.ingest.pipeline import load_chunks
from src.store.index import load_retriever
from src.utils.logging import setup_logging

TAXONOMY = {
    "gold_too_narrow": (
        "The retrieved chunk genuinely answers the question, but the golden set "
        "labelled a different chunk as gold. A defect in the *evaluation*, not "
        "the system."
    ),
    "chunking_split_the_answer": (
        "The answer spans a chunk boundary, so no single chunk contains enough "
        "to be scored relevant. Fix: larger chunks or neighbour expansion."
    ),
    "embedding_semantic_gap": (
        "The query and the gold chunk are semantically related but the "
        "bi-encoder does not place them close. Fix: a stronger embedding model."
    ),
    "reranker_misordered": (
        "The gold chunk was in the candidate pool but the cross-encoder ranked "
        "it below the cut. Fix: a larger re-ranker, or a deeper final k."
    ),
    "query_underspecified": (
        "The question is genuinely ambiguous and several documents answer it "
        "plausibly. Arguably not a failure at all."
    ),
    "retrieval_miss": (
        "The gold chunk never entered the candidate pool at any stage. The most "
        "serious class: no amount of re-ranking can recover it."
    ),
}


def classify(question, retrieved_ids: list[str], pool_ids: list[str],
             chunks_by_id: dict) -> str:
    """Assign a root cause. Ordered from most to least specific."""
    gold = question.relevant
    if not gold:
        return "gold_too_narrow"

    if not (set(pool_ids) & gold):
        return "retrieval_miss"
    if set(pool_ids) & gold and not (set(retrieved_ids[:3]) & gold):
        # It was retrievable and the re-ranker had it -- it simply did not
        # survive the cut.
        return "reranker_misordered"

    gold_docs = {chunks_by_id[c]["doc_id"] for c in gold if c in chunks_by_id}
    got_docs = {chunks_by_id[c]["doc_id"] for c in retrieved_ids[:3] if c in chunks_by_id}
    if gold_docs & got_docs:
        # Right document, wrong chunk -- the answer straddles a boundary or the
        # gold label picked a neighbouring chunk.
        return "chunking_split_the_answer"
    if question.category == "ambiguous":
        return "query_underspecified"
    return "embedding_semantic_gap"


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.path("logs_dir"))
    chunks = load_chunks(cfg)
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    questions, _ = resolve_grades(
        load_golden_set(cfg.root / "data" / "golden_set.jsonl"), chunks
    )
    retriever = load_retriever(cfg)

    failures: list[dict] = []
    answerable = [q for q in questions if q.answerable and q.grades]

    for q in answerable:
        hits, _ = retriever.retrieve(q.question, top_k=10, mode="hybrid", rerank=True)
        retrieved = [h.chunk_id for h in hits]
        pool, _ = retriever.retrieve(q.question, top_k=25, mode="hybrid", rerank=False)
        pool_ids = [h.chunk_id for h in pool]

        if set(retrieved[:3]) & q.relevant:
            continue

        cause = classify(q, retrieved, pool_ids, chunks_by_id)
        top = hits[0] if hits else None
        failures.append({
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "cause": cause,
            "retrieved_top3": [
                f"{chunks_by_id.get(c, {}).get('doc_title', '?')} > "
                f"{chunks_by_id.get(c, {}).get('section_heading', '?')}"
                for c in retrieved[:3]
            ],
            "expected": [
                f"{chunks_by_id.get(c, {}).get('doc_title', '?')} > "
                f"{chunks_by_id.get(c, {}).get('section_heading', '?')}"
                for c in sorted(q.relevant)[:3]
            ],
            "gold_in_pool": bool(set(pool_ids) & q.relevant),
            "top_score": round(top.score, 3) if top else 0.0,
        })

    counts = Counter(f["cause"] for f in failures)
    payload = {
        "configuration": "A5 -- hybrid + cross-encoder re-rank",
        "n_questions": len(answerable),
        "n_failures": len(failures),
        "hit_rate_at_3": round(1 - len(failures) / len(answerable), 3) if answerable else 0,
        "causes": dict(counts),
        "failures": failures,
    }

    out = cfg.path("reports_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Failure Analysis",
        "",
        "Every golden question the best configuration (**A5 — hybrid + cross-encoder",
        "re-ranking**) still fails at k=3, with a diagnosed root cause.",
        "",
        "Honest failure reporting is a deliverable, not an embarrassment: a system",
        "with an unexamined 100% is far less trustworthy than one that says exactly",
        "where and why it breaks. Critically, the taxonomy separates defects in the",
        "**system** from defects in the **golden set** — a distinction that changes",
        "what you should do about them.",
        "",
        "## Summary",
        "",
        f"- Questions evaluated: **{payload['n_questions']}**",
        f"- Failures at k=3: **{payload['n_failures']}**",
        f"- Hit Rate@3: **{payload['hit_rate_at_3']}**",
        "",
        "| Root cause | n | Meaning |",
        "|---|---|---|",
    ]
    for cause, n in counts.most_common():
        lines.append(f"| `{cause}` | {n} | {TAXONOMY.get(cause, '')} |")

    lines += ["", "## Individual failures", ""]
    for f in failures:
        lines += [
            f"### `{f['qid']}` — {f['cause']}",
            "",
            f"**Question** ({f['category']}): *{f['question']}*",
            "",
            f"- Retrieved: {'; '.join(f['retrieved_top3']) or '(nothing)'}",
            f"- Expected: {'; '.join(f['expected'])}",
            f"- Gold chunk was in the stage-1 candidate pool: "
            f"**{'yes' if f['gold_in_pool'] else 'no'}**",
            f"- Top score: {f['top_score']}",
            "",
        ]

    lines += [
        "## What this points at next",
        "",
        "- `gold_too_narrow` dominating means the *golden set* is the limiting",
        "  factor, not retrieval — the fix is to broaden gold labelling, and the",
        "  reported metrics are a pessimistic lower bound on true performance.",
        "- `retrieval_miss` is the only class re-ranking can never repair, since",
        "  the chunk never enters the pool. It is the first thing to attack.",
        "- `chunking_split_the_answer` is fixed at ingestion (larger chunks or",
        "  neighbour expansion), not in the retriever.",
        "",
    ]
    (out / "failure_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(failures)} failure(s) of {len(answerable)} -> reports/failure_analysis.md")
    for cause, n in counts.most_common():
        print(f"  {cause:32s} {n}")


if __name__ == "__main__":
    main()
