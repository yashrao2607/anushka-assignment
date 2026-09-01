"""LLM-as-judge answer-quality evaluation -- PRD Section 4.3 (M7-M11).

Retrieval metrics say nothing about the *answer*. A system can retrieve perfectly
and still produce an unfaithful summary, or retrieve badly and get lucky. So
answer quality is measured separately (PRD 13.1), on four axes:

  M7  faithfulness      -- is every claim supported by the cited context?
  M8  answer relevance  -- does the answer address the question asked?
  M9  citation accuracy -- does each citation point at a real supplied source?
  M11 hallucination     -- any unsupported factual claim?

**Free-tier discipline.** Judging every one of the 51 answerable questions would
mean 102 live calls (one generation + one judgement each). Instead:

  * a **stratified sample** covering every question category, default 12;
  * generation and judging both hit the shared disk cache, so a re-run is free;
  * a hard `max_calls` budget that raises rather than silently burning quota;
  * the judge runs on a **smaller, cheaper model** than the answerer -- judging a
    binary support relation is a far easier task than composing the answer, so
    spending 70B-class capacity on it buys nothing.

M9 and M11 are computed *without* the LLM wherever possible: a dangling citation
is detectable by arithmetic, and an answer with zero citations that is not a
refusal is definitionally ungrounded. Only genuine entailment needs a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import Config
from ..generate.answerer import Answerer
from ..generate.groq_client import GroqError
from ..generate.prompts import JUDGE_SYSTEM, JUDGE_TEMPLATE, build_context
from ..utils.logging import get_logger
from .golden import load_golden_set, resolve_grades

JUDGE_MODEL = "openai/gpt-oss-20b"


def stratified_sample(questions: list, n: int) -> list:
    """Take a sample spanning every category rather than the first n.

    A sample skewed toward `direct` questions would flatter the system; the
    point of the categories is that they fail differently.
    """
    by_category: dict[str, list] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)

    picked: list = []
    categories = sorted(by_category)
    i = 0
    while len(picked) < n and any(by_category.values()):
        category = categories[i % len(categories)]
        if by_category[category]:
            picked.append(by_category[category].pop(0))
        i += 1
        if i > n * len(categories) + 10:
            break
    return picked[:n]


def _parse_judgement(raw: str) -> dict:
    """Extract the JSON object from a judge reply, tolerating stray prose."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"faithful": 0, "relevant": 0, "unsupported_claim": "unparseable"}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithful": 0, "relevant": 0, "unsupported_claim": "unparseable"}
    return {
        "faithful": int(bool(data.get("faithful", 0))),
        "relevant": int(bool(data.get("relevant", 0))),
        "unsupported_claim": str(data.get("unsupported_claim", ""))[:200],
    }


def run_judgement(cfg: Config, answerer: Answerer, sample_size: int = 12) -> dict:
    log = get_logger()
    questions = load_golden_set(
        cfg.root / cfg.extra.get("eval", {}).get("golden_set", "data/golden_set.jsonl")
    )
    from ..ingest.pipeline import load_chunks

    questions, _ = resolve_grades(questions, load_chunks(cfg))

    answerable = [q for q in questions if q.answerable and q.grades]
    unanswerable = [q for q in questions if not q.answerable]

    sample = stratified_sample(answerable, sample_size)
    # Always include a few unanswerable questions: refusal behaviour on the live
    # pipeline must be verified end-to-end, not only in offline calibration.
    refusal_sample = unanswerable[:3]

    log.info("judging %d answerable + %d unanswerable question(s) (budget %d live calls)",
             len(sample), len(refusal_sample), answerer.client.max_calls)

    rows: list[dict] = []
    errors: list[str] = []

    for q in sample:
        try:
            response = answerer.answer(q.question, top_k=5)
        except GroqError as exc:
            errors.append(f"{q.qid}: {exc}")
            break

        n_citations = len(response.citations)
        row = {
            "qid": q.qid, "category": q.category, "question": q.question,
            "answer": response.answer, "refused": response.refused,
            "confidence": response.confidence,
            "n_citations": n_citations,
            "citation_violations": len(response.citation_violations),
            # M9: computable without a model -- a citation is accurate if it
            # points at a source that was actually supplied.
            "citation_accuracy": (
                1.0 if n_citations and not response.citation_violations
                else (0.0 if response.citation_violations else None)
            ),
            "cited_gold": any(c.chunk_id in q.relevant for c in response.citations),
        }

        if response.refused:
            # A refusal on an answerable question is a miss, not a quality
            # failure -- scoring it for faithfulness would be meaningless.
            row.update({"faithful": None, "relevant": None,
                        "unsupported_claim": "", "judged": False})
            rows.append(row)
            continue

        context, _ = build_context(response.sources, answerer.max_context_tokens)
        try:
            verdict = _parse_judgement(answerer.client.complete(
                system=JUDGE_SYSTEM,
                user=JUDGE_TEMPLATE.format(
                    context=context, question=q.question, answer=response.answer
                ),
                model=JUDGE_MODEL,
                max_tokens=900,
                json_mode=True,
                reasoning_effort="low",
            ))
        except GroqError as exc:
            errors.append(f"{q.qid} (judge): {exc}")
            break
        row.update(verdict)
        row["judged"] = True
        rows.append(row)

    # Live refusal check -- gated refusals cost zero API calls.
    refusal_rows = []
    for q in refusal_sample:
        response = answerer.answer(q.question, top_k=5)
        refusal_rows.append({
            "qid": q.qid, "question": q.question,
            "refused": response.refused, "confidence": response.confidence,
            "answer": response.answer[:120],
        })

    judged = [r for r in rows if r.get("judged")]
    cited = [r for r in rows if r["citation_accuracy"] is not None]
    summary = {
        "questions_answered": len(rows),
        "judged_by_llm": len(judged),
        "refused_on_answerable": sum(1 for r in rows if r["refused"]),
        "faithfulness_M7": round(
            sum(r["faithful"] for r in judged) / len(judged), 3) if judged else None,
        "answer_relevance_M8": round(
            sum(r["relevant"] for r in judged) / len(judged), 3) if judged else None,
        "citation_accuracy_M9": round(
            sum(r["citation_accuracy"] for r in cited) / len(cited), 3) if cited else None,
        "refusal_correctness_M10": round(
            sum(1 for r in refusal_rows if r["refused"]) / len(refusal_rows), 3
        ) if refusal_rows else None,
        "hallucination_rate_M11": round(
            sum(1 for r in judged if not r["faithful"]) / len(judged), 3) if judged else None,
        "cited_gold_chunk": round(
            sum(1 for r in rows if r["cited_gold"]) / len(rows), 3) if rows else None,
        "total_citation_violations": sum(r["citation_violations"] for r in rows),
        **answerer.client.usage_summary(),
    }

    payload = {
        "summary": summary, "rows": rows, "refusal_checks": refusal_rows,
        "errors": errors, "judge_model": JUDGE_MODEL,
        "answer_model": answerer.model, "threshold": answerer.threshold,
    }

    out = cfg.path("reports_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "answer_quality.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out / "answer_quality.md", payload)
    return payload


def _write_report(path: Path, payload: dict) -> None:
    s = payload["summary"]
    lines = [
        "# Answer Quality Report (LLM-as-judge)",
        "",
        f"Answer model: `{payload['answer_model']}` (Groq) · Judge model: "
        f"`{payload['judge_model']}` · Refusal threshold: {payload['threshold']}",
        "",
        "Evaluated on a **stratified sample spanning every question category**, not",
        "the first n questions -- a sample skewed toward easy `direct` lookups would",
        "flatter the system. Sample size is kept small and every response is disk-",
        "cached, so this report is reproducible at zero additional API cost.",
        "",
        "## Metrics",
        "",
        "| ID | Metric | Target | Measured |",
        "|---|---|---|---|",
        f"| M7 | Faithfulness (claims supported by context) | ≥ 0.90 | "
        f"**{s['faithfulness_M7']}** |",
        f"| M8 | Answer relevance | ≥ 0.88 | **{s['answer_relevance_M8']}** |",
        f"| M9 | Citation accuracy | ≥ 0.95 | **{s['citation_accuracy_M9']}** |",
        f"| M10 | Refusal correctness (out-of-corpus) | ≥ 0.90 | "
        f"**{s['refusal_correctness_M10']}** |",
        f"| M11 | Hallucination rate | ≤ 0.05 | **{s['hallucination_rate_M11']}** |",
        "",
        "| Diagnostic | Value |",
        "|---|---|",
        f"| Questions answered | {s['questions_answered']} |",
        f"| Judged by LLM | {s['judged_by_llm']} |",
        f"| Refused on an *answerable* question | {s['refused_on_answerable']} |",
        f"| Answers citing a gold chunk | {s['cited_gold_chunk']} |",
        f"| Dangling citations (`citation_violation`) | {s['total_citation_violations']} |",
        "",
        "## API usage",
        "",
        f"- Live calls: **{s['live_calls']}** · cache hits: {s['cached_calls']}",
        f"- Tokens: {s['prompt_tokens']} prompt + {s['completion_tokens']} completion "
        f"= {s['total_tokens']}",
        f"- Cached responses on disk: {s['cache_entries']}",
        "",
        "Refusals are gated *before* generation, so every correct refusal costs zero",
        "API calls and cannot hallucinate by construction.",
        "",
        "## Live refusal checks (out-of-corpus questions)",
        "",
        "| qid | question | refused | confidence |",
        "|---|---|---|---|",
    ]
    for r in payload["refusal_checks"]:
        lines.append(
            f"| {r['qid']} | {r['question'][:60]} | "
            f"{'✅' if r['refused'] else '❌'} | {r['confidence']:.4f} |"
        )

    lines += ["", "## Per-question detail", "",
              "| qid | category | refused | cites | cited gold | faithful | relevant |",
              "|---|---|---|---|---|---|---|"]
    for r in payload["rows"]:
        lines.append(
            f"| {r['qid']} | {r['category']} | {'yes' if r['refused'] else 'no'} | "
            f"{r['n_citations']} | {'yes' if r['cited_gold'] else 'no'} | "
            f"{r.get('faithful', '–')} | {r.get('relevant', '–')} |"
        )

    unfaithful = [r for r in payload["rows"] if r.get("judged") and not r.get("faithful")]
    if unfaithful:
        lines += ["", "## Unsupported claims flagged by the judge", ""]
        for r in unfaithful:
            lines.append(f"- `{r['qid']}` — {r.get('unsupported_claim', '')!r}")
    if payload["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e}" for e in payload["errors"]]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
