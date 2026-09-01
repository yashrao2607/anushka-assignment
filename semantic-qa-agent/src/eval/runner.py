"""Evaluation runner and report writer -- PRD Sections 13.1-13.4.

Runs the golden set through one or more retrieval configurations and emits:

  reports/eval_report.md   headline metrics + per-category breakdown
  reports/ablation.md      the A0-A4 comparison table
  reports/eval_results.json machine-readable results for later phases

Retrieval quality and answer quality are reported separately (PRD 13.1). Phase 2
covers retrieval only; the answer-quality half arrives with generation in
Phase 3. Reporting a single blended number would hide which half is broken.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..retrieve.retriever import Mode, Retriever
from ..utils.logging import get_logger
from .golden import GoldenQuestion, golden_set_stats, load_golden_set, resolve_grades
from .metrics import aggregate, evaluate_query

K_VALUES = (1, 3, 5, 10)


@dataclass
class ConfigResult:
    """Metrics for one retrieval configuration (one ablation row)."""

    name: str
    description: str
    mode: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    n_queries: int = 0
    failures: list[dict] = field(default_factory=list)
    bm25_top_n: int | None = None
    reranked: bool = False


def precision_ceiling(questions: list[GoldenQuestion], k: int) -> float:
    """The highest Precision@k any system could achieve on this golden set.

    Precision@k divides by k, so a question with only 1 relevant chunk caps at
    1/k no matter how perfect the ranking. Reporting raw P@3 without this
    ceiling makes a near-optimal system look broken. This is the context that
    turns 0.379 from "a failure" into "81% of the achievable maximum".
    """
    answerable = [q for q in questions if q.answerable and q.grades]
    if not answerable:
        return 0.0
    return sum(min(len(q.relevant), k) / k for q in answerable) / len(answerable)


def evaluate_config(
    retriever: Retriever,
    questions: list[GoldenQuestion],
    *,
    name: str,
    description: str,
    mode: Mode,
    top_k: int = 10,
    rerank: bool = False,
) -> ConfigResult:
    """Run every answerable golden question through one retrieval mode."""
    answerable = [q for q in questions if q.answerable and q.grades]
    per_query: list[dict[str, float]] = []
    by_category: dict[str, list[dict[str, float]]] = {}
    latencies: list[float] = []
    failures: list[dict] = []

    for q in answerable:
        t0 = time.perf_counter()
        hits, _ = retriever.retrieve(q.question, top_k=top_k, mode=mode, rerank=rerank)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = [h.chunk_id for h in hits]

        scores = evaluate_query(retrieved, q.relevant, q.grades, K_VALUES)
        per_query.append(scores)
        by_category.setdefault(q.category, []).append(scores)

        # A miss at k=3 is what the failure analysis in Phase 3 will triage.
        if scores["hit_rate@3"] == 0.0:
            failures.append({
                "qid": q.qid,
                "question": q.question,
                "category": q.category,
                "retrieved_top3": retrieved[:3],
                "expected_any_of": sorted(q.relevant)[:4],
            })

    latencies.sort()
    return ConfigResult(
        name=name,
        description=description,
        mode=mode,
        metrics=aggregate(per_query),
        per_category={c: aggregate(v) for c, v in sorted(by_category.items())},
        category_counts={c: len(v) for c, v in sorted(by_category.items())},
        bm25_top_n=retriever.bm25_top_n if mode in ("sparse", "hybrid") else None,
        reranked=rerank,
        latency_ms={
            "p50": round(statistics.median(latencies), 1) if latencies else 0.0,
            "p95": round(latencies[int(len(latencies) * 0.95) - 1], 1) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        },
        n_queries=len(answerable),
        failures=failures,
    )


# --------------------------------------------------------------------------- #
# Ablation definitions -- PRD Section 13.3, rows A0-A4 (Phase 2 scope)
# --------------------------------------------------------------------------- #

# (name, description, mode, bm25_top_n override, rerank)
ABLATIONS: list[tuple[str, str, Mode, int | None, bool]] = [
    ("A0", "Keyword baseline (BM25 only)", "sparse", None, False),
    ("A1", "Dense only (MiniLM bi-encoder)", "dense", None, False),
    ("A4", "Hybrid (dense + BM25, RRF)", "hybrid", None, False),
    # A4b tests a specific hypothesis rather than adding a variant for its own
    # sake: if hybrid underperforms dense on paraphrase queries, the cause
    # should be BM25 *rank dilution* -- on a small corpus BM25's top-25 is over
    # half of everything, so most of its ranking is noise that RRF nonetheless
    # rewards. Narrowing the sparse leg to its top-5 should recover paraphrase
    # performance while keeping the exact-identifier wins. If it does, the
    # diagnosis is confirmed; if it does not, the hypothesis was wrong.
    ("A4b", "Hybrid, sparse leg narrowed to top-5", "hybrid", 5, False),
    # Phase 3: the two-stage architecture. A5 tests whether a cross-encoder
    # closes the rank-quality gap that Phase 2 measured and diagnosed; A6 checks
    # whether re-ranking a dense-only pool does as well, which would mean the
    # sparse leg is carrying no weight once a re-ranker is present.
    ("A5", "Hybrid + cross-encoder re-rank", "hybrid", None, True),
    ("A6", "Dense + cross-encoder re-rank", "dense", None, True),
]


def run_evaluation(cfg: Config, retriever: Retriever, only: str | None = None) -> dict:
    """Run the golden set across every Phase 2 ablation arm."""
    log = get_logger()
    golden_path = cfg.root / cfg.extra.get("eval", {}).get(
        "golden_set", "data/golden_set.jsonl"
    )
    questions = load_golden_set(golden_path)

    from ..ingest.pipeline import load_chunks

    chunks = load_chunks(cfg)
    questions, warnings = resolve_grades(questions, chunks)
    stats = golden_set_stats(questions)

    log.info("golden set: %d question(s), %d answerable, %d resolved against %d chunk(s)",
             stats["total"], stats["answerable"], stats["resolved"], len(chunks))
    for warning in warnings:
        log.warning("golden set: %s", warning)

    arms = [a for a in ABLATIONS if only is None or a[0] == only]
    results: list[ConfigResult] = []
    original_bm25_top_n = retriever.bm25_top_n
    for name, description, mode, bm25_top_n, rerank in arms:
        log.info("evaluating %s -- %s", name, description)
        retriever.bm25_top_n = bm25_top_n or original_bm25_top_n
        results.append(
            evaluate_config(retriever, questions, name=name,
                            description=description, mode=mode, rerank=rerank)
        )
    retriever.bm25_top_n = original_bm25_top_n

    ceilings = {f"precision@{k}": round(precision_ceiling(questions, k), 4)
                for k in K_VALUES}

    payload = {
        "golden_set": stats,
        "warnings": warnings,
        "n_chunks": len(chunks),
        "config_fingerprint": cfg.fingerprint(),
        "embedding_model": cfg.extra.get("embedding", {}).get("model"),
        "precision_ceiling": ceilings,
        "results": [
            {
                "name": r.name, "description": r.description, "mode": r.mode,
                "metrics": r.metrics, "per_category": r.per_category,
                "category_counts": r.category_counts, "bm25_top_n": r.bm25_top_n,
                "reranked": r.reranked,
                "latency_ms": r.latency_ms, "n_queries": r.n_queries,
                "n_failures": len(r.failures),
                "precision@3_pct_of_ceiling": round(
                    100 * r.metrics.get("precision@3", 0) / ceilings["precision@3"], 1
                ) if ceilings["precision@3"] else 0.0,
            }
            for r in results
        ],
    }

    _write_json(cfg, payload)
    _write_report(cfg, payload, results)
    _write_ablation(cfg, payload, results)
    return payload


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #

def _write_json(cfg: Config, payload: dict) -> Path:
    path = cfg.path("reports_dir") / "eval_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _write_report(cfg: Config, payload: dict, results: list[ConfigResult]) -> Path:
    best = max(results, key=lambda r: r.metrics.get("precision@3", 0.0))
    stats = payload["golden_set"]
    lines: list[str] = [
        "# Retrieval Evaluation Report",
        "",
        "*Auto-generated by `python -m src.cli evaluate`. Every number below is",
        "measured, not estimated.*",
        "",
        "## Run context",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Chunks indexed | {payload['n_chunks']} |",
        f"| Embedding model | `{payload['embedding_model']}` |",
        f"| Config fingerprint | `{payload['config_fingerprint']}` |",
        f"| Golden questions | {stats['total']} ({stats['answerable']} answerable, "
        f"{stats['unanswerable']} unanswerable) |",
        f"| Questions with resolved gold | {stats['resolved']} |",
        f"| Avg. relevant chunks / question | {stats['avg_relevant_per_q']} |",
        "",
        "> Unanswerable questions are held back for the Phase 3 refusal test (M10);",
        "> they are not part of the retrieval metrics below.",
        "",
        "## Headline metrics",
        "",
        "| Config | Mode | P@1 | P@3 | P@5 | R@5 | R@10 | MRR@10 | nDCG@10 | Hit@3 | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| **{r.name}** {r.description} | {r.mode} | {_fmt(m['precision@1'])} | "
            f"{_fmt(m['precision@3'])} | {_fmt(m['precision@5'])} | {_fmt(m['recall@5'])} | "
            f"{_fmt(m['recall@10'])} | {_fmt(m['mrr@10'])} | {_fmt(m['ndcg@10'])} | "
            f"{_fmt(m['hit_rate@3'])} | {r.latency_ms['p95']} |"
        )

    ceil = payload["precision_ceiling"]
    lines += [
        "",
        f"**Best configuration by Precision@3: {best.name} — {best.description}**",
        "",
        "### Reading Precision@k correctly",
        "",
        f"Precision@k divides by k, so a question with only one relevant chunk caps at",
        f"1/k however perfect the ranking. On this golden set the average question has",
        f"**{stats['avg_relevant_per_q']} relevant chunks**, which puts a hard ceiling on P@k:",
        "",
        "| Metric | Ceiling | Best achieved | % of ceiling |",
        "|---|---|---|---|",
    ]
    for k in (1, 3, 5):
        key = f"precision@{k}"
        achieved = max(r.metrics.get(key, 0.0) for r in results)
        pct = 100 * achieved / ceil[key] if ceil[key] else 0.0
        lines.append(f"| P@{k} | {_fmt(ceil[key])} | {_fmt(achieved)} | {pct:.0f}% |")

    lines += [
        "",
        "Raw P@3 therefore understates performance badly. **Recall@10, MRR and nDCG are",
        "the metrics to judge this system on**, because they are not capped by the",
        "number of relevant chunks a question happens to have.",
        "",
        "## Per-category breakdown",
        "",
        "Averaged metrics hide which *kind* of question is failing. This is the table",
        "that says where the system is actually weak.",
        "",
    ]
    for r in results:
        lines += [
            f"### {r.name} — {r.description}",
            "",
            "| Category | n | P@3 | R@5 | MRR@10 | nDCG@10 | Hit@3 |",
            "|---|---|---|---|---|---|---|",
        ]
        for category, m in r.per_category.items():
            n = r.category_counts.get(category, 0)
            lines.append(
                f"| {category} | {n} | {_fmt(m['precision@3'])} | {_fmt(m['recall@5'])} | "
                f"{_fmt(m['mrr@10'])} | {_fmt(m['ndcg@10'])} | {_fmt(m['hit_rate@3'])} |"
            )
        lines.append("")

    lines += ["## Retrieval failures (no relevant chunk in the top 3)", ""]
    for r in results:
        lines.append(f"**{r.name}** — {len(r.failures)} of {r.n_queries} queries")
        for f in r.failures[:8]:
            lines.append(
                f"- `{f['qid']}` ({f['category']}) — *{f['question']}* → "
                f"got `{', '.join(f['retrieved_top3'][:2])}`"
            )
        lines.append("")

    if payload["warnings"]:
        lines += ["## Golden-set warnings", "",
                  "Unresolvable spans are defects in the *golden set*, not in retrieval,",
                  "and are listed here so they cannot silently depress the metrics.", ""]
        lines += [f"- {w}" for w in payload["warnings"]]
        lines.append("")

    path = cfg.path("reports_dir") / "eval_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    get_logger().info("wrote evaluation report -> %s", path)
    return path


def _write_ablation(cfg: Config, payload: dict, results: list[ConfigResult]) -> Path:
    baseline = next((r for r in results if r.name == "A0"), results[0])
    base_p3 = baseline.metrics.get("precision@3", 0.0)

    lines = [
        "# Ablation Matrix — Phase 2 (rows A0–A4)",
        "",
        "Each row is one command: `python -m src.cli evaluate --only A4`.",
        "Rows A2/A3 (alternative embedding models) and A5–A9 (re-ranking, chunk-size",
        "arms, HyDE) are filled in Phase 3.",
        "",
        "| # | Configuration | P@3 | P@5 | R@10 | MRR@10 | nDCG@10 | p95 ms | Δ P@3 vs A0 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.metrics
        delta = m["precision@3"] - base_p3
        delta_s = "—" if r.name == "A0" else f"{delta:+.3f}"
        lines.append(
            f"| {r.name} | {r.description} | {_fmt(m['precision@3'])} | "
            f"{_fmt(m['precision@5'])} | {_fmt(m['recall@10'])} | {_fmt(m['mrr@10'])} | "
            f"{_fmt(m['ndcg@10'])} | {r.latency_ms['p95']} | {delta_s} |"
        )

    lines += [
        "",
        "## Reading the table",
        "",
        "- **A0 (BM25 only)** is the keyword baseline the whole project exists to beat.",
        "- **A1 (dense only)** isolates the contribution of semantic embeddings.",
        "- **A4 (hybrid + RRF)** should beat both, because the two legs fail on",
        "  different query types: dense loses on rare literal tokens, sparse loses on",
        "  paraphrase. The per-category table in `eval_report.md` is where that",
        "  complementarity is visible — check `exact_id` versus `paraphrase`.",
        "",
    ]
    path = cfg.path("reports_dir") / "ablation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    get_logger().info("wrote ablation table -> %s", path)
    return path
