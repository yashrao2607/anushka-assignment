"""Refusal-threshold calibration -- PRD Section 9.8.

The refusal threshold decides whether the system answers or says "I don't know".
Guessing it is not acceptable: too low and the system hallucinates on
out-of-corpus questions; too high and it refuses questions it could actually
answer. Either failure destroys user trust, in opposite directions.

So it is **measured**. The golden set already contains both classes:

  * 51 answerable questions   -> the system SHOULD answer
  * 9  unanswerable questions -> the system SHOULD refuse

Sweeping the threshold over both groups and scoring each candidate value with F1
produces the operating point, plus the full precision/recall trade-off curve.

**This entire procedure needs zero LLM calls.** It works purely on cross-encoder
relevance scores, which is why the gate sits *before* generation in the pipeline:
it can be calibrated and validated for free, and it protects the API quota rather
than consuming it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from ..generate.answerer import Answerer
from ..retrieve.reranker import sigmoid
from ..utils.logging import get_logger
from .golden import load_golden_set, resolve_grades


def collect_confidences(
    answerer: Answerer, questions: list, top_k: int = 5
) -> tuple[list[float], list[float]]:
    """Top-chunk confidence for answerable and unanswerable questions."""
    answerable: list[float] = []
    unanswerable: list[float] = []
    for q in questions:
        hits, _ = answerer.retriever.retrieve(
            q.question, top_k=top_k, mode="hybrid", rerank=True
        )
        if not hits:
            confidence = 0.0
        else:
            top = hits[0]
            confidence = (
                sigmoid(top.rerank_score) if top.rerank_score is not None
                else float(top.score)
            )
        (answerable if q.answerable else unanswerable).append(confidence)
    return answerable, unanswerable


def sweep(
    answerable: list[float], unanswerable: list[float], steps: int = 89
) -> list[dict]:
    """Score every candidate threshold from 0.01 to 0.89.

    Definitions (framing "answer" as the positive class):
      TP  answerable question, correctly answered   (conf >= tau)
      FN  answerable question, wrongly refused      (conf <  tau)
      TN  unanswerable question, correctly refused  (conf <  tau)
      FP  unanswerable question, wrongly answered   (conf >= tau)  <- the
          dangerous one: this is where hallucination happens.
    """
    rows: list[dict] = []
    for i in range(1, steps + 1):
        tau = round(i / 100, 2)
        tp = sum(1 for c in answerable if c >= tau)
        fn = len(answerable) - tp
        fp = sum(1 for c in unanswerable if c >= tau)
        tn = len(unanswerable) - fp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        refusal_correctness = tn / len(unanswerable) if unanswerable else 0.0
        answer_correctness = tp / len(answerable) if answerable else 0.0

        rows.append({
            "threshold": tau, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "answer_precision": round(precision, 4),
            "answer_recall": round(recall, 4),
            "f1": round(f1, 4),
            "refusal_correctness": round(refusal_correctness, 4),
            "answer_correctness": round(answer_correctness, 4),
            "balanced": round((refusal_correctness + answer_correctness) / 2, 4),
        })
    return rows


def run_calibration(cfg: Config, answerer: Answerer) -> dict:
    log = get_logger()
    questions = load_golden_set(
        cfg.root / cfg.extra.get("eval", {}).get("golden_set", "data/golden_set.jsonl")
    )
    from ..ingest.pipeline import load_chunks

    questions, _ = resolve_grades(questions, load_chunks(cfg))

    answerable, unanswerable = collect_confidences(answerer, questions)
    log.info("calibration: %d answerable, %d unanswerable (0 LLM calls)",
             len(answerable), len(unanswerable))

    rows = sweep(answerable, unanswerable)
    # Balanced accuracy is the selection criterion rather than raw F1: with 51
    # answerable against 9 unanswerable, F1 is dominated by the majority class
    # and would happily pick a threshold that never refuses anything.
    best = max(rows, key=lambda r: (r["balanced"], r["f1"]))

    payload = {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answerable_confidence": {
            "min": round(min(answerable), 4) if answerable else 0.0,
            "median": round(sorted(answerable)[len(answerable) // 2], 4) if answerable else 0.0,
            "max": round(max(answerable), 4) if answerable else 0.0,
        },
        "unanswerable_confidence": {
            "min": round(min(unanswerable), 4) if unanswerable else 0.0,
            "median": round(sorted(unanswerable)[len(unanswerable) // 2], 4) if unanswerable else 0.0,
            "max": round(max(unanswerable), 4) if unanswerable else 0.0,
        },
        "recommended_threshold": best["threshold"],
        "at_recommended": best,
        "curve": rows,
    }

    out = cfg.path("reports_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out / "calibration.md", payload)
    log.info("recommended refusal threshold: %.2f (balanced accuracy %.3f)",
             best["threshold"], best["balanced"])
    return payload


def _write_report(path: Path, payload: dict) -> None:
    best = payload["at_recommended"]
    ans, una = payload["answerable_confidence"], payload["unanswerable_confidence"]
    lines = [
        "# Refusal-Threshold Calibration",
        "",
        "The threshold that decides *answer* versus *I don't know*, measured rather",
        "than guessed. Produced with **zero LLM calls** -- it operates purely on",
        "cross-encoder relevance scores, which is why the gate sits before generation.",
        "",
        "## Confidence distributions",
        "",
        "| Group | n | min | median | max |",
        "|---|---|---|---|---|",
        f"| Answerable (should answer) | {payload['n_answerable']} | {ans['min']} | "
        f"{ans['median']} | {ans['max']} |",
        f"| Unanswerable (should refuse) | {payload['n_unanswerable']} | {una['min']} | "
        f"{una['median']} | {una['max']} |",
        "",
        "A clean separation between the two medians is what makes a threshold",
        "meaningful at all. Overlap between the distributions is the irreducible",
        "error any single threshold must trade against.",
        "",
        f"## Recommended threshold: **{payload['recommended_threshold']}**",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answer correctness (answerable correctly answered) | {best['answer_correctness']} |",
        f"| Refusal correctness (unanswerable correctly refused) | {best['refusal_correctness']} |",
        f"| Balanced accuracy | {best['balanced']} |",
        f"| F1 (answer as positive class) | {best['f1']} |",
        f"| False answers on out-of-corpus questions | {best['fp']} |",
        f"| Wrongly refused answerable questions | {best['fn']} |",
        "",
        "Balanced accuracy is the selection criterion rather than raw F1: with",
        f"{payload['n_answerable']} answerable against {payload['n_unanswerable']} unanswerable,",
        "F1 is dominated by the majority class and would pick a threshold that",
        "essentially never refuses.",
        "",
        "## Trade-off curve",
        "",
        "| τ | answer correctness | refusal correctness | balanced | F1 |",
        "|---|---|---|---|---|",
    ]
    for row in payload["curve"]:
        if round(row["threshold"] * 100) % 5 == 0:
            marker = " ←" if row["threshold"] == payload["recommended_threshold"] else ""
            lines.append(
                f"| {row['threshold']:.2f} | {row['answer_correctness']:.3f} | "
                f"{row['refusal_correctness']:.3f} | {row['balanced']:.3f} | "
                f"{row['f1']:.3f}{marker} |"
            )
    lines += [
        "",
        "**How to read this.** Raising τ makes the system more cautious: refusal",
        "correctness rises while answer correctness falls. The recommended value is",
        "the point where the two are best balanced. A deployment with a lower",
        "tolerance for wrong answers than for unhelpful ones should deliberately",
        "choose a higher τ -- the curve makes that a documented decision rather than",
        "an accident.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
