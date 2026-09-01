"""Golden-set loading and span-based relevance resolution -- PRD Section 7.3.

**The central design decision of Phase 2.** The obvious way to write a golden
set is to label gold *chunk ids*. That is a trap: chunk ids depend on the
chunking configuration, so the moment the ablation sweep changes chunk size from
512 to 256, every gold label points at an id that no longer exists and the whole
golden set silently evaluates to zero. The labels would have to be redone for
every ablation arm -- which in practice means the ablation never gets run.

So gold is labelled by **text span** instead: a distinctive phrase that answers
the question. At evaluation time, spans are resolved against whatever chunk set
currently exists. The golden set is authored once and stays valid across every
chunking configuration, which is precisely what makes the ablation affordable.

Graded relevance (0-3), required for nDCG:

    3  the chunk contains a primary span -- it directly answers the question
    2  the chunk contains a supporting span -- useful corroborating context
    1  the chunk shares a section heading with a grade-3 chunk in a gold doc
    0  everything else
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Whitespace- and case-insensitive form used for span matching.

    Cleaning collapses newlines, so a span authored across a line break in the
    source would never match the cleaned chunk. Normalising both sides makes
    span authoring robust to how the source happens to be wrapped.
    """
    return _WS.sub(" ", text.lower()).strip()


@dataclass
class GoldenQuestion:
    qid: str
    question: str
    category: str
    gold_docs: list[str] = field(default_factory=list)
    primary_spans: list[str] = field(default_factory=list)
    supporting_spans: list[str] = field(default_factory=list)
    answerable: bool = True

    # resolved at evaluation time against the current chunk set
    grades: dict[str, int] = field(default_factory=dict)

    @property
    def relevant(self) -> set[str]:
        """Chunks counted as relevant for binary metrics (grade >= 2).

        Grade-1 chunks are same-section context: genuinely useful to a reader,
        but counting them as 'relevant' would inflate precision. They still
        contribute to nDCG, where their lower gain is handled correctly.
        """
        return {cid for cid, g in self.grades.items() if g >= 2}


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            questions.append(
                GoldenQuestion(
                    qid=record["qid"],
                    question=record["question"],
                    category=record.get("category", "direct"),
                    gold_docs=record.get("gold_docs", []),
                    primary_spans=record.get("primary_spans", []),
                    supporting_spans=record.get("supporting_spans", []),
                    answerable=record.get("answerable", True),
                )
            )
    return questions


def resolve_grades(
    questions: list[GoldenQuestion], chunks: list[dict]
) -> tuple[list[GoldenQuestion], list[str]]:
    """Attach graded relevance to each question. Returns (questions, warnings).

    Warnings are returned rather than raised: an unresolvable span is a defect in
    the *golden set*, and it must be visible in the report, because a question
    with no resolvable gold silently scores 0 on every metric and would drag the
    reported average down for a reason that has nothing to do with retrieval.
    """
    normalized = [(c["chunk_id"], normalize(c["text"]), c) for c in chunks]
    warnings: list[str] = []

    for q in questions:
        if not q.answerable:
            continue
        grades: dict[str, int] = {}
        heading_of_primary: set[tuple[str, str]] = set()

        for span in q.primary_spans:
            needle = normalize(span)
            matched = [cid for cid, text, _ in normalized if needle in text]
            if not matched:
                warnings.append(f"{q.qid}: primary span not found in any chunk: {span[:60]!r}")
            for cid in matched:
                grades[cid] = 3
            for cid, _, rec in normalized:
                if cid in matched and rec.get("section_heading"):
                    heading_of_primary.add((rec["doc_id"], rec["section_heading"]))

        for span in q.supporting_spans:
            needle = normalize(span)
            matched = [cid for cid, text, _ in normalized if needle in text]
            if not matched:
                warnings.append(f"{q.qid}: supporting span not found: {span[:60]!r}")
            for cid in matched:
                grades[cid] = max(grades.get(cid, 0), 2)

        # Grade 1: same document + same section as a directly-answering chunk.
        for cid, _, rec in normalized:
            if cid in grades:
                continue
            key = (rec.get("doc_id"), rec.get("section_heading"))
            if key in heading_of_primary:
                grades[cid] = 1

        if not grades:
            warnings.append(f"{q.qid}: no relevant chunks resolved -- question will score 0")
        q.grades = grades

    return questions, warnings


def golden_set_stats(questions: list[GoldenQuestion]) -> dict:
    by_category: dict[str, int] = {}
    for q in questions:
        by_category[q.category] = by_category.get(q.category, 0) + 1
    answerable = [q for q in questions if q.answerable]
    resolved = [q for q in answerable if q.grades]
    return {
        "total": len(questions),
        "answerable": len(answerable),
        "unanswerable": len(questions) - len(answerable),
        "resolved": len(resolved),
        "unresolved": len(answerable) - len(resolved),
        "by_category": by_category,
        "avg_relevant_per_q": round(
            sum(len(q.relevant) for q in resolved) / len(resolved), 2
        ) if resolved else 0.0,
    }
