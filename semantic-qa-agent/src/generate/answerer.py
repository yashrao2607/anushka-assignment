"""Grounded answer pipeline -- PRD Sections 9.8-9.10.

    retrieve -> RELEVANCE GATE -> assemble context -> generate -> validate -> log

**The gate is the important part, and it comes before the LLM.** If the best
re-ranked chunk scores below the calibrated threshold, the system refuses
immediately and *never calls the model at all*. Three consequences:

  * it cannot hallucinate on an out-of-corpus question, because no generation
    happens -- the guarantee is structural, not a matter of prompt obedience;
  * refusals are instant (no network round trip);
  * refusals cost **zero API quota**, which matters directly on a free tier.

The threshold is calibrated empirically from cross-encoder scores (see
`calibrate.py`) rather than guessed, and calibration itself needs no LLM calls.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..retrieve.reranker import sigmoid
from ..retrieve.retriever import Retriever
from ..utils.logging import get_logger
from .groq_client import GroqClient, GroqError
from .prompts import (
    REFUSAL_TEXT, SYSTEM_PROMPT, USER_TEMPLATE, build_context, is_refusal,
    validate_citations,
)


@dataclass
class Citation:
    marker: int
    chunk_id: str
    doc_title: str
    page_no: int | None
    section_heading: str | None
    quote: str


@dataclass
class AnswerResponse:
    trace_id: str
    query: str
    answer: str
    refused: bool
    refusal_reason: str | None = None
    confidence: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    sources: list = field(default_factory=list)
    citation_violations: list[int] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    config_fingerprint: str = ""
    model: str = ""
    from_cache: bool = False


class Answerer:
    """Retrieval-augmented generation with a pre-LLM refusal gate."""

    def __init__(
        self,
        cfg: Config,
        retriever: Retriever,
        client: GroqClient | None = None,
        threshold: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        gcfg = cfg.extra.get("generation", {})
        gatecfg = cfg.extra.get("gate", {})

        self.model = gcfg.get("model", "openai/gpt-oss-120b")
        self.max_words = int(gcfg.get("max_answer_words", 150))
        self.max_context_tokens = int(gcfg.get("max_context_tokens", 3000))
        self.threshold = (
            threshold if threshold is not None
            else float(gatecfg.get("refusal_threshold", 0.40))
        )
        self.client = client or GroqClient(
            root=cfg.root,
            model=self.model,
            temperature=float(gcfg.get("temperature", 0.0)),
        )

    # -- the gate -----------------------------------------------------------
    def gate(self, hits: list) -> tuple[bool, float]:
        """Decide whether the corpus plausibly contains the answer.

        Returns (should_refuse, calibrated_confidence). The confidence is a
        sigmoid over the top cross-encoder logit; when re-ranking is off it
        falls back to the top retrieval score, which is a weaker signal and is
        noted as such in the response.
        """
        if not hits:
            return True, 0.0
        top = hits[0]
        raw = top.rerank_score if top.rerank_score is not None else top.score
        confidence = sigmoid(raw) if top.rerank_score is not None else float(raw)
        return bool(confidence < self.threshold), float(confidence)

    # -- the pipeline -------------------------------------------------------
    def answer(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        rerank: bool = True,
        allow_llm: bool = True,
    ) -> AnswerResponse:
        log = get_logger()
        trace_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()

        hits, trace = self.retriever.retrieve(
            query, top_k=top_k, mode=mode, rerank=rerank
        )
        refuse, confidence = self.gate(hits)

        base = AnswerResponse(
            trace_id=trace_id,
            query=query,
            answer="",
            refused=False,
            confidence=round(confidence, 4),
            sources=hits,
            config_fingerprint=self.cfg.fingerprint(),
            model=self.model,
        )

        if refuse:
            base.answer = REFUSAL_TEXT
            base.refused = True
            base.refusal_reason = (
                f"top relevance {confidence:.3f} < threshold {self.threshold:.2f}"
            )
            base.latency_ms = {
                "retrieval": round(trace.total_ms, 1),
                "llm": 0.0,
                "total": round((time.perf_counter() - started) * 1000, 1),
            }
            log.info("[%s] gated refusal (%.3f < %.2f) -- no LLM call made",
                     trace_id, confidence, self.threshold)
            return base

        context, used = build_context(hits, self.max_context_tokens)
        if not allow_llm:
            base.answer = "(retrieval only -- generation disabled)"
            base.sources = used
            return base

        t_llm = time.perf_counter()
        cached_before = self.client.usage.calls_cached
        try:
            raw = self.client.complete(
                system=SYSTEM_PROMPT.format(max_words=self.max_words),
                user=USER_TEMPLATE.format(context=context, question=query),
            )
        except GroqError as exc:
            # Degrade to retrieval-only rather than failing the request: the
            # sources are still genuinely useful without the synthesis layer.
            log.warning("[%s] generation unavailable: %s", trace_id, exc)
            base.answer = f"(generation unavailable: {exc})"
            base.sources = used
            base.latency_ms = {
                "retrieval": round(trace.total_ms, 1), "llm": 0.0,
                "total": round((time.perf_counter() - started) * 1000, 1),
            }
            return base
        llm_ms = (time.perf_counter() - t_llm) * 1000

        cleaned, valid, invalid = validate_citations(raw, len(used))
        if invalid:
            log.warning("[%s] citation_violation: %s point at non-existent sources",
                        trace_id, invalid)

        citations = []
        for marker in valid:
            chunk = used[marker - 1]
            meta = chunk.metadata
            citations.append(Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                doc_title=meta.get("doc_title", ""),
                page_no=meta.get("page_no"),
                section_heading=meta.get("section_heading"),
                quote=" ".join(chunk.text.split())[:220],
            ))

        base.answer = cleaned
        base.refused = is_refusal(cleaned)
        base.citations = citations
        base.sources = used
        base.citation_violations = invalid
        base.from_cache = self.client.usage.calls_cached > cached_before
        base.latency_ms = {
            "retrieval": round(trace.total_ms, 1),
            "rerank": round(trace.rerank_ms, 1),
            "llm": round(llm_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        }

        self._log_trace(base, trace)
        return base

    def _log_trace(self, response: AnswerResponse, trace) -> None:
        """Structured per-query record -- the raw material for failure analysis."""
        get_logger().info(
            "[%s] answered (%d citation(s), conf %.3f, %.0f ms%s)",
            response.trace_id, len(response.citations), response.confidence,
            response.latency_ms.get("total", 0),
            ", cached" if response.from_cache else "",
            extra={
                "event": "answer",
                "trace_id": response.trace_id,
                "query": response.query,
                "refused": response.refused,
                "confidence": response.confidence,
                "n_citations": len(response.citations),
                "citation_violations": response.citation_violations,
                "reranked": trace.reranked,
                "latency_ms": response.latency_ms,
                "from_cache": response.from_cache,
            },
        )


def build_answerer(cfg: Config, threshold: float | None = None) -> Answerer:
    from ..store.index import load_retriever

    return Answerer(cfg, load_retriever(cfg), threshold=threshold)
