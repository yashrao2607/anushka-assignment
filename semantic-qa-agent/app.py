"""Streamlit UI -- PRD US-6.2.

Deliberately shows the machinery rather than hiding it. A demo that renders only
an answer is indistinguishable from a chatbot; this one surfaces the retrieval
trace, the confidence against the calibrated threshold, and every source chunk,
so a reviewer can verify each claim in the answer against its evidence.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config           # noqa: E402
from src.generate.answerer import Answerer   # noqa: E402
from src.store.index import load_retriever   # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402

st.set_page_config(page_title="Semantic Q&A Agent", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading models and index…")
def get_answerer(threshold: float):
    """Models are cached across reruns -- reloading the cross-encoder on every
    keystroke would make the UI unusable."""
    cfg = load_config()
    setup_logging(cfg.path("logs_dir"))
    return Answerer(cfg, load_retriever(cfg), threshold=threshold), cfg


st.title("🔎 Semantic Search / Intelligent Q&A Agent")
st.caption(
    "Meaning-based retrieval → cross-encoder re-ranking → grounded answer with "
    "citations, and a calibrated refusal when the corpus does not contain the answer."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Sources to retrieve (k)", 1, 10, 5)
    mode = st.selectbox("Retrieval mode", ["hybrid", "dense", "sparse"], index=0,
                        help="hybrid = dense + BM25 fused with Reciprocal Rank Fusion")
    rerank = st.checkbox("Cross-encoder re-ranking", value=True,
                         help="Second-stage precision. Measured: MRR 0.852 → 0.941")
    threshold = st.slider(
        "Refusal threshold", 0.0, 1.0, 0.02, 0.01,
        help="Calibrated on the golden set, not guessed. Below this the system "
             "refuses without calling the LLM at all.",
    )
    use_llm = st.checkbox("Generate an answer (uses Groq API)", value=True,
                          help="Uncheck for retrieval only — makes zero API calls.")
    st.divider()
    st.caption(
        "**Free-tier note.** Responses are disk-cached and generation is pinned to "
        "temperature 0, so repeating a question costs nothing. Refusals never call "
        "the API."
    )

answerer, cfg = get_answerer(threshold)
answerer.threshold = threshold

EXAMPLES = [
    "Can I get my money back if a client cancels a trip at the last minute?",
    "My laptop will not turn on. What should I do?",
    "What does error code ERR_4092 mean?",
    "How many casual leaves do interns get?",
    "What is the policy on employees keeping pets in the office?",
]

st.write("**Try one of these:**")
cols = st.columns(len(EXAMPLES))
picked = None
for col, example in zip(cols, EXAMPLES):
    label = example if len(example) < 34 else example[:31] + "…"
    if col.button(label, help=example, use_container_width=True):
        picked = example

query = st.text_input("Ask a question about the document corpus:",
                      value=picked or "", placeholder="e.g. how do I claim travel expenses?")

if query:
    started = time.perf_counter()
    with st.spinner("Retrieving and generating…"):
        response = answerer.answer(query, top_k=top_k, mode=mode,
                                   rerank=rerank, allow_llm=use_llm)
    elapsed = (time.perf_counter() - started) * 1000

    if response.refused:
        st.warning(f"**{response.answer}**")
        st.caption(
            f"Refused because the best match scored {response.confidence:.3f}, below "
            f"the calibrated threshold of {threshold:.2f}. **No LLM call was made** — "
            f"the system cannot hallucinate here by construction."
        )
    else:
        st.success(response.answer)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{response.confidence:.3f}",
              delta=f"{response.confidence - threshold:+.3f} vs threshold")
    m2.metric("Sources cited", len(response.citations))
    m3.metric("Latency", f"{elapsed:.0f} ms")
    m4.metric("API calls", 0 if (response.refused or response.from_cache) else 1,
              help="Cached and refused queries cost nothing.")

    if response.citation_violations:
        st.error(
            f"Citation violation: the model cited {response.citation_violations}, "
            f"which do not exist among the supplied sources. Those markers were "
            f"stripped from the answer."
        )

    if response.citations:
        st.subheader("Citations")
        for c in response.citations:
            with st.expander(
                f"[{c.marker}] {c.doc_title} › {c.section_heading} (page {c.page_no})"
            ):
                st.write(c.quote)
                st.caption(f"chunk_id: `{c.chunk_id}`")

    if response.sources:
        st.subheader("Retrieved context")
        for i, chunk in enumerate(response.sources, start=1):
            meta = chunk.metadata
            legs = []
            if chunk.dense_rank:
                legs.append(f"dense #{chunk.dense_rank} ({chunk.dense_score:.3f})")
            if chunk.bm25_rank:
                legs.append(f"bm25 #{chunk.bm25_rank} ({chunk.bm25_score:.2f})")
            if chunk.rerank_score is not None:
                legs.append(f"**rerank {chunk.rerank_score:+.2f}**")
            with st.expander(
                f"[{i}] {meta.get('doc_title')} › {meta.get('section_heading')} "
                f"— {' · '.join(legs) if legs else 'n/a'}"
            ):
                st.write(chunk.text)
                st.caption(
                    f"`{chunk.chunk_id}` · page {meta.get('page_no')} · "
                    f"chars {meta.get('char_start')}–{meta.get('char_end')} · "
                    f"{meta.get('token_count')} tokens"
                )

    with st.expander("Trace"):
        st.json({
            "trace_id": response.trace_id,
            "latency_ms": response.latency_ms,
            "config_fingerprint": response.config_fingerprint,
            "model": response.model,
            "from_cache": response.from_cache,
            "api_usage": answerer.client.usage_summary(),
        })

    st.divider()
    feedback = st.radio("Was this answer useful?", ["—", "👍 yes", "👎 no"],
                        horizontal=True, index=0)
    if feedback != "—":
        path = cfg.root / "data" / "feedback.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            import json

            fh.write(json.dumps({
                "trace_id": response.trace_id, "query": query,
                "useful": feedback.startswith("👍"),
                "confidence": response.confidence,
                "refused": response.refused,
            }) + "\n")
        st.toast("Feedback recorded — thank you.")
