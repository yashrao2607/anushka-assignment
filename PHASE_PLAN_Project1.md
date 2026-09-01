# Project 1 — Execution Plan: 3 Phases × 4 Parts

**Project:** Semantic Search / Intelligent Q&A Agent
**Reference PRD:** `PRD_Project1_Semantic_Search_QA_Agent.md`
**Start:** 01 September 2026 · **Duration:** 14 working days
**Status legend:** ✅ done · 🔄 in progress · ⬜ not started

---

## Why three phases

Each phase is a **shippable, demonstrable milestone** — not an arbitrary time-slice. If work stopped at the end of any phase, there would still be something working to show the manager. Each phase ends with a demo and a written exit criterion, and each phase is split into 4 parts so progress is visible daily rather than weekly.

| Phase | Theme | Days | The one thing it proves |
|---|---|---|---|
| **Phase 1** | Foundation & Ingestion | 1–4 | *"Documents become clean, traceable, retrievable units."* |
| **Phase 2** | Retrieval Core & Measurement | 5–9 | *"Semantic search beats keyword search — and here is the number."* |
| **Phase 3** | Intelligence & Delivery | 10–14 | *"It answers with citations, refuses when it should, and every design choice is justified by an ablation."* |

---

## PHASE 1 — Foundation & Ingestion Pipeline (Days 1–4) ✅

**Goal:** Turn a folder of messy documents into clean, metadata-rich, deduplicated chunks that are ready to embed — with full provenance so every future citation can point back to an exact page and character span.

**Why this is Phase 1:** retrieval quality is capped by chunk quality. Every downstream metric in the PRD inherits the errors made here. This phase is deliberately built and tested *before* a single embedding is computed.

| Part | Deliverable | Exit criterion | Status |
|---|---|---|---|
| **1.1** | Repo scaffold, typed config system, dependency manifest, logging, CLI skeleton | `python -m src.cli --help` works; config loads and validates; `pytest` green | ✅ |
| **1.2** | Document loaders (PDF/DOCX/MD/TXT/HTML/CSV) + text cleaner | Every supported format parses; unsupported/corrupt files are skipped with a warning, never a crash; cleaner de-hyphenates, strips boilerplate and normalises unicode | ✅ |
| **1.3** | Chunking engine — recursive-character split, token-aware, overlap, heading-aware enrichment | Chunks respect size/overlap config; no chunk splits a word; every chunk carries `char_start`/`char_end` back into its source | ✅ |
| **1.4** | Seed corpus + ingestion pipeline + manifest + verification run | `python -m src.cli ingest` produces `chunks.jsonl` + `manifest.csv` + a summary table; re-running is idempotent; ≥ 20 tests pass | ✅ |

**Phase 1 exit demo:** run one command on a folder of documents → get a statistics table (files, pages, chunks, tokens, duration) and a `chunks.jsonl` where any chunk can be traced to its source file, page, and character offset.

---

## PHASE 2 — Retrieval Core & Measurement (Days 5–9) ✅

**Goal:** Make the corpus searchable by meaning, and — critically — build the measuring instrument *before* tuning anything, so every later improvement is provable rather than felt.

**Why this order:** the golden set is built in Part 2.3, *before* the hybrid retriever in Part 2.4, so the hybrid retriever's value is measured, not assumed. Building the evaluation harness early is the single decision that turns this from a demo into engineering.

| Part | Deliverable | Exit criterion |
|---|---|---|
| **2.1** | Embedding layer — Sentence-Transformers, batching, L2 normalisation, SQLite embedding cache | ✅ 48 chunks in 16.8 s cold, near-instant warm |
| **2.2** | ChromaDB persistent store + dense cosine retrieval + CLI `query` | ✅ ChromaStore live; `--compare` shows all three modes side by side |
| **2.3** | **60-question golden set** (graded relevance 0–3) + evaluation harness computing P@k, R@k, MRR, nDCG, Hit@1 | ✅ 51 answerable, 0 unresolved; metrics hand-verified |
| **2.4** | BM25 sparse index + **Reciprocal Rank Fusion** hybrid retrieval | ✅ `ERR_4092` ranks #1; A0/A1/A4 measured, plus hypothesis arm A4b |

**Phase 2 exit demo:** side-by-side — the same query through keyword search and semantic search, with the metrics table proving the difference.

---

## PHASE 3 — Intelligence, Rigour & Delivery (Days 10–14) ✅

**Goal:** Turn retrieval into grounded answers, prove every architectural decision with an ablation, and package it so the reviewer can reproduce it in five minutes.

| Part | Deliverable | Exit criterion |
|---|---|---|
| **3.1** | **Cross-encoder re-ranker** (two-stage retrieval) + toggle | ✅ P@3 0.379→0.431, MRR 0.852→0.941, R@10→1.000; fallback tested |
| **3.2** | **Groq** grounded generation + inline citations + validation + **calibrated refusal gate** | ✅ faithfulness 1.000, hallucination 0.000, 0 violations, refusal 1.000; τ calibrated to 0.02 (PRD guessed 0.40) |
| **3.3** | Interfaces — Streamlit UI + CLI (`ask`/`calibrate`/`judge`) | ✅ `streamlit run app.py` with retrieval trace, citations, API-call counter |
| **3.4** | **Full ablation (A0–A6)**, per-category results, failure analysis, reports | ✅ all tables measured; 2 failures of 51 diagnosed (Hit@3 0.961); 109 tests pass |

**Phase 3 exit demo:** the full 3-minute demo — ingest → semantic query beating keyword → cited answer → correct refusal → the ablation table that explains why the system is built the way it is.

---

## Dependency graph (what blocks what)

```
1.1 config ──► 1.2 loaders ──► 1.3 chunker ──► 1.4 pipeline ──┐
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
              2.1 embeddings ──► 2.2 vector store ──► 2.3 GOLDEN SET + HARNESS ──► 2.4 hybrid
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
              3.1 re-ranker ──► 3.2 generation + refusal ──► 3.3 interfaces ──► 3.4 ablation + report
```

**Critical path:** 1.4 → 2.3 → 3.1 → 3.4.
**The real bottleneck is 2.3 (the golden set)** — nothing can be measured before it exists, so it is scheduled as early as it can possibly be built and time-boxed to one day.
**Slack policy:** if a slip occurs, 3.3 (FastAPI/Docker) is dropped first. The evaluation work (2.3, 3.4) is never cut — it is the differentiator.

---

## Phase 1 completion record

Delivered in `semantic-qa-agent/`:

| Artefact | Path |
|---|---|
| Typed, validated config | `config/default.yaml`, `src/config.py` |
| Structured logging | `src/utils/logging.py` |
| Document loaders (6 formats) | `src/ingest/loaders.py` |
| Text cleaner | `src/ingest/cleaner.py` |
| Chunking engine | `src/ingest/chunker.py` |
| Ingestion pipeline + manifest | `src/ingest/pipeline.py` |
| Data models | `src/models.py` |
| CLI | `src/cli.py` |
| Seed corpus | `data/raw/*.md`, `*.txt` |
| Test suite | `tests/` |
| Phase report | `semantic-qa-agent/reports/PHASE1_REPORT.md` |

## PROJECT COMPLETE — all three phases delivered ✅

| Phase | Headline measured result |
|---|---|
| 1 | 15 docs → 48 chunks, 100% heading-enriched, full provenance, idempotent |
| 2 | Semantic beats keyword: MRR 0.738 → 0.898; paraphrase Hit@3 0.462 → 1.000 |
| 3 | **MRR 0.941 · nDCG 0.942 · R@10 1.000 · faithfulness 1.000 · hallucination 0.000** |

**Against the keyword baseline: +37% Precision@3, +27% MRR, +28% nDCG, perfect
Recall@10.** Every out-of-corpus question was correctly refused without spending
a single API call. 109 tests pass. Full detail in
`semantic-qa-agent/reports/PHASE3_REPORT.md`.
