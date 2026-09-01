# Semantic Search / Intelligent Q&A Agent

A question-answering agent over a custom document corpus that retrieves by
**meaning** rather than keyword match, re-ranks for precision, and answers with
**inline citations** — refusing to answer when the corpus does not contain the
information.

**Reference PRD:** [`../PRD_Project1_Semantic_Search_QA_Agent.md`](../PRD_Project1_Semantic_Search_QA_Agent.md)
**Execution plan:** [`../PHASE_PLAN_Project1.md`](../PHASE_PLAN_Project1.md)

---

## Status

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Foundation & ingestion — config, loaders, cleaner, chunker, pipeline | ✅ **Complete** |
| **Phase 2** | Embeddings + cache, ChromaDB, 60-question golden set, evaluation harness, BM25 + RRF hybrid | ✅ **Complete** |
| **Phase 3** | Cross-encoder re-ranking, Groq grounded generation, citations, calibrated refusal, UI, full ablation | ✅ **Complete** |

### Headline results (measured, 51 golden questions · 109 tests passing)

| # | Configuration | P@3 | R@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|---|
| A0 | BM25 keyword baseline | 0.314 | 0.853 | 0.738 | 0.738 |
| A1 | Dense semantic | 0.366 | 0.971 | 0.898 | 0.886 |
| A4 | Hybrid + RRF | 0.379 | 0.980 | 0.852 | 0.861 |
| **A5** | **Hybrid + cross-encoder re-rank** ← shipped | **0.431** | **1.000** | **0.941** | **0.942** |

**Against the keyword baseline: +37% Precision@3, +27% MRR, +28% nDCG, perfect
Recall@10.** On paraphrased questions keyword search misses entirely more than
half the time; the shipped system finds the answer 92% of the time.

| Answer quality | Target | Measured |
|---|---|---|
| Faithfulness (M7) | ≥ 0.90 | **1.000** |
| Answer relevance (M8) | ≥ 0.88 | **0.909** |
| Citation accuracy (M9) | ≥ 0.95 | **1.000** |
| Refusal correctness (M10) | ≥ 0.90 | **1.000** |
| Hallucination rate (M11) | ≤ 0.05 | **0.000** |

Phase reports: [Phase 1](reports/PHASE1_REPORT.md) ·
[Phase 2](reports/PHASE2_REPORT.md) (includes a **refuted** hypothesis) ·
[Phase 3](reports/PHASE3_REPORT.md)

---

## Quickstart

```bash
pip install -r requirements.txt

# Phase 1 -- ingestion
python -m src.cli ingest            # parse -> clean -> chunk -> chunks.jsonl
python -m src.cli stats             # chunk-set statistics
python -m src.cli inspect --n 3     # eyeball a few chunks for QA

# Phase 2 -- retrieval + measurement
python -m src.cli index             # embed + build dense (Chroma) and BM25 indexes
python -m src.cli query "can I get money back for a cancelled trip?" --compare
python -m src.cli evaluate          # golden set -> reports/eval_report.md + ablation.md

# Phase 3 -- answers, citations, refusal, UI
python -m src.cli calibrate         # calibrate the refusal threshold (0 API calls)
python -m src.cli ask "can I get money back for a cancelled trip?"
python -m src.cli ask "do we allow pets in the office?"   # refuses, 0 API calls
python scripts/failure_analysis.py  # diagnose every remaining failure
streamlit run app.py                # web UI

python -m pytest tests/ -q          # 109 tests
```

**Groq setup.** Put your key in `semantic-qa-agent/.env` (gitignored):
`GROQ_API_KEY=gsk_...`. Everything except `ask` and `judge` runs with **zero API
calls** — including the entire refusal calibration.

`--compare` runs BM25, dense and hybrid side by side on the same query — the
fastest way to see the vocabulary-mismatch problem this project exists to solve.

Runs entirely on CPU. Phase 1 needs no model downloads at all; Phase 2 downloads
one 80 MB embedding model on first use and is fully offline thereafter.

### Tuning without editing code

```bash
python -m src.cli ingest --chunk-size 256 --overlap 32
python -m src.cli ingest --path /some/other/corpus
```

All defaults live in [`config/default.yaml`](config/default.yaml) — PRD
principle #4, *config over code*.

---

## What Phase 1 does

```
data/raw/*.{pdf,docx,md,txt,html,csv}
   │
   ├─ [1] LOADER      per-page extraction; real page numbers, never estimated
   ├─ [2] CLEANER     unicode/ligatures, de-hyphenation, boilerplate, page numbers
   ├─ [3] CHUNKER     recursive split, token-aware, overlapping, heading-enriched
   └─ [4] PIPELINE    SHA-256 dedupe, manifest, unparsed report, run summary
        │
        ▼
   data/processed/chunks.jsonl   ← ready for Phase 2 embedding
   data/manifest.csv             ← auditable per-document record
   reports/unparsed.csv          ← every failure, with its reason
   logs/ingest.jsonl             ← structured events for failure analysis
```

### Design decisions worth knowing

- **Provenance is captured at ingestion or never.** Every chunk records its
  file, page, section heading and exact `char_start`/`char_end` span. Phase 3's
  citation feature is only possible because of this.
- **`text` vs `embed_text`.** Users are shown `text`; the retriever embeds
  `embed_text`, which is prefixed with `[Document > Section]`. A chunk reading
  *"This does not apply to contractors."* is meaningless alone but retrievable
  once its parent heading travels with it.
- **Overlap snaps to word boundaries.** A raw character slice produces fragments
  like `"te. Delays in"` — broken tokens that add noise to the embedding for no
  benefit.
- **Idempotent by content hash.** Re-running ingestion never duplicates a chunk,
  so it is always safe to re-run during development.
- **Failures are data, not exceptions.** An unparseable file is logged with a
  reason and reported; it never stops the run.
- **Config fingerprint.** Every chunk carries a hash of the settings that
  produced it, so any number in any later report is traceable to its exact
  configuration (PRD NFR-8).

---

## Layout

```
config/default.yaml         all tunables, including Phase 2/3 sections
src/config.py               typed load + validation + fingerprint
src/models.py               Chunk / RawDocument / IngestStats
src/utils/logging.py        console + JSONL sinks, table renderer
src/ingest/loaders.py       6 formats, lazy optional deps, graceful failure
src/ingest/cleaner.py       ordered cleaning transforms
src/ingest/chunker.py       recursive splitter, overlap, heading detection
src/ingest/pipeline.py      orchestration, dedupe, manifest, reports
src/embed/embedder.py       bi-encoder + SQLite embedding cache
src/store/base.py           VectorStore ABC · ChromaStore · NumpyStore · BM25Index
src/store/index.py          index build/load, the Phase 1 -> Phase 2 bridge
src/retrieve/retriever.py   dense / sparse / hybrid retrieval + RRF fusion
src/eval/metrics.py         P@k, R@k, MRR, nDCG, MAP -- written from scratch
src/eval/golden.py          span-based gold resolution + graded relevance
src/eval/runner.py          ablation runner + report writers
src/cli.py                  ingest · stats · inspect · index · query · evaluate
data/golden_set.jsonl       60 hand-authored questions, graded relevance
tests/test_phase1.py        34 tests   ·   tests/test_phase2.py  37 tests
reports/PHASE1_REPORT.md    ·  PHASE2_REPORT.md  ·  eval_report.md  ·  ablation.md
```

---

## What Phase 2 adds

```
data/processed/chunks.jsonl
   │
   ├─ [5] EMBEDDER    all-MiniLM-L6-v2, 384-d, L2-normalised, SQLite-cached
   │        ├──────► ChromaDB (HNSW, cosine)   ── dense index
   │        └──────► BM25Okapi (pickled)       ── sparse index
   │
   └─ QUERY ─┬─ dense  top-25 ─┐
             └─ bm25   top-25 ─┴─► RRF fusion (k=60) ─► top-k results
                                          │
                                          ▼
                        golden set (60 Qs) ─► P@k · R@k · MRR · nDCG · Hit@k
                                          ─► reports/eval_report.md + ablation.md
```

### Design decisions worth knowing

- **Gold is labelled by text span, not chunk id.** Chunk ids change whenever
  chunk size changes, which would invalidate the entire golden set on every
  ablation arm. Spans are resolved against whatever chunk set exists, so the
  golden set is authored once and survives every configuration.
- **Graded relevance 0–3**, so nDCG is real rather than a binary proxy. Grade-1
  (same-section context) counts toward nDCG but is excluded from precision.
- **Metrics written from scratch** and unit-tested against hand-computed values
  in the test docstrings — which caught an arithmetic error in my own working.
- **RRF over weighted blending.** Cosine and BM25 scores are on incomparable
  scales; RRF uses ranks only, so it needs no tuning and no normalisation.
- **The golden set validates itself.** A test fails the build if any gold span
  cannot be resolved, because an unresolvable span silently scores zero and
  would depress the report for reasons unrelated to retrieval.

---

## Verified run

```
$ python -m src.cli ingest
15 files ingested · 1 skipped (unsupported) · 0 failed · 48 chunks · 48 with headings

$ python -m src.cli index
48 chunks · dim 384 · backend ChromaStore · bm25 rank_bm25.BM25Okapi

$ python -m src.cli evaluate
60 questions · 51 answerable · 51 resolved · 0 warnings
A0 BM25    P@3 0.314  R@10 0.853  MRR 0.738  nDCG 0.738
A1 Dense   P@3 0.366  R@10 0.971  MRR 0.898  nDCG 0.886
A4 Hybrid  P@3 0.379  R@10 0.980  MRR 0.852  nDCG 0.861

$ python -m pytest tests/ -q
71 passed in 0.85s
```

Full detail: [`reports/PHASE1_REPORT.md`](reports/PHASE1_REPORT.md) ·
[`reports/PHASE2_REPORT.md`](reports/PHASE2_REPORT.md)
