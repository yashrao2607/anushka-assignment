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
| Phase 2 | Embeddings, ChromaDB, golden set, evaluation harness, BM25 hybrid + RRF | ⬜ Next |
| Phase 3 | Cross-encoder re-ranking, grounded generation, refusal gate, UI, ablations | ⬜ |

---

## Quickstart

```bash
pip install -r requirements.txt

python -m src.cli ingest            # parse -> clean -> chunk -> chunks.jsonl
python -m src.cli stats             # chunk-set statistics
python -m src.cli inspect --n 3     # eyeball a few chunks for QA
python -m pytest tests/ -q          # 34 tests
```

Everything in Phase 1 runs on CPU with no model downloads and no network access.

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
src/cli.py                  ingest · stats · inspect (query/evaluate/serve stubbed)
tests/test_phase1.py        34 tests across all four Phase 1 parts
reports/PHASE1_REPORT.md    verified Phase 1 completion record
```

---

## Verified Phase 1 run

```
INGESTION SUMMARY                      CHUNK QUALITY
Files found                     6      Chunks                        5
Files ingested                  5      Tokens: median              237
Files skipped (unsupported)     1      Chunks with a heading         5
Files failed (parse error)      0      Config fingerprint 6b87171666a8
Chunks produced                 5
Duration (s)                 0.01      34 tests passed
```

Full detail in [`reports/PHASE1_REPORT.md`](reports/PHASE1_REPORT.md).
