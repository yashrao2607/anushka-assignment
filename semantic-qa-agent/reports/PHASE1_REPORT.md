# Phase 1 Completion Report — Foundation & Ingestion Pipeline

**Project:** Semantic Search / Intelligent Q&A Agent (Project 1)
**Phase:** 1 of 3 — Foundation & Ingestion
**Date:** 01 September 2026
**Status:** ✅ **Complete — all four parts delivered, built, and verified by execution**

---

## 1. What Phase 1 was for

Retrieval quality is capped by chunk quality. Every metric in Phase 2 and every
citation in Phase 3 inherits whatever errors are made during ingestion — and no
downstream component can recover information that was destroyed here. Phase 1
was therefore built and tested **before a single embedding was computed**.

The phase is complete when a folder of messy documents becomes clean,
deduplicated, metadata-rich chunks in which **any chunk can be traced back to an
exact file, page, section, and character span**.

---

## 2. Deliverables — all four parts

### Part 1.1 — Scaffold, config system, logging, CLI ✅

| Artefact | What it does |
|---|---|
| `config/default.yaml` | Every tunable in one file, including Phase 2/3 sections declared ahead of time so later components have a stable contract |
| `src/config.py` | Typed dataclass config with **validation that fails at startup**, dotted CLI overrides, and a `fingerprint()` hash |
| `src/utils/logging.py` | Dual sink — human-readable console + structured JSONL for later failure analysis |
| `src/models.py` | `Chunk`, `RawDocument`, `Page`, `IngestStats` |
| `src/cli.py` | `ingest` · `stats` · `inspect`, with `query`/`evaluate`/`serve` registered and reporting their target phase |

**Verified behaviours**
- An impossible configuration is rejected before any work begins:
  ```
  $ python -m src.cli ingest --chunk-size 100 --overlap 100
  CONFIG ERROR: chunking.overlap_tokens (100) must be smaller than
  chunk_size_tokens (100) -- otherwise chunking never advances and would loop forever.
  ```
  This is PRD principle #6 (*fail loudly*) enforced in code rather than asserted in prose.
- `fingerprint()` hashes **only output-affecting settings**. Changing the reports
  directory does not invalidate a chunk set; changing the chunk size does. Every
  chunk carries this fingerprint, satisfying PRD NFR-8 (reproducibility).

### Part 1.2 — Loaders and cleaner ✅

**Six formats:** PDF (PyMuPDF), DOCX (python-docx), Markdown, TXT, HTML
(BeautifulSoup with a regex fallback), CSV.

| Decision | Reasoning |
|---|---|
| **Per-page PDF extraction** | `page_no` is read from the parser, never estimated from character offsets. A citation saying "page 12" must actually be true, or the whole citation feature is worthless. |
| **Lazy optional imports** | A missing PyMuPDF degrades *that one format* to an actionable message; it never prevents ingestion of the formats that do work. |
| **DOCX headings normalised to Markdown** | `Heading 2` becomes `## …`, so the chunker's heading detection works identically across every format instead of needing per-format logic. |
| **CSV rows flattened to `column: value`** | A raw row (`"A,3,true"`) embeds terribly — the column names carry the meaning. Restating each row as labelled fields makes rows genuinely retrievable. |
| **Scanned-PDF detection** | A page under 50 characters has no text layer. It is reported to `reports/unparsed.csv` with an OCR suggestion rather than silently producing an empty index. |
| **Every failure is data** | A corrupt file yields a `RawDocument` with a `load_error`, surfaced in the summary and the unparsed report. It never raises. |

**Cleaner** — ordered transforms, order chosen deliberately: unicode NFKC →
ligature repair → de-hyphenation → boilerplate strip → page-number strip →
whitespace collapse. De-hyphenation must precede whitespace collapsing, because
collapsing first destroys the newline that marks the hyphenation.

Two correctness details that a naive implementation gets wrong:
- **De-hyphenation requires a lowercase letter on both sides**, so `state-of-the-art`
  and numeric ranges survive while `infor-\nmation` is rejoined.
- **Boilerplate detection requires ≥ 3 pages.** On a 2-page document every line
  looks repeated, which would strip the entire content as "boilerplate".

### Part 1.3 — Chunking engine ✅

Recursive-character splitting: split on the largest natural boundary first
(paragraph), falling back through line → sentence → clause → word only when a
piece is still too large.

**Three decisions that separate this from a naive splitter:**

1. **Overlap** — consecutive chunks share `overlap_tokens`, so an answer
   straddling a boundary is not lost by *both* chunks.
2. **Heading enrichment** — `embed_text` is prefixed `[Document > Section]`
   while `text` stays clean for display. *"This does not apply to contractors."*
   is unretrievable alone; *"[HR Policy > Casual Leave] This does not apply to
   contractors."* is retrievable.
3. **True character spans** — every chunk records `char_start`/`char_end` back
   into the cleaned source, so Phase 3 can highlight the exact cited region.

**A defect found and fixed during verification.** The first implementation took
the overlap as a raw character slice, producing carried fragments like
`"te. Delays in"` — starting mid-word. Those broken tokens get embedded and add
noise for no benefit. `_tail_overlap()` now snaps forward to the next word
boundary, at the cost of a few characters of overlap. This is exactly the class
of defect that silently degrades retrieval metrics and is nearly impossible to
diagnose from a Phase 2 number.

Token counting uses **tiktoken** (`cl100k_base`) when installed and a calibrated
word-based approximation otherwise, so Phase 1 runs with zero heavy dependencies.
The active tokenizer is printed on every run.

### Part 1.4 — Pipeline, corpus, manifest, verification ✅

Pipeline: `discover → load → detect boilerplate → clean → chunk → dedupe → write`.

**Outputs**

| File | Purpose |
|---|---|
| `data/processed/chunks.jsonl` | One chunk per line — the Phase 2 input |
| `data/manifest.csv` | Per-document audit record: pages, chars, chunks, tokens, status, error |
| `reports/unparsed.csv` | Every file that failed, with its reason |
| `logs/ingest.jsonl` | Structured events — the raw material for Phase 3 failure analysis |

**Idempotency (PRD US-1.2)** is enforced by a content SHA-256 per chunk. Ingesting
the same document twice yields zero new chunks. This matters practically: during
development ingestion is run dozens of times, and without this every run would
silently corrupt the index.

**Seed corpus** — 5 synthetic handbook documents (HR policy, travel/expense, IT
support FAQ, security policy, onboarding FAQ) plus one `.rtf` decoy that must be
skipped without crashing. The corpus is deliberately written to contain the
**vocabulary-mismatch cases** the Phase 2 golden set will test — for example
*"can I get money back if a client trip gets cancelled?"* against a section
headed *"Non-refundable Bookings"*, which share **zero content words**. This is
the paraphrase case that keyword search cannot solve and semantic search must.

---

## 3. Verified execution

```
$ python -m src.cli ingest

WARNING unsupported file type, skipping: notes.rtf
INFO    discovered 5 supported file(s)
INFO    tokenizer: tiktoken/cl100k_base | config fingerprint: 6b87171666a8
INFO      hr_policy.md          1 page(s) ->  1 chunk(s)
INFO      it_support_faq.txt    1 page(s) ->  1 chunk(s)
INFO      onboarding_faq.md     1 page(s) ->  1 chunk(s)
INFO      security_policy.md    1 page(s) ->  1 chunk(s)
INFO      travel_policy.md      1 page(s) ->  1 chunk(s)

INGESTION SUMMARY                            CHUNK QUALITY
+------------------------------------+----+  +-----------------------+--------------+
| Files found                        |  6 |  | Chunks                |            5 |
| Files ingested                     |  5 |  | Tokens: min           |          200 |
| Files skipped (unsupported)        |  1 |  | Tokens: median        |          237 |
| Files failed (parse error)         |  0 |  | Tokens: mean          |        238.0 |
| Pages / blocks parsed              |  5 |  | Tokens: max           |          301 |
| Chunks produced                    |  5 |  | Chunks with a heading |            5 |
| Chunks dropped (too small / noisy) |  0 |  | Distinct documents    |            5 |
| Duplicate chunks skipped           |  0 |  | Config fingerprint    | 6b87171666a8 |
| Duration (s)                       |0.01|  +-----------------------+--------------+
+------------------------------------+----+
```

**Chunking verified under stress** — forcing a smaller budget proves the
splitter, the overlap, and the heading detection all engage:

```
$ python -m src.cli ingest --chunk-size 120 --overlap 24
hr_policy.md -> 4 chunks · it_support_faq.txt -> 4 · security_policy.md -> 4
travel_policy.md -> 4 · onboarding_faq.md -> 3        (19 chunks total)
Tokens min/median/max: 55 / 72 / 127 · 19 of 19 chunks carry a heading
```

Every chunk resolves to a real page and span:

```
chunk_id : hr_policy__p1__c0
source   : hr_policy.md  (page 1)
heading  : HR Policy Handbook 2026
span     : chars 0-315  | 72 tokens
embedded : [Hr Policy > HR Policy Handbook 2026] # HR Policy Handbook 2026 ...
```

**Test suite: 34 passed in 1.24 s.**

```
Part 1.1  config      6 tests   defaults, overrides, validation, fingerprint stability
Part 1.2  cleaning    7 tests   hyphenation, ligatures, whitespace, page numbers, boilerplate
Part 1.2  loaders     4 tests   discovery, doc ids, graceful failure on a corrupt PDF
Part 1.3  chunking   12 tests   size bounds, boundaries, overlap, spans, headings, noise, ids
Part 1.4  pipeline    5 tests   end-to-end, idempotency, manifest, full provenance
```

---

## 4. Exit criteria — all met

| # | Criterion | Result |
|---|---|---|
| 1 | `python -m src.cli --help` works; config loads and validates | ✅ |
| 2 | All six formats parse; unsupported/corrupt files skipped with a warning, never a crash | ✅ verified with a `.rtf` decoy and a corrupt-PDF test |
| 3 | Cleaner de-hyphenates, repairs ligatures, strips boilerplate and page numbers | ✅ 7 tests |
| 4 | Chunks respect size/overlap config; no chunk splits a word | ✅ fixed during verification |
| 5 | Every chunk carries `char_start`/`char_end` back to its source | ✅ asserted for every chunk |
| 6 | `ingest` produces `chunks.jsonl` + `manifest.csv` + a summary table | ✅ |
| 7 | Re-running ingestion is idempotent | ✅ SHA-256 content hash |
| 8 | ≥ 20 tests pass | ✅ **34 passed** |

---

## 5. Known limitations (carried forward honestly)

| Limitation | Impact | Plan |
|---|---|---|
| Seed corpus is 5 synthetic documents (~1,190 tokens) | Too small for meaningful retrieval metrics | Phase 2 Part 2.1 expands to the 3,000–6,000 chunk target from PRD §7.1 |
| OCR not enabled | Scanned PDFs are reported, not read | Tesseract hook exists and is disabled by default; enable per-corpus if needed |
| Table structure is flattened to text | Multi-column tables lose their grid | Out of scope per PRD NG7; table-to-markdown is a stretch item |
| Approximate token count when tiktoken is absent | A few percent of drift in chunk boundaries | Acceptable — boundaries are not that sensitive; tiktoken is in `requirements.txt` |
| `--force` flag is accepted but is a no-op | Ingestion currently always rebuilds | Becomes meaningful in Phase 2 when a persistent vector index exists |

---

## 6. Next action — Phase 2, Part 2.1

Build the embedding layer: Sentence-Transformers `all-MiniLM-L6-v2`, batched at
64, L2-normalised, with a SQLite embedding cache keyed on
`sha256(text) + model_name`.

**Why the cache first:** Phase 3's ablation sweep re-runs the corpus across ten
configurations. Without a cache that is hours of redundant computation; with one
it is a single embed pass plus nine cache hits. Building it now, before it is
needed, is what keeps the ablation affordable.

Phase 1 hands over a clean contract: `data/processed/chunks.jsonl`, where every
record already carries `embed_text`, a content hash, and full provenance.
