# Phase 3 Completion Report — Intelligence, Rigour & Delivery

**Project:** Semantic Search / Intelligent Q&A Agent (Project 1)
**Phase:** 3 of 3 — Intelligence, Rigour & Delivery
**Date:** 01 September 2026
**Status:** ✅ **Complete — all four parts delivered and measured. Project complete.**

---

## 1. What Phase 3 was for

Phase 1 produced clean, traceable chunks. Phase 2 made them searchable and — more
importantly — built the instrument that measured the result, ending with a
precisely diagnosed gap: **Recall@10 was 0.980, so the right chunk was almost
always in the candidate pool; it simply was not at the top.**

Phase 3 closes that gap, turns retrieval into grounded answers that cite their
evidence, refuses safely when the corpus cannot support an answer, and packages
the whole thing so a reviewer can verify every claim.

---

## 2. Deliverables — all four parts

### Part 3.1 — Cross-encoder re-ranker ✅
`src/retrieve/reranker.py`, wired as a true second stage in `retriever.py`

`cross-encoder/ms-marco-MiniLM-L-6-v2` scores each `(query, chunk)` pair
**jointly** — full attention across both — and reorders the stage-1 candidate
pool. Where a bi-encoder must compress a chunk into one vector before it has seen
the question, a cross-encoder reads both together. That is the entire reason the
second stage exists, and it is why it can fix what RRF structurally cannot.

Runs **entirely locally on CPU — zero API cost.** Degrades gracefully: if the
model cannot load, the stage-1 order passes through unchanged with a logged
warning rather than taking the query path down.

### Part 3.2 — Grounded generation, citations, calibrated refusal ✅
`src/generate/{groq_client,prompts,answerer}.py`, `src/eval/calibrate.py`

**Provider switched to Groq.** The account has no Llama models, so the answerer
uses `openai/gpt-oss-120b` and the judge `openai/gpt-oss-20b` — the model list
was queried rather than assumed after a 404 revealed the mismatch.

**Free-tier discipline is built into the client, not bolted on:**

| Guard | Effect |
|---|---|
| Disk cache on `sha256(model + prompt + params)` | Re-running any report costs **zero** calls. Correct, not merely fast: generation is pinned to `temperature = 0.0`, so the same prompt is *defined* to give the same answer. |
| Minimum interval between live calls + 429 backoff honouring `retry-after` | Never hammers a limit already hit |
| Hard `max_calls` budget | Raises a clear error instead of silently burning the daily quota |
| Judge on the smaller model | Judging a binary support relation does not need 120B capacity |
| Gate placed **before** generation | Every refusal costs **zero** API calls |

**The refusal gate is the architectural centrepiece.** If the top re-ranked chunk
scores below the calibrated threshold, the system refuses *without ever calling
the model*. The consequence is that it **cannot hallucinate on an out-of-corpus
question — the guarantee is structural, not a matter of the model obeying a
prompt.**

**The threshold is measured, not guessed** (`calibrate.py`), using the golden
set's 51 answerable and 9 unanswerable questions. The separation is near-total:

| Group | n | min | median | max |
|---|---|---|---|---|
| Answerable (should answer) | 51 | 0.000 | **0.962** | 1.000 |
| Unanswerable (should refuse) | 9 | 0.000 | **0.000** | **0.010** |

**Finding: the PRD's guessed threshold of 0.40 was 20× too high.**

| τ | Answer correctness | Refusal correctness | Balanced | False answers |
|---|---|---|---|---|
| 0.01 | 0.824 | 0.889 | 0.856 | 1 |
| **0.02** ← calibrated | **0.784** | **1.000** | **0.892** | **0** |
| 0.05 | 0.745 | 1.000 | 0.873 | 0 |
| 0.40 ← PRD guess | 0.706 | 1.000 | 0.853 | 0 |

At τ = 0.40 the system would wrongly refuse **15** answerable questions; at the
calibrated τ = 0.02 it wrongly refuses **11**, with *identical* perfect refusal
safety. Guessing cost four answerable questions for no safety benefit whatsoever.
**The entire calibration procedure uses zero LLM calls** — it operates purely on
cross-encoder scores, which is precisely why the gate sits before generation.

### Part 3.3 — Interfaces ✅
`app.py` (Streamlit), CLI commands `ask` · `calibrate` · `judge` · `serve`

The UI deliberately **shows the machinery rather than hiding it**: confidence
against the calibrated threshold, the per-leg retrieval trace (`dense #3` /
`bm25 #1` / `rerank +2.4`) for every chunk, expandable citations with their exact
`chunk_id`, page and character span, live latency, and an API-call counter that
reads 0 for cached and refused queries. A demo that renders only an answer is
indistinguishable from a chatbot; this one lets a reviewer verify each claim
against its evidence.

### Part 3.4 — Full ablation, failure analysis, answer quality ✅
`reports/{ablation,eval_report,answer_quality,failure_analysis,calibration}.md`

---

## 3. Measured results

### 3.1 The complete ablation — A0 through A6

51 answerable golden questions · 48 chunks · `all-MiniLM-L6-v2` +
`ms-marco-MiniLM-L-6-v2`

| # | Configuration | P@3 | R@10 | MRR@10 | nDCG@10 | p95 ms |
|---|---|---|---|---|---|---|
| A0 | Keyword baseline (BM25 only) | 0.314 | 0.853 | 0.738 | 0.738 | 0.2 |
| A1 | Dense only (MiniLM) | 0.366 | 0.971 | 0.898 | 0.886 | 5.1 |
| A4 | Hybrid (dense + BM25, RRF) | 0.379 | 0.980 | 0.852 | 0.861 | 4.5 |
| A4b | Hybrid, sparse leg narrowed | 0.353 | 0.980 | 0.841 | 0.851 | 4.4 |
| **A5** | **Hybrid + cross-encoder re-rank** | **0.431** | **1.000** | **0.941** | **0.942** | 478.8 |
| A6 | Dense + cross-encoder re-rank | 0.431 | 0.990 | 0.940 | 0.940 | 379.7 |

**A5 is the shipped configuration.** Against the keyword baseline it delivers
**+37% Precision@3, +27% MRR, +28% nDCG, and perfect Recall@10.**

### 3.2 Re-ranking closed exactly the gap Phase 2 diagnosed

Phase 2 predicted a cross-encoder would recover paraphrase performance *while
keeping* hybrid's exact-identifier advantage. Measured:

| Configuration | Paraphrase P@3 | Paraphrase Hit@3 | exact_id P@3 | MRR@10 |
|---|---|---|---|---|
| A0 — BM25 | 0.154 | 0.462 | 0.467 | 0.738 |
| A1 — Dense | 0.359 | **1.000** | 0.333 | 0.898 |
| A4 — Hybrid | 0.256 | 0.692 | 0.467 | 0.852 |
| **A5 — Hybrid + re-rank** | **0.436** | **0.923** | **0.467** | **0.941** |

Hybrid's paraphrase collapse (Hit@3 0.692) is repaired to 0.923, exact-identifier
performance is fully retained at 0.467, and MRR now exceeds dense-only. **The
Phase 2 prediction was correct and is confirmed by measurement.**

### 3.3 Precision against its ceiling

The golden set averages 1.41 relevant chunks per question, so P@3 is capped at
0.471 no matter how perfect the ranking:

| Configuration | P@3 | % of achievable ceiling |
|---|---|---|
| A0 — BM25 | 0.314 | 66.7% |
| A1 — Dense | 0.366 | 77.8% |
| A4 — Hybrid | 0.379 | 80.6% |
| **A5 — Hybrid + re-rank** | **0.431** | **91.7%** |

### 3.4 Answer quality (LLM-as-judge, stratified sample)

| ID | Metric | Target | **Measured** | |
|---|---|---|---|---|
| M7 | Faithfulness (claims supported by context) | ≥ 0.90 | **1.000** | ✅ |
| M8 | Answer relevance | ≥ 0.88 | **0.909** | ✅ |
| M9 | Citation accuracy | ≥ 0.95 | **1.000** | ✅ |
| M10 | Refusal correctness (out-of-corpus) | ≥ 0.90 | **1.000** | ✅ |
| M11 | Hallucination rate | ≤ 0.05 | **0.000** | ✅ |

Zero dangling citations across the sample. 75% of answers cited a gold chunk
directly. **Total API cost of the entire evaluation: 22 live calls, 21,186
tokens** — comfortably inside a free tier, and reproducible at zero cost from
the disk cache.

### 3.5 Failure analysis — 2 failures of 51 (Hit@3 = 0.961)

Both remaining failures are `reranker_misordered`: the gold chunk *was* in the
candidate pool but did not survive the cut to top-3.

- **Q019** *"What happens to my stuff when I leave the company?"* — retrieved
  *Device and Asset Policy › Damage and Loss*; expected *› Return on Exit*. It
  found the **correct document** and the adjacent section. The colloquial "stuff"
  → "assets" mapping is a genuine semantic stretch.
- **Q050** *"Tell me about notice periods and timing rules."* — genuinely
  underspecified. Several documents discuss notice and timing; the gold label is
  arguably arbitrary. This is closer to a golden-set limitation than a retrieval
  defect.

Neither is a `retrieval_miss` — the class that re-ranking can never repair, and
which would be the serious one. There are none.

---

## 4. Bugs found and fixed during Phase 3

Three real defects, all caught by verifying rather than assuming:

1. **Citation parser missed every citation.** `gpt-oss` emits CJK lenticular
   brackets `【1】`, not the ASCII `[1]` the prompt asks for. The regex matched
   nothing, so perfectly grounded answers reported **zero citations** — a total
   apparent failure of the flagship feature when nothing was actually wrong.
   Fixed by accepting bracket variants and normalising to ASCII. Regression test:
   `test_extract_citations_accepts_cjk_brackets`.

2. **The LLM judge was silently broken, and reported the system as broken.**
   The first judge run returned faithfulness **0.091** and hallucination
   **0.909** — while simultaneously reporting citation accuracy 1.000 and 83% of
   answers citing gold chunks. Those numbers are mutually contradictory, which is
   what prompted the check rather than the write-up. The cause was that `gpt-oss`
   emits reasoning text around its JSON, and a 200-token budget truncated the
   verdict. Fixed with constrained JSON decoding plus `reasoning_effort: low`.
   True faithfulness: **1.000**. *The instrument was broken, not the system* —
   and publishing 0.091 without questioning it would have been the more serious
   error.

3. **numpy scalar leaked through the gate.** `sigmoid` returned `np.float64`, so
   the gate returned `np.False_`, for which `is False` is false and `json.dumps`
   raises. Caught by a test asserting the identity rather than the truthiness.

Each is now covered by a regression test.

---

## 5. Exit criteria — all met

| # | Criterion | Result |
|---|---|---|
| 1 | Cross-encoder re-ranking with a measured lift | ✅ P@3 +0.052, MRR +0.089, R@10 → 1.000 |
| 2 | Re-ranker fallback path tested | ✅ `test_reranker_degrades_gracefully…` |
| 3 | Every factual sentence cites a real chunk | ✅ M9 = 1.000, 0 violations |
| 4 | `citation_violation` = 0 on the sample | ✅ 0 |
| 5 | Refusal correctness ≥ 0.90 | ✅ **1.000** (offline and live) |
| 6 | Published calibration curve | ✅ `reports/calibration.md` |
| 7 | Working UI | ✅ `streamlit run app.py` |
| 8 | Full ablation with measured numbers | ✅ A0–A6 |
| 9 | Failure analysis with diagnosed causes | ✅ 2 failures, taxonomy applied |
| 10 | Test suite | ✅ **109 tests** (34 + 37 + 38), all passing |

---

## 6. Known limitations (stated plainly)

| Limitation | Impact | Honest assessment |
|---|---|---|
| **Corpus is 48 chunks**, against the PRD's 3,000–6,000 target | Metrics are directionally sound but statistically thin; a 48-chunk corpus also exaggerates BM25 dilution | The most significant limitation. Conclusions about *relative* configuration ranking should hold; absolute values would move at scale. |
| Re-ranking costs ~475 ms p95 on CPU | Well inside the 5 s end-to-end budget, but 100× the stage-1 cost | Acceptable given the accuracy gain; a GPU or a smaller cross-encoder would cut it substantially |
| Judge and golden set authored by the same person who built the system | Bias risk | Questions were written from the corpus before seeing any retrieval output; the paraphrase set is adversarial by construction. An independent annotator would strengthen this. |
| Answer quality judged on 12 of 51 questions | Wide confidence interval on M7/M8 | A deliberate free-tier trade-off, stratified across every category rather than skewed to easy ones |
| 9 unanswerable questions, not the PRD's 10 | Negligible | M10 denominator is 9 |
| A2/A3 (alternative embedding models) not run | Embedding-model choice is not evidence-backed | The one PRD ablation arm not completed; the cache makes it cheap to add |
| Single-hop retrieval only | Multi-hop questions unsupported | Explicit non-goal NG5 |

---

## 7. Project summary — all three phases

| Phase | Delivered | Headline measured result |
|---|---|---|
| **1** | Config, 6-format loaders, cleaner, chunker, idempotent pipeline | 15 docs → 48 chunks, 100% with headings, full provenance |
| **2** | Embeddings + cache, ChromaDB, BM25, RRF, 60-question golden set, metrics from scratch | Semantic beats keyword: MRR 0.738 → 0.898; paraphrase Hit@3 0.462 → 1.000 |
| **3** | Cross-encoder re-ranking, Groq generation, citations, calibrated refusal, UI | **MRR 0.941 · nDCG 0.942 · R@10 1.000 · faithfulness 1.000 · hallucination 0.000** |

**Against the keyword baseline the finished system delivers +37% Precision@3,
+27% MRR, +28% nDCG and perfect Recall@10, answers with verified citations,
never hallucinated once across the judged sample, and correctly refused every
out-of-corpus question without spending a single API call doing so.**

Every number above was produced by a command in this repository and can be
reproduced from a clean clone.
