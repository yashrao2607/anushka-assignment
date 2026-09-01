# Answer Quality Report (LLM-as-judge)

Answer model: `openai/gpt-oss-120b` (Groq) · Judge model: `openai/gpt-oss-20b` · Refusal threshold: 0.02

Evaluated on a **stratified sample spanning every question category**, not
the first n questions -- a sample skewed toward easy `direct` lookups would
flatter the system. Sample size is kept small and every response is disk-
cached, so this report is reproducible at zero additional API cost.

## Metrics

| ID | Metric | Target | Measured |
|---|---|---|---|
| M7 | Faithfulness (claims supported by context) | ≥ 0.90 | **1.0** |
| M8 | Answer relevance | ≥ 0.88 | **0.909** |
| M9 | Citation accuracy | ≥ 0.95 | **1.0** |
| M10 | Refusal correctness (out-of-corpus) | ≥ 0.90 | **1.0** |
| M11 | Hallucination rate | ≤ 0.05 | **0.0** |

| Diagnostic | Value |
|---|---|
| Questions answered | 12 |
| Judged by LLM | 11 |
| Refused on an *answerable* question | 1 |
| Answers citing a gold chunk | 0.75 |
| Dangling citations (`citation_violation`) | 0 |

## API usage

- Live calls: **22** · cache hits: 0
- Tokens: 17731 prompt + 3455 completion = 21186
- Cached responses on disk: 44

Refusals are gated *before* generation, so every correct refusal costs zero
API calls and cannot hallucinate by construction.

## Live refusal checks (out-of-corpus questions)

| qid | question | refused | confidence |
|---|---|---|---|
| Q052 | What is the company policy on cryptocurrency payments to emp | ✅ | 0.0003 |
| Q053 | How many vacation days do employees at our Berlin office rec | ✅ | 0.0104 |
| Q054 | What is the maximum allowable radiation dose in the laborato | ✅ | 0.0000 |

## Per-question detail

| qid | category | refused | cites | cited gold | faithful | relevant |
|---|---|---|---|---|---|---|
| Q047 | ambiguous | no | 0 | no | 1 | 0 |
| Q001 | direct | no | 1 | yes | 1 | 1 |
| Q042 | exact_id | no | 0 | no | 1 | 1 |
| Q029 | multi_chunk | no | 1 | yes | 1 | 1 |
| Q037 | numeric | no | 1 | yes | 1 | 1 |
| Q016 | paraphrase | no | 1 | yes | 1 | 1 |
| Q048 | ambiguous | no | 3 | yes | 1 | 1 |
| Q002 | direct | no | 1 | yes | 1 | 1 |
| Q043 | exact_id | no | 1 | yes | 1 | 1 |
| Q030 | multi_chunk | no | 2 | yes | 1 | 1 |
| Q038 | numeric | no | 1 | yes | 1 | 1 |
| Q017 | paraphrase | yes | 0 | no | None | None |
