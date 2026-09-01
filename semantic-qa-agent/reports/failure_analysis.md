# Failure Analysis

Every golden question the best configuration (**A5 — hybrid + cross-encoder
re-ranking**) still fails at k=3, with a diagnosed root cause.

Honest failure reporting is a deliverable, not an embarrassment: a system
with an unexamined 100% is far less trustworthy than one that says exactly
where and why it breaks. Critically, the taxonomy separates defects in the
**system** from defects in the **golden set** — a distinction that changes
what you should do about them.

## Summary

- Questions evaluated: **51**
- Failures at k=3: **2**
- Hit Rate@3: **0.961**

| Root cause | n | Meaning |
|---|---|---|
| `reranker_misordered` | 2 | The gold chunk was in the candidate pool but the cross-encoder ranked it below the cut. Fix: a larger re-ranker, or a deeper final k. |

## Individual failures

### `Q019` — reranker_misordered

**Question** (paraphrase): *What happens to my stuff when I leave the company?*

- Retrieved: Device And Asset Policy > Damage and Loss; Security Policy > Incident Reporting; Hr Policy > 1. Leave Entitlement
- Expected: Device And Asset Policy > Return on Exit
- Gold chunk was in the stage-1 candidate pool: **yes**
- Top score: -6.487

### `Q050` — reranker_misordered

**Question** (ambiguous): *Tell me about notice periods and timing rules.*

- Retrieved: Customer Support Sla > Escalation Path; Customer Support Sla > Refunds and Service Credits; Payroll Policy > Provident Fund and Deductions
- Expected: Hr Policy > 3. Parental Leave; Payroll Policy > Variable Pay and Bonus
- Gold chunk was in the stage-1 candidate pool: **yes**
- Top score: -7.334

## What this points at next

- `gold_too_narrow` dominating means the *golden set* is the limiting
  factor, not retrieval — the fix is to broaden gold labelling, and the
  reported metrics are a pessimistic lower bound on true performance.
- `retrieval_miss` is the only class re-ranking can never repair, since
  the chunk never enters the pool. It is the first thing to attack.
- `chunking_split_the_answer` is fixed at ingestion (larger chunks or
  neighbour expansion), not in the retriever.
