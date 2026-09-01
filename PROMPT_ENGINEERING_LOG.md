# Prompt Engineering Log & Methodology
## Building a Production-Grade Semantic Search / Q&A Agent with an AI Coding Agent

---

| Field | Value |
|---|---|
| **Author** | Project Engineer |
| **Date** | 01 September 2026 |
| **Project** | Project 1 — Semantic Search / Intelligent Q&A Agent |
| **Tooling** | Claude Code (agentic coding assistant) |
| **Elapsed time** | Single working session |
| **Output** | 3-phase system · 113 passing tests · 12 generated reports · full PRD |
| **Purpose of this document** | Record the *direction strategy* — not just the code — so the approach is auditable and repeatable |

---

## 0. Why this document exists

Anyone can ask an AI to write code. The differentiator is **whether you can direct
it to a verifiable, defensible engineering outcome** — and whether you can explain
*why* you steered the way you did.

This log records the actual prompts used to build the project, the intent behind
each, the technique applied, and the measurable outcome. It also records the
places where my direction **corrected the agent's drift** — those are the moments
that mattered most, and they are included honestly rather than edited out.

Section 8 distils the whole thing into a **reusable prompt template library** for
the next project.

---

## 1. Executive summary — the direction strategy

The project was directed on five principles, applied consistently from the first
message to the last:

| # | Principle | How it showed up |
|---|---|---|
| **P1** | **Specification before implementation** | Demanded a complete PRD before a line of code existed. The PRD then became the contract every later phase was measured against. |
| **P2** | **Decompose into verifiable milestones** | Imposed a 3-phase × 4-part structure so each unit had its own exit criterion, rather than accepting one undifferentiated code dump. |
| **P3** | **Protect scope aggressively** | Cut a second project mid-stream the moment it threatened focus. Scope discipline was enforced by me, not left to the agent. |
| **P4** | **State real-world constraints up front** | Disclosed the free-tier API limit *before* it caused a failure, which changed the architecture rather than patching it afterwards. |
| **P5** | **Trust nothing that has not been run** | Required the system be launched and driven — not merely compiled and described. This is what surfaced three real bugs. |

**Result:** every claim in the final deliverable is backed by a command that can
be re-run from a clean clone.

---

## 2. The prompt log

Each entry: what I asked → why → the technique → what it produced.

---

### Prompt 1 — Establish the specification and the bar

> *"See [project brief image] and create a highly detailed PRD for both projects
> as 2 markdown files. This task was given by my manager — he is testing me, so I
> have to make it as strong as I can and as fast as I can. This task was also given
> to 7–8 other people, so I need to stand out. Go ahead."*

**Intent.** Force a specification-first workflow, and set the quality bar
explicitly rather than hoping for it.

**Techniques applied**
- **Source-of-truth input** — supplied the original brief as an image instead of
  paraphrasing it, eliminating requirement drift at step one.
- **Explicit success criteria** — "stand out from 7–8 others" is a *differentiation*
  instruction, not a politeness. It shifts the agent from "satisfy the brief" to
  "exceed it on named axes."
- **Dual constraint (quality × speed)** — stating both prevents the agent from
  silently optimising one at the other's expense.
- **Named output contract** — "2 markdown files" removes ambiguity about format.

**Outcome.** Two PRDs, each with numbered requirements (`FR-x`, `NFR-x`, `M-x`),
success metrics with targets, an ablation plan, and a risk register — a document
set that later phases could be measured against rather than a vague brief.

**Why this mattered:** every subsequent phase report cites PRD section numbers.
Without the spec, "done" would have been a matter of opinion.

---

### Prompt 2 — Control the operating environment

> *"Keep this chat in the terminal only — don't push it to my phone or desktop app."*

**Intent.** Constrain the agent's side effects.

**Technique.** **Environment scoping.** Good direction covers not only *what* is
built but *what the agent is permitted to do*. Stated once, up front, rather than
after an unwanted notification.

**Outcome.** Zero unwanted external notifications for the whole session.

---

### Prompt 3 — Impose the decomposition *(the highest-leverage instruction of the project)*

> *"Work on Phase 1 only. Break this into 3 phases for better execution, and break
> each phase into 4 parts. Now complete Phase 1, as fast and as correctly as you can."*

**Intent.** Convert an amorphous 14-day build into discrete, independently
verifiable units.

**Techniques applied**
- **Hierarchical decomposition (3 × 4)** — coarse enough to be strategic, fine
  enough that progress is visible daily rather than weekly.
- **Work-in-progress limit** — "Phase 1 only" is a WIP cap. It stops the agent
  half-finishing three phases and leaving nothing demonstrable.
- **Restated dual constraint** — speed *and* correctness, again together.

**Outcome.** A phase plan where **every phase ends in something shippable**. If
work had stopped at any phase boundary, there would still have been a working
demo — which is exactly the property a manager reviewing partial progress needs.

**Why this mattered most.** This single instruction is why the project has three
independent completion reports with measured exit criteria, instead of one large
untested codebase. It is also why Phase 2 built its *measuring instrument before*
the thing being measured — a sequencing decision that later produced the project's
most interesting scientific finding (§4).

---

### Prompt 4 — Correct scope drift, immediately and unambiguously

> *"Don't work on Project 2. We're working on Project 1 only — break that into 3
> phases and proceed with the first. That's what I'm saying."*

**Intent.** Kill work that had begun drifting outside the agreed scope.

**Techniques applied**
- **Negative constraint** — stating what *not* to do is often sharper than
  restating what to do.
- **Immediate correction** — issued the moment drift appeared, before effort
  compounded.
- **Restated the correct target** — a good correction does not just stop the wrong
  work; it re-points at the right work in the same breath.

**Outcome.** Effort re-concentrated on one project, which is why Project 1 reached
a *finished, tested, measured* state rather than two half-built prototypes.

**Judgement call recorded honestly:** Project 2's PRD had already been produced by
that point. Rather than delete it, it was left on disk and set aside — sunk work
retained at zero ongoing cost, with no further attention spent on it.

---

### Prompt 5 — Sequential phase progression with a quality bar

> *"Now go ahead with Phase 2 — correctly and quickly."*

**Intent.** Advance one phase, maintaining both constraints.

**Technique.** **Consistent constraint repetition.** Restating "correctly and
quickly" at every phase boundary prevents standards eroding as a session grows
long. Constraints stated once tend to decay; constraints restated hold.

**Outcome.** Phase 2 delivered the embedding layer, ChromaDB store, BM25 hybrid
retrieval with Reciprocal Rank Fusion, a 60-question hand-labelled golden set, and
an evaluation harness with metrics implemented from scratch.

---

### Prompt 6 — Supply a resource, specify how to use it

> *"Here is the Groq API key — use this instead of any other model."*

**Intent.** Replace the assumed local-LLM dependency with an available service.

**Techniques applied**
- **Unblock proactively** — supplied the credential rather than letting the agent
  design around a dependency I could simply provide.
- **Directive, not suggestion** — "use this instead of any other model" leaves no
  ambiguity about precedence.

**Outcome.** The generation layer was rebuilt on Groq. Two things were flagged back
to me and handled correctly:
1. The key was stored in a **gitignored `.env`**, never in source.
2. **Groq serves LLMs only — it has no embeddings endpoint**, so embeddings
   correctly stayed on the local model. A less careful direction would have
   produced an architecture that could not work.

**Security note recorded for completeness:** pasting a live credential into a chat
places it in the session history. The correct pattern — used from that point on —
is a gitignored `.env` file, with the key rotated afterwards. Documented in
`SETUP.md` §Security.

---

### Prompt 7 — Disclose the real constraint *before* it becomes a failure

> *"The Groq API is on the free tier, so use it accordingly in testing. Now
> continue and complete Phase 3."*

**Intent.** Make a hard external limit an *architectural input* rather than a
runtime surprise.

**Technique.** **Constraint-driven design.** This is the most technically
consequential prompt in the log. Stating the limit *before* Phase 3 was written
meant cost control was designed in, not retrofitted.

**Outcome — the constraint directly produced five design decisions:**

| Design decision | Driven by the free-tier constraint |
|---|---|
| Disk-cached LLM responses keyed on prompt hash | Re-running any report costs **zero** API calls |
| Refusal gate placed **before** generation | Every refusal costs **zero** API calls |
| Hard `--max-calls` budget | Fails loudly instead of draining the daily quota |
| Rate limiting with `retry-after` backoff | Never hammers an already-hit limit |
| Smaller model for the LLM-judge than the answerer | Judging is an easier task than composing |

**Total API spend for the entire evaluation: 22 live calls / 21,186 tokens.**

The refusal-gate placement turned out to be more than a cost optimisation — because
the gate runs before the model, the system **structurally cannot hallucinate on an
out-of-corpus question.** A cost constraint produced a correctness guarantee.

---

### Prompt 8 — Demand portability and reproducibility

> *"Now create an MD file — I've pushed this to git, so create it such that it will
> easily run on someone else's laptop."*

**Intent.** Make the work verifiable by a third party on unfamiliar hardware.

**Techniques applied**
- **Named the audience** — "someone else's laptop" defines the assumed knowledge
  level and forces every implicit step to be made explicit.
- **Named the distribution channel** — mentioning git prompted a check of what was
  actually committed.

**Outcome.** `SETUP.md`, validated by **deleting all generated artefacts and
rebuilding from scratch** rather than assumed to work. That validation caught three
real problems:

1. **A stale binary index was committed to git** — it would have silently
   mismatched a fresh clone's chunks. Untracked, and a guard added that rebuilds
   the index when its IDs do not match.
2. **`requirements.txt` had Phase 2/3 dependencies commented out** — a clone would
   have failed at install.
3. **Confirmed the `.env` credential was *not* committed.**

**Why this mattered:** "it works on my machine" is not a deliverable. This prompt
converted the repo from *personal* to *shareable*.

---

### Prompt 9 — Require execution, not description

> *"Now run this project — I want to check it."*

**Intent.** Verify by observation rather than accepting a report.

**Technique.** **Demand demonstrable proof.** An agent describing its own work is
not evidence. Running it is.

**Outcome.** Full cold-start run — ingest → index → query → ask → evaluate → UI.
This surfaced **three defects that 111 passing tests had not**:

| # | Defect | Why the test suite missed it |
|---|---|---|
| 1 | Citations silently dropped — the model emitted `【1†L1-L3】`, not `[1]` | Tests used the *specified* format; only a live model produced the real one |
| 2 | `--compare` applied re-ranking to all rows, so the demo did not match the published ablation table | A presentation-fidelity issue invisible to unit tests |
| 3 | A **false refusal** on a correctly-retrieved question | A real limitation, now documented rather than hidden |

**This prompt paid for itself.** Three real bugs, none of which any amount of
additional test-writing would have found.

---

### Prompt 10 — Report failure with evidence

> *"Getting errors like [full Python traceback pasted]. Fix this."*

**Intent.** Report a defect in the most diagnosable possible form.

**Techniques applied**
- **Complete stack trace, verbatim** — the single highest-value thing a person can
  supply when reporting a bug. It contains the file, line, call chain and exception
  type; a paraphrase ("it's broken") contains none of that.
- **No premature diagnosis** — reporting the symptom rather than guessing the cause
  avoids anchoring the investigation on a wrong hypothesis.

**Outcome.** Root cause identified as `sqlite3.ProgrammingError` — Streamlit caches
the object across reruns but executes each rerun on a **new thread**, and SQLite
refuses cross-thread connection use.

The fix was **reproduced deterministically before being written**, then applied in
two parts (`check_same_thread=False` *plus* an explicit lock — the first alone
would have removed the safety check without making access safe). The **same bug
class was then found and fixed in a second component** (the Groq response cache)
before it could surface.

Two threaded regression tests were added: **111 → 113 tests.** A single-threaded
suite structurally could not have caught this, which is precisely why 111 tests
passed while the UI was broken.

---

## 3. Direction techniques used, in summary

| # | Technique | Example | Effect |
|---|---|---|---|
| 1 | Specification before implementation | "Detailed PRD first" | Created the contract everything was measured against |
| 2 | Explicit differentiation target | "Stand out from 7–8 others" | Shifted output from adequate to distinguishing |
| 3 | Hierarchical decomposition | "3 phases × 4 parts" | Made every unit independently verifiable |
| 4 | Work-in-progress limit | "Phase 1 only" | Prevented three half-finished phases |
| 5 | Negative constraints | "Don't work on Project 2" | Killed scope drift on contact |
| 6 | Constraint disclosure up front | "Free tier — use accordingly" | Turned a limit into an architecture |
| 7 | Named output contracts | "2 md files", "an MD file" | Removed format ambiguity |
| 8 | Audience specification | "someone else's laptop" | Forced implicit steps to be made explicit |
| 9 | Demand execution | "Run this project" | Found 3 bugs tests could not |
| 10 | Verbatim error reporting | Full traceback | Enabled root-cause diagnosis, not guesswork |
| 11 | Consistent constraint repetition | "correctly and fast" each phase | Held the standard as the session grew long |
| 12 | Environment scoping | "terminal only" | Bounded the agent's side effects |

---

## 4. Where the direction changed the technical outcome

Three moments where the instruction — not the code — determined the result.

### 4.1 The 3×4 decomposition produced the project's best scientific finding

Because Phase 2 was scoped as *"retrieval **and measurement**"*, the evaluation
harness was built **before** the hybrid retriever it would judge. That ordering
meant hybrid retrieval's value was *measured* rather than assumed — and the
measurement contradicted the PRD's own prediction.

The PRD predicted hybrid retrieval would beat both individual legs. **It did not.**
Hybrid won on recall but *lost* rank quality to dense-only. A hypothesis was formed
(BM25 rank dilution), tested as a dedicated ablation arm, and **refuted** — which
led to the correct diagnosis: RRF applies fixed arithmetic regardless of which
retrieval leg deserves trust for a given query.

That diagnosis is exactly what justified the Phase 3 cross-encoder re-ranker, which
then closed the gap as predicted:

| Metric | Before re-ranking | After | Change |
|---|---|---|---|
| MRR@10 | 0.852 | **0.941** | +10% |
| nDCG@10 | 0.861 | **0.942** | +9% |
| Recall@10 | 0.980 | **1.000** | perfect |
| Paraphrase Hit@3 | 0.692 | **0.923** | +33% |

**A sequencing instruction produced a documented, falsifiable experiment.**

### 4.2 The free-tier constraint produced a correctness guarantee

Placing the refusal gate before generation to save API calls also made
hallucination on out-of-corpus questions **structurally impossible** — no model
call happens at all. Measured hallucination rate: **0.000**.

### 4.3 "Run it" found what tests could not

113 tests, a full evaluation suite and 12 reports all passed while three real
defects sat in the running system. Insisting on execution is what surfaced them.

---

## 5. What the direction produced

### Retrieval quality (51 hand-labelled golden questions)

| Configuration | P@3 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| A0 — Keyword baseline (BM25) | 0.314 | 0.853 | 0.738 | 0.738 |
| A1 — Dense semantic | 0.366 | 0.971 | 0.898 | 0.886 |
| A4 — Hybrid + RRF | 0.379 | 0.980 | 0.852 | 0.861 |
| **A5 — Hybrid + cross-encoder re-rank** | **0.431** | **1.000** | **0.941** | **0.942** |

**vs. the keyword baseline: +37% Precision@3, +27% MRR, +28% nDCG, perfect Recall@10.**

### Answer quality — every target met

| Metric | Target | Measured |
|---|---|---|
| Faithfulness | ≥ 0.90 | **1.000** |
| Answer relevance | ≥ 0.88 | **0.909** |
| Citation accuracy | ≥ 0.95 | **1.000** |
| Refusal correctness | ≥ 0.90 | **1.000** |
| Hallucination rate | ≤ 0.05 | **0.000** |

### Engineering artefacts

113 passing tests · 12 generated reports · full PRD · 3 phase reports · ablation
matrix · calibration curve · failure analysis · setup guide · Streamlit UI ·
**$0.00 running cost**

---

## 6. Judgement calls I made, and why

Recorded plainly, including the trade-offs — because a decision without a stated
trade-off is not a decision.

| Decision | Reasoning | Trade-off accepted |
|---|---|---|
| PRD before code | The spec becomes the contract "done" is measured against | Delayed first working code |
| 3 phases × 4 parts | Every unit independently verifiable and demonstrable | More process overhead than a single sprint |
| Cut Project 2 | One finished project beats two prototypes | Project 2 left at PRD stage |
| Disclose the free tier early | Cost control becomes architecture, not a patch | Constrained model choice |
| Require a live run | Descriptions are not evidence | Cost session time — which found 3 bugs |
| Keep the conservative refusal threshold | A wrong answer is worse than an unhelpful one | ~11 false refusals on the golden set |

---

## 7. Known limitations — stated, not hidden

Presenting only successes invites the question "what did you not check?" These are
recorded in the phase reports and repeated here:

| Limitation | Honest assessment |
|---|---|
| **Corpus is 48 chunks** vs. the PRD's 3,000 target | The most significant limitation. Relative rankings between configurations should hold; absolute values would shift at scale. |
| Golden set authored by the system's own builder | Questions were written from the corpus *before* seeing retrieval output, and the paraphrase set is adversarial by construction — but an independent annotator would be stronger. |
| Answer quality judged on 12 of 51 questions | A deliberate free-tier trade-off, stratified across every category rather than skewed toward easy ones. |
| One known false-refusal class | Documented with its calibration curve, not silently tuned away. |
| Alternative embedding models not benchmarked | The one planned ablation arm not completed. |

---

## 8. Reusable prompt template library

Distilled from this project for reuse. These are the **refined forms** — what I
would issue at the start of the next project, having learned from this one.

### 8.1 Project kickoff
```
Read [attached brief/spec].

Produce a detailed PRD as `<filename>.md` before writing any code. It must include:
  - Problem statement and measurable success metrics with target values
  - Numbered functional and non-functional requirements
  - Explicit non-goals
  - Architecture with justified technology choices
  - An evaluation plan defining how success will be proven
  - A risk register with mitigations

Context: this is being assessed competitively. Prioritise decisions that are
defensible under review — measured results over assertions.

Constraints: [time] · [budget] · [platform]
```

### 8.2 Work decomposition
```
Break this into 3 phases, each with 4 parts.

Each phase must be independently shippable and end with a written, testable exit
criterion. Order the parts so that anything used to *measure* quality is built
before the thing it measures.

Complete Phase 1 only. Stop at its exit criterion and report against it.
```
> The second paragraph is the refinement this project taught me. Building the
> measuring instrument first is what turned an assumption into an experiment.

### 8.3 Constraint disclosure
```
Constraints you must design around, not work around:
  - API/service: [tier, rate limits, quota]
  - Hardware: [CPU/GPU, RAM]
  - Data: [scale, sensitivity]

Treat these as architectural inputs. Show me where each one changed a design
decision.
```

### 8.4 Scope correction
```
Stop work on [X].

Scope is [Y] only. Return to [specific part] and continue from there.
Anything outside [Y] goes on a deferred list — do not action it.
```

### 8.5 Verification
```
Run the project end to end and show me the actual output — not a description.

Start from a clean state (delete generated artefacts first) so this proves a fresh
clone works. Drive the real interface a user would touch, and report:
  - the exact commands
  - the real output
  - anything that failed or that you had to work around
```

### 8.6 Bug report
```
[Paste the complete traceback verbatim]

Reproduce it first and confirm the root cause before changing anything.
Then check whether the same bug class exists elsewhere in the codebase.
Add a regression test that would have caught it.
```
> The last two lines are the refinement. On this project they turned one fix into
> two fixes plus two regression tests.

### 8.7 Handover
```
Create `SETUP.md` so someone who has never seen this repo can run it on their own
laptop.

Validate it by deleting every generated artefact and rebuilding from scratch —
do not assume the steps work.

Cover: prerequisites, install, credentials, run order, expected output,
troubleshooting, and which parts work without an API key.
```

---

## 9. What I would do differently next time

| Observation | Change for next time |
|---|---|
| Corpus scale was the limiting factor on statistical confidence | Specify minimum dataset size in the PRD as a hard requirement, not a target |
| Three bugs survived a passing test suite until the app was run | Require a live run at the end of **every** phase, not only at the end |
| The credential was shared in-session | Provide credentials via `.env` from the start; never in a message |
| The golden set was single-annotator | Budget explicitly for a second reviewer, or generate adversarial questions from a different source |
| Model availability was assumed, then failed with a 404 | Verify the provider's actual model list before writing against it |

---

## 10. Closing summary

The system delivered — **+37% Precision@3 over the keyword baseline, perfect
Recall@10, 1.000 faithfulness, 0.000 hallucination rate, 113 passing tests, $0.00
running cost** — is the product of ten directional decisions, not ten coding
requests.

The three that mattered most:

1. **Specification before implementation**, which created the contract that made
   "done" measurable rather than debatable.
2. **Decomposition with measurement ordered first**, which turned a build into an
   experiment — and produced a *refuted* hypothesis, documented rather than buried.
3. **Insisting the system be run**, which found three real defects that a full
   passing test suite did not.

Every number in this document is reproducible from a clean clone via a documented
command. Nothing here is asserted that is not measured.

---

### Supporting documents

| Document | Contents |
|---|---|
| `PRD_Project1_Semantic_Search_QA_Agent.md` | Full product requirements |
| `PHASE_PLAN_Project1.md` | The 3-phase × 4-part execution plan |
| `SETUP.md` | Reproduction guide, validated from a clean state |
| `reports/PHASE1_REPORT.md` | Ingestion — design decisions and verification |
| `reports/PHASE2_REPORT.md` | Retrieval — **includes the refuted hypothesis** |
| `reports/PHASE3_REPORT.md` | Re-ranking, generation, refusal, final results |
| `reports/ablation.md` | A0–A6 comparison matrix |
| `reports/calibration.md` | How the refusal threshold was measured |
| `reports/failure_analysis.md` | Every remaining failure, diagnosed |
