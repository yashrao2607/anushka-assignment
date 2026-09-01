# Prompt Playbook — Project 2 (CV: Detection, Tracking & Re-Identification)

**Project:** P2-CV-DET-REID
**Author:** Project Engineer · **Date:** 01 September 2026
**Purpose:** the prompt sequence and prompting strategy used to take
`PRD_Project2_Computer_Vision_ObjectDetection_ReID.md` from an 826-line
specification to a working, tested, measured repository.

> **How to read this.** Each prompt below is given in its canonical form,
> followed by *why it is written that way* and *what it produced*. The prompts
> are refined write-ups of the ones actually issued — the intent and sequence
> are exactly as executed; the wording is tightened for reuse as a template on
> the next project.

---

## 0. The operating principles behind every prompt

Five rules did most of the work. Everything below is an application of them.

| # | Principle | Why it matters |
|---|---|---|
| **P1** | **Decompose before you build.** Ask for a phase plan first, never for code first. | A model given a 826-line PRD and told "build it" will produce the easy 60% and silently skip the evaluation harness — the part that carries the marks. |
| **P2** | **Make each phase independently demoable.** | If work stops anywhere, there is still something to show. It also forces the model to sequence dependencies honestly instead of leaving integration to the end. |
| **P3** | **Demand the measuring instrument before the thing measured.** | Build the metrics harness in Phase 1, *before* the detector is trained in Phase 2. Otherwise every later "improvement" is felt rather than proven. |
| **P4** | **Ask for honesty explicitly, and reward it.** | Left alone, models report the numbers that look good. Asking for limitations *by name* is what surfaced the seven real bugs in §5. |
| **P5** | **Require execution, not just code.** | "Write it" produces plausible code. "Run it and show me the output" produces *working* code. The gap between the two is where every real bug lived. |

---

## 1. Phase decomposition

### Prompt 1 — Structure the work

```
Read PRD_Project2_Computer_Vision_ObjectDetection_ReID.md.

Break it into 3 phases, each phase into 3–4 parts, optimised for
execution speed and output quality.

Constraints:
- Each phase must end in a shippable, demonstrable milestone — if work
  stopped there, I must still have something to show.
- Each part needs a written exit criterion that can be checked, not
  a description of activity.
- Sequence by dependency, and state explicitly what blocks what.
- Map every phase back to the PRD's goal IDs, user stories, metric IDs
  and ablation rows, so nothing in the spec is silently dropped.
- Identify the critical path and the real bottleneck, and state the
  slack policy: if something slips, what gets cut first and what is
  never cut.

Then begin Phase 1.
```

**Why it is written this way**

- *"3 phases, each into 3–4 parts"* — bounded granularity. Ask for "a plan" and
  you get either three vague bullets or a forty-item list; neither is executable.
- *"shippable, demonstrable milestone"* — forces vertical slices over horizontal
  layers. Without it you get "Phase 1: all the data code, Phase 2: all the model
  code", which has nothing to demo until the end.
- *"exit criterion that can be checked, not a description of activity"* — the
  single highest-leverage clause here. It converts *"implement the splitter"*
  into *"`tests/test_no_leakage.py` fails the build on any scene spanning two
  splits."*
- *"map back to goal IDs, user stories, metric IDs"* — a traceability
  requirement. It is what stops the model quietly dropping M7 or US-4.3.
- *"what gets cut first and what is never cut"* — pre-commits the triage
  decision while thinking is cheap, instead of under time pressure.

**Produced:** `PHASE_PLAN_Project2.md` — three phases × four parts, a dependency
graph, the critical path (1.3 → 1.4 → 2.1 → 2.4 → 3.1), and a slack policy that
protects the evaluation work because it is the differentiator.

---

### Prompt 2 — Audit the environment before planning around it

```
Before writing any code, audit this machine and record the result in the
plan: Python, PyTorch build, CUDA availability, GPU presence, and which
of the PRD's libraries are installed.

If any PRD assumption is false here, say so in the plan as a live risk
with the mitigation adopted — not as a footnote discovered on day five.
```

**Why**: the PRD assumes a T4 GPU. This machine is CPU-only. Discovering that
in Phase 2 wastes two phases of planning; discovering it in Prompt 2 turns it
into a documented risk (PRD **R9**) with an adopted mitigation, and every
subsequent FPS number gets labelled as the honest CPU worst case.

**Produced:** the *Environment reality check* table in the phase plan, and
device-agnostic code (`device: auto` → cuda → mps → cpu) throughout.

---

## 2. Phase execution

### Prompt 3 — The execution prompt (reused per phase)

```
Continue with Phase N. Complete, fast, and correct.

Non-negotiables:
- Run everything you write. Show me real output, not expected output.
- Build the measuring instrument before the thing it measures.
- Config over code: no magic numbers in source; every tunable in
  configs/default.yaml.
- Every number you report must be reproducible from a committed command,
  and must carry its provenance (config fingerprint, git commit, device).
- Where a PRD assumption does not hold here, implement the closest honest
  alternative and label it — never silently substitute.
```

**Why it is written this way**

- *"Run everything you write"* (**P5**) — this clause alone caught the ragged
  table crash, the broken string literal, the Windows encoding failure and the
  `dataclasses.fields()` annotation bug. None would have appeared in review.
- *"Show me real output, not expected output"* — closes the most common failure
  mode, where a model narrates what a command *would* print.
- *"measuring instrument before the thing it measures"* (**P3**) — this is why
  `eval/detection_metrics.py` and its hand-computed tests exist before any
  training code.
- *"never silently substitute"* — the clause that produced the ReID backbone
  disclosure. `torchreid` is unavailable, so the extractor falls back to an
  ImageNet ResNet18 **and names the active backbone in every ReID report**.

---

### Prompt 4 — Force verifiable tests, not smoke tests

```
For every metric you implement, write tests whose expected values are
derived by hand in the docstring above them.

A metrics harness validated against another implementation only proves the
two agree. I need proof that the implementation matches the definition the
report claims to use.

Include the adversarial cases: duplicate detections, ignore regions, an
empty split, ID switches across a gap, and a deliberately leaked manifest
to prove the leakage guard actually fires.
```

**Why**: *"a guard that has never been seen to fail is not a guard."* Testing
that the leakage check passes on clean data proves nothing — the test must also
be shown to fail on leaked data.

**Produced:** 121 tests. The hand-derivation requirement caught five of my own
arithmetic errors in the *expected* values, each of which confirmed the
implementation was right and the test was wrong — which is exactly the failure
mode a self-consistent test suite hides.

---

## 3. The rigour prompts — where the marks actually are

These separate a submission that works from one that is *believable*.

### Prompt 5 — Slice, never average

```
Do not report a single averaged mAP. Report it sliced by the difficulty
attributes captured at sampling time — lighting, blur, crowding, object
size — because an averaged number hides exactly the failures that matter
operationally.

Where a slice cannot be computed, say "n/a" and say why. Never print 0.0
for "not measurable": "there are no small objects here" and "the model
missed every small object" are opposite findings.
```

**Why**: this is PRD differentiator **D4**, and the `n/a` vs `0.0` distinction
is the kind of detail a reviewer notices. It also caught a real design flaw —
the difficulty attributes must be captured *at sampling time*, because they
cannot be reconstructed afterwards.

### Prompt 6 — Calibrate, do not guess

```
Any threshold that affects a reported metric must be calibrated on data
and published as a curve, not chosen as a plausible-looking constant.

Sweep tau_reid over real occlusion events found in the ground truth,
pick the F1-optimal value, write it back into the config, and publish the
sweep as a figure. State what each error direction actually costs.

Calibrate on train/val only. Choosing a threshold on the test split and
then reporting test-split performance is selecting on the test set.
```

**Why**: PRD §9.4 asks for exactly this. The last paragraph is the one that
matters — it is the same class of error as data leakage, one stage later, and
it is easy to commit without noticing.

**Produced:** `reports/calibration.md` + `reports/figures/tau_calibration.png` —
41 real occlusion events, 163 candidate pairs, a clear F1 peak at τ = 0.30.

### Prompt 7 — Ablate honestly

```
Build the ablation so each row changes exactly ONE thing from the row
above, and every row runs on IDENTICAL detections. A tracker comparison
on different detections measures nothing.

Report the result even if a component makes things worse. A negative
result that is measured is worth more than a positive one that is
assumed.
```

**Why**: this produced the single most valuable finding in the project — adding
appearance embeddings **hurt** tracking (IDF1 0.259 → 0.219), because the
fallback backbone is not ReID-trained. Without the "report it even if worse"
clause, that row would have been quietly framed as a wash.

### Prompt 8 — Diagnose failures automatically

```
Build the failure gallery with root causes assigned automatically from the
geometry of each failure, not by eye, and pair each cause with its
remediation.

Hand-labelled causes drift with the labeller, and the frequency table is
only useful across iterations if the labelling rule is fixed. The table
should read as the next iteration's work plan, not as a list of images.
```

**Produced:** 40 diagnosed cases with a root-cause frequency table.

---

## 4. The honesty prompt — the one that mattered most

```
Before you tell me it is done, list what you did NOT do and why:

- Which PRD metrics are unmeasured, and what blocks them
- Which targets FAILED, with the actual number
- Where a substitution was made and what it costs
- Which numbers are optimistic because of how the data was built

Put this in the README as "Honest limitations", not in a footnote. A
report that does not say where it fails cannot be trusted where it
succeeds.
```

**Why**: this is the highest-value prompt in the set, and the least intuitive.
It inverts the default incentive. It produced:

- **UOCA 125% error** against an 8% target, reported prominently rather than
  omitted — and correctly diagnosed as a *detection* problem, not a tracking one.
- **ONNX parity FAIL** (Δ 0.033 > 0.01 tolerance) — the gate caught a genuine
  postprocessing difference that would otherwise have shipped.
- **M7 reported as `n/a`**, not 0.0.
- Four of six classes reported as `no GT` rather than given fabricated scores.

A reviewer who sees a project self-report its own failures trusts the numbers
it does claim.

---

## 5. What the prompting actually caught

Seven real bugs, each surfaced by a specific clause rather than by luck.

| # | Bug | The clause that caught it |
|---|---|---|
| 1 | Blur metric rated **night** scenes as more blurred than the blurred one — a dark frame has weak second derivatives everywhere. Would have silently merged two difficulty slices. | "slice by lighting **and** blur" forced both to be computed, exposing that they were the same partition |
| 2 | The "camera motion" slice was measuring object motion — the busiest *static* scene outscored the panning one. | "state what each slice actually measures" |
| 3 | Test split was 100% night; every headline number was a night number wearing an average's name. | "report the composition of each split" |
| 4 | Test split had **zero** occlusion events, making M17 unmeasurable. | "report events found separately from events scored" |
| 5 | Rank-1 was 0.97 because the protocol matched objects to themselves one frame later. | "state how hard this test actually is" |
| 6 | Occlusion events silently discarded as unscorable when the detector missed the boundary frame — charging a detection failure to ReID. | "attribute each failure to the component that caused it" |
| 7 | `render_table` crashed with an `IndexError` on a ragged row. | "run everything you write" |

---

## 6. Reusable prompt template

For the next PRD, in order:

```
1. PLAN     "Read <PRD>. Break into 3 phases x 3-4 parts. Each phase a
             shippable milestone. Checkable exit criteria. Map to the
             spec's goal/metric IDs. State the critical path, the real
             bottleneck, and the slack policy."

2. AUDIT    "Audit the environment first. Record any false PRD assumption
             as a live risk with the mitigation adopted."

3. BUILD    "Continue with Phase N. Run everything you write; show real
             output. Measuring instrument before the thing measured.
             Config over code. Never silently substitute."

4. TEST     "Expected values hand-derived in the docstring. Include the
             adversarial cases and prove each guard fires."

5. RIGOUR   "Slice, never average. Calibrate, never guess — and never on
             the test split. Ablate one variable at a time on identical
             inputs, and report negative results."

6. HONESTY  "Before you say done: what did you NOT do, which targets
             failed with what number, and which results are optimistic
             because of how the data was built?"

7. SHIP     "Setup guide for a fresh clone. Troubleshooting from the
             errors we actually hit, not imagined ones."
```

---

## 7. What I would prompt differently next time

Stated because a playbook without one of these has not been used in anger.

1. **Audit the environment in Prompt 1, not Prompt 2.** The CPU-only constraint
   reshaped the whole delivery. It should have been an input to the plan rather
   than an amendment to it.
2. **Specify the data-difficulty distribution up front.** Three separate fixes
   (test split all-night, no occlusion events in test, no small objects at all)
   were the same root cause: I let the evaluation data be built without stating
   what it needed to be able to measure. One clause — *"the test split must
   contain at least one instance of every difficulty axis the metrics report"* —
   would have prevented all three.
3. **Ask for the negative-result framing earlier.** The ReID-hurts-tracking
   finding is the most interesting result in the project, and it only surfaced
   at ablation time. Asking "what would falsify this design?" during Phase 1
   would have surfaced it sooner and shaped the experiment.

---

## Appendix — Measured results this prompting produced

| Metric | Result | Target | Verdict |
|---|---|---|---|
| Scene-level leakage gate | PASS | zero overlap | ✅ |
| Test suite | 121 passing | green | ✅ |
| M18 cross-camera match rate | 1.000 (8/8) | ≥ 0.60 | ✅ |
| M15 Rank-5 | 0.935 | ≥ 0.90 | ✅ |
| M22 ONNX model size | 10.21 MB | ≤ 25 MB | ✅ |
| Failure gallery | 40 diagnosed | ≥ 20 | ✅ |
| M14 Rank-1 | 0.741 | ≥ 0.75 | ❌ |
| M16 ReID mAP | 0.640 | ≥ 0.65 | ❌ |
| NFR-6 ONNX parity | Δ 0.033 | ≤ 0.01 | ❌ |
| UOCA (North Star) | 125% error | ≤ 8% | ❌ |

The failures share one root cause — **no GPU on the development machine, so the
detector was never fine-tuned**. The committed numbers are the B0 zero-shot
baseline that Phase 2's training exists to beat. That is stated in the README,
the phase plan and every affected report rather than left for a reviewer to
work out.
