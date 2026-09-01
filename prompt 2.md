# Prompt Playbook
## Project 2 — Computer Vision: Object Detection, Tracking & Re-Identification

---

| Field | Value |
|---|---|
| **Author** | Project Engineer |
| **Date** | 01 September 2026 |
| **Project** | Project 2 — Object Detection, Tracking and ReID on live/recorded video |
| **Status** | **Prospective** — this is the direction plan, written *before* execution |
| **Derived from** | The methodology proven on Project 1 (see `PROMPT_ENGINEERING_LOG.md`) |
| **Reference spec** | `PRD_Project2_Computer_Vision_ObjectDetection_ReID.md` |
| **Estimated effort** | 14 working days · 3 phases × 4 parts |

---

## 0. What this document is

Project 1 produced a measured result — **+37% Precision@3 over the keyword
baseline, 1.000 faithfulness, 0.000 hallucination rate, 113 passing tests** — and,
more usefully, a *repeatable way of directing the work* that produced it.

This playbook applies that method to Project 2 **before** execution begins, and
adapts it to the failure modes that are specific to computer vision. It is written
to be executed top-to-bottom: every prompt is copy-pasteable.

**Why write it in advance.** On Project 1 the highest-value instructions were the
structural ones — decompose into verifiable phases, build the measuring instrument
before the thing it measures, disclose constraints as architectural inputs. Those
are all decisions that must be made *at the start* or not at all. Writing the
playbook first is how they get made deliberately rather than accidentally.

---

## 1. Carried-over principles (validated on Project 1)

| # | Principle | Evidence from Project 1 |
|---|---|---|
| **P1** | Specification before implementation | The PRD became the contract that made "done" measurable; every phase report cites its section numbers |
| **P2** | Decompose into independently verifiable milestones | 3 phases × 4 parts produced three shippable states and three measured completion reports |
| **P3** | Build the measuring instrument *before* the thing it measures | Produced a falsifiable — and actually **refuted** — hypothesis instead of an assumption |
| **P4** | Disclose real constraints as architectural inputs | The free-tier limit produced a design that made hallucination *structurally impossible* |
| **P5** | Trust nothing that has not been run | A live run found 3 defects that 111 passing tests did not |

**P3 and P5 are the two that transfer hardest into computer vision**, for reasons
Section 2 explains.

---

## 2. What is different about a CV project — and what that changes

Project 1's dominant risk was retrieval quality. Project 2's dominant risks are
**data integrity** and **metric choice**. The direction strategy has to change
accordingly.

| Risk unique to CV | Why it is dangerous | The prompt-level guardrail |
|---|---|---|
| **Data leakage from random frame splits** | Consecutive video frames are ~99% identical. A random split puts near-duplicates in train *and* test, inflating every metric. The model looks excellent and is worthless. **This is the single most common fatal flaw in video-CV projects.** | Mandate **scene-level splitting** in the very first data prompt, enforced by an automated test |
| **Averaged metrics hide operational failure** | A model at 0.85 mAP overall can be at 0.30 at night — which is exactly when it is needed | Require **difficulty-sliced** reporting (day/night, occluded, small objects) from the first evaluation |
| **Detection metrics say nothing about tracking** | mAP measures the detector. It is blind to identity switches. | Mandate **MOTA / IDF1 / HOTA** as separate, non-substitutable metrics |
| **Tracking is not ReID** | Frame-to-frame association is not re-identification after a long occlusion or across cameras | Specify ReID as its own deliverable with its own metrics (Rank-1, mAP, CMC) |
| **Annotation is the true bottleneck** | Labelling silently consumes the schedule, then the model is blamed | Front-load annotation and require **model-assisted pre-labelling** |
| **Class imbalance** | A strong `car` AP hides a broken `bicycle` AP | Require **per-class AP**, never only the mean |
| **Accuracy claims without latency** | A model that cannot hit real-time is not a solution to a live-feed problem | Require every accuracy number to be paired with its FPS cost |

---

## 3. The decomposition

Same 3 × 4 structure, sequenced so the measuring instrument precedes the thing
measured, and so the true bottleneck (annotation) is front-loaded.

| Phase | Theme | Days | The one thing it proves |
|---|---|---|---|
| **1** | Data Foundation | 1–5 | *"The dataset is real, clean, and provably free of leakage."* |
| **2** | Detection Core & Measurement | 6–9 | *"The detector works, and here is the number — sliced by difficulty."* |
| **3** | Tracking, ReID & Delivery | 10–14 | *"Objects keep their identity through occlusion, and every design choice is justified."* |

| Phase | Part | Deliverable | Exit criterion |
|---|---|---|---|
| 1 | 1.1 | Repo scaffold, config, environment + GPU verification | `torch.cuda.is_available()` resolved; `pytest` green on scaffold |
| 1 | 1.2 | Footage collection, frame sampling (~2 fps), EDA | ≥ 3,000 candidate frames; class and condition distribution plotted |
| 1 | 1.3 | Annotation guide + model-assisted labelling | Guide committed *before* labelling starts; ≥ 3,000 frames labelled |
| 1 | 1.4 | **Scene-level splits + leakage test + dataset card** | Automated leakage test passes in CI; label consistency ≥ 0.85 |
| 2 | 2.1 | **Zero-shot COCO baseline (B0)** | Domain-gap baseline measured before any fine-tuning |
| 2 | 2.2 | Training pipeline + augmentation policy | Reproducible run from config; train/val curves logged |
| 2 | 2.3 | **Evaluation harness: IoU, mAP@0.5, mAP@0.5:0.95, per-class AP** | One command emits the full detection report + PR curves |
| 2 | 2.4 | Model-size and image-size sweep | Accuracy-vs-FPS Pareto curve; the knee becomes the default |
| 3 | 3.1 | Tracker integration + **MOTA / IDF1 / HOTA** | Stable IDs rendered; tracking metrics measured on identical detections |
| 3 | 3.2 | **ReID embeddings + gallery re-association** | Post-occlusion identity recovery ≥ 0.70, threshold calibrated |
| 3 | 3.3 | Live inference demo + ONNX export | ≥ 25 FPS sustained; exported mAP within 1% of PyTorch |
| 3 | 3.4 | Full ablation + failure gallery + reports | Every table populated with measured numbers |

**Critical path:** 1.4 → 2.1 → 2.3 → 3.1 → 3.4.
**The bottleneck is annotation (1.2–1.3)**, deliberately front-loaded.

---

## 4. The prompt sequence

Copy-pasteable, in execution order.

---

### Prompt 1 — Kickoff and specification

```
Read the attached project brief for Project 2 (Computer Vision: Object Detection
and ReID).

Before writing any code, produce a detailed PRD as
`PRD_Project2_Computer_Vision_ObjectDetection_ReID.md`. It must include:

  - Problem statement, and a committed primary use case with justification
  - Success metrics with target values, split into three separate families:
      * detection  — IoU, mAP@0.5, mAP@0.5:0.95, per-class AP
      * tracking   — MOTA, IDF1, HOTA, ID switches
      * re-ID      — Rank-1, Rank-5, mAP, post-occlusion recovery rate
  - Numbered functional and non-functional requirements, including an FPS target
  - Explicit non-goals
  - A dataset strategy covering sourcing, annotation protocol, splitting and
    class imbalance
  - An ablation plan naming every configuration to be compared
  - A risk register with mitigations

Do not report a single blended "accuracy" number anywhere — detection, tracking
and ReID fail differently and must be measured separately.

Constraints: 14 working days · single GPU (Colab/Kaggle acceptable) · CPU
inference must also be benchmarked.
```

> **Why the three metric families are named explicitly.** mAP is blind to identity
> switches; MOTA is dominated by detection errors. A brief that says only
> "evaluate the model" will get one number that hides which component is broken.

---

### Prompt 2 — Decomposition

```
Break this into 3 phases, each with 4 parts.

Each phase must be independently demonstrable and end with a written, testable
exit criterion.

Sequencing requirements:
  - Annotation is the real bottleneck. Front-load it in Phase 1.
  - Build the evaluation harness BEFORE the models it will judge.
  - Measure a zero-shot COCO-pretrained baseline before any fine-tuning, so the
    value of fine-tuning is provable rather than assumed.

Complete Phase 1 only. Stop at its exit criterion and report against it.
```

---

### Prompt 3 — Data integrity *(the single most important prompt in this project)*

```
For the dataset, these are hard requirements, not preferences:

1. SPLIT BY SCENE, NEVER BY FRAME.
   Consecutive video frames are near-duplicates. A random frame split leaks
   near-identical images into train and test and inflates every metric.
   Split by source video/scene: 70/15/15.

2. Write an automated test that asserts no video ID appears in more than one
   split, and make it fail the build. I want leakage to be impossible, not
   merely avoided.

3. Sample frames at ~2 fps, not every frame. Consecutive frames add labelling
   cost without adding information.

4. Write the annotation guide BEFORE labelling begins. It must state the rule for
   occluded objects, truncated objects, minimum box size, and every ambiguous
   class boundary, with examples.

5. Record per-frame difficulty attributes at annotation time: lighting
   (day/dusk/night), weather, occlusion level, object size, crowd density.
   These cannot be reconstructed later and they are what makes sliced evaluation
   possible.

6. Report the class distribution across all three splits. If any class is missing
   from a split, tell me before proceeding.
```

> **Why this prompt exists.** Every other decision in the project can be revised.
> A leaked split cannot — it invalidates every number produced afterwards, and it
> is invisible unless explicitly tested for.

---

### Prompt 4 — Annotation throughput

```
Annotation is the schedule bottleneck. Use model-assisted pre-labelling: run a
COCO-pretrained detector over the sampled frames to generate candidate boxes,
then correct them rather than labelling from scratch.

Double-annotate a 200-frame subset and report label consistency as an IoU-based
agreement score. If agreement is below 0.85, the annotation guide is ambiguous —
fix the guide, not the labels.

Report how long labelling actually took against the estimate.
```

---

### Prompt 5 — Constraint disclosure

```
Constraints you must design around, not work around:

  - Compute: single GPU (T4-class). Training must converge in under 6 hours.
  - Deployment: inference must be benchmarked on BOTH GPU and CPU. Treat CPU as
    the honest worst case and publish those numbers too.
  - Real-time: the full pipeline (detect + track + ReID) must sustain the FPS
    target in the PRD.
  - Data: no facial recognition and no biometric identification. ReID operates on
    appearance embeddings within a session only.

Show me where each constraint changed a design decision.
```

---

### Prompt 6 — Baseline before fine-tuning

```
Before any training, run the COCO-pretrained model zero-shot over the test split
and record the result as baseline B0.

This is the domain-gap baseline. Without it there is no way to prove that
fine-tuning was worth doing.

Also produce a working end-to-end demo from this baseline in the first session —
a working pipeline on borrowed weights beats a perfect pipeline that has nothing
to show.
```

---

### Prompt 7 — Evaluation harness

```
Build the detection evaluation harness now, before tuning anything.

Implement the metrics from scratch rather than importing them, and unit-test each
one against a hand-computed fixture — I want to be able to defend every number.

Required outputs from one command:
  - mAP@0.5, mAP@0.5:0.95, precision, recall, mean IoU
  - PER-CLASS AP — never only the mean. A strong majority class must not be able
    to hide a broken rare class.
  - Precision-recall curves and a confusion matrix
  - DIFFICULTY-SLICED results: day / dusk / night, clear / occluded,
    small / medium / large objects, crowded / sparse

The sliced table is the one that matters operationally. An averaged mAP hides
exactly the conditions the system will fail in.
```

---

### Prompt 8 — Tracking, measured separately

```
Integrate tracking now.

Compare, on IDENTICAL detections so the comparison is fair:
  - a plain IoU/SORT baseline
  - ByteTrack (motion + two-stage association)
  - BoT-SORT without appearance
  - BoT-SORT with appearance embeddings

Report MOTA, IDF1, HOTA, ID switches, MT/ML and fragmentation.

Do NOT report mAP as evidence that tracking works — mAP is blind to identity
switches. If IDF1 and mAP disagree, tell me and explain why.
```

---

### Prompt 9 — ReID as its own deliverable

```
Tracking is not re-identification. Build ReID as a distinct capability:

  - An appearance-embedding gallery holding recently-lost tracks
  - Cosine matching to RESTORE an original ID after a long occlusion, rather than
    issuing a new one
  - Class-gated matching — a car must never be re-identified as a person
  - The match threshold CALIBRATED on annotated occlusion events, not guessed.
    Publish the calibration curve.

Measure: Rank-1, Rank-5, ReID mAP, and post-occlusion recovery rate.

Then demonstrate the same mechanism across two camera views.
```

> **Calibrate, do not guess.** On Project 1 the PRD's guessed refusal threshold
> was **20× too high** — calibration recovered four answerable questions at zero
> safety cost. The same discipline applies to every threshold in this project.

---

### Prompt 10 — Ablation with the cost attached

```
Produce the full ablation table with measured numbers, one row per configuration:
detector size × image size × tracker × ReID on/off.

Every accuracy number must be paired with its FPS cost. Plot the
accuracy-vs-latency Pareto frontier and mark the recommended operating point.

If any configuration performs WORSE than expected, report it and diagnose it.
An ablation that only confirms expectations is not an experiment.
```

> On Project 1 the ablation **refuted** the PRD's own prediction about hybrid
> retrieval. That negative result was the most valuable finding in the project and
> it directly motivated the fix that followed. Invite the same here.

---

### Prompt 11 — Verification

```
Run the project end to end and show me the actual output — not a description.

Start from a clean state so this proves a fresh clone works.

Demonstrate all three input paths: webcam, RTSP stream, and a video file.

Save an annotated output video and show me frames from it. A blank or box-free
frame is a failure to launch, not a pass.

Report the exact commands, the real output, and anything that failed or that you
had to work around.
```

---

### Prompt 12 — Failure gallery

```
Produce a failure gallery: at least 20 saved failure cases with predicted and
ground-truth boxes drawn, each labelled with a root cause from a fixed taxonomy
(missed small object, low-light miss, motion blur, duplicate box, class
confusion, ID switch on crossing, ID switch on occlusion, track fragmentation,
ReID false match, label error).

Include a frequency table of causes.

Be honest about failures — I would rather know exactly where this breaks than see
an unexamined 100%.
```

---

### Prompt 13 — Handover

```
Create `SETUP.md` so someone who has never seen this repo can run it on their own
laptop.

Validate it by deleting every generated artefact — weights, splits, caches — and
rebuilding from scratch. Do not assume the steps work.

Cover: prerequisites, GPU vs CPU paths, dataset acquisition, training, inference,
expected output, troubleshooting, and which parts run without a GPU.
```

---

### Prompt 14 — Bug reporting (use throughout)

```
[Paste the complete traceback or the failing metric verbatim]

Reproduce it and confirm the root cause before changing anything.
Then check whether the same bug class exists elsewhere in the codebase.
Add a regression test that would have caught it.
```

> On Project 1 the final two lines turned one SQLite threading fix into two fixes
> plus two regression tests, and caught a latent instance of the same bug in a
> second component before it ever surfaced.

---

## 5. Guardrail prompts — deploy when these situations arise

| Situation | Prompt |
|---|---|
| **Metrics look suspiciously high** | *"These numbers look too good. Verify there is no train/test leakage — check that no source video appears in two splits, and that no near-duplicate frames cross the boundary. Show me the check."* |
| **Only mean mAP is reported** | *"Give me per-class AP and the difficulty-sliced table. The mean is hiding something."* |
| **Tracking claimed without tracking metrics** | *"mAP does not measure tracking. Report IDF1 and HOTA on annotated sequences, or state plainly that tracking is unmeasured."* |
| **A threshold appears without justification** | *"Where did that threshold come from? Calibrate it against labelled data and publish the curve."* |
| **Accuracy quoted without speed** | *"What is the FPS cost of that? Every accuracy number needs its latency attached."* |
| **Scope drifting toward segmentation/3D** | *"Stop. That is a documented non-goal. Return to [current part] and continue."* |
| **Annotation is running over** | *"Annotation is over budget. Reduce the class count — merge rare classes into `other` — rather than cutting the evaluation work."* |
| **Agent reports success without running** | *"Show me it running, not a description of it running."* |

---

## 6. Anti-patterns to avoid

Failure modes worth naming so they are recognised early.

| Anti-pattern | Why it fails | Instead |
|---|---|---|
| "Just train a YOLO model on this data" | Produces a model with no baseline, no split discipline and no evaluation — an unverifiable number | Specify baseline, splits and metrics up front |
| Accepting one blended accuracy figure | Hides which of the three components is broken | Demand three separate metric families |
| Random train/test split | Silent, catastrophic leakage | Scene-level split, enforced by test |
| Labelling before writing the annotation guide | Inconsistent labels discovered halfway through, corrupting the dataset | Guide first, then label |
| Tuning before the evaluation harness exists | Improvements cannot be distinguished from noise | Harness first, always |
| Reporting only the best configuration | Reads as cherry-picking | Publish the whole ablation, including what failed |
| Building the UI before the model works | Time spent on presentation while the core is unproven | Working pipeline first, interface last |

---

## 7. Definition of done

Project 2 is complete when **every** item is true:

- [ ] Dataset annotated, **split by scene**, with the leakage test passing in CI
- [ ] Annotation guide committed, label consistency ≥ 0.85 reported
- [ ] Zero-shot baseline B0 measured before any fine-tuning
- [ ] Detection metrics meet PRD targets, reported with **per-class AP**
- [ ] **Difficulty-sliced** results published (lighting, occlusion, object size)
- [ ] Tracking measured with **MOTA / IDF1 / HOTA**, not mAP
- [ ] ReID measured with **Rank-1 / mAP / post-occlusion recovery**
- [ ] ReID threshold **calibrated**, with the curve published
- [ ] Ablation table complete, every accuracy paired with its FPS
- [ ] Accuracy-vs-latency Pareto plot with a marked operating point
- [ ] Live demo verified on webcam, RTSP and file
- [ ] ONNX export within 1% of PyTorch mAP
- [ ] Failure gallery: ≥ 20 diagnosed cases with a cause-frequency table
- [ ] `SETUP.md` validated by a clean-state rebuild
- [ ] Test suite passing, including the leakage test
- [ ] Limitations stated plainly in the README

---

## 8. What "good" looks like at each phase gate

| Gate | A weak submission shows | This playbook should produce |
|---|---|---|
| **Phase 1** | "I collected some images and labelled them." | A dataset card, a written annotation guide, a measured label-consistency score, and an automated leakage test that fails the build |
| **Phase 2** | "The model got 0.85 mAP." | A zero-shot baseline, per-class AP, a difficulty-sliced table showing where it degrades, and PR curves |
| **Phase 3** | "It tracks objects." | IDF1 and HOTA against a tracker baseline, a calibrated ReID threshold with its curve, a Pareto plot, and a diagnosed failure gallery |

---

## 9. Carry-forward lessons from Project 1

Applied directly to the prompts above.

| Lesson from Project 1 | How it is encoded here |
|---|---|
| The measuring instrument must precede the thing measured | Prompt 2 sequencing requirement; Prompt 7 before any tuning |
| A guessed threshold was 20× wrong | Prompt 9 mandates calibration with a published curve |
| A refuted hypothesis was the most valuable finding | Prompt 10 explicitly invites negative results |
| 113 passing tests missed 3 real bugs | Prompt 11 demands a live run showing actual frames |
| Dataset scale was the limiting factor | Prompt 1 sets dataset size as a hard requirement, not a target |
| A model was assumed available, then 404'd | Prompt 5 requires environment and capability verification up front |
| Single-threaded tests missed a threading bug | Prompt 14 requires checking for the same bug class elsewhere |

---

## 10. Summary

Project 1 demonstrated that the outcome is determined less by the code requested
than by **the structure imposed on the work**: specification before
implementation, decomposition into verifiable milestones, measurement built before
the thing it measures, constraints treated as architectural inputs, and nothing
trusted until it has been run.

This playbook applies that structure to Project 2 in advance, with three CV-specific
additions that Project 1 did not need:

1. **Scene-level splitting, enforced by an automated test** — because leakage is
   silent, catastrophic, and invisible unless explicitly checked for.
2. **Three separate metric families** — because detection, tracking and
   re-identification fail differently and a single number hides which is broken.
3. **Difficulty-sliced reporting** — because an averaged mAP conceals exactly the
   conditions under which the system will be relied upon and fail.

---

### Related documents

| Document | Contents |
|---|---|
| `PRD_Project2_Computer_Vision_ObjectDetection_ReID.md` | Full specification for Project 2 |
| `PROMPT_ENGINEERING_LOG.md` | The Project 1 methodology this playbook is derived from |
| `PROMPTS.md` | The Project 1 prompt sequence, condensed |
| `SETUP.md` | Project 1 reproduction guide (the standard to match) |
