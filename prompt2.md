# Prompt Log — Project 2
## Computer Vision: Object Detection, Tracking & Re-Identification

**Author:** Project Engineer · **Date:** 01 September 2026
**What this is:** the actual prompts I used to take an 826-line PRD to a working,
tested, fully measured repository — in order, exactly as written.

**What they produced:** 18 Python modules · 121 passing tests · 8 measured reports ·
a live browser demo · a working cross-camera re-identification system.

---

## The strategy in one line

**Plan first, build in phases, run everything, then force it to admit what failed.**

Six prompts. Each one does a specific job, and each one is deliberately short —
a good prompt does not need to be long, it needs to remove the model's room to
skip the hard part.

---

## Prompt 1 — Turn the spec into a plan

> now see "PRD_Project2_Computer_Vision_ObjectDetection_ReID.md" break it in 3 phases and each phase into 3-4 parts for better execution and great output and fast ofcourse then do continue with phase 1

**Why this works:** I did not say "build this." A model handed a huge spec and
told to build it will do the easy 60% and quietly skip the evaluation work —
which is the part that actually carries the marks. Asking for the **plan first**
forces it to sequence the dependencies honestly and commit to what "done" means
before a single line is written.

The three details that did the heavy lifting:
- **"3 phases, 3–4 parts"** — bounded size. Ask for "a plan" and you get either
  3 vague bullets or a 40-item list. Neither is executable.
- **"better execution and great output"** — quality and speed named together, so
  neither gets sacrificed silently.
- **"then continue with phase 1"** — plan *and* start, in one instruction. No
  waiting around for approval on an obvious next step.

**Delivered:** `PHASE_PLAN_Project2.md` — 3 phases × 4 parts, a dependency graph,
the critical path, and a slack policy saying what gets cut first if time runs out
(and what is never cut).

---

## Prompt 2 — Build Phase 2

> now do continue with phase 2 complete , fast correct and great !!

**Why this works:** three words doing three different jobs.

- **"complete"** — no half-finished stubs, no "TODO: implement later."
- **"correct"** — this is the one that matters. It licenses the model to stop and
  fix things instead of racing to a green checkmark.
- **"fast"** — keeps it from gold-plating.

Short prompts work here because Prompt 1 already established the standard. I set
the bar once, then referred back to it.

**Delivered:** the full tracking stack — Kalman filter, camera-motion compensation,
Hungarian matching, and three swappable trackers (IoU baseline, ByteTrack,
BoT-SORT), plus the MOTA / IDF1 / HOTA metrics to score them.

---

## Prompt 3 — Build Phase 3

> now do continue with phase 3 complete , fast correct and great !!

**Why this works:** same bar, deliberately unchanged. Consistency is the point —
the standard does not drop just because the work got harder.

**Delivered:** the re-identification layer — the part the brief actually asks for
and most submissions skip. Appearance embeddings, a gallery that restores an
object's identity after it disappears behind an obstacle, a calibrated matching
threshold, cross-camera hand-off, ONNX export, and an auto-diagnosed failure
gallery.

---

## Prompt 4 — Make it actually run

> do it fast ! and run it locally also and tell me how to test it

**This is the most important prompt in the set.**

"Write the code" gets you plausible code. **"Run it"** gets you *working* code.
The gap between those two is where every real bug lives.

This one prompt caught seven genuine bugs that no code review would have found,
including one that would have quietly corrupted the results: the standard blur
measurement was rating **night** scenes as more blurred than the actually-blurred
ones. That would have merged two separate difficulty categories into one and made
the whole analysis wrong — invisibly.

**"tell me how to test it"** is the second half. It forces the work to be
*verifiable by someone else*, not just by the person who wrote it.

**Delivered:** the whole pipeline executed end to end on this machine, 121 tests
passing, and a testing guide anyone can follow.

---

## Prompt 5 — See it with my own eyes

> visually run it for me i want to see how it is running

**Why this works:** numbers in a table can be wrong in ways that look fine.
A video cannot hide.

I asked to *watch* it, and that is how I confirmed the headline claim was real:
an object walks behind a pole, disappears completely, comes back — and keeps its
**original ID number**, tagged `R` for "restored." That is the difference between
tracking and true re-identification, and I saw it happen rather than reading that
it did.

**Delivered:** an annotated video with live FPS, object IDs, motion trails and
unique-object counters — plus a browser UI at `localhost:8502` where the whole
thing can be driven by hand.

---

## Prompt 6 — Package it for other people

> create a setup2.md for running this since i am sending this to someone else via github

**Why this works:** work that only runs on my machine is not finished work.
I asked for the handover guide as an explicit deliverable rather than assuming
someone else could figure it out.

**Delivered:** `SETUP2.md` — install steps, a 10-minute quick path, a full command
reference, and a troubleshooting table built from the errors we **actually hit**,
not imagined ones.

---

## The six prompts, together

| # | Prompt | Job it does |
|---|---|---|
| 1 | Break the PRD into 3 phases × 3–4 parts, then start Phase 1 | Structure the work before building it |
| 2 | Continue with Phase 2 — complete, fast, correct | Hold the standard |
| 3 | Continue with Phase 3 — complete, fast, correct | Hold the standard |
| 4 | Run it locally and tell me how to test it | **Prove it works** |
| 5 | Visually run it, I want to see it running | **Confirm the claim with my own eyes** |
| 6 | Create a setup guide for GitHub handover | Make it usable by someone else |

**The pattern: Plan → Build → Build → Prove → See → Ship.**

That sequence is reusable on any spec. The two prompts most people skip are
**4 and 5** — and those are the two that turn a demo into something you can
defend in a review.

---

## What these prompts produced

| Result | Number |
|---|---|
| Python modules | 18 |
| Tests passing | 121 |
| Measured reports generated | 8 |
| Real bugs caught by insisting it run | 7 |
| Cross-camera re-identification match rate | **100%** (8 of 8 identities) |
| ID switches, naive tracker → proper tracker | **133 → 3** |
| Diagnosed failure cases | 40 |
| Model size after export | 10.2 MB (target ≤ 25 MB) |

The ablation table also produced something more valuable than a good score: a
**measured negative result**. Adding appearance matching made tracking *worse*
(identity score 0.273 → 0.219), because the ReID-specific model was unavailable
and the substitute was not designed for the job. That is now a documented,
evidence-backed finding rather than an assumption — which is exactly what an
ablation is for.

---

## What I would prompt differently next time

1. **Check the machine before planning.** I found out mid-project that there was
   no GPU here. That reshaped the whole delivery. It should have been the first
   question, not the fifth.

2. **Say what the test data must be able to prove.** Three separate problems —
   the test set being all night footage, having no occlusion events, and having
   no small objects — were the same mistake: I let the evaluation data get built
   without stating what it needed to measure. One sentence would have prevented
   all three: *"the test set must contain at least one example of every difficulty
   type the metrics report."*

3. **Ask "what would prove this design wrong?" earlier.** The most interesting
   finding in the project — that the appearance matching hurt — only appeared at
   the very end. Asking that question in Phase 1 would have surfaced it sooner.
