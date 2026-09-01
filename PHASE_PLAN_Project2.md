# Project 2 — Execution Plan: 3 Phases × 4 Parts

**Project:** Computer Vision — Object Detection, Tracking & Re-Identification (ReID)
**Reference PRD:** `PRD_Project2_Computer_Vision_ObjectDetection_ReID.md`
**Project code:** P2-CV-DET-REID · **Start:** 01 September 2026 · **Duration:** 14 working days
**Status legend:** ✅ done · 🔄 in progress · ⬜ not started

---

## Why three phases

Each phase is a **shippable, demonstrable milestone**, not a time-slice. If work stopped at the end of any phase, there is still a working artefact to show. Each phase ends with a demo and a written exit criterion, and splits into 4 parts so progress is visible daily.

The ordering follows one rule taken from the PRD's guiding principles: **build the measuring instrument before the thing being measured.** The detection metrics harness is built in Phase 1 *before* the detector is fine-tuned in Phase 2; the tracking metrics harness is built alongside the tracker, *before* ReID is added in Phase 3. Every improvement is therefore provable rather than felt.

| Phase | Theme | Days | The one thing it proves |
|---|---|---|---|
| **Phase 1** | Foundation, Data & Detection Baseline | 1–5 | *"Raw video becomes a leak-free, difficulty-attributed dataset — and a detector already draws measured boxes on it."* |
| **Phase 2** | Detector Training & Tracking | 6–10 | *"The fine-tuned detector beats the zero-shot baseline by a measured margin, and objects hold one stable ID."* |
| **Phase 3** | Re-Identification, Rigour & Delivery | 11–14 | *"Identity survives full occlusion and a camera hand-off — and every design choice is backed by an ablation row."* |

---

## PHASE 1 — Foundation, Data & Detection Baseline (Days 1–5) 🔄

**Goal:** Turn raw video into sampled, attributed, leak-free splits, and stand up the detection measuring instrument — then run the zero-shot COCO baseline (B0) through it so there is a number on the board and a demo on day one.

**Why this is Phase 1:** every metric in the PRD inherits the errors made here. Principle 4 (*no leakage*) is the single most common fatal flaw in video-CV projects — adjacent frames are ~99% redundant, so a random frame split silently inflates every downstream number. That split is therefore enforced by a CI test *before* any model is trained. Appendix E's "working pipeline first, custom data second" is honoured: B0 runs before annotation begins, which both guarantees an early demo and produces the domain-gap baseline that makes fine-tuning's value measurable.

| Part | Deliverable | Exit criterion | Status |
|---|---|---|---|
| **1.1** | Repo scaffold, typed + validated config system, structured logging, CLI skeleton, device/environment report | `python -m src.cli --help` works; config loads and validates; `python -m src.cli env` prints torch/CUDA/OpenCV/Ultralytics versions and the resolved device; `pytest` green | ⬜ |
| **1.2** | Frame sampler (~2 fps, never every frame) + **difficulty-attribute extraction** (blur score, brightness → day/dusk/night) + `manifest.csv` | `python -m src.cli sample` writes frames + a manifest carrying `source_video, frame_no, lighting, blur_score, …`; re-running is idempotent | ⬜ |
| **1.3** | **Scene-level splitter** (70/15/15 by source video) + label-format validator + **no-leakage test** + `annotation_guide.md` + `DATASET_CARD.md` | `tests/test_no_leakage.py` asserts zero video-ID overlap across splits and fails the build otherwise; per-split class distribution reported; every label is normalised and in-range | ⬜ |
| **1.4** | **Detection metrics harness** (IoU, per-class AP, mAP@0.5, mAP@0.5:0.95, P/R, mean-IoU, size-sliced AP) + **B0 zero-shot baseline** run through it + annotated output video | `pytest tests/test_metrics.py` passes against hand-computed worked examples; `python -m src.cli baseline` writes `reports/eval_report.md` with the B0 row and an annotated `.mp4` | ⬜ |

**Phase 1 exit demo:** one command turns a folder of videos into a versioned dataset with a leakage-proof split; a second command runs the COCO-pretrained detector over a held-out clip and prints a real mAP table plus an annotated video — the domain-gap baseline that Phase 2 must beat.

**Maps to PRD:** G1 · US-1.1, US-1.2, US-1.3 · M1–M7 (harness) · row B0 of §13.2 · §7.1, §7.3, §7.6 · Delivery-plan Days 1–2 and the §13.1 evaluation philosophy.

---

## PHASE 2 — Detector Training & Tracking (Days 6–10) ⬜

**Goal:** Fine-tune the detector and prove the gain against B0, then make identity persist across frames — with MOTA/IDF1/HOTA measured, not asserted.

**Why this order:** the tracker consumes detections, so detection quality is fixed and measured first; otherwise a tracking failure and a detection failure are indistinguishable. Trackers are then compared **on identical detections** (US-3.3), which is the only way the comparison means anything.

| Part | Deliverable | Exit criterion |
|---|---|---|
| **2.1** | Config-driven training pipeline (YOLO11n/s, COCO warm start, cosine LR, early stop on val mAP50-95) + run registry recording config, git SHA, dataset hash, versions | `python train.py --config configs/yolo11s.yaml` reproduces the reported model from a clean clone; every run writes `runs/<run_id>/` |
| **2.2** | Full detection evaluation: per-class AP, PR curves, confusion matrix, small-object slice; **EXP-1** (n/s/m) and **EXP-2** (imgsz 480/640/960) | M1–M7 met or the gap explained; B1/B2/B7 ablation rows filled with measured numbers |
| **2.3** | Tracking: IoU-only baseline → ByteTrack → **BoT-SORT** (Kalman + camera-motion compensation + two-stage association), config-swappable | Stable IDs rendered on a test clip; tracker swap is a one-flag change; `min_hits`/`track_buffer` configurable |
| **2.4** | Tracking metrics **MOTA / IDF1 / HOTA / IDSW / MT-ML / Frag** + MOT-challenge-format ground truth + **EXP-9** (`track_buffer` sweep) | M8–M13 met; B3/B4 rows filled; `results.csv` is MOT-compatible so TrackEval runs without a converter |

**Phase 2 exit demo:** the same clip through the zero-shot baseline and the fine-tuned + tracked pipeline, side by side, with the detection *and* tracking metric tables that quantify the difference.

**Maps to PRD:** G2, G3 · US-2.1–2.4, US-3.1–3.3 · M1–M13 · EXP-1, 2, 3, 5, 7, 9, 10 · rows B1–B4, B7 · Days 5–9.

---

## PHASE 3 — Re-Identification, Rigour & Delivery (Days 11–14) ⬜

**Goal:** Deliver the D1 differentiator — identity **recovery** after occlusion and across cameras — then prove the whole system with the ablation, the difficulty slices and the Pareto curve, and package it for the reviewer.

| Part | Deliverable | Exit criterion |
|---|---|---|
| **3.1** | **OSNet embeddings** (batched, 512-d L2-normed, EMA-smoothed) + **ReID gallery re-association** (class-gated, TTL-bounded) + **τ_reid calibration sweep** | M17 (post-occlusion recovery ≥ 0.70) met; the calibration curve is a published figure, not a guessed constant; B5/B6 rows filled |
| **3.2** | **Cross-camera ReID demo** + analytics (unique counts, dwell time, line crossing) + threaded reader, renderer, live webcam/RTSP/file demo, Streamlit UI | M18 (cross-camera match ≥ 0.60) met on the two-view clip; ≥ 25 FPS sustained; all three `--source` modes verified; **UOCA error ≤ 8%** on the held-out clip |
| **3.3** | **Full ablation B0–B10** + **difficulty-sliced results** (day/dusk/night, occlusion, size, crowding, camera motion) + Pareto/latency breakdown + ONNX export & parity | Every table populated with measured numbers; hard slices within 15 relative % of easy, or the gap explained with a named remediation; exported mAP within 1% of PyTorch |
| **3.4** | **Failure gallery** (≥ 20 diagnosed cases + root-cause frequency table) + README + ethics/privacy doc + `--blur-faces` + demo video | Every Definition-of-Done box in §4.6 ticked; failure taxonomy counted, not just listed |

**Phase 3 exit demo:** the 3-minute video — live detection with stable IDs → an object disappears behind an occluder and **returns with its original ID** → the same object matched across two camera views → the ablation table that explains why each component is there.

**Maps to PRD:** G4–G8 · US-4.1–4.3, US-5.1–5.4, US-6.1–6.2 · M14–M25 · EXP-4, 6, 8 · rows B5, B6, B8–B10 · §13.3–13.5, §17 · Days 10–14.

---

## Dependency graph (what blocks what)

```
1.1 config ──► 1.2 sampler + attributes ──► 1.3 SCENE SPLIT + leakage test ──┐
       │                                                                     │
       └──────────────► 1.4 DETECTION METRICS HARNESS ──► B0 baseline ───────┤
                                                                             ▼
                        2.1 training ──► 2.2 detection eval (EXP-1/2) ──► 2.3 tracker
                                                                             │
                        ┌────────────────────────────────────────────────────┘
                        ▼
        2.4 TRACKING METRICS ──► 3.1 ReID gallery + tau calibration ──► 3.2 cross-camera + demo
                                                                             │
                        ┌────────────────────────────────────────────────────┘
                        ▼
        3.3 ablation + slices + Pareto + ONNX ──► 3.4 failure gallery + README + video
```

**Critical path:** 1.3 (clean splits) → 1.4 (metrics harness) → 2.1 (first model) → 2.4 (tracking metrics) → 3.1 (ReID).

**The real bottleneck is annotation (1.2–1.3).** It is deliberately front-loaded and de-risked three ways: model-assisted pre-labelling (run B0 to auto-generate boxes, then only *correct* them — 3–5× faster than labelling from scratch, PRD R1); sampling at 2 fps rather than every frame; and a synthetic-clip harness with exact programmatic ground truth so the *entire* pipeline — splitter, metrics, tracker, ReID gallery, occlusion recovery — is runnable and testable on day one, independent of annotation progress.

**Slack policy:** if a slip occurs, cut in this order — (1) Streamlit UI, (2) TensorRT/INT8 arm, (3) cross-camera ReID demoted from *must* to *should*, (4) class count reduced to 3 (PRD R1 fallback). **The evaluation work (1.4, 2.4, 3.3) is never cut — it is the differentiator (D2, D3, D4).**

---

## Environment reality check (recorded up front, not discovered on Day 5)

Measured on this machine at project start:

| Fact | Value | Consequence |
|---|---|---|
| Python | 3.11.9 | Matches the PRD stack. |
| PyTorch | 2.7.0 **+cpu** | CPU-only build; `torch.cuda.is_available()` → `False`. |
| GPU | none detected (`nvidia-smi` absent) | **PRD risk R9 is live from Day 1, not a contingency.** |

**Adopted mitigation (PRD R9):** the local machine is the *development and CPU-benchmark* environment; **training runs on Colab/Kaggle free GPU**, with checkpoints pulled back into `runs/`. All code is written device-agnostic (`device: auto` resolving cuda→mps→cpu) and every component is exercised on CPU at reduced scale in CI. M19/M24 (GPU FPS and training time) are reported from the Colab run; **M20 (CPU ≥ 12 FPS with frame-skip) is measured locally and is treated as the honest worst case** the reviewer can reproduce without a GPU.

---

## Progress log

| Date | Phase.Part | Note |
|---|---|---|
| 2026-09-01 | — | Plan written; environment audited (CPU-only, no GPU → R9 mitigation adopted); Phase 1 started. |
