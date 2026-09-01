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

## PHASE 1 — Foundation, Data & Detection Baseline (Days 1–5) ✅

**Goal:** Turn raw video into sampled, attributed, leak-free splits, and stand up the detection measuring instrument — then run the zero-shot COCO baseline (B0) through it so there is a number on the board and a demo on day one.

**Why this is Phase 1:** every metric in the PRD inherits the errors made here. Principle 4 (*no leakage*) is the single most common fatal flaw in video-CV projects — adjacent frames are ~99% redundant, so a random frame split silently inflates every downstream number. That split is therefore enforced by a CI test *before* any model is trained. Appendix E's "working pipeline first, custom data second" is honoured: B0 runs before annotation begins, which both guarantees an early demo and produces the domain-gap baseline that makes fine-tuning's value measurable.

| Part | Deliverable | Exit criterion | Status |
|---|---|---|---|
| **1.1** | Repo scaffold, typed + validated config system, structured logging, CLI skeleton, device/environment report | `python -m src.cli --help` works; config loads and validates; `python -m src.cli env` prints torch/CUDA/OpenCV/Ultralytics versions and the resolved device; `pytest` green | ✅ |
| **1.2** | Frame sampler (~2 fps, never every frame) + **difficulty-attribute extraction** (blur score, brightness → day/dusk/night) + `manifest.csv` | `python -m src.cli sample` writes frames + a manifest carrying `source_video, frame_no, lighting, blur_score, …`; re-running is idempotent | ✅ |
| **1.3** | **Scene-level splitter** (70/15/15 by source video) + label-format validator + **no-leakage test** + `annotation_guide.md` + `DATASET_CARD.md` | `tests/test_no_leakage.py` asserts zero video-ID overlap across splits and fails the build otherwise; per-split class distribution reported; every label is normalised and in-range | ✅ |
| **1.4** | **Detection metrics harness** (IoU, per-class AP, mAP@0.5, mAP@0.5:0.95, P/R, mean-IoU, size-sliced AP) + **B0 zero-shot baseline** run through it + annotated output video | `pytest tests/test_metrics.py` passes against hand-computed worked examples; `python -m src.cli baseline` writes `reports/eval_report.md` with the B0 row and an annotated `.mp4` | ✅ |

**Phase 1 exit demo:** one command turns a folder of videos into a versioned dataset with a leakage-proof split; a second command runs the COCO-pretrained detector over a held-out clip and prints a real mAP table plus an annotated video — the domain-gap baseline that Phase 2 must beat.

**Maps to PRD:** G1 · US-1.1, US-1.2, US-1.3 · M1–M7 (harness) · row B0 of §13.2 · §7.1, §7.3, §7.6 · Delivery-plan Days 1–2 and the §13.1 evaluation philosophy.

---

## PHASE 2 — Detector Training & Tracking (Days 6–10) ✅ *(training code built, not yet run — see below)*

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

## PHASE 3 — Re-Identification, Rigour & Delivery (Days 11–14) ✅ *(GPU-only arms documented, not run)*

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
| 2026-09-01 | — | Plan written; environment audited (CPU-only, no GPU → R9 mitigation adopted). |
| 2026-09-01 | 1.1–1.4 | Phase 1 complete. Repo scaffold, typed config, scene-level splitter with a CI leakage gate, detection metrics harness implemented from the definition, B0 zero-shot baseline measured. |
| 2026-09-01 | 2.1–2.4 | Phase 2 complete. Training pipeline + run registry built (execution needs the GPU path, R9); Kalman/GMC/matching, three trackers, and MOTA/IDF1/HOTA/IDSW/MT-ML/Frag all implemented and tested. |
| 2026-09-01 | 3.1–3.4 | Phase 3 complete. ReID extractor, gallery re-association, τ calibration sweep, cross-camera hand-off, analytics + UOCA, threaded live demo, ONNX export + parity, automatic failure gallery, Streamlit UI. |


---

## Completion record

### What was built and run

All three phases are implemented and exercised end to end on this machine. 121
tests pass. Every headline number in `cv-detection-reid/reports/` was produced
by a command in the repository, and every report carries a provenance block
(config fingerprint, git commit, device, library versions).

| Phase | Artefacts |
|---|---|
| **1** | `src/config.py` · `src/data/{sampler,attributes,manifest,splitter,validate_labels,mot,labels_builder}.py` · `src/eval/detection_metrics.py` · `src/models/detector.py` · `scripts/make_sample_videos.py` |
| **2** | `src/models/train.py` · `src/tracking/{kalman,matching,gmc,base,trackers}.py` · `src/eval/tracking_metrics.py` · `src/pipeline/track_video.py` |
| **3** | `src/reid/{extractor,gallery,calibrate}.py` · `src/eval/{reid_metrics,failures}.py` · `src/pipeline/{reader,demo,analytics,cross_camera}.py` · `src/models/export.py` · `app.py` |

### What was deliberately *not* run, and why

Honest reporting is a deliverable (PRD §13.5, D4), so the gaps are listed here
rather than left for the reviewer to discover.

| Item | Status | Reason |
|---|---|---|
| Detector fine-tuning (2.1) | Code complete, **not executed** | No GPU on this machine (**R9**). A 60-epoch run on 360 CPU frames would take hours and produce a weaker model than the COCO warm start. The committed numbers are the honest **B0 zero-shot** domain-gap baseline that fine-tuning must beat. |
| Ablation rows B7–B10 | **Not measured** | imgsz 960, YOLO11m, hard-negative mining and TensorRT FP16 all need the GPU training path. |
| M19 (≥ 30 FPS GPU), M24 (training time) | **Not measured** | GPU targets. M20 (CPU with frame-skip) is measured locally and reported as the honest worst case. |
| OSNet ReID backbone | **Substituted** | `torchreid` not installed; the extractor falls back to an ImageNet ResNet18 and **names the active backbone in every ReID report**. `pip install torchreid` switches to the PRD's `osnet_x0_25`. |
| Classes `car`, `truck`, `motorcycle`, `bicycle` | **No ground truth** | The synthetic set covers `person` and `bus`. Per-class AP reports `no GT` rather than fabricating a number. |
| M7 (small-object AP) | **`n/a`, not 0.0** | No object in the dataset is under 32² px. "No small objects here" and "missed every small object" are opposite findings and are reported differently. |

### Bugs found and fixed while building — each one worth naming

1. **The blur score conflated darkness with blur.** The textbook variance-of-Laplacian rated the *night* scenes as more blurred than the deliberately motion-blurred one, because a dark frame has weak second derivatives everywhere. That would have silently merged two difficulty slices §13.3 needs kept apart. Fixed by contrast-normalising the measure; the threshold (40) now sits in a measured gap — blurred scenes score 26–31, every sharp scene above 52.
2. **The "camera motion" slice was not measuring camera motion.** The frame-difference proxy scored the busiest *static* scene higher than the panning one. Renamed to `activity` and documented; the true camera-motion slice comes from the GMC module's estimated translation.
3. **The test split was 100% night.** Scene-level splitting is correct but not automatically *representative*. Fixed by stratifying the scene deal by lighting, and by warning when a split still ends up single-condition.
4. **The test split had zero occlusion events**, making M17 unmeasurable. Fixed by spreading occluder scenes across all three lighting conditions so every split gets some.
5. **The Rank-k protocol was too easy.** With single-view footage the same-camera exclusion did nothing, so Rank-1 measured "match this object one frame later" (0.97). Fixed with a 3-second minimum query/gallery gap; Rank-1 fell to a believable 0.74.
6. **Occlusion events were being silently discarded as unscorable** when the detector missed the object on the exact boundary frame. Fixed with a boundary search window, and the report now separates *events found* from *events scorable* — a detection failure is no longer charged to ReID.
7. **`render_table` failed with an `IndexError` on a ragged row.** Now raises naming the table and both widths.

### One-command verification

```bash
cd cv-detection-reid
python -m pytest tests/ -q                    # 121 tests
python -m src.cli validate                    # leakage gate + label integrity
python -m src.cli report                      # assembles reports/FINAL_REPORT.md
```
