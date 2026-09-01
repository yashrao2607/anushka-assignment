# Product Requirements Document (PRD)
## Project 2 — Computer Vision: Object Detection, Tracking & Re-Identification (ReID)

---

| Field | Value |
|---|---|
| **Document Title** | PRD — Multi-Object Detection, Tracking and Re-Identification on Live / Recorded Video |
| **Project Code** | P2-CV-DET-REID |
| **Version** | v1.0 (Baseline, submitted for review) |
| **Status** | Draft → Awaiting Manager Sign-off |
| **Date Created** | 01 September 2026 |
| **Author / Owner** | Project Engineer (Individual Contributor) |
| **Reviewer / Approver** | Reporting Manager |
| **Document Type** | Engineering PRD + Technical Design + ML Experiment Plan |
| **Related Docs** | `PRD_Project1_Semantic_Search_QA_Agent.md`, `EXECUTION_PLAN_3_PHASES.md` |
| **Estimated Effort** | 14 working days (2 sprints of 7 days) |

---

## 0. How to Read This Document

Every requirement carries a stable ID (`US-x`, `NFR-x`, `M-x`, `EXP-x`) so commits, tests, and review comments can reference it directly.

**Reading paths:**
- *Manager / Reviewer (10 min):* Sections 1, 2, 3, 4, 15, 18, 19.
- *Engineer building it:* Everything, in order.
- *ML reviewer:* Sections 4, 7, 10, 12, 13.

---

## 1. Executive Summary

### 1.1 One-Paragraph Summary
We will build a **real-time computer-vision pipeline that detects, localises, tracks and re-identifies objects** in a video feed — live (webcam/RTSP) or recorded. A YOLO-family single-stage detector, fine-tuned on a custom annotated dataset, produces per-frame bounding boxes. A multi-object tracker (**BoT-SORT**, with ByteTrack as a measured baseline) associates those detections across frames into persistent tracks with stable IDs, fusing motion prediction (Kalman filter + camera-motion compensation) with **deep appearance embeddings**. A dedicated **ReID head (OSNet)** produces appearance vectors that let the system **recover an identity after occlusion and re-identify the same object across camera views** — the capability that separates true ReID from naive frame-to-frame tracking. The system is evaluated rigorously with **IoU, mAP@0.5, mAP@0.5:0.95** for detection, **MOTA / IDF1 / HOTA / ID-switches** for tracking, and **Rank-1 / mAP / CMC** for ReID, and ships as an inference demo running at ≥ 25 FPS with an ONNX/TensorRT export path.

### 1.2 Why This Matters (Business Framing)
Detection alone answers *"what is in this frame?"*. Tracking answers *"is this the same thing as before?"*. Only the second question supports the decisions that create value: counting unique vehicles rather than frame-by-frame boxes, measuring dwell time, flagging a defect once instead of 300 times, or following a subject across a camera hand-off. A detector without identity produces an unusable flood of duplicate events; the identity layer is what converts raw pixels into countable, auditable business events.

### 1.3 What Makes This Submission Different

| # | Differentiator | Why it separates this build from a baseline |
|---|---|---|
| **D1** | **True ReID, not just tracking** — a dedicated OSNet appearance-embedding gallery with cross-camera and post-occlusion re-association | Most submissions will run `model.track()` and stop. Handling identity *recovery* after a full occlusion, and matching across two camera views, is the part the brief actually names and the part almost nobody implements. |
| **D2** | **Full tracking-metric suite (MOTA, IDF1, HOTA, ID-switches) — not just mAP** | mAP measures the detector; it says nothing about identity quality. Reporting HOTA/IDF1 proves the tracking claim is measured rather than asserted. |
| **D3** | **A systematic ablation across detector size × tracker × ReID on/off × image size**, reported as an accuracy-vs-FPS Pareto curve | Turns model selection into evidence. A reviewer can see exactly which knob bought which point of accuracy at which latency cost. |
| **D4** | **Deliberate handling of real-world noisy data** — motion blur, night/low light, occlusion, small objects, class imbalance, and a documented hard-negative mining loop | The brief explicitly asks for the "ability to handle real-world noisy data". This is addressed with a stratified *difficulty-sliced* evaluation, not a single averaged number. |
| **D5** | **Deployment realism** — ONNX/TensorRT export, INT8/FP16 quantisation, batched + frame-skipped inference, and a measured accuracy-vs-speed trade-off table | Shows the model can actually run on a live feed, not only in a notebook. |
| **D6** | **Privacy-by-design for a surveillance-class system** — optional face/plate blurring, embedding-only storage, retention policy, and a documented bias check | Surveillance CV carries real ethical risk. Addressing it unprompted signals engineering maturity. |

---

## 2. Problem Statement

### 2.1 Chosen Use Case (committed, with a secondary)
The brief allows product defects, potholes, or surveillance. **Tracking and ReID only carry meaning when objects move and persist**, so:

- **Primary use case — Traffic / premises monitoring:** detect and track `person`, `car`, `motorcycle`, `bus`, `truck`, `bicycle` in a video feed; maintain a stable ID per object; re-identify after occlusion and across two camera views. Produces unique counts, dwell time, and directional flow.
- **Secondary use case — Road-surface defect detection (potholes/cracks):** a static-defect dataset run through the *same* pipeline to prove generality and to exercise the small-object and class-imbalance problems. Tracking here de-duplicates the same pothole seen across ~60 consecutive dashcam frames into **one** reported defect — a concrete demonstration of why identity matters.

### 2.2 Pain Points

| Pain | Concrete Failure Case |
|---|---|
| **No object permanence** | A car passes behind a pole for 8 frames and returns as a brand-new ID. A "unique vehicle" count of 40 becomes 96. |
| **Duplicate event flood** | The same pothole appears in 60 dashcam frames → 60 maintenance tickets for one hole. |
| **Identity swaps in crowds** | Two pedestrians cross; the tracker swaps their IDs; every downstream trajectory statistic is corrupted. |
| **Degradation on real footage** | A model trained on clean daylight images collapses at dusk, in rain, or under motion blur — and an averaged mAP hides exactly that. |
| **Small/far objects missed** | Objects under 32×32 px are the majority in wide-angle traffic views and the majority of detector misses. |
| **No cross-camera continuity** | Camera A and Camera B report the same person as two different subjects. |

### 2.3 Problem Statement (formal)
> Existing per-frame detection cannot answer identity-dependent questions — how many *unique* objects appeared, how long each persisted, whether an object seen now is the same one seen 10 seconds or one camera ago. We need a pipeline that detects objects accurately under real-world noise, maintains stable identities through occlusion, re-identifies objects across time and viewpoints, and proves each of those claims with the appropriate metric rather than a single averaged score.

---

## 3. Goals, Non-Goals & Guiding Principles

### 3.1 Goals (G)

| ID | Goal | Type |
|---|---|---|
| **G1** | Curate and annotate a custom dataset covering the target classes under realistic, noisy conditions. | Data |
| **G2** | Train and fine-tune an object detector achieving the accuracy targets in §4.2. | Model |
| **G3** | Track detected objects across frames with stable IDs, surviving occlusion. | Model |
| **G4** | Re-identify objects after occlusion and across two camera views using appearance embeddings. | Model |
| **G5** | Evaluate with IoU, mAP@0.5, mAP@0.5:0.95 (detection) and MOTA/IDF1/HOTA (tracking) and Rank-1/mAP (ReID). | Scientific |
| **G6** | Deliver a live-feed inference demo (webcam / RTSP / video file) at ≥ 25 FPS. | UX |
| **G7** | Publish an accuracy-vs-latency Pareto analysis with an ONNX/TensorRT export path. | Ops |
| **G8** | Report performance **sliced by difficulty** (lighting, occlusion, object size), not just averaged. | Rigour |

### 3.2 Non-Goals (out of scope for v1.0)

| ID | Non-Goal | Rationale |
|---|---|---|
| NG1 | Designing a novel detector architecture from scratch. | Fine-tuning a strong pretrained backbone is the correct engineering choice; time is better spent on data, tracking, ReID, and evaluation. |
| NG2 | Instance segmentation / pose estimation. | Boxes satisfy the brief. The Ultralytics `-seg` variant is a documented one-line upgrade. |
| NG3 | 3D detection, depth estimation, or metric distance from a monocular camera. | Requires calibration out of scope. Pixel-space analytics only. |
| NG4 | Large-scale multi-camera networks (> 2 cameras) with global topology reasoning. | Two-camera hand-off proves the ReID capability; N-camera graph matching is v2. |
| NG5 | Facial recognition or biometric identification of individuals. | **Deliberate ethical exclusion.** ReID operates on clothing/shape/colour appearance embeddings within a session, never on identity databases. See §17. |
| NG6 | Production edge deployment on physical hardware (Jetson/Coral). | Export + benchmark path is delivered; physical device deployment is documented, not executed. |
| NG7 | Training from scratch on COCO-scale data. | No compute budget; transfer learning is the plan. |

### 3.3 Guiding Principles
1. **Data quality beats model size.** A day spent fixing labels beats a day spent on a bigger backbone. Every model failure is triaged as a data problem first.
2. **Measure the right thing.** Detection metrics for the detector, tracking metrics for the tracker, ReID metrics for ReID. Never one blended number.
3. **Reproducibility is a requirement.** Fixed seeds, pinned versions, versioned dataset splits, every run logged with its exact config.
4. **No leakage.** Splits are by **video/scene**, never by random frame — adjacent frames are near-duplicates and random splitting silently inflates every metric. This single decision is the most common fatal flaw in video-CV projects.
5. **Speed is a feature.** Every accuracy claim is paired with its FPS cost.
6. **Fail visibly.** Low-confidence and ambiguous frames are logged for review, not silently dropped.

---

## 4. Success Metrics & Acceptance Criteria

### 4.1 North Star Metric
> **Unique-Object Counting Accuracy (UOCA)** — on a held-out 5-minute test video with a known ground-truth count of unique objects, the absolute percentage error between reported and true unique-object counts.
> **Target: UOCA error ≤ 8%**
>
> This is the North Star deliberately: it can only be achieved if detection, tracking, **and** ReID all work. It is the one number that cannot be gamed by any single component.

### 4.2 Detection Metrics (held-out test split)

| ID | Metric | Definition | **Target (v1.0)** | Stretch |
|---|---|---|---|---|
| M1 | **mAP@0.5** | Mean Average Precision at IoU 0.5 | **≥ 0.75** | 0.85 |
| M2 | **mAP@0.5:0.95** | COCO-style mAP averaged over IoU 0.50→0.95 | **≥ 0.50** | 0.60 |
| M3 | **Precision** @ conf 0.25 | TP / (TP+FP) | ≥ 0.80 | 0.88 |
| M4 | **Recall** @ conf 0.25 | TP / (TP+FN) | ≥ 0.75 | 0.85 |
| M5 | **Mean IoU** of matched boxes | Localisation tightness | ≥ 0.78 | 0.85 |
| M6 | **Per-class AP@0.5** | Every class individually | ≥ 0.60 for all classes | 0.75 |
| M7 | **Small-object AP** (area < 32²px) | The hardest slice | ≥ 0.35 | 0.45 |

### 4.3 Tracking Metrics (on annotated test sequences)

| ID | Metric | What it measures | **Target** |
|---|---|---|---|
| M8 | **MOTA** | Overall accuracy: FN + FP + ID-switch penalties | ≥ 0.65 |
| M9 | **IDF1** | Identity-preservation F1 — the key identity metric | ≥ 0.70 |
| M10 | **HOTA** | Balanced detection + association quality (modern standard) | ≥ 0.55 |
| M11 | **ID Switches (IDSW)** | Count of identity changes on the same true object | ≤ 15 per 1,000 frames |
| M12 | **MT / ML** | Mostly-Tracked / Mostly-Lost trajectory ratio | MT ≥ 60%, ML ≤ 15% |
| M13 | **Fragmentation** | Track interruptions per trajectory | ≤ 2.0 avg |

### 4.4 ReID Metrics (query/gallery protocol)

| ID | Metric | **Target** |
|---|---|---|
| M14 | **Rank-1 accuracy** — correct identity is the top gallery match | ≥ 0.75 |
| M15 | **Rank-5 accuracy** | ≥ 0.90 |
| M16 | **ReID mAP** | ≥ 0.65 |
| M17 | **Post-occlusion recovery rate** — identity correctly restored after a ≥ 1 s full occlusion | ≥ 0.70 |
| M18 | **Cross-camera match rate** — same object matched between Camera A and Camera B | ≥ 0.60 |

### 4.5 Performance / System Metrics

| ID | Metric | Target |
|---|---|---|
| M19 | Inference FPS, 640×640, full pipeline (detect + track + ReID), **GPU** | ≥ 30 FPS |
| M20 | Inference FPS, **CPU** fallback, with frame-skip | ≥ 12 FPS |
| M21 | End-to-end per-frame latency, p95 (GPU) | ≤ 40 ms |
| M22 | Model size (deployed, ONNX FP16) | ≤ 25 MB |
| M23 | Peak GPU VRAM at inference | ≤ 4 GB |
| M24 | Training time to convergence | ≤ 6 h on a single T4/Colab GPU |
| M25 | Cold start (weights loaded, first frame out) | ≤ 15 s |

### 4.6 Definition of Done

- [ ] Dataset annotated, split **by scene**, versioned, with a documented label protocol and a manifest.
- [ ] `python train.py --config configs/base.yaml` reproduces the reported detector from a clean clone.
- [ ] `python demo.py --source 0` runs a live webcam demo with boxes, IDs, class labels, confidence, and an FPS badge.
- [ ] `python demo.py --source video.mp4 --save` writes an annotated output video.
- [ ] `python evaluate.py` emits `reports/eval_report.md` with every metric in §4.2–4.5.
- [ ] The **ablation table (§13.2)** is populated with real measured numbers.
- [ ] A **difficulty-sliced** results table (§13.3) exists — day/night, occluded/clear, small/large.
- [ ] Cross-camera ReID demo works on the two-view test clip.
- [ ] ONNX export verified: exported-model mAP within 1% of PyTorch.
- [ ] Failure gallery (§13.5) with ≥ 20 diagnosed failure cases.
- [ ] README with quickstart, architecture diagram, and headline results; 3-minute demo video.

---

## 5. Users, Personas & User Stories

### 5.1 Personas
**P1 — Operations Supervisor.** Watches a live feed; needs alerts and counts, not raw video. *Success = trustworthy unique counts and no duplicate alerts.*
**P2 — Maintenance Planner (defect use case).** Needs one ticket per real defect with a location and a confidence score. *Success = de-duplicated defect list.*
**P3 — ML Engineer (me).** Needs to retrain, swap trackers, and prove improvements. *Success = config-driven training and a one-command evaluation harness.*
**P4 — Reviewing Manager.** Needs to verify the claims quickly. *Success = a reproducible demo and an honest, sliced metrics report.*

### 5.2 Epics → User Stories → Acceptance Criteria

#### EPIC-1: Data
- **US-1.1** *As P3, I need a labelled dataset covering the target classes under varied conditions.*
  **AC:** ≥ 3,000 annotated frames; every class has ≥ 200 instances; ≥ 25% of frames are "hard" (night, rain, blur, or occlusion); label protocol documented in `docs/annotation_guide.md`.
- **US-1.2** *As P3, I need splits that don't leak.*
  **AC:** Train/val/test = 70/15/15, split **by source video/scene**; an automated test asserts zero video overlap between splits and fails CI if violated.
- **US-1.3** *As P3, I need to trust my labels.*
  **AC:** A 200-frame subset is double-annotated; agreement reported as IoU-based label consistency ≥ 0.85; disagreements resolved and the guide updated.

#### EPIC-2: Detection
- **US-2.1** *As P1, I want objects detected and localised in each frame.* **AC:** Meets M1–M5.
- **US-2.2** *As P1, I want to tune the sensitivity.* **AC:** Confidence and NMS-IoU thresholds are configurable; a precision–recall curve is published so a threshold can be chosen for the operating point rather than guessed.
- **US-2.3** *As P3, I want the model to survive real-world noise.* **AC:** mAP@0.5 on the "hard" slice is within 15 relative percent of the "easy" slice (§13.3).
- **US-2.4** *As P3, I want small distant objects detected.* **AC:** Meets M7; tiling/higher-imgsz inference evaluated as an ablation arm.

#### EPIC-3: Tracking
- **US-3.1** *As P1, I want each object to keep one ID for as long as it is visible.* **AC:** Meets M8–M13.
- **US-3.2** *As P1, I want an object that is briefly hidden to keep its ID.* **AC:** Meets M17; `track_buffer` (frames an identity survives unseen) is configurable and tuned empirically.
- **US-3.3** *As P3, I want to compare trackers.* **AC:** BoT-SORT vs. ByteTrack vs. a plain IoU/SORT baseline, all measured on identical detections (§13.2).

#### EPIC-4: Re-Identification
- **US-4.1** *As P1, I want the same object recognised after a long occlusion.* **AC:** Appearance embeddings compared by cosine distance against a gallery of recently-lost tracks; meets M17.
- **US-4.2** *As P1, I want the same object matched across two camera views.* **AC:** Meets M18; demonstrated on the two-view test clip in the demo video.
- **US-4.3** *As P3, I want ReID to be toggleable.* **AC:** `reid.enabled: true|false`; both paths measured so ReID's exact contribution to IDF1 is quantified, not assumed.

#### EPIC-5: Inference Demo
- **US-5.1** Live webcam / RTSP / file input via one `--source` flag. **AC:** All three verified.
- **US-5.2** Annotated overlay: box, class, confidence, track ID, trail, live FPS and unique-count counters. **AC:** Rendered at ≥ 25 FPS.
- **US-5.3** Output artefacts: annotated MP4, per-frame CSV/JSON of `frame, track_id, class, conf, x1,y1,x2,y2`, and a summary of unique objects and dwell times. **AC:** Files written on `--save`.
- **US-5.4** Streamlit UI for upload-and-analyse. **AC:** Video upload → processed → results shown with downloadable outputs.

#### EPIC-6: Evaluation
- **US-6.1** One command produces the full metrics report. **AC:** `evaluate.py` writes markdown + JSON + figures (PR curves, confusion matrix, per-class AP bars, Pareto plot).
- **US-6.2** A failure gallery of the worst cases with diagnosed causes. **AC:** ≥ 20 annotated failure images in `reports/failures/`.

---

## 6. Scope & Release Plan

### 6.1 MVP (v1.0) — must ship
Custom annotated dataset with scene-level splits · YOLO fine-tuning pipeline · detection evaluation (IoU/mAP) · BoT-SORT tracking with ReID embeddings · tracking evaluation (MOTA/IDF1/HOTA) · post-occlusion identity recovery · live + file inference demo with overlays · ablation table · difficulty-sliced results · failure gallery · README + demo video.

### 6.2 v1.1 — should ship if time allows
Cross-camera ReID demo · ONNX/TensorRT export + quantisation benchmarks · Streamlit UI · zone/line-crossing counting · dwell-time analytics · hard-negative mining round 2 · test-time augmentation.

### 6.3 v2.0 — stretch / documented future
Multi-camera (>2) topology reasoning · segmentation masks · active learning loop (auto-select frames for labelling by model uncertainty) · temporal smoothing with a lightweight video model · Jetson deployment · anomaly/event detection on trajectories · self-supervised ReID fine-tuning on unlabelled footage.

### 6.4 MoSCoW

| Must | Should | Could | Won't (v1) |
|---|---|---|---|
| Dataset + splits, detector training, detection & tracking metrics, BoT-SORT + ReID, live demo, ablation, sliced results | Cross-camera ReID, ONNX export, Streamlit, counting analytics | TensorRT INT8, TTA, active learning | Face recognition, 3D, segmentation, >2 cameras, physical edge deployment |

---

## 7. Dataset Strategy *(the section that decides the project's ceiling)*

### 7.1 Data Sources

| Source | Role | Volume |
|---|---|---|
| **Public base — a COCO/traffic-domain subset** (e.g. filtered COCO for the vehicle/person classes, or an open traffic-surveillance set) | Pre-training / warm start; supplies class diversity for free | ~5,000 images |
| **Custom captured / curated video** — 8–12 clips from varied scenes, times of day and weather | The core custom contribution; guarantees the domain matches the deployment | ~2,000–3,000 annotated frames |
| **Two-view clip** — the same scene from two angles/positions | Required to evaluate cross-camera ReID (M18) | 1 clip, ~2 min, both views |
| **Secondary defect set** (potholes/cracks) | Generality demo, small-object and imbalance stress test | ~800 images |
| **Hard-negative set** — frames with confusable objects and no targets | Suppresses false positives | ~300 images |

**Frame sampling rule:** sample at ~2 fps from source video, never every frame. Consecutive frames are ~99% redundant; annotating them wastes labelling budget and inflates apparent dataset size without adding information.

### 7.2 Annotation Protocol (documented in `docs/annotation_guide.md`)
- **Tool:** CVAT or Roboflow (both export YOLO-format `class cx cy w h`, normalised).
- **Rules — written down before labelling starts, because ambiguity discovered mid-way corrupts a dataset:**
  1. Box the **full visible extent**; do not extrapolate hidden parts.
  2. Objects occluded > 70% are labelled `ignore` (excluded from loss, excluded from metrics) rather than mislabelled or dropped.
  3. Minimum box size 8×8 px; smaller objects go to `ignore`.
  4. Truncated objects at the frame edge **are** labelled.
  5. Reflections, screen images, and posters are **not** labelled — and are added as hard negatives.
  6. One class per box; the ambiguous `van` → `truck` vs `car` boundary is resolved by an explicit written rule with example images.
- **Quality control:** a 200-frame subset is annotated twice; consistency is reported (US-1.3). Every `ignore` decision is logged so the rule can be audited.

### 7.3 Splits — no leakage (Principle 4)
Split **by source video**, 70/15/15. Test videos are held out entirely and never viewed during model selection. `tests/test_no_leakage.py` asserts the intersection of video IDs across splits is empty and fails the build otherwise. Class distribution across splits is reported so a split is not accidentally missing a class.

### 7.4 Class Balance & Imbalance Handling
Expected distribution is heavily skewed (`car` ≫ `bicycle`, potholes ≪ background). Mitigations, applied in order and each measured: (1) targeted collection for rare classes; (2) oversampling rare-class images in the sampler; (3) class-weighted loss; (4) heavy augmentation on rare classes; (5) report **per-class AP** so a strong `car` score can never hide a broken `bicycle` score (M6).

### 7.5 Augmentation Policy — mapped to real failure modes
Augmentation is chosen to simulate the *specific* noise the deployment has, not applied by default.

| Augmentation | Real-world condition it simulates | Setting |
|---|---|---|
| HSV jitter (h .015 / s .7 / v .4) | Time-of-day, weather, exposure shifts | on |
| Mosaic | Scale variation, context diversity, more objects/img | on; **disabled for the final 10 epochs** (`close_mosaic`) because it distorts box statistics near convergence |
| MixUp (0.1) | Occlusion and cluttered overlap | light |
| Random scale (±50%) / translate (0.1) | Distance and camera-position variance | on |
| Horizontal flip (0.5) | Direction invariance | on |
| Vertical flip | — | **off** (traffic scenes have a fixed gravity prior; flipping vertically teaches nonsense) |
| Motion blur, Gaussian noise, JPEG compression | Fast motion, cheap sensors, stream compression | on (albumentations) |
| Random erasing / CoarseDropout | Partial occlusion | on |
| Low-light gamma / brightness reduction | Night and dusk operation | on |

Every choice is justified; the *absence* of vertical flip is as deliberate as the presence of mosaic, and both are stated for the reviewer.

### 7.5.1 Hard-Negative Mining Loop
After the first training round, run the model over unlabelled footage, collect high-confidence **false positives**, add them as background/hard-negative examples, and retrain. This is measured as its own ablation row — it typically buys a meaningful precision gain at zero labelling cost for new positives.

### 7.6 Dataset Manifest
`data/manifest.csv`: `image_id, source_video, frame_no, split, timestamp, lighting{day,dusk,night}, weather{clear,rain}, occlusion_level{none,partial,heavy}, n_objects, classes, blur_score`. These attribute columns are what make the **difficulty-sliced evaluation** in §13.3 possible — they must be captured at annotation time, not reconstructed later.

---

## 8. Solution Overview & Architecture

### 8.1 Architecture Diagram

```
┌──────────────────── TRAINING PIPELINE (offline) ─────────────────────────────┐
│ raw video / images                                                           │
│      ▼                                                                       │
│ [1] FRAME SAMPLER (~2 fps)  →  [2] ANNOTATION (CVAT/Roboflow, YOLO format)   │
│      ▼                                                                       │
│ [3] SPLITTER  (BY SCENE — leakage test enforced in CI)                       │
│      ▼                                                                       │
│ [4] AUGMENTATION PIPELINE (§7.5)                                             │
│      ▼                                                                       │
│ [5] DETECTOR TRAINING — Ultralytics YOLO11, COCO-pretrained warm start       │
│     AdamW/SGD · cosine LR · warmup · early stopping on val mAP50-95          │
│      ▼                                                                       │
│ [6] VALIDATION → mAP50, mAP50-95, per-class AP, PR curves, confusion matrix  │
│      ▼                                                                       │
│ [7] EXPORT → best.pt → ONNX → (TensorRT FP16/INT8)                           │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────── INFERENCE PIPELINE (online, per frame) ──────────────────┐
│ SOURCE: webcam(0) | RTSP | .mp4                                              │
│      ▼                                                                       │
│ [8]  FRAME READER (OpenCV, threaded queue — decode never blocks inference)   │
│      ▼                                                                       │
│ [9]  PREPROCESS: letterbox → 640×640, BGR→RGB, /255, NCHW                    │
│      ▼                                                                       │
│ [10] DETECTOR  → raw boxes                                                   │
│      ▼                                                                       │
│ [11] POSTPROCESS: conf filter (0.25) → class-wise NMS (IoU 0.45) → boxes     │
│      ▼                                                                       │
│ [12] REID FEATURE EXTRACTOR — OSNet on each box crop → 512-d L2-normed emb.  │
│      ▼                                                                       │
│ [13] TRACKER — BoT-SORT                                                      │
│        · Kalman filter motion prediction                                     │
│        · Camera-Motion Compensation (GMC) — critical for moving cameras      │
│        · cost = λ·IoU_distance + (1−λ)·cosine_appearance_distance            │
│        · Hungarian assignment; two-stage association (high-conf then low-conf)│
│        · unmatched tracks → LOST (kept alive for track_buffer frames)        │
│      ▼                                                                       │
│ [14] REID GALLERY / RE-ASSOCIATION                                           │
│        · gallery of embeddings for LOST + cross-camera tracks                │
│        · new unmatched detection → cosine match vs gallery                   │
│        · match < τ_reid → RESTORE the original ID (not a new one)            │
│      ▼                                                                       │
│ [15] TRACK MANAGER — birth/confirm/death, trajectory history, unique counter │
│      ▼                                                                       │
│ [16] ANALYTICS — unique counts, dwell time, line-crossing, zone occupancy    │
│      ▼                                                                       │
│ [17] RENDERER — boxes, class+conf, ID, trail, FPS badge, counters            │
│      ▼                                                                       │
│  DISPLAY  +  annotated .mp4  +  per-frame results.csv/json                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──── EVALUATION HARNESS ────────────────────────────────────────────────────┐
│ detection: IoU, mAP50, mAP50-95, per-class AP, PR curves                    │
│ tracking : MOTA, IDF1, HOTA, IDSW, MT/ML, Frag  (TrackEval / motmetrics)    │
│ reid     : Rank-1/5, mAP, CMC curve, post-occlusion recovery                │
│ slices   : day/night · clear/occluded · small/med/large  → reports/         │
└────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Why This Design
- **Single-stage detector (YOLO)** over two-stage (Faster R-CNN): real-time is a hard requirement (M19); modern YOLO closes most of the accuracy gap at a fraction of the latency.
- **Tracking-by-detection** over end-to-end joint detection-and-tracking: modular, debuggable, and it lets the detector and tracker be evaluated — and improved — independently. A joint model would make it impossible to say *which* component caused a failure.
- **BoT-SORT** over ByteTrack as the default: ByteTrack's low-confidence second association stage is excellent, but BoT-SORT adds **appearance embeddings + camera-motion compensation** — the two things that make identity survive occlusion and camera movement. Both are measured (§13.2), so the choice is evidence-backed rather than fashionable.
- **A separate ReID gallery layer** on top of the tracker: an internal tracker's appearance memory is short-lived. An explicit gallery is what enables *long* occlusion recovery and *cross-camera* matching — the capabilities the brief names.

---

## 9. Detailed Component Specifications

### 9.1 [10] Detector
- **Family:** Ultralytics YOLO11 (fallback YOLOv8 — identical API, both documented in the prescribed Ultralytics course).
- **Variants trained:** `n` (nano, speed baseline), `s` (**default**), `m` (accuracy arm). The chosen default is whichever sits on the accuracy-vs-FPS Pareto knee (§13.4) — decided by data, not preference.
- **Input:** 640×640 letterboxed (an 960 arm is run for the small-object slice).
- **Warm start:** COCO-pretrained weights. Rationale: the target classes overlap COCO heavily, so transfer learning converges in ~1/10 the epochs and materially outperforms random init at this dataset size.
- **Training config (starting point, tuned by EXP-1):**
  `epochs 100 (early stop patience 20) · batch 16 (auto-scaled to VRAM) · optimizer AdamW · lr0 0.001 · lrf 0.01 · cosine schedule · warmup 3 epochs · weight_decay 5e-4 · momentum 0.937 · box/cls/dfl loss gains 7.5/0.5/1.5 · amp true · seed 42 · workers 8 · close_mosaic 10`
- **Model selection:** best checkpoint by **val mAP@0.5:0.95**, never by training loss. The test split is touched exactly once, at the end.
- **Postprocess:** confidence 0.25, class-wise NMS at IoU 0.45; both published as a swept PR curve so the operating point is a stated choice.

### 9.2 [12] ReID Feature Extractor
- **Model:** OSNet (`osnet_x0_25` for speed / `osnet_x1_0` for accuracy) from `torchreid`, or the Ultralytics-bundled ReID weights.
- **Output:** 512-d L2-normalised embedding per detection crop, so cosine similarity is a dot product.
- **Batching:** all crops in a frame are batched into a single forward pass — the difference between ~4 ms and ~25 ms per frame at 10 objects, and the main reason a naive implementation is slow.
- **Crop handling:** boxes are expanded 5%, clipped to frame, resized to 256×128, and normalised with ImageNet statistics. Crops under 16×32 px are skipped (too degraded to embed meaningfully) and fall back to motion-only association.
- **Embedding smoothing:** each track keeps an exponential moving average of its embeddings (`α = 0.9`) rather than only the latest — a single blurred frame then cannot poison the track's identity.

### 9.3 [13] Tracker — BoT-SORT
- **Motion:** Kalman filter with a constant-velocity model on `(x, y, aspect, height)` plus velocities.
- **Camera-Motion Compensation (GMC):** sparse optical flow / ECC between consecutive frames estimates global camera motion and warps predicted track positions before association. **Essential** for dashcam/handheld footage; without it, every track's prediction is wrong whenever the camera pans, and ID switches explode.
- **Association cost:** `C = λ · d_IoU + (1 − λ) · d_cosine`, default `λ = 0.5`, tuned in EXP-3.
- **Two-stage association (from ByteTrack):** match high-confidence detections first, then use the remaining *low*-confidence detections to rescue tracks that would otherwise die. This is the single most effective trick against occlusion-driven fragmentation, because a partially occluded object is precisely a low-confidence detection.
- **Track lifecycle:** `NEW → TRACKED` after `min_hits = 3` consecutive matches (suppresses flicker from spurious detections) → `LOST` when unmatched → `REMOVED` after `track_buffer = 30` frames (~1 s at 30 fps) unmatched. `track_buffer` is tuned empirically in EXP-3: too short loses identities through occlusion, too long causes ID reuse on genuinely departed objects.
- **Assignment:** Hungarian algorithm on the cost matrix with a match threshold of 0.8.

### 9.4 [14] ReID Gallery & Re-Association — *the D1 differentiator*
- A gallery holds `{track_id → smoothed embedding, last_seen_frame, class, camera_id}` for every LOST track and, in cross-camera mode, for every track from the other view.
- When a detection matches no active track, its embedding is compared against the gallery by cosine distance, **restricted to the same class** (a car can never be re-identified as a person — an easy, large accuracy win).
- If `min_distance < τ_reid`, the original ID is **restored** rather than a new one issued. `τ_reid` is **calibrated empirically**, not guessed: sweep τ over the annotated occlusion events and pick the value maximising post-occlusion recovery F1. The calibration curve is a published figure.
- Gallery entries expire after `gallery_ttl = 300` frames to bound memory and prevent stale matches.
- Cross-camera mode uses the same mechanism with a `camera_id` field and a widened temporal window, which is exactly why the design generalises from occlusion recovery to camera hand-off with no new machinery.

### 9.5 [8] Frame Reader & Pipeline Throughput
Decoding and inference run on **separate threads** with a bounded queue (`maxsize=4`, drop-oldest on overflow). A single-threaded read→infer loop is bottlenecked by decode and typically wastes 30–40% of achievable FPS. For live feeds, dropping stale frames is correct behaviour — a real-time system must prefer the newest frame over a complete-but-lagging one.

### 9.6 [17] Renderer
Per-object: box in a deterministic per-ID colour (`hash(id)`), `class conf% #ID` label, and a fading motion trail from the last 30 centroids. Global HUD: live FPS, frame index, per-class unique count, active-track count. `--blur-faces` applies face/plate blurring in the render path (§17).

### 9.7 Configuration & Reproducibility
All hyperparameters live in `configs/*.yaml`. Every run writes `runs/<run_id>/` containing the resolved config, the git commit SHA, the dataset version hash, library versions, hardware info, and all metrics — so any number in any report can be traced back to the exact run that produced it.

---

## 10. Experiment Plan

| ID | Experiment | Question it answers | Success signal |
|---|---|---|---|
| **EXP-1** | Model-size sweep: YOLO11n / s / m | What accuracy does each FPS tier buy? | A Pareto curve; the knee becomes the default |
| **EXP-2** | Image size: 480 / 640 / 960 | Does higher resolution rescue small objects (M7)? | Small-object AP gain vs. FPS cost |
| **EXP-3** | Tracker: IoU-only / ByteTrack / BoT-SORT / BoT-SORT+ReID | How much identity quality does each stage add? | IDF1 and HOTA deltas on identical detections |
| **EXP-4** | ReID backbone: none / osnet_x0_25 / osnet_x1_0 | Is the larger ReID model worth its latency? | M17 recovery rate vs. ms/frame |
| **EXP-5** | Augmentation: baseline / +noise-blur-lowlight | Does noise augmentation help the *hard* slice? | Hard-slice mAP gain (§13.3) |
| **EXP-6** | Hard-negative mining round | Does adding FP background crops raise precision? | Precision gain at fixed recall |
| **EXP-7** | Freeze-backbone vs. full fine-tune | Best transfer strategy at this dataset size? | val mAP50-95 and training time |
| **EXP-8** | Export: PyTorch / ONNX / TensorRT-FP16 / INT8 | What does quantisation cost in accuracy and buy in speed? | mAP delta ≤ 1% with ≥ 2× speedup |
| **EXP-9** | `track_buffer` sweep: 15 / 30 / 60 / 90 | How long should an identity survive unseen? | Recovery rate vs. ID-reuse errors |
| **EXP-10** | Confidence-threshold sweep | Where is the right operating point? | PR curve; F1-optimal threshold |

Every experiment is a config file plus one command, logged to the same results table — so the ablation in §13.2 assembles itself rather than being hand-transcribed.

---

## 11. Data Model & Interfaces

### 11.1 Core Records
```python
@dataclass
class Detection:
    frame_id: int
    bbox_xyxy: tuple[float, float, float, float]
    conf: float
    cls_id: int
    cls_name: str
    embedding: np.ndarray | None      # 512-d, L2-normalised

@dataclass
class Track:
    track_id: int
    cls_id: int
    state: str                        # new | tracked | lost | removed
    bbox_xyxy: tuple
    kalman_state: np.ndarray
    smoothed_embedding: np.ndarray    # EMA over the track's history
    history: list[tuple]              # centroid trail
    hits: int
    age: int
    frames_since_update: int
    first_seen_frame: int
    last_seen_frame: int
    camera_id: str
    reid_restored: bool               # True if this ID came back via the gallery
```

### 11.2 Per-Frame Output (`results.csv` / `.json`)
`frame, timestamp, track_id, class, conf, x1, y1, x2, y2, reid_restored, camera_id`
Chosen to be **MOT-challenge compatible**, so standard evaluation tooling (TrackEval) works without a converter.

### 11.3 CLI Contract
```bash
python train.py    --config configs/yolo11s.yaml
python evaluate.py --weights runs/best.pt --split test --slices
python demo.py     --source 0                     # webcam
python demo.py     --source rtsp://…              # live stream
python demo.py     --source clip.mp4 --save --tracker botsort --reid --blur-faces
python reid_demo.py --cam-a a.mp4 --cam-b b.mp4   # cross-camera hand-off
python export.py   --weights runs/best.pt --format onnx --half
```

---

## 12. Non-Functional Requirements

| ID | Requirement | Target | Verification |
|---|---|---|---|
| NFR-1 | Real-time GPU throughput | ≥ 30 FPS @ 640, full pipeline | benchmark script, 1,000 frames |
| NFR-2 | CPU fallback | ≥ 12 FPS with frame-skip 2 | same |
| NFR-3 | p95 per-frame latency (GPU) | ≤ 40 ms | per-stage timing breakdown |
| NFR-4 | VRAM at inference | ≤ 4 GB | `nvidia-smi` sampling |
| NFR-5 | Training reproducibility | mAP within ±0.5 across 2 seeded runs | two full runs, diffed |
| NFR-6 | ONNX parity | exported mAP within 1% of PyTorch | parity test in CI |
| NFR-7 | Robustness | no crash on corrupt frame, dropped stream, 0 detections, or resolution change | fault-injection tests |
| NFR-8 | Stream resilience | auto-reconnect on RTSP drop, ≤ 5 s | simulated drop test |
| NFR-9 | Portability | Windows + Linux, CUDA and CPU-only paths | tested on both |
| NFR-10 | Determinism | fixed seeds; deterministic eval | assert equal metrics across reruns |
| NFR-11 | Memory stability | no leak over a 30-min run | RSS/VRAM sampled over time |
| NFR-12 | Model size | ≤ 25 MB ONNX FP16 | `ls -lh` |

---

## 13. Evaluation Framework *(the section that wins the review)*

### 13.1 Evaluation Philosophy
Three independent capabilities are claimed, so three independent metric families are reported. A single "accuracy" number for a detect-track-ReID system is meaningless: a perfect detector with a broken tracker and a broken detector with a perfect tracker can produce the same blended score, while being completely different engineering problems.

**Metric definitions:**
- **IoU** = area(intersection)/area(union) of predicted and ground-truth boxes.
- **AP** = area under the precision–recall curve for a class; **mAP@0.5** averages AP over classes at IoU 0.5; **mAP@0.5:0.95** averages over ten IoU thresholds — the stricter, localisation-sensitive metric.
- **MOTA** = 1 − (FN + FP + IDSW)/GT — penalises all three error types but is dominated by detection errors.
- **IDF1** = identity-based F1 over trajectory matching — the metric that actually reflects identity quality.
- **HOTA** = geometric mean of detection and association accuracy — the modern balanced standard, reported because MOTA alone would flatter a good detector with a weak tracker.
- **Rank-k / CMC** = probability the correct gallery identity appears within the top-k matches.

### 13.2 Ablation Matrix — *to be filled with measured numbers*

| # | Configuration | mAP50 | mAP50-95 | IDF1 | HOTA | IDSW | FPS | Notes |
|---|---|---|---|---|---|---|---|---|
| B0 | COCO-pretrained, **zero fine-tuning** | | | | | | | domain-gap baseline |
| B1 | YOLO11n fine-tuned, IoU-only tracker | | | | | | | speed floor |
| B2 | YOLO11s fine-tuned, IoU-only tracker | | | | | | | detector effect isolated |
| B3 | B2 + **ByteTrack** | | | | | | | motion-only association |
| B4 | B2 + **BoT-SORT (no ReID)** | | | | | | | + camera-motion compensation |
| B5 | B2 + **BoT-SORT + OSNet ReID** *(proposed final)* | | | | | | | appearance association |
| B6 | B5 + **ReID gallery re-association** | | | | | | | long-occlusion recovery |
| B7 | B5 with imgsz 960 | | | | | | | small-object arm |
| B8 | YOLO11m + B6 | | | | | | | accuracy ceiling |
| B9 | B6 + hard-negative mining | | | | | | | precision arm |
| B10 | B6 exported to TensorRT FP16 | | | | | | | deployment arm |

### 13.3 Difficulty-Sliced Results — *the D4 differentiator*
An averaged mAP hides exactly the failures that matter operationally. Every model is therefore also reported per slice, using the attribute columns captured in the manifest (§7.6):

| Slice | mAP50 | Recall | IDF1 | Interpretation |
|---|---|---|---|---|
| Daylight, clear | | | | the easy case — the upper bound |
| Dusk / low light | | | | exposure robustness |
| Night | | | | the hardest lighting case |
| Rain / wet glare | | | | weather robustness |
| Heavy occlusion | | | | where tracking and ReID earn their place |
| Small objects (< 32² px) | | | | resolution limit |
| Crowded (> 15 objects/frame) | | | | association stress |
| Camera motion (dashcam) | | | | where GMC earns its place |

**Acceptance rule (US-2.3):** the hard slices must stay within 15 relative percent of the easy slice, or the gap must be explained with a named remediation experiment.

### 13.4 Pareto / Deployment Analysis
An accuracy-vs-FPS scatter over every configuration in §13.2, with the Pareto frontier drawn, plus a per-stage latency breakdown (decode / preprocess / detect / ReID embed / associate / render). This converts "which model should we deploy?" from an opinion into a chart with an explicitly marked recommended operating point.

### 13.5 Failure Gallery (a required deliverable)
≥ 20 saved failure images/clips in `reports/failures/`, each annotated with the predicted and ground-truth boxes and a root cause from a fixed taxonomy: `missed_small_object` · `low_light_miss` · `motion_blur_miss` · `duplicate_box_nms` · `class_confusion` · `id_switch_crossing` · `id_switch_occlusion` · `track_fragmentation` · `reid_false_match` · `label_error`. Each entry names the remediation. **A frequency table of these causes is what turns the next iteration into a plan instead of a guess** — and honest failure reporting is a strength in review, not a weakness.

---

## 14. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Ecosystem |
| DL framework | **PyTorch 2.11** | Named in the brief; the Ultralytics backend |
| Detector | **Ultralytics YOLO11 / v8** | Named in the brief; training, validation, export, and tracking in one API |
| Tracking | **BoT-SORT** (reference: `niraharon/bot-sort`), ByteTrack baseline | Named in the brief; appearance + camera-motion compensation |
| ReID | `torchreid` OSNet | Strong, small, purpose-built person/vehicle ReID backbone |
| CV ops | **OpenCV** | Named in the brief; I/O, GMC, drawing, video writing |
| Augmentation | Albumentations | Noise/blur/weather transforms beyond the built-ins |
| Tracking metrics | TrackEval / `motmetrics` | Reference implementations of MOTA/IDF1/HOTA |
| Annotation | CVAT / Roboflow | YOLO-format export, review workflow |
| Experiment tracking | Weights & Biases (or Ultralytics CSV + matplotlib offline) | Every run logged and comparable |
| Serving/demo | Streamlit + FastAPI | Upload-and-analyse UI and a programmatic endpoint |
| Export | ONNX Runtime, TensorRT | Deployment path and quantisation benchmarks |
| Quality | pytest, ruff, black, mypy, pre-commit | Enforced in CI |
| Packaging | Docker (CUDA base) | Reproducibility for the reviewer |

*Note on TensorFlow:* the brief lists TensorFlow among the libraries. PyTorch is chosen as the primary framework because Ultralytics and BoT-SORT — the other two named tools — are PyTorch-native, and mixing frameworks would add risk for no accuracy gain. TensorFlow familiarity is demonstrated through the **TFLite/SavedModel export arm** in EXP-8, which is a supported Ultralytics export target. This is a deliberate, stated trade-off rather than an omission.

**Hardware:** training on a single T4/A100-class GPU (Colab/Kaggle acceptable); inference benchmarked on **both** GPU and CPU, with CPU treated as the honest worst case.

---

## 15. Repository Structure & Delivery Plan

### 15.1 Repository Structure
```
cv-detection-reid/
├── README.md · PRD_Project2_Computer_Vision_ObjectDetection_ReID.md
├── requirements.txt · Dockerfile · .pre-commit-config.yaml
├── configs/
│   ├── data.yaml                 # classes + split paths
│   ├── yolo11n.yaml · yolo11s.yaml · yolo11m.yaml
│   ├── tracker_botsort.yaml · tracker_bytetrack.yaml
│   └── experiments/B0.yaml … B10.yaml
├── data/
│   ├── raw_videos/ · frames/ · labels/
│   ├── splits/{train,val,test}/{images,labels}
│   ├── manifest.csv              # incl. difficulty attributes (§7.6)
│   └── DATASET_CARD.md
├── docs/  annotation_guide.md · architecture.md · ethics.md
├── src/
│   ├── data/     sampler.py · splitter.py · augment.py · validate_labels.py
│   ├── models/   train.py · export.py · detector.py
│   ├── tracking/ botsort.py · bytetrack.py · kalman.py · gmc.py · matching.py
│   ├── reid/     extractor.py · gallery.py · calibrate.py
│   ├── eval/     detection_metrics.py · tracking_metrics.py · reid_metrics.py
│   │             slices.py · report.py
│   ├── pipeline/ reader.py · inference.py · analytics.py · renderer.py
│   └── utils/    config.py · viz.py · logging.py
├── train.py · evaluate.py · demo.py · reid_demo.py · export.py · benchmark.py
├── app.py                        # Streamlit UI
├── tests/  test_no_leakage.py · test_metrics.py · test_tracker.py · test_pipeline.py
├── reports/ eval_report.md · ablation.md · slices.md · failures/ · figures/
└── notebooks/ 01_eda.ipynb · 02_error_analysis.ipynb · 03_results.ipynb
```

### 15.2 Delivery Plan (14 working days)

**Sprint 1 — Data & Detection (Days 1–7)**

| Day | Deliverable | Exit criterion |
|---|---|---|
| 1 | Repo scaffold, configs, environment, GPU verified | `import torch; torch.cuda.is_available()` → True; `pytest` green |
| 2 | Video collection + frame sampling + EDA | ≥ 3,000 candidate frames; class/condition distribution plotted |
| 3 | Annotation guide + labelling round 1 | ≥ 1,500 frames labelled; guide committed |
| 4 | Labelling round 2 + QC + **scene-level splits** | Leakage test passes in CI; label-consistency ≥ 0.85 |
| 5 | Baseline training (YOLO11n + s) + B0 zero-shot baseline | First mAP numbers recorded |
| 6 | Detection evaluation harness + PR/confusion figures | `evaluate.py` emits a full detection report |
| 7 | Augmentation tuning + EXP-1/EXP-2 | B1/B2/B7 rows filled → *Sprint 1 demo* |

**Sprint 2 — Tracking, ReID & Delivery (Days 8–14)**

| Day | Deliverable | Exit criterion |
|---|---|---|
| 8 | ByteTrack + BoT-SORT integration | Stable IDs rendered on a test clip |
| 9 | Tracking metrics (MOTA/IDF1/HOTA) + MOT-format ground truth | B3/B4 rows filled |
| 10 | **OSNet ReID embeddings + gallery re-association + τ calibration** | M17 met; calibration curve plotted; B5/B6 filled |
| 11 | **Cross-camera ReID demo** + counting/dwell analytics | M18 met on the two-view clip |
| 12 | Live demo (webcam/RTSP), threaded reader, renderer, Streamlit | ≥ 25 FPS sustained; all three sources verified |
| 13 | **Full ablation + difficulty slices + Pareto + ONNX export** | All tables populated; ONNX parity within 1% |
| 14 | Failure gallery, README, demo video, final report | Every DoD box ticked → *Final demo* |

**Critical path:** Day 4 (clean splits) → Day 5 (first model) → Day 9 (tracking metrics) → Day 10 (ReID). **Annotation is the true bottleneck** and is deliberately front-loaded across Days 2–4; if it slips, the mitigation is to reduce class count (drop `bicycle`/`bus` into an `other` class) rather than to compress the evaluation work.

---

## 16. Risks & Mitigations

| ID | Risk | Likelihood | Impact | Mitigation | Trigger / fallback |
|---|---|---|---|---|---|
| R1 | Annotation takes longer than budgeted | **High** | High | Model-assisted pre-labelling: run COCO-pretrained YOLO to auto-generate boxes, then only *correct* them (3–5× faster than labelling from scratch) | If Day 4 is missed: cut to 3 classes and use a public dataset for the rest |
| R2 | Insufficient data → overfitting | Med | High | Transfer learning, heavy augmentation, early stopping, freeze-backbone arm (EXP-7) | If train/val mAP gap > 0.15: more augmentation, smaller model, more data |
| R3 | **Data leakage from random frame splits** | Med | **Critical** | Scene-level splitting enforced by an automated CI test | Any leak found → rebuild splits and re-run every experiment; numbers before the fix are void |
| R4 | Frequent ID switches in crowds | High | High | BoT-SORT + ReID + tuned `track_buffer` (EXP-9) + two-stage association | If IDF1 < 0.60: raise ReID weight λ, upgrade to `osnet_x1_0`, add per-class gating |
| R5 | Small objects missed | High | Med | Higher imgsz, tiled inference, scale augmentation, small-object-specific slice reporting | If small-object AP < 0.30: SAHI-style tiled inference |
| R6 | Real-time target missed | Med | High | Frame-skip with track interpolation, ONNX/TensorRT, FP16, batched ReID, threaded decode | Drop to YOLO11n; publish both operating points honestly |
| R7 | ReID false matches (wrong ID restored) | Med | Med | Class-gated matching, calibrated τ, embedding EMA smoothing, gallery TTL | Tighten τ; report the precision/recall trade-off rather than hiding it |
| R8 | Class imbalance leaves rare classes broken | High | Med | Targeted collection, oversampling, class-weighted loss, **per-class AP reporting** | Merge rare classes into `other` and state the change |
| R9 | No GPU available | Low | High | Colab/Kaggle free GPU; training checkpoints committed to Drive | Train `n` variant only; publish CPU-realistic numbers |
| R10 | Motion blur / low light collapse | Med | Med | Targeted augmentation (EXP-5) + night footage in the training set | If the night slice is > 25% below the day slice: collect more night data |
| R11 | Scope creep into segmentation/3D/multi-camera | High | Med | Non-goals written down (§3.2) and enforced at each daily exit criterion | New ideas go to v2.0, not the sprint |
| R12 | Ethical/privacy objection to a surveillance demo | Med | Med | Face/plate blurring, no biometric identification, documented ethics section (§17) | Use vehicle-only or defect footage for the public demo |

---

## 17. Ethics, Privacy & Responsible AI

This is a surveillance-class system. Treating that seriously is part of the engineering, not an appendix.

| Area | Commitment |
|---|---|
| **No biometric identification** | ReID matches *appearance embeddings within a session* (clothing, shape, colour). It is explicitly **not** facial recognition and holds no identity database (NG5). This limit is stated in the README, not buried. |
| **Anonymisation** | `--blur-faces` applies face and licence-plate blurring in the render path; enabled by default for any shared demo output. |
| **Data minimisation** | Only embeddings and box coordinates are retained for analytics; raw crops are held in memory only for the gallery TTL and never written to disk by default. |
| **Retention** | Gallery entries expire after `gallery_ttl`; output CSVs contain no imagery. A retention policy is stated in `docs/ethics.md`. |
| **Consent & sourcing** | Footage is either self-captured in public/permitted spaces or from openly-licensed datasets. Every source and its licence is recorded in the dataset card. |
| **Bias check** | Detection recall is reported across lighting conditions and object scales. Where person detection is involved, uneven performance across visually distinct groups is a known risk class in CV; the sliced evaluation is the mechanism for surfacing it, and any gap found is reported rather than averaged away. |
| **Failure disclosure** | The README states plainly where the system fails: heavy crowds, extreme low light, very small objects, and long occlusions beyond the gallery TTL. |
| **Human in the loop** | Outputs are framed as decision *support*. The system produces counts and flags; it does not take automated action on any individual. |

---

## 18. Testing Strategy

| Level | Coverage |
|---|---|
| **Unit** | IoU computation against hand-computed boxes; NMS behaviour on overlapping boxes; Kalman predict/update on a synthetic constant-velocity trajectory; Hungarian assignment on a known cost matrix; cosine distance and gallery matching; AP calculation against a worked example; config validation |
| **Data** | **Leakage test** (no video appears in two splits) · label-format validation (normalised coords in [0,1], valid class ids) · corrupt-image detection · class-distribution report |
| **Integration** | End-to-end on a 100-frame fixture clip: detections produced, IDs stable, CSV schema correct · tracker swap (BoT-SORT ↔ ByteTrack) · ReID on/off parity · ONNX vs. PyTorch output parity |
| **Regression gate** | Test-set mAP50 logged in CI; the build **fails** if it regresses more than 2 absolute points versus the committed baseline |
| **Performance** | 1,000-frame benchmark reporting FPS p50/p95 and a per-stage latency breakdown, on both GPU and CPU |
| **Fault injection** | Corrupt frame · zero detections · RTSP disconnect mid-stream · resolution change mid-video · empty gallery · a 10-minute soak run checking for memory growth |
| **Manual UAT** | A scripted walkthrough on 3 unseen clips, recorded in `reports/uat.md` |

---

## 19. Deliverables

1. **Source code** — clean, typed, tested, documented repository (§15.1).
2. **`PRD_Project2_Computer_Vision_ObjectDetection_ReID.md`** — this document.
3. **Annotated dataset + `DATASET_CARD.md` + `annotation_guide.md`** — with scene-level splits and difficulty attributes.
4. **Trained weights** — `best.pt` plus ONNX (and TensorRT where available).
5. **`reports/eval_report.md`** — all detection, tracking, and ReID metrics.
6. **`reports/ablation.md`** — the completed §13.2 matrix plus the Pareto plot.
7. **`reports/slices.md`** — the difficulty-sliced results table.
8. **`reports/failures/`** — the diagnosed failure gallery plus a root-cause frequency table.
9. **Inference demo** — live webcam / RTSP / file, plus the Streamlit UI.
10. **Cross-camera ReID demo** — the two-view hand-off clip.
11. **Annotated output video + results CSV/JSON.**
12. **README** with quickstart, architecture diagram, and headline results; **3-minute demo video**.

---

## 20. Learning Objectives Mapped to the Prescribed Courses

| Prescribed resource | Where it is applied | Concrete artefact |
|---|---|---|
| **docs.ultralytics.com** | §9.1 training/validation/export API, augmentation hyperparameters, `model.track()`, tracker YAML configs, ONNX/TensorRT/TFLite export | `src/models/train.py`, `export.py`, `configs/yolo11s.yaml` |
| **docs.pytorch.org (2.11)** | §9.2 custom ReID inference module, tensor ops, AMP, `DataLoader`, device management, checkpointing | `src/reid/extractor.py`, the custom training utilities |
| **github.com/niraharon/bot-sort** | §9.3 BoT-SORT internals — Kalman filter, camera-motion compensation, appearance-fused association cost, two-stage matching | `src/tracking/botsort.py`, `gmc.py`, `matching.py` |
| **github.com/opencv/opencv** | §8/§9.5/§9.6 video I/O, RTSP capture, letterboxing, optical flow for GMC, drawing, video writing | `src/pipeline/reader.py`, `renderer.py`, `src/tracking/gmc.py` |

**Additional self-directed learning:** MOT-challenge evaluation protocol (MOTA/IDF1/HOTA), the CMC/Rank-k ReID protocol, Hungarian assignment, non-maximum suppression variants, quantisation trade-offs, and dataset-bias auditing.

---

## 21. Open Questions for the Manager

| # | Question | My default if unanswered |
|---|---|---|
| Q1 | Which **use case** should be primary — traffic/surveillance, product defects, or potholes? | Traffic/premises monitoring as primary (tracking + ReID only carry meaning on moving objects), with potholes as the secondary generality demo. |
| Q2 | Is there **company footage** I should use instead of self-captured/public video? | Proceed with self-captured + openly-licensed data; swapping in real footage costs one annotation round. |
| Q3 | Is **cross-camera ReID** in scope, or is single-camera occlusion recovery sufficient? | Build single-camera recovery first (must), cross-camera second (should). |
| Q4 | What is the **deployment target** — cloud GPU, laptop CPU, or edge device? | Optimise for a single GPU; publish CPU numbers honestly and deliver the export path for edge. |
| Q5 | Is there an **annotation budget** (tool licence, labelling help), or is this solo? | Assume solo; use model-assisted pre-labelling to make the volume feasible. |
| Q6 | Any **privacy constraints** on the demo footage I should apply up front? | Blur faces and plates by default in all shared output. |

---

## Appendix A — `configs/yolo11s.yaml` (training)
```yaml
model: yolo11s.pt          # COCO-pretrained warm start
data: configs/data.yaml
epochs: 100
patience: 20               # early stop on val mAP50-95
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
cos_lr: true
warmup_epochs: 3
weight_decay: 0.0005
momentum: 0.937
box: 7.5
cls: 0.5
dfl: 1.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
flipud: 0.0                # deliberate: traffic scenes have a gravity prior
mosaic: 1.0
close_mosaic: 10           # disable mosaic for the final 10 epochs
mixup: 0.1
amp: true
seed: 42
workers: 8
project: runs
name: yolo11s_base
```

## Appendix B — `configs/tracker_botsort.yaml`
```yaml
tracker_type: botsort
track_high_thresh: 0.5     # first-stage association threshold
track_low_thresh: 0.1      # second-stage rescue of occluded objects
new_track_thresh: 0.6
track_buffer: 30           # frames an identity survives unseen (EXP-9)
match_thresh: 0.8
min_hits: 3                # confirmations before a track is displayed
gmc_method: sparseOptFlow  # camera-motion compensation
with_reid: true
reid_model: osnet_x0_25
appearance_weight: 0.5     # lambda in the fused cost (EXP-3)
reid_gallery:
  enabled: true
  threshold: 0.35          # tau_reid — calibrated, see 9.4
  ttl_frames: 300
  class_gated: true        # never re-identify across classes
  embedding_ema: 0.9
```

## Appendix C — `configs/data.yaml`
```yaml
path: data/splits
train: train/images
val: val/images
test: test/images
names:
  0: person
  1: car
  2: motorcycle
  3: bus
  4: truck
  5: bicycle
# secondary defect set uses a separate data card:
# names: {0: pothole, 1: crack}
```

## Appendix D — Glossary
**IoU** — intersection over union of two boxes. · **mAP** — mean average precision across classes (and IoU thresholds for 0.5:0.95). · **NMS** — non-maximum suppression, removing duplicate boxes. · **MOTA / IDF1 / HOTA** — multi-object tracking accuracy, identity F1, and higher-order tracking accuracy. · **IDSW** — identity switch. · **ReID** — re-identification: matching the same object across time, occlusion, or cameras via appearance embeddings. · **Kalman filter** — recursive state estimator predicting an object's next position. · **GMC** — global/camera-motion compensation. · **Hungarian algorithm** — optimal one-to-one assignment on a cost matrix. · **CMC curve** — cumulative matching characteristic, Rank-k accuracy versus k. · **Hard negative** — a background example the model wrongly fires on. · **Letterbox** — aspect-preserving resize with padding.

## Appendix E — Risk-Free Fast Start (first 3 hours)
1. `pip install ultralytics opencv-python torch torchreid` and verify CUDA.
2. Run COCO-pretrained `yolo11n` with `model.track()` on one test clip — this is the **B0 baseline** and it produces a working demo within the first hour.
3. Only then begin annotation. *Working pipeline first, custom data second* — this guarantees there is always something demonstrable, and it produces the domain-gap baseline that makes fine-tuning's value measurable.

---

**End of PRD — Project 2**
