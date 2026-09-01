# cv-detection-reid

**Object detection, multi-object tracking and re-identification on video.**
Implementation of `PRD_Project2_Computer_Vision_ObjectDetection_ReID.md`
(project code `P2-CV-DET-REID`), built to the 3-phase plan in
`PHASE_PLAN_Project2.md`.

Detection answers *"what is in this frame?"*. Tracking answers *"is this the
same thing as before?"*. **Re-identification** answers *"is this the same thing
as ten seconds ago, or on the other camera?"* — and that is the question this
repository is built around, because it is the one that turns pixels into
countable business events.

---

## Quickstart (5 minutes, no GPU required)

```bash
pip install -r requirements.txt

# 1. Check the environment (prints torch/CUDA/OpenCV versions + resolved device)
python -m src.cli env

# 2. Build the dataset. Drop real .mp4 files into data/raw_videos/ first, or
#    generate the synthetic scene set with exact ground truth:
python scripts/make_sample_videos.py

python -m src.cli sample      # ~2 fps frame sampling + difficulty attributes
python -m src.cli labels      # MOT ground truth -> YOLO labels
python -m src.cli split       # SCENE-level 70/15/15 split (leakage-proof)
python -m src.cli validate    # labels, images, and the leakage gate

# 3. Measure. Every command writes a markdown + JSON report into reports/.
python -m src.cli eval-det   --split test --label B0
python -m src.cli eval-track --tracker botsort --max-frames 300
python -m src.cli calibrate  --apply
python -m src.cli reid-eval  --split test --max-frames 300
python -m src.cli cross-camera
python -m src.cli failures
python -m src.cli ablate     --max-frames 200
python -m src.cli report     # assembles reports/FINAL_REPORT.md

# 4. See it work
python -m src.cli demo --source data/raw_videos/scene04_camA.mp4 \
                       --tracker botsort --reid --gallery --save --max-frames 300
python -m src.cli demo --source 0 --show          # live webcam
```

---

## How to test it

### Run the test suite

```bash
python -m pytest tests/ -q                 # everything
python -m pytest tests/ -v                 # with names, to read what is asserted
python -m pytest tests/test_no_leakage.py  # the CI gate on its own
```

The tests are not smoke tests. Every expected value in `test_metrics.py` and
`test_tracking_metrics.py` is **hand-derived in the docstring above it** — a
metrics harness validated against another implementation only proves the two
agree, whereas these prove the implementation matches the definition the
reports claim to use.

| File | What it pins down |
|---|---|
| `tests/test_metrics.py` | IoU on hand-computed boxes; greedy confidence matching; duplicate boxes counted as false positives; `ignore` regions absorbed; 101-point COCO AP against a worked example; empty size bands reporting `n/a` rather than `0.0` |
| `tests/test_tracking_metrics.py` | MOTA/FP/FN arithmetic; ID switches counted across a gap; two objects swapping ids costing two switches; IDF1 punishing a mid-track relabel harder than MOTA does; HOTA separating detection quality from association quality |
| `tests/test_no_leakage.py` | The leakage gate — including a **deliberately leaked** manifest, to prove the assertion actually fires, and a demonstration that `mode: random` leaks by construction |
| `tests/test_tracker.py` | Kalman predict/update on a synthetic constant-velocity trajectory; Hungarian assignment on a known cost matrix; the trackers keeping a stable id and recovering one through an occlusion |
| `tests/test_pipeline.py` | Config validation rejecting bad values; manifest round-trips; MOT↔YOLO conversion; the ReID gallery's class gate, threshold and TTL |

### Verify the claims by hand

Each headline claim has one command that produces the evidence:

```bash
# "Splits do not leak"  ->  must print PASS, and the test must fail on a leak
python -m src.cli validate
python -m pytest tests/test_no_leakage.py -v

# "Identity survives occlusion"  ->  watch #IDs on the occluder scene
python -m src.cli demo --source data/raw_videos/scene04_camA.mp4 \
                       --tracker botsort --reid --gallery --save --max-frames 300
#   then open reports/demo/scene04_camA_demo.mp4 and watch an object go behind
#   the pole and come back with the SAME id, flagged "R" for restored.

# "The threshold is calibrated, not guessed"
python -m src.cli calibrate            # prints the sweep; writes the figure
open reports/figures/tau_calibration.png

# "Each component earns its place"
python -m src.cli ablate --max-frames 200      # B0 -> B4 on identical detections
```

### Reproducibility

Every report carries a **provenance block**: config fingerprint, git commit,
dataset hash, device and library versions (PRD §9.7). Two runs of the same
config on the same data produce the same numbers — seeds are fixed in
`src/utils/device.py::seed_everything` and the split is deterministic for a
fixed seed (asserted in `tests/test_no_leakage.py`).

---

## What is actually here

```
src/
  config.py            typed, validated config; fingerprint for provenance
  cli.py, cli_phase3.py  the command surface
  data/      sampler · attributes · manifest · splitter · validate_labels · mot
  models/    detector · train · export
  tracking/  kalman · matching · gmc · base · trackers
  reid/      extractor · gallery · calibrate
  eval/      detection_metrics · tracking_metrics · reid_metrics · failures · report · runner
  pipeline/  reader · track_video · demo · analytics · cross_camera
  utils/     config · device · logging · viz
```

### The pieces that matter

**Detection metrics implemented from the definition** (`eval/detection_metrics.py`).
COCO 101-point AP, greedy confidence-ordered matching, duplicate detections
counted as false positives so weak NMS is visible, and `ignore` regions that
absorb a detection without scoring it either way.

**A ReID gallery, not just a tracker with appearance** (`reid/gallery.py`).
An internal tracker's appearance memory is short-lived. The explicit gallery is
what lets an identity survive a *long* occlusion and cross a camera hand-off,
with four guards: class gating, a calibrated threshold, TTL expiry, and
EMA-smoothed embeddings.

**A calibrated threshold, with the curve published** (`reid/calibrate.py`).
`tau_reid` is swept over real occlusion events found in the ground truth's
visibility column and chosen to maximise recovery F1. Too tight and genuine
returns are rejected (unique counts inflate); too loose and two objects merge
into one identity (every trajectory statistic corrupts, silently).

**Cross-camera hand-off with no new machinery** (`pipeline/cross_camera.py`).
Same gallery, same cosine distance, same class gate, same threshold — plus a
`camera_id` and a wider window. That the hand-off falls out of the occlusion
design is the claim; the M18 table is the test of it.

**Automatic failure diagnosis** (`eval/failures.py`). Causes are assigned from
the geometry of each failure, not by eye, so the frequency table is comparable
across iterations and reads as a work plan rather than a list.

---

## Honest limitations

Stated here rather than buried, because a report that does not say where it
fails cannot be trusted where it succeeds.

| Limitation | Detail |
|---|---|
| **No GPU on the development machine** | `torch` is a CPU build and `nvidia-smi` is absent, so PRD risk **R9** is live rather than contingent. All FPS numbers in the reports are **CPU** numbers and are the honest worst case; the M19 GPU target (≥ 30 FPS) and M24 (training time) require the Colab/Kaggle path. Every component is device-agnostic (`device: auto`). |
| **ReID backbone is not OSNet by default** | `torchreid` is not installed, so the extractor falls back to an ImageNet ResNet18 — **not** a ReID-trained model. It was trained to collapse intra-class variation, which is the opposite of what ReID needs. Every ReID report names the active backbone. `pip install torchreid` switches to the PRD's `osnet_x0_25`. |
| **Synthetic backgrounds** | Object pixels are real crops cut from photographs, so detector scores are meaningful, but the backgrounds are procedural. Absolute mAP here is **not** comparable to a road-scene benchmark and is never presented as such. Drop real `.mp4` files into `data/raw_videos/` and the whole pipeline consumes them unchanged. |
| **Only 2 of 6 classes have ground truth** | The synthetic set covers `person` and `bus`. `car`, `truck`, `motorcycle` and `bicycle` have zero instances, so their per-class AP reports `no GT` rather than a fabricated number, and M6 is unverifiable for them until real footage is annotated. |
| **No small objects** | Every synthetic object is larger than 32² px, so **M7 (small-object AP) reports `n/a`, not 0.0** — "no small objects here" and "missed every small object" are opposite findings. |
| **HOTA is a faithful reimplementation, not TrackEval** | Per-frame correspondence is solved by Hungarian assignment on IoU; official TrackEval jointly optimises detection and association. The two agree closely on well-behaved sequences and can diverge by about a point where association is poor. |
| **The detector is not fine-tuned yet** | The committed numbers are the **B0 zero-shot** domain-gap baseline. `python -m src.cli train` runs the fine-tuning, but a meaningful run needs the GPU path. |

---

## Privacy and ethics (PRD §17)

This is a surveillance-class system and the constraints are part of the
engineering, not an appendix.

- **No biometric identification.** ReID matches appearance embeddings *within a
  session* — shape, colour, clothing. There is no identity database and no face
  recognition. This is a designed limit (PRD NG5), not a missing feature.
- **`--blur-faces`** blurs faces inside person boxes and plate-height bands on
  vehicles. It blurs the region **whether or not** a face detector fires — a
  privacy guarantee that only holds when a detector succeeds is not a guarantee.
- **Data minimisation.** Only embeddings and box coordinates are retained.
  Gallery entries expire after `gallery_ttl` frames; the results CSV contains no
  imagery.
- **Bias.** Detection recall is reported sliced by lighting and object scale.
  Uneven performance across visually distinct groups is a known risk class in
  person detection; the sliced evaluation is the mechanism for surfacing it, and
  any gap found is reported rather than averaged away.

---

## Configuration

Everything tunable lives in `configs/default.yaml` — no magic numbers in
source (PRD §9.7). Notable entries:

| Key | Meaning |
|---|---|
| `sampling.target_fps` | 2 fps. Consecutive frames are ~99% redundant. |
| `splits.mode` | `by_video`. `random` exists only so the leakage test can prove it fails. |
| `attributes.blur_hard_below` | 40. **Measured**, not guessed: the blurred scenes score 26–31 and every sharp scene above 52. |
| `tracking.track_buffer` | Frames an identity survives unseen (EXP-9). |
| `tracking.appearance_weight` | λ in `C = λ·d_IoU + (1−λ)·d_cosine`. |
| `reid.gallery.threshold` | `tau_reid` — written by `calibrate --apply`, never typed by hand. |
