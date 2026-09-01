# SETUP — Running `cv-detection-reid` from a fresh clone

**Project:** P2-CV-DET-REID — Object Detection, Tracking & Re-Identification
**Audience:** anyone who has just cloned this repository and wants it running.
**Time to first result:** ~10 minutes on CPU. No GPU required.

If you only read one thing: run the six commands in [§3 Quick path](#3-quick-path-10-minutes)
and open `reports/FINAL_REPORT.md`.

---

## 1. Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 (3.10–3.12 fine) | `python --version` |
| pip | recent | `python -m pip install -U pip` |
| Disk | ~2 GB | weights, generated video, frames |
| RAM | 4 GB+ | 8 GB comfortable |
| GPU | **optional** | everything runs on CPU; see [§7](#7-gpu--colab-path) for the training path |
| Internet | first run only | to download `yolo11n.pt` (5.4 MB) and the ResNet18 ReID weights (45 MB) |

Works on **Windows, Linux and macOS**. Windows is the primary development
platform for this repo, and the Windows-specific gotchas are handled in code
(see [§8](#8-troubleshooting)).

---

## 2. Install

```bash
git clone <your-repo-url>
cd cv-detection-reid

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux / macOS:
source .venv/bin/activate

python -m pip install -U pip
pip install -r requirements.txt
```

Verify the install and see which device was resolved:

```bash
python -m src.cli env
```

Expected output — a table of versions plus the resolved device. If it says
`resolved_device cpu` and prints a note about PRD risk R9, that is **correct
and expected** on a machine without an NVIDIA GPU. Nothing is broken.

```bash
python -m pytest tests/ -q       # expect: 121 passed
```

If the tests pass, the install is good. They need no data, no weights and no
network — they exercise the metrics, the trackers, the gallery and the leakage
gate against hand-computed values.

---

## 3. Quick path (10 minutes)

This builds the bundled synthetic dataset, then measures everything.

```bash
# 1. Generate 18 scenes with pixel-exact ground truth (~2 min).
#    Real footage? Skip this and drop .mp4 files into data/raw_videos/ instead.
python scripts/make_sample_videos.py

# 2. Build the dataset  (~1 min total)
python -m src.cli sample      # sample frames at 2 fps + difficulty attributes
python -m src.cli labels      # MOT ground truth -> YOLO labels
python -m src.cli split       # SCENE-level 70/15/15 split
python -m src.cli validate    # labels, images, and the leakage gate

# 3. See it work  (~2 min)
python -m src.cli demo --source data/raw_videos/scene04_camA.mp4 \
       --tracker botsort --reid --gallery --save --max-frames 260
#    -> reports/demo/scene04_camA_demo.mp4
```

**What to look for in that video:** objects walk behind the dark vertical pole
and come back carrying the **same ID with an `R` suffix** — `R` means the ReID
gallery *restored* the identity instead of issuing a new one. The `unique`
counter in the HUD should stay flat while that happens. That is the whole point
of the project in one clip.

---

## 4. Full evaluation

Each command writes a markdown **and** a JSON report into `reports/`, both
carrying a provenance block (config fingerprint, git commit, device, library
versions) so any number traces back to the run that produced it.

```bash
python -m src.cli eval-det   --split test --label B0      # ~1 min
python -m src.cli eval-track --tracker botsort --max-frames 300   # ~4 min
python -m src.cli calibrate  --apply                      # ~3 min
python -m src.cli reid-eval  --split test --max-frames 300 # ~8 min
python -m src.cli cross-camera                            # ~1 min
python -m src.cli failures   --split test                 # ~1 min
python -m src.cli export     --verify --limit 30          # ~2 min
python -m src.cli ablate     --max-frames 200             # ~15 min
python -m src.cli report                                  # instant
```

Then open **`reports/FINAL_REPORT.md`** — it stitches all of the above into one
document.

> **Order matters for one pair.** `calibrate --apply` writes the chosen
> `tau_reid` into `configs/default.yaml`, and `reid-eval` reads it. Run
> calibrate first, or pass `--tau` explicitly.

### Command reference

| Command | Purpose | Output |
|---|---|---|
| `env` | versions + resolved device | console |
| `sample` | frames @ 2 fps + difficulty attributes | `data/frames/`, `data/manifest.csv` |
| `labels` | MOT ground truth → YOLO labels | `data/labels/` |
| `split` | scene-level 70/15/15 | `data/splits/`, manifest `split` column |
| `validate` | label + image integrity, leakage gate | console, exit 1 on failure |
| `eval-det` | mAP, per-class AP, difficulty slices | `reports/eval_report.md` |
| `train` | fine-tune the detector | `runs/<id>/`, `run.json` |
| `track` | detect + track one video | `reports/tracks/` |
| `eval-track` | MOTA / IDF1 / HOTA / IDSW / MT-ML | `reports/tracking_report.md` |
| `calibrate` | sweep `tau_reid` over real occlusions | `reports/calibration.md` + figure |
| `reid-eval` | Rank-k / CMC / mAP + occlusion recovery | `reports/reid_report.md` |
| `cross-camera` | two-view hand-off (M18) | `reports/cross_camera.md` |
| `demo` | live webcam / RTSP / file | `reports/demo/` |
| `export` | ONNX export + mAP parity gate | `reports/export.md` |
| `failures` | auto-diagnosed failure gallery | `reports/failures/` + `.md` |
| `ablate` | B0→B6 on identical detections | `reports/ablation.md` |
| `report` | assemble everything | `reports/FINAL_REPORT.md` |

Every command takes `--config <path>` and `-v` for debug logging.

---

## 5. The browser UI

```bash
streamlit run app.py
# -> http://localhost:8501
```

Upload a video or pick a bundled scene, toggle tracker / ReID gallery / privacy
blur, and press **Run**. It plays the annotated result, shows unique counts and
dwell times, and offers the per-frame CSV for download. Any report in
`reports/` can be read at the bottom of the page.

Port already in use? `streamlit run app.py --server.port 8502`.

---

## 6. Using your own footage

The pipeline does not know or care that the bundled data is synthetic.

```bash
# 1. Name files <scene>_<camera>.mp4 so two views of one scene stay together
#    in the same split. This matters -- see docs/annotation_guide.md §5.
cp /path/to/*.mp4 data/raw_videos/

# 2. Annotate. Read docs/annotation_guide.md BEFORE labelling starts --
#    the rules exist because ambiguity discovered mid-way corrupts a dataset.
#    Export MOT format to data/gt/<video-stem>_gt.txt:
#        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
#    `frame` is 1-indexed; `class` is the index in dataset.classes.

# 3. Same three commands as before
python -m src.cli sample && python -m src.cli labels && python -m src.cli split
python -m src.cli validate     # MUST print PASS before you trust any metric
```

To change the class list, edit `dataset.classes` in `configs/default.yaml`.
**Never reorder that list** without remapping every existing label file — the
index *is* the class id. The validator catches out-of-range ids but cannot
detect a silent reordering.

---

## 7. GPU / Colab path

This repo was developed on a CPU-only machine, so the committed detector
numbers are the **B0 zero-shot COCO baseline**. To fine-tune:

```bash
# On a machine (or Colab/Kaggle) with CUDA:
pip install -r requirements.txt          # installs the CUDA torch wheel there
python -m src.cli env                    # confirm cuda_available True

python -m src.cli train --model yolo11n.pt --epochs 60 --batch 16
# -> runs/<run_id>/weights/best.pt  plus runs/<run_id>/run.json

# Bring best.pt back and re-measure everything against it:
python -m src.cli eval-det   --split test --weights runs/<run_id>/weights/best.pt
python -m src.cli eval-track --weights runs/<run_id>/weights/best.pt
python -m src.cli ablate     --weights runs/<run_id>/weights/best.pt
```

Passing `--weights` to `ablate` unlocks the fine-tuned rows (B2, B4f, B6f) so
the table shows the fine-tuning delta directly against the B0 baseline.

Optional, for the ReID backbone the design actually calls for:

```bash
pip install torchreid        # switches the extractor to OSNet (osnet_x0_25)
```

Without it the pipeline falls back to an ImageNet ResNet18 and **names the
active backbone in every ReID report**, so no number is ever misattributed.

---

## 8. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `no videos found in data/raw_videos` | Run `python scripts/make_sample_videos.py`, or copy your own `.mp4` files in. |
| `manifest not found` | Run `python -m src.cli sample` first. |
| `no frames assigned to split 'test'` | Run `python -m src.cli split`. |
| `resolved_device cpu` + R9 note | Expected without an NVIDIA GPU. Not an error. |
| `torchreid is not installed; falling back` | Expected. `pip install torchreid` for the OSNet backbone. |
| `onnxslim not installed` during export | Handled — the export continues without graph simplification. |
| `UnicodeEncodeError: charmap` on Windows | Fixed in `src/cli.py` (stdout forced to UTF-8). If you see it in your own script, add `sys.stdout.reconfigure(encoding="utf-8")`. |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| Weight download fails | The first run needs internet for `yolo11n.pt` and ResNet18. Behind a proxy, set `HTTPS_PROXY`, or place `yolo11n.pt` in the repo root manually. |
| **`LeakageError` from any command** | **Stop.** A scene or video spans two splits. Per PRD risk R3 every metric computed since the leak is void. Re-run `split`, then `validate`, then re-run the experiments. |
| Everything is slow | Expected on CPU: ~5–13 FPS. Use `--max-frames` to bound runs, and `detection.frame_skip: 2` in the config for the CPU-fallback path. |

---

## 9. Verifying it actually works

Four checks a reviewer can run in under five minutes:

```bash
# 1. The test suite -- hand-computed expected values, no data needed
python -m pytest tests/ -q

# 2. The leakage gate, including proof it fires on a deliberately leaked manifest
python -m pytest tests/test_no_leakage.py -v

# 3. The gallery restoring an identity through an occlusion longer than track_buffer
python -m pytest tests/test_tracker.py::test_gallery_restores_the_original_id_after_a_long_occlusion -v

# 4. Reproducibility -- the same config on the same data gives the same numbers
python -m src.cli eval-det --split test
python -m src.cli eval-det --split test        # identical mAP, identical fingerprint
```

---

## 10. Repository map

```
configs/default.yaml     every tunable; no magic numbers in source
data/
  raw_videos/            input .mp4 files
  gt/                    MOT-format ground truth
  frames/ labels/        sampled frames and YOLO labels
  splits/                train/val/test image+label directories
  manifest.csv           one row per frame, with difficulty attributes
docs/
  annotation_guide.md    READ BEFORE LABELLING
  DATASET_CARD.md        what the data is, and its limitations
  ethics.md              privacy commitments, each naming its module
src/
  config.py  cli.py  cli_phase3.py
  data/      sampler attributes manifest splitter validate_labels mot labels_builder
  models/    detector train export
  tracking/  kalman matching gmc base trackers
  reid/      extractor gallery calibrate
  eval/      detection_metrics tracking_metrics reid_metrics failures report runner
  pipeline/  reader track_video demo analytics cross_camera
tests/       121 tests, expected values hand-derived in the docstrings
reports/     generated markdown + JSON, each with a provenance block
app.py       Streamlit UI
```

---

## 11. Known limitations

Read `README.md` §"Honest limitations" before quoting any number. The short
version:

- **No GPU on the reference machine** → the detector is *not* fine-tuned; the
  committed numbers are the B0 zero-shot baseline. All FPS figures are CPU.
- **ReID backbone is ResNet18, not OSNet** → every ReID report names it.
- **Synthetic backgrounds** → absolute mAP is not comparable to a road-scene
  benchmark.
- **Only `person` and `bus` have ground truth** → the other four classes report
  `no GT` rather than a fabricated number.
- **No objects under 32² px** → M7 reports `n/a`, never `0.0`.
