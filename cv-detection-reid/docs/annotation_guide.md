# Annotation Guide

**Project:** P2-CV-DET-REID · **Reference:** PRD §7.2 · **Status:** binding for all labelling

These rules are written down **before** labelling starts. Ambiguity discovered
half-way through a labelling round corrupts a dataset: the frames labelled
before the rule was settled and the frames labelled after it are two different
datasets wearing the same name, and no amount of later model work recovers it.

Every rule below is enforced or checked somewhere in code, and the check is
named. A rule nobody verifies is a suggestion.

---

## 1. Tooling and format

| Item | Value |
|---|---|
| Tool | CVAT or Roboflow (both export YOLO format) |
| Export format | YOLO: one `.txt` per image, `class_id cx cy w h` |
| Coordinates | **Normalised to [0, 1]** against image width/height |
| Class ids | The index in `configs/default.yaml → dataset.classes`. **Never reorder that list** without remapping every existing label file. |
| Validation | `python -m src.cli validate` — checks field count, class range, normalisation, degenerate boxes and out-of-frame boxes |

Model-assisted pre-labelling is expected (PRD R1): run the COCO-pretrained
detector to generate boxes, then **correct** them. Correcting is 3–5× faster
than drawing from scratch. Two cautions:

- Delete the confidence column before export. A stray sixth field is the most
  common export error and is caught as `bad_field_count`.
- Pre-labelling biases the annotator towards accepting the model's mistakes,
  especially its systematic ones. Every pre-labelled frame is reviewed
  box-by-box, not skimmed.

---

## 2. The rules

### Rule 1 — Box the full **visible** extent
Draw the box tightly around the pixels you can actually see. Do **not**
extrapolate the hidden part of a partly occluded object. A box drawn around
where you imagine the object continues teaches the detector to predict boxes
larger than the evidence supports, which lowers IoU on every future match.

### Rule 2 — Objects occluded more than 70% are `ignore`
Not labelled as a normal object, and not deleted. An `ignore` region is
excluded from the loss **and** from the metrics: a detection landing on one is
scored neither as a true positive nor as a false positive.

Both alternatives are worse. Labelling a barely-visible sliver as a full object
teaches the model to hallucinate; deleting it turns a correct detection into a
false positive and punishes the model for being right.

*Enforced by:* `src/data/mot.py::IGNORE_VISIBILITY` (0.30 visibility), applied
in `to_yolo_lines`; honoured in `detection_metrics.match_predictions` and
`tracking_metrics.clear_mot`.

### Rule 3 — Minimum box size 8×8 px
Smaller goes to `ignore`. Below roughly this size there is no appearance
information left to learn from, and the annotation itself is unreliable —
two annotators will disagree on a 5-pixel blob far more often than they agree.

### Rule 4 — Truncated objects at the frame edge **are** labelled
Clip the box to the frame boundary. A car half-out of the frame is a car, and
excluding them teaches the detector that objects vanish at the border — which
is precisely where objects enter and leave in every real deployment.

### Rule 5 — Reflections, screens and posters are **not** labelled
A person on a billboard is not a person in the scene. Add these frames to the
**hard-negative** set instead (PRD §7.5.1), where they actively suppress the
false positives they would otherwise cause.

### Rule 6 — One class per box, with the ambiguous boundaries written down
The `van` problem is the usual one. The rule, decided once:

| Situation | Class |
|---|---|
| Passenger van, minivan, SUV — windows along the passenger area, car-like body | `car` |
| Panel van, box van, pickup with a cargo bed, any goods vehicle | `truck` |
| Vehicle carrying passengers with more than ~8 seats, bus body | `bus` |
| Two wheels, engine, rider straddling | `motorcycle` |
| Two wheels, pedals, no engine | `bicycle` |
| A rider **on** a bicycle or motorcycle | Two boxes: one `person`, one for the vehicle |

The last row matters and is easy to get wrong in both directions. The person
and the vehicle are separate objects with separate trajectories; merging them
makes the tracker's job incoherent when the rider dismounts.

---

## 3. Quality control (US-1.3)

- A **200-frame subset is annotated twice**, independently.
- Agreement is reported as IoU-based label consistency; the acceptance bar is
  **≥ 0.85**.
- Every disagreement is resolved by discussion, and the resolution is written
  back into this guide as a new row or example. A disagreement that is settled
  but not written down will recur.
- Every `ignore` decision is logged so it can be audited later — the ignore
  rule is the one with the most room for silent inconsistency.

---

## 4. Attributes captured at annotation time (PRD §7.6)

These are recorded per frame in `data/manifest.csv` and **cannot be
reconstructed afterwards**. They are what makes the difficulty-sliced
evaluation (§13.3) possible.

| Attribute | Values | How it is obtained |
|---|---|---|
| `lighting` | day / dusk / night | Automatic: mean HSV-V, thresholds in config |
| `blur_level` | sharp / blurred | Automatic: contrast-normalised Laplacian variance |
| `occlusion_level` | none / partial / heavy | Annotator, or automatic from crowding |
| `weather` | clear / rain | Annotator |
| `n_objects`, `classes` | — | Derived from the labels by the validator |

The automatic ones are computed by `src/data/attributes.py` at sampling time.
Note that the blur measure is **contrast-normalised**: the raw variance of the
Laplacian rates dark frames as blurred, which would silently merge the "night"
and "blurred" slices. See the docstring in that module for the measured
numbers behind the threshold.

---

## 5. Splitting — the rule that outranks all the others

Splits are assigned **by scene**, never by frame (PRD Principle 4, Risk R3
rated *Critical*). Adjacent frames are ~99% redundant; a random frame split
puts near-duplicates of the test set into training and inflates every metric
in the project.

Two camera views of the same scene belong to the **same** scene and therefore
to the same split. Naming convention: `<scene>_<camera>.mp4`, e.g.
`scene06_camA.mp4` and `scene06_camB.mp4` are one scene.

*Enforced by:* `tests/test_no_leakage.py`, which fails the build on any scene
or video spanning two splits. **If it ever fails, every metric produced since
the leak was introduced is void and the experiments must be re-run.**
