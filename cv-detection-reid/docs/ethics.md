# Ethics, Privacy & Responsible AI

**Reference:** PRD §17 · **Status:** binding

This is a surveillance-class system. Treating that seriously is part of the
engineering, not an appendix — so each commitment below names the module that
implements it, rather than asserting it.

## Commitments

### No biometric identification
ReID matches **appearance embeddings within a session** — shape, colour,
clothing. There is no identity database, no enrolment, and no face recognition.
This is a designed exclusion (PRD NG5), not a missing feature.

Concretely: `src/reid/gallery.py` holds `{track_id → embedding, last_seen_frame,
class, camera_id}` and nothing else. Entries expire after `gallery_ttl` frames,
nothing is written to disk, and there is no path by which an embedding could be
compared against a person named anywhere.

### Anonymisation
`--blur-faces` (`src/pipeline/demo.py::blur_regions`) blurs faces inside person
boxes and plate-height bands on vehicles.

It blurs the region **whether or not a face detector fires**. A privacy
guarantee that only holds when a detector succeeds is not a guarantee — it
fails exactly in the hard cases where it matters most. The Haar cascade runs as
an *additional* pass, never as the mechanism.

Enable it for any shared output.

### Data minimisation
Only embeddings and box coordinates are retained. Raw crops exist in memory for
the duration of one forward pass and are never written to disk. The results CSV
(`frame, timestamp, track_id, class, conf, x1..y2, reid_restored, camera_id`)
contains no imagery.

### Retention
Gallery entries expire after `reid.gallery.ttl_frames` (default 300 frames,
~10 s at 30 fps). This is a memory bound **and** a retention policy: an identity
the system saw ten seconds ago is no longer recoverable from it.

### Consent and sourcing
The committed dataset is synthetic — object pixels come from two openly-licensed
sample photographs, and **no real people were surveilled to build it**. Any real
footage must be self-captured in public or permitted spaces, or drawn from an
openly licensed dataset, with source and licence recorded in
`docs/DATASET_CARD.md`.

### Bias
Detection recall is reported **sliced by lighting and object scale**, never only
averaged. Uneven performance across visually distinct groups is a known risk
class in person detection; the sliced evaluation is the mechanism for surfacing
it, and any gap found is reported rather than averaged away.

The current dataset **cannot** support a demographic bias audit — its people
come from two photographs. That is a stated limitation of this dataset, not a
claim that the system is unbiased.

### Failure disclosure
The README states plainly where the system fails: heavy crowds, extreme low
light, very small objects, occlusions longer than the gallery TTL, and — on the
current CPU-only, not-yet-fine-tuned configuration — person detection on the
occluder scenes. `reports/failures.md` enumerates diagnosed cases with a
root-cause frequency table.

### Human in the loop
Outputs are decision *support*. The system produces counts, dwell times and
flags. It takes no automated action on any individual, and nothing in the
codebase provides an actuation path.

## What this system must not be used for

- Identifying named individuals, or linking observations to an identity database.
- Tracking specific people across sessions, days, or unrelated deployments.
- Any automated decision affecting an individual without human review.
- Deployment in a space where the people observed have not been informed.

These are limits of intended use. The absence of any biometric database makes
the first two hard by construction; the last two are the operator's obligation.
