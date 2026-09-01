# Cross-Camera Re-Identification (PRD §9.4, M18)

*Generated 2026-09-01T06:35:36+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 35d0b15      |
| tau_reid           | 0.3          |
| reid_backend       | resnet18     |
| reid_model         | resnet18     |
| embedding_dim      | 512          |
| device             | cpu          |
| crop_size          | 256x128      |

## Headline (M18)

| ID  | Metric                  | Measured | Target | Verdict |
|-----|-------------------------|----------|--------|---------|
| M18 | Cross-camera match rate | 1.0000   | ≥ 0.6  | PASS    |

## Per camera pair

| Camera A         | Camera B         | Identities scored | Matched | Rate  |
|------------------|------------------|-------------------|---------|-------|
| scene06_camA.mp4 | scene06_camB.mp4 | 4                 | 4       | 1.000 |
| scene16_camA.mp4 | scene16_camB.mp4 | 4                 | 4       | 1.000 |

## Per identity

| Cam-A track | Expected cam-B track | Predicted | Cosine distance | Matched |
|-------------|----------------------|-----------|-----------------|---------|
| 1           | 1                    | 1         | 0.0512          | yes     |
| 3           | 3                    | 3         | 0.1353          | yes     |
| 4           | 4                    | 4         | 0.0803          | yes     |
| 2           | 2                    | 2         | 0.0538          | yes     |
| 1           | 1                    | 1         | 0.0138          | yes     |
| 3           | 3                    | 3         | 0.006           | yes     |
| 2           | 2                    | 2         | 0.0066          | yes     |
| 4           | 4                    | 4         | 0.0171          | yes     |

## Design note

Cross-camera matching reuses the occlusion-recovery gallery unchanged — same cosine distance, same class gate, same calibrated `tau_reid`, with a `camera_id` field and a wider temporal window (PRD §9.4). No new machinery was added for the hand-off, which is the design claim this table tests.

## How hard this test actually is

**Read this before quoting the number.** On the bundled two-view scenes the second camera is the same scene under a horizontal pan, so the two views share lighting, scale and viewpoint. That makes the hand-off substantially easier than a real camera pair, where viewpoint, exposure and colour balance all differ — the conditions a ReID-trained backbone exists to handle and an ImageNet backbone does not.

A high rate here demonstrates that the **mechanism** works end to end: embeddings are built per camera, gated by class, compared by cosine distance and thresholded at the calibrated `tau_reid`. It does not establish the rate that would hold on genuinely different viewpoints. Re-run this command on a real two-camera clip before treating M18 as a deployment number.
