# Re-Identification Evaluation (PRD §4.4)

*Generated 2026-09-01T06:37:37+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 76ee71c      |
| split              | test         |
| tau_reid           | 0.3          |
| reid_backend       | resnet18     |
| reid_model         | resnet18     |
| embedding_dim      | 512          |
| device             | cpu          |
| crop_size          | 256x128      |

## Headline ReID metrics — backend `resnet18`, tau 0.3, split `test`

| ID  | Metric          | Measured | Target | Verdict |
|-----|-----------------|----------|--------|---------|
| M14 | Rank-1 accuracy | 0.7410   | ≥ 0.75 | FAIL    |
| M15 | Rank-5 accuracy | 0.9353   | ≥ 0.9  | PASS    |
| M16 | ReID mAP        | 0.6395   | ≥ 0.65 | FAIL    |

## Post-occlusion recovery per sequence (M17)

| Sequence         | Events found | Scorable | Recovered | Rate | Gallery restores | FPS  |
|------------------|--------------|----------|-----------|------|------------------|------|
| scene04_camA.mp4 | 1            | 0        | 0         | n/a  | 4                | 3.5  |
| scene07_camA.mp4 | 8            | 0        | 0         | n/a  | 3                | 2.91 |
| scene13_camA.mp4 | 2            | 0        | 0         | n/a  | 3                | 3.05 |

*Events found* are genuine occlusions in the ground truth. *Scorable* are those where the detector saw the object on both sides of the gap — an event that is not scorable is a **detection** failure, and charging it to ReID would measure the wrong component.

## Query/gallery protocol (M14–M16)

| Key              | Value                                                                |
|------------------|----------------------------------------------------------------------|
| queries          | 139                                                                  |
| gallery items    | 141                                                                  |
| identities       | 19                                                                   |
| CMC (rank 1..10) | 0.741, 0.813, 0.863, 0.899, 0.935, 0.950, 0.950, 0.957, 0.957, 0.957 |

## Unrecovered occlusions (the failures worth reading)

*No identity was lost across a scored occlusion.*
