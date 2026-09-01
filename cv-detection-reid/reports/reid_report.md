# Re-Identification Evaluation (PRD §4.4)

*Generated 2026-09-01T06:23:56+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | e001cbd      |
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

| Sequence         | Occlusion events scored | Recovered | Rate | Gallery restores | FPS  |
|------------------|-------------------------|-----------|------|------------------|------|
| scene04_camA.mp4 | 0                       | 0         | n/a  | 4                | 2.57 |
| scene07_camA.mp4 | 0                       | 0         | n/a  | 3                | 1.97 |
| scene13_camA.mp4 | 0                       | 0         | n/a  | 3                | 4.46 |

## Query/gallery protocol (M14–M16)

| Key              | Value                                                                |
|------------------|----------------------------------------------------------------------|
| queries          | 139                                                                  |
| gallery items    | 141                                                                  |
| identities       | 19                                                                   |
| CMC (rank 1..10) | 0.741, 0.813, 0.863, 0.899, 0.935, 0.950, 0.950, 0.957, 0.957, 0.957 |

## Unrecovered occlusions (the failures worth reading)

*No identity was lost across a scored occlusion.*
