# Re-Identification Evaluation (PRD §4.4)

*Generated 2026-09-01T06:13:21+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | a37bb45      |
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
| M14 | Rank-1 accuracy | 0.9683   | ≥ 0.75 | PASS    |
| M15 | Rank-5 accuracy | 0.9841   | ≥ 0.9  | PASS    |
| M16 | ReID mAP        | 0.9148   | ≥ 0.65 | PASS    |

## Post-occlusion recovery per sequence (M17)

| Sequence         | Occlusion events scored | Recovered | Rate | Gallery restores | FPS  |
|------------------|-------------------------|-----------|------|------------------|------|
| scene03_camA.mp4 | 0                       | 0         | n/a  | 1                | 6.69 |
| scene08_camA.mp4 | 0                       | 0         | n/a  | 0                | 6.38 |

## Query/gallery protocol (M14–M16)

| Key              | Value                                                                |
|------------------|----------------------------------------------------------------------|
| queries          | 63                                                                   |
| gallery items    | 63                                                                   |
| identities       | 7                                                                    |
| CMC (rank 1..10) | 0.968, 0.984, 0.984, 0.984, 0.984, 1.000, 1.000, 1.000, 1.000, 1.000 |

## Unrecovered occlusions (the failures worth reading)

*No identity was lost across a scored occlusion.*
