# Tracking Evaluation — botsort

*Generated 2026-09-01T05:58:45+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | a3fdbe1      |
| tracker            | botsort      |
| weights            | yolo11n.pt   |
| mean FPS           | 12.16        |

## Headline tracking metrics — tracker `botsort`, split `test` (pooled)

| ID   | Metric                  | Measured | Target | Verdict |
|------|-------------------------|----------|--------|---------|
| M8   | MOTA                    | -0.2420  | ≥ 0.65 | FAIL    |
| M9   | IDF1                    | 0.4615   | ≥ 0.7  | FAIL    |
| M10  | HOTA                    | 0.4550   | ≥ 0.55 | FAIL    |
| M11  | ID switches / 1k frames | 0.0000   | ≤ 15.0 | PASS    |
| M12  | MT ratio                | 0.4286   | ≥ 0.6  | FAIL    |
| M12b | ML ratio                | 0.2857   | ≤ 0.15 | FAIL    |
| M13  | Fragmentation (avg)     | 0.2857   | ≤ 2.0  | PASS    |

## Per-sequence

| Sequence         | MOTA   | IDF1  | HOTA  | IDSW | MT | ML | Frag | FPS   |
|------------------|--------|-------|-------|------|----|----|------|-------|
| scene03_camA.mp4 | 0.222  | 0.681 | 0.597 | 0    | 2  | 0  | 0.67 | 11.3  |
| scene08_camA.mp4 | -0.605 | 0.272 | 0.325 | 0    | 1  | 2  | 0.00 | 13.02 |

## Error breakdown

| Key                     | Value           |
|-------------------------|-----------------|
| TP                      | 497             |
| FP                      | 723             |
| FN                      | 437             |
| ID switches             | 0               |
| GT trajectories         | 7               |
| predicted trajectories  | 11              |
| MOTP (mean matched IoU) | 0.8711          |
| DetA / AssA             | 0.2805 / 0.7443 |
