# Detection Evaluation — B0

*Generated 2026-09-01T06:18:00+00:00*

## Provenance

| Key                | Value                                               |
|--------------------|-----------------------------------------------------|
| config fingerprint | f0a93d2104cf                                        |
| git commit         | a37bb45                                             |
| split              | test                                                |
| weights            | yolo11n.pt                                          |
| device             | cpu                                                 |
| device_name        | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| half               | False                                               |
| model_classes      | 80                                                  |
| coco_pretrained    | True                                                |
| mapped_classes     | 6                                                   |
| imgsz              | 640                                                 |
| conf               | 0.25                                                |
| iou                | 0.45                                                |

## Headline metrics — configuration `B0`, split `test`

| ID | Metric                | Measured | Target | Verdict      |
|----|-----------------------|----------|--------|--------------|
| M1 | mAP@0.5               | 0.4147   | ≥ 0.75 | FAIL         |
| M2 | mAP@0.5:0.95          | 0.2997   | ≥ 0.5  | FAIL         |
| M3 | Precision @ conf 0.25 | 0.2674   | ≥ 0.8  | FAIL         |
| M4 | Recall @ conf 0.25    | 0.4435   | ≥ 0.75 | FAIL         |
| M5 | Mean IoU (matched)    | 0.8323   | ≥ 0.78 | PASS         |
| M7 | Small-object AP       | n/a      | ≥ 0.35 | NOT MEASURED |

## Per-class AP (M6)

| Class      | GT boxes | AP@0.5 | AP@0.5:0.95 | M6 (>= 0.60) |
|------------|----------|--------|-------------|--------------|
| bicycle    | 0        | n/a    | n/a         | no GT        |
| bus        | 147      | 0.7098 | 0.5472      | PASS         |
| car        | 0        | n/a    | n/a         | no GT        |
| motorcycle | 0        | n/a    | n/a         | no GT        |
| person     | 216      | 0.1197 | 0.0523      | FAIL         |
| truck      | 0        | n/a    | n/a         | no GT        |

## Object-size bands (M7)

| Band   | AP@0.5              |
|--------|---------------------|
| small  | n/a (no GT in band) |
| medium | 0.1721              |
| large  | 0.5321              |

## Difficulty slices (PRD §13.3)

| Slice                | Images | GT boxes | mAP@0.5 | Recall | Precision |
|----------------------|--------|----------|---------|--------|-----------|
| activity:high        | 31     | 209      | 0.3948  | 0.4354 | 0.2871    |
| activity:low         | 59     | 154      | 0.4466  | 0.4545 | 0.2456    |
| blur:sharp           | 90     | 363      | 0.4147  | 0.4435 | 0.2674    |
| crowded:<=15 objects | 90     | 363      | 0.4147  | 0.4435 | 0.2674    |
| lighting:day         | 53     | 286      | 0.3951  | 0.4126 | 0.2554    |
| lighting:dusk        | 7      | 9        | 0.6906  | 0.8889 | 0.4211    |
| lighting:night       | 30     | 68       | 0.5258  | 0.5147 | 0.2893    |

## Confusion matrix

| GT \ Pred  | person | car | motorcycle | bus | truck | bicycle | background |
|------------|--------|-----|------------|-----|-------|---------|------------|
| person     | 89     | 0   | 0          | 0   | 0     | 0       | 127        |
| car        | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| motorcycle | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| bus        | 0      | 15  | 0          | 66  | 16    | 0       | 50         |
| truck      | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| bicycle    | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| background | 389    | 5   | 0          | 9   | 13    | 0       | 0          |

## Counts

| Key                | Value           |
|--------------------|-----------------|
| images             | 90              |
| ground-truth boxes | 363             |
| predictions        | 6065            |
| TP / FP / FN       | 161 / 441 / 202 |

## Throughput

| Key             | Value  |
|-----------------|--------|
| frames          | 90     |
| fps             | 9.3    |
| latency_p50_ms  | 64.16  |
| latency_p95_ms  | 80.52  |
| latency_mean_ms | 107.57 |
