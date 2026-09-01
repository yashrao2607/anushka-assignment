# Detection Evaluation — B0

*Generated 2026-09-01T05:57:46+00:00*

## Provenance

| Key                | Value                                               |
|--------------------|-----------------------------------------------------|
| config fingerprint | f0a93d2104cf                                        |
| git commit         | a3fdbe1                                             |
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
| M1 | mAP@0.5               | 0.5867   | ≥ 0.75 | FAIL         |
| M2 | mAP@0.5:0.95          | 0.5039   | ≥ 0.5  | PASS         |
| M3 | Precision @ conf 0.25 | 0.3910   | ≥ 0.8  | FAIL         |
| M4 | Recall @ conf 0.25    | 0.7093   | ≥ 0.75 | FAIL         |
| M5 | Mean IoU (matched)    | 0.8970   | ≥ 0.78 | PASS         |
| M7 | Small-object AP       | n/a      | ≥ 0.35 | NOT MEASURED |

## Per-class AP (M6)

| Class      | GT boxes | AP@0.5 | AP@0.5:0.95 | M6 (>= 0.60) |
|------------|----------|--------|-------------|--------------|
| bicycle    | 0        | n/a    | n/a         | no GT        |
| bus        | 76       | 0.8063 | 0.7444      | PASS         |
| car        | 0        | n/a    | n/a         | no GT        |
| motorcycle | 0        | n/a    | n/a         | no GT        |
| person     | 96       | 0.3671 | 0.2635      | FAIL         |
| truck      | 0        | n/a    | n/a         | no GT        |

## Object-size bands (M7)

| Band   | AP@0.5              |
|--------|---------------------|
| small  | n/a (no GT in band) |
| medium | 0.3568              |
| large  | 0.4034              |

## Difficulty slices (PRD §13.3)

| Slice                | Images | GT boxes | mAP@0.5 | Recall | Precision |
|----------------------|--------|----------|---------|--------|-----------|
| activity:high        | 11     | 33       | 0.5486  | 0.7879 | 0.3467    |
| activity:low         | 49     | 139      | 0.6025  | 0.6906 | 0.4051    |
| blur:blurred         | 30     | 101      | 0.4315  | 0.5446 | 0.2926    |
| blur:sharp           | 30     | 71       | 0.8292  | 0.9437 | 0.5403    |
| crowded:<=15 objects | 60     | 172      | 0.5867  | 0.7093 | 0.3910    |
| lighting:day         | 30     | 101      | 0.4315  | 0.5446 | 0.2926    |
| lighting:night       | 30     | 71       | 0.8292  | 0.9437 | 0.5403    |

## Confusion matrix

| GT \ Pred  | person | car | motorcycle | bus | truck | bicycle | background |
|------------|--------|-----|------------|-----|-------|---------|------------|
| person     | 68     | 0   | 0          | 0   | 0     | 0       | 28         |
| car        | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| motorcycle | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| bus        | 0      | 11  | 0          | 51  | 3     | 0       | 11         |
| truck      | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| bicycle    | 0      | 0   | 0          | 0   | 0     | 0       | 0          |
| background | 172    | 2   | 0          | 3   | 2     | 0       | 0          |

## Counts

| Key                | Value          |
|--------------------|----------------|
| images             | 60             |
| ground-truth boxes | 172            |
| predictions        | 2892           |
| TP / FP / FN       | 122 / 190 / 50 |

## Throughput

| Key             | Value |
|-----------------|-------|
| frames          | 60    |
| fps             | 10.87 |
| latency_p50_ms  | 57.95 |
| latency_p95_ms  | 87.21 |
| latency_mean_ms | 92.01 |
