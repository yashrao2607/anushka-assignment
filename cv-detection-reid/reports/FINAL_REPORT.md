# cv-detection-reid — Consolidated Results

*Generated 2026-09-01T06:50:21+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 8dcc920      |

## Detection  —  `reports/eval_report.md`

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

## Tracking  —  `reports/tracking_report.md`

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

## ReID threshold calibration  —  `reports/calibration.md`

## Provenance

| Key                | Value                                                                                                                                                                                                                                                                        |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| config fingerprint | f0a93d2104cf                                                                                                                                                                                                                                                                 |
| git commit         | a37bb45                                                                                                                                                                                                                                                                      |
| split              | trainval                                                                                                                                                                                                                                                                     |
| videos             | scene01_camA.mp4, scene02_camA.mp4, scene03_camA.mp4, scene05_camA.mp4, scene06_camA.mp4, scene06_camB.mp4, scene08_camA.mp4, scene09_camA.mp4, scene10_camA.mp4, scene11_camA.mp4, scene12_camA.mp4, scene14_camA.mp4, scene15_camA.mp4, scene16_camA.mp4, scene16_camB.mp4 |

## Chosen threshold

| Key               | Value    |
|-------------------|----------|
| tau_reid          | 0.3      |
| F1 at tau         | 0.5435   |
| ReID backend      | resnet18 |
| occlusion events  | 41       |
| candidate pairs   | 163      |
| calibration split | trainval |

## Sweep

| tau  | TP | FP (identity merge) | FN (new id issued) | precision | recall | F1    |
|------|----|---------------------|--------------------|-----------|--------|-------|
| 0.05 | 0  | 0                   | 40                 | 0.000     | 0.000  | 0.000 |
| 0.10 | 1  | 0                   | 39                 | 1.000     | 0.025  | 0.049 |
| 0.15 | 1  | 0                   | 39                 | 1.000     | 0.025  | 0.049 |
| 0.20 | 8  | 9                   | 32                 | 0.471     | 0.200  | 0.281 |
| 0.25 | 20 | 14                  | 20                 | 0.588     | 0.500  | 0.540 |
| 0.30 | 25 | 27                  | 15                 | 0.481     | 0.625  | 0.543 |
| 0.35 | 33 | 51                  | 7                  | 0.393     | 0.825  | 0.532 |
| 0.40 | 38 | 82                  | 2                  | 0.317     | 0.950  | 0.475 |
| 0.45 | 40 | 113                 | 0                  | 0.261     | 1.000  | 0.414 |
| 0.50 | 40 | 121                 | 0                  | 0.248     | 1.000  | 0.398 |
| 0.55 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.60 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.65 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.70 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.75 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.80 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.85 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.90 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 0.95 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |
| 1.00 | 40 | 123                 | 0                  | 0.245     | 1.000  | 0.394 |

## Reading this

A **false positive** here is an identity *merge*: two different objects collapsed into one id. A **false negative** is a genuine return rejected, which issues a new id and inflates the unique count. F1 balances the two; the curve is published so the shape, not only the argmax, is auditable.

![calibration](figures/tau_calibration.png)

## Re-identification  —  `reports/reid_report.md`

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

## Cross-camera hand-off  —  `reports/cross_camera.md`

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

## Ablation matrix  —  `reports/ablation.md`

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 05b18e5      |
| split              | test         |

## Measured rows

| #  | Configuration                                 | mAP50 | mAP50-95 | IDF1  | HOTA  | IDSW | ID restores | FPS  |
|----|-----------------------------------------------|-------|----------|-------|-------|------|-------------|------|
| B0 | COCO-pretrained, zero fine-tuning, no tracker | 0.415 | 0.300    | -     | -     | -    | -           | 8.4  |
| B1 | B0 + IoU-only tracker                         | 0.415 | 0.300    | 0.128 | 0.195 | 133  | -           | 13.0 |
| B3 | B0 + ByteTrack (Kalman + two-stage)           | 0.415 | 0.300    | 0.273 | 0.285 | 3    | -           | 10.7 |
| B4 | B0 + BoT-SORT (GMC, no ReID)                  | 0.415 | 0.300    | 0.259 | 0.278 | 3    | -           | 7.5  |
| B5 | B4 + OSNet/ReID appearance cost               | 0.415 | 0.300    | 0.219 | 0.249 | 10   | -           | 4.8  |
| B6 | B5 + ReID gallery re-association              | 0.415 | 0.300    | 0.222 | 0.249 | 11   | 4           | 5.1  |

## Notes

Rows B2/B3f/B4f require fine-tuned weights (`--weights runs/<id>/weights/best.pt`); B5-B10 (ReID gallery, imgsz 960, YOLO11m, hard negatives, TensorRT) land in Phase 3.

FPS is measured on this machine's **CPU** (PRD R9: no GPU present), so it is the honest worst case rather than the M19 GPU target.

## Export & parity  —  `reports/export.md`

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 9ebe158      |

## Result

| Key              | Value        |
|------------------|--------------|
| source weights   | yolo11n.pt   |
| format           | onnx         |
| output           | yolo11n.onnx |
| size (MB)        | 10.21        |
| M22 target       | <= 25 MB     |
| M22 verdict      | PASS         |
| PyTorch mAP@0.5  | 0.4218       |
| exported mAP@0.5 | 0.4544       |
| absolute delta   | 0.0326       |
| NFR-6 tolerance  | 0.01         |
| NFR-6 verdict    | FAIL         |
| PyTorch FPS      | 6.83         |
| exported FPS     | 11.91        |
| speedup          | 1.74         |

## Note

FPS here is CPU-only (PRD R9). The TensorRT FP16/INT8 arm of EXP-8 needs CUDA and is documented rather than measured.

## Failure gallery  —  `reports/failures.md`

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 9ebe158      |
| split              | test         |
| weights            | yolo11n.pt   |

## Root-cause frequency — this table is the next iteration's work plan

| Root cause                | Count | Remediation                                                            |
|---------------------------|-------|------------------------------------------------------------------------|
| false_positive_background | 394   | hard-negative mining round (7.5.1, EXP-6)                              |
| localisation_drift        | 122   | more box-loss weight; check annotation tightness (label protocol 7.2)  |
| occlusion_miss            | 41    | lower conf threshold; two-stage association already mitigates (US-3.2) |
| class_confusion           | 31    | targeted collection for the confused pair; class-weighted loss (7.4)   |
| duplicate_box_nms         | 22    | tune NMS IoU; class-wise NMS is already on (EXP-10)                    |
| low_light_miss            | 14    | night footage + low-light gamma augmentation (EXP-5, R10)              |

## Saved cases

| #  | Cause                     | Image                | Detail                                          | Lighting | Blur  | IoU   | Conf  |
|----|---------------------------|----------------------|-------------------------------------------------|----------|-------|-------|-------|
| 0  | occlusion_miss            | scene04_camA_f000000 | no overlapping prediction in a clear frame      | day      | sharp | 0.008 | 0.0   |
| 1  | false_positive_background | scene04_camA_f000000 | conf 0.78 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.784 |
| 2  | localisation_drift        | scene04_camA_f000015 | box found but IoU only 0.20                     | day      | sharp | 0.199 | 0.0   |
| 3  | class_confusion           | scene04_camA_f000030 | predicted truck on a bus                        | day      | sharp | 0.984 | 0.356 |
| 4  | duplicate_box_nms         | scene04_camA_f000060 | conf 0.41 truck with no unclaimed ground truth  | day      | sharp | 0.0   | 0.41  |
| 5  | low_light_miss            | scene13_camA_f000000 | brightness 15                                   | night    | sharp | 0.0   | 0.0   |
| 6  | occlusion_miss            | scene04_camA_f000015 | no overlapping prediction in a clear frame      | day      | sharp | 0.004 | 0.0   |
| 7  | false_positive_background | scene04_camA_f000000 | conf 0.26 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.259 |
| 8  | localisation_drift        | scene04_camA_f000030 | box found but IoU only 0.42                     | day      | sharp | 0.416 | 0.0   |
| 9  | class_confusion           | scene04_camA_f000045 | predicted truck on a bus                        | day      | sharp | 0.981 | 0.492 |
| 10 | duplicate_box_nms         | scene04_camA_f000075 | conf 0.38 bus with no unclaimed ground truth    | day      | sharp | 0.0   | 0.377 |
| 11 | low_light_miss            | scene13_camA_f000090 | brightness 16                                   | night    | sharp | 0.675 | 0.0   |
| 12 | occlusion_miss            | scene04_camA_f000045 | no overlapping prediction in a clear frame      | day      | sharp | 0.001 | 0.0   |
| 13 | false_positive_background | scene04_camA_f000015 | conf 0.84 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.84  |
| 14 | localisation_drift        | scene04_camA_f000030 | box found but IoU only 0.50                     | day      | sharp | 0.499 | 0.0   |
| 15 | class_confusion           | scene04_camA_f000075 | predicted truck on a bus                        | day      | sharp | 0.966 | 0.412 |
| 16 | duplicate_box_nms         | scene04_camA_f000075 | conf 0.36 truck with no unclaimed ground truth  | day      | sharp | 0.0   | 0.364 |
| 17 | low_light_miss            | scene13_camA_f000090 | brightness 16                                   | night    | sharp | 0.0   | 0.0   |
| 18 | occlusion_miss            | scene04_camA_f000075 | no overlapping prediction in a clear frame      | day      | sharp | 0.0   | 0.0   |
| 19 | false_positive_background | scene04_camA_f000015 | conf 0.67 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.669 |
| 20 | localisation_drift        | scene04_camA_f000045 | box found but IoU only 0.28                     | day      | sharp | 0.277 | 0.0   |
| 21 | class_confusion           | scene04_camA_f000120 | predicted truck on a bus                        | day      | sharp | 0.844 | 0.35  |
| 22 | duplicate_box_nms         | scene04_camA_f000090 | conf 0.65 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.652 |
| 23 | low_light_miss            | scene13_camA_f000120 | brightness 16                                   | night    | sharp | 0.0   | 0.0   |
| 24 | occlusion_miss            | scene04_camA_f000090 | no overlapping prediction in a clear frame      | day      | sharp | 0.0   | 0.0   |
| 25 | false_positive_background | scene04_camA_f000015 | conf 0.65 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.645 |
| 26 | localisation_drift        | scene04_camA_f000045 | box found but IoU only 0.48                     | day      | sharp | 0.477 | 0.0   |
| 27 | class_confusion           | scene04_camA_f000135 | predicted car on a bus                          | day      | sharp | 0.629 | 0.294 |
| 28 | duplicate_box_nms         | scene04_camA_f000090 | conf 0.34 truck with no unclaimed ground truth  | day      | sharp | 0.0   | 0.341 |
| 29 | low_light_miss            | scene13_camA_f000150 | brightness 16                                   | night    | sharp | 0.075 | 0.0   |
| 30 | occlusion_miss            | scene04_camA_f000105 | no overlapping prediction in a clear frame      | day      | sharp | 0.0   | 0.0   |
| 31 | false_positive_background | scene04_camA_f000030 | conf 0.80 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.796 |
| 32 | localisation_drift        | scene04_camA_f000060 | box found but IoU only 0.40                     | day      | sharp | 0.396 | 0.0   |
| 33 | class_confusion           | scene04_camA_f000210 | predicted truck on a bus                        | day      | sharp | 0.968 | 0.315 |
| 34 | duplicate_box_nms         | scene04_camA_f000105 | conf 0.67 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.669 |
| 35 | low_light_miss            | scene13_camA_f000165 | brightness 16                                   | night    | sharp | 0.079 | 0.0   |
| 36 | occlusion_miss            | scene04_camA_f000120 | no overlapping prediction in a clear frame      | day      | sharp | 0.0   | 0.0   |
| 37 | false_positive_background | scene04_camA_f000030 | conf 0.71 person with no unclaimed ground truth | day      | sharp | 0.0   | 0.714 |
| 38 | localisation_drift        | scene04_camA_f000075 | box found but IoU only 0.42                     | day      | sharp | 0.415 | 0.0   |
| 39 | class_confusion           | scene04_camA_f000210 | predicted truck on a bus                        | day      | sharp | 0.552 | 0.281 |

## How causes are assigned

Automatically, from the geometry of each failure — size band, lighting and blur attributes from the manifest, and whether an overlapping-but-below-threshold box existed. Hand-labelled causes drift with the labeller, and the frequency table is only useful across iterations if the labelling rule is fixed.
