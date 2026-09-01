# Failure Gallery (PRD §13.5)

*Generated 2026-09-01T06:35:58+00:00*

## Provenance

| Key                | Value        |
|--------------------|--------------|
| config fingerprint | f0a93d2104cf |
| git commit         | 8194df7      |
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
