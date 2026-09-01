# Ablation Matrix (PRD §13.2)

*Generated 2026-09-01T06:47:21+00:00*

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
