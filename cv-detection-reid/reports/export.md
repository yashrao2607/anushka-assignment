# Export & Parity (PRD EXP-8, NFR-6, M22)

*Generated 2026-09-01T06:39:34+00:00*

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
