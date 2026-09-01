"""Detector wrapper around Ultralytics YOLO.

PRD 9.1. Two responsibilities that justify a wrapper rather than calling
`YOLO()` inline everywhere:

1. **Class-space mapping.** The B0 baseline (PRD 13.2) runs a *COCO-pretrained*
   model against *our* six-class label space. COCO's `car` is id 2, ours is id
   1; without an explicit remap the zero-shot baseline scores near zero and the
   domain-gap number is meaningless. The map lives in `dataset.coco_id_map` and
   is applied here, once.

2. **A stable `PredBox` contract.** The tracker (Phase 2), the ReID extractor
   (Phase 3) and the metrics harness all consume detections. They consume this
   dataclass, not an Ultralytics `Results` object, so swapping the detector for
   an ONNX session in Phase 3.3 changes one file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import Config
from ..eval.detection_metrics import PredBox
from ..utils.device import DeviceInfo, resolve_device
from ..utils.logging import get_logger

log = get_logger("models.detector")

# A model whose head has this many classes is COCO-pretrained and untouched.
COCO_N_CLASSES = 80


@dataclass
class DetectorStats:
    frames: int = 0
    total_ms: float = 0.0
    per_frame_ms: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / (self.total_ms / 1000.0) if self.total_ms > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        arr = np.array(self.per_frame_ms) if self.per_frame_ms else np.array([0.0])
        return {
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
            "latency_mean_ms": round(float(arr.mean()), 2),
        }


class Detector:
    """Loads a YOLO checkpoint and emits `PredBox` in our class space."""

    def __init__(self, cfg: Config, weights: str | Path | None = None):
        from ultralytics import YOLO  # imported lazily: heavy, and CLI `env` must work without it

        self.cfg = cfg
        self.weights = str(weights or cfg.detection.weights)
        self.device: DeviceInfo = resolve_device(cfg.detection.device)
        self.model = YOLO(self.weights)
        self.model_names: dict[int, str] = dict(self.model.names)
        self.class_names: tuple[str, ...] = cfg.dataset.classes
        self.stats = DetectorStats()

        self.is_coco = len(self.model_names) == COCO_N_CLASSES
        self._id_map = self._build_id_map()
        if self.is_coco:
            kept = sorted({self.class_names[v] for v in self._id_map.values()})
            log.info(
                f"COCO-pretrained head detected; mapping {len(self._id_map)} COCO ids "
                f"onto our classes: {kept}. Everything else is discarded.",
                extra={"event": "coco_class_map", "n_mapped": len(self._id_map)},
            )

        # FP16 is a CUDA-only win; forcing it on CPU is either ignored or slower.
        self.half = bool(cfg.detection.half and self.device.is_gpu)
        if cfg.detection.half and not self.device.is_gpu:
            log.warning(
                "detection.half requested but device is not a GPU; running FP32",
                extra={"event": "half_disabled"},
            )

    def _build_id_map(self) -> dict[int, int]:
        """model class id -> our class id."""
        if self.is_coco:
            return {
                int(coco_id): self.class_names.index(name)
                for coco_id, name in self.cfg.dataset.coco_id_map.items()
                if name in self.class_names
            }
        # A fine-tuned model already predicts our classes; match on name so a
        # reordered `dataset.classes` fails visibly instead of silently
        # relabelling every box.
        out: dict[int, int] = {}
        for mid, name in self.model_names.items():
            if name in self.class_names:
                out[int(mid)] = self.class_names.index(name)
            else:
                log.warning(
                    f"model class {name!r} (id {mid}) is not in dataset.classes; dropped",
                    extra={"event": "unmapped_model_class", "model_class": name},
                )
        return out

    # -- inference ---------------------------------------------------------
    def predict(
        self,
        source: np.ndarray | str | Path,
        image_id: str,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> list[PredBox]:
        """Run the detector on one image and return boxes in our class space."""
        t0 = time.perf_counter()
        results = self.model.predict(
            source=source,
            conf=conf if conf is not None else self.cfg.detection.conf,
            iou=iou if iou is not None else self.cfg.detection.iou,
            imgsz=imgsz or self.cfg.detection.imgsz,
            max_det=self.cfg.detection.max_det,
            device=self.device.device,
            verbose=False,
            # Ultralytics deprecated `half`; only pass it when it is actually
            # requested so a CPU run does not emit a warning per frame.
            **({"half": True} if self.half else {}),
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        self.stats.frames += 1
        self.stats.total_ms += elapsed
        self.stats.per_frame_ms.append(elapsed)

        out: list[PredBox] = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                mapped = self._id_map.get(int(k))
                if mapped is None:
                    continue  # a COCO class we do not model, e.g. traffic light
                out.append(
                    PredBox(
                        image_id=image_id,
                        cls_id=mapped,
                        xyxy=(float(x1), float(y1), float(x2), float(y2)),
                        conf=float(c),
                    )
                )
        return out

    def predict_batch(
        self, sources: Sequence[np.ndarray | str | Path], image_ids: Sequence[str], **kw: Any
    ) -> dict[str, list[PredBox]]:
        """Convenience loop. Batched GPU inference lands with the export arm in Phase 3.3."""
        if len(sources) != len(image_ids):
            raise ValueError("sources and image_ids must be the same length")
        return {iid: self.predict(src, iid, **kw) for src, iid in zip(sources, image_ids)}

    def describe(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "device": self.device.device,
            "device_name": self.device.name,
            "half": self.half,
            "model_classes": len(self.model_names),
            "coco_pretrained": self.is_coco,
            "mapped_classes": len(self._id_map),
            "imgsz": self.cfg.detection.imgsz,
            "conf": self.cfg.detection.conf,
            "iou": self.cfg.detection.iou,
        }
