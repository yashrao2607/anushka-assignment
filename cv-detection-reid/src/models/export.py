"""ONNX export and the parity check.

PRD NFR-6 and EXP-8: "exported mAP within 1% of PyTorch", verified in CI.

The parity check is the point of this module. An export that loads and runs is
not an export that is *correct* -- opset mismatches, a wrong `imgsz`, or a
silently different NMS produce a model that returns plausible boxes with
different numbers. So the exported model is re-scored through the same metrics
harness on the same split, and the two mAPs are compared. That comparison is
the deliverable; the `.onnx` file on its own proves nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..utils.device import resolve_device
from ..utils.logging import get_logger

log = get_logger("models.export")

# PRD NFR-6: the tolerance the build gate enforces.
PARITY_TOLERANCE = 0.01


@dataclass
class ExportResult:
    weights: str
    fmt: str
    output: str = ""
    size_mb: float = 0.0
    ok: bool = False
    error: str = ""
    pytorch_map50: float | None = None
    exported_map50: float | None = None
    delta: float | None = None
    parity_ok: bool | None = None
    timing: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def export_model(
    cfg: Config,
    weights: str | Path | None = None,
    fmt: str = "onnx",
    half: bool = False,
    imgsz: int | None = None,
    simplify: bool = True,
) -> ExportResult:
    """Export the detector. Returns a result rather than raising on failure.

    A missing exporter dependency is a reportable state, not a crash: the rest
    of the deliverables do not depend on the export arm, and a hard failure
    here would take the whole report down with it.
    """
    from ultralytics import YOLO

    src = str(weights or cfg.detection.weights)
    device = resolve_device(cfg.detection.device)
    result = ExportResult(weights=src, fmt=fmt)

    if half and not device.is_gpu:
        # FP16 export on a CPU-only box produces a model that no CPU runtime
        # will execute efficiently, and the "half the size" claim would be
        # accompanied by an unmeasurable speed number.
        log.warning("FP16 export requested without a GPU; exporting FP32 instead",
                    extra={"event": "half_export_skipped"})
        half = False

    try:
        model = YOLO(src)
        out = model.export(
            format=fmt,
            imgsz=imgsz or cfg.detection.imgsz,
            half=half,
            simplify=simplify,
            device=device.device,
            verbose=False,
        )
        path = Path(str(out))
        result.output = str(path)
        result.ok = path.exists()
        if result.ok:
            result.size_mb = round(path.stat().st_size / 1024**2, 2)
            log.info(f"exported {fmt} -> {path.name} ({result.size_mb} MB)",
                     extra={"event": "export_done", "format": fmt, "size_mb": result.size_mb})
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        log.warning(f"export to {fmt} failed: {result.error}", extra={"event": "export_failed"})
    return result


def verify_parity(
    cfg: Config,
    pytorch_weights: str | Path,
    exported_path: str | Path,
    split: str = "test",
    limit: int | None = None,
    tolerance: float = PARITY_TOLERANCE,
) -> ExportResult:
    """Score both models through the same harness and compare mAP@0.5."""
    from ..eval.runner import run_detection_eval

    result = ExportResult(weights=str(pytorch_weights), fmt=Path(exported_path).suffix.lstrip("."),
                          output=str(exported_path), ok=True)
    try:
        torch_run = run_detection_eval(cfg, split=split, weights=pytorch_weights,
                                       limit=limit, with_slices=False)
        onnx_run = run_detection_eval(cfg, split=split, weights=exported_path,
                                      limit=limit, with_slices=False)
    except Exception as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.pytorch_map50 = torch_run.overall.map50
    result.exported_map50 = onnx_run.overall.map50
    result.delta = round(abs(result.pytorch_map50 - result.exported_map50), 4)
    result.parity_ok = result.delta <= tolerance
    result.timing = {
        "pytorch_fps": torch_run.timing.get("fps"),
        "exported_fps": onnx_run.timing.get("fps"),
        "speedup": (
            round(onnx_run.timing["fps"] / torch_run.timing["fps"], 2)
            if torch_run.timing.get("fps") else None
        ),
    }
    log.info(
        f"parity: PyTorch mAP50 {result.pytorch_map50:.4f} vs exported "
        f"{result.exported_map50:.4f} (delta {result.delta:.4f}, "
        f"{'PASS' if result.parity_ok else 'FAIL'} at tolerance {tolerance})",
        extra={"event": "parity", "delta": result.delta, "ok": result.parity_ok},
    )
    return result
