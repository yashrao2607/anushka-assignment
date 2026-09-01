"""Device resolution and an environment report.

PRD NFR-9 (portability, Windows + Linux, CUDA and CPU paths) and R9 (no GPU
available). This machine is CPU-only, so `resolve_device` must degrade
gracefully rather than assume CUDA -- and `environment_report` records exactly
what produced a number, because "which device was this benchmarked on?" is the
first question asked of any FPS claim.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    device: str            # cpu | cuda | mps  (or cuda:N)
    name: str
    total_memory_gb: float | None
    is_gpu: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve `auto` to the best available backend: cuda -> mps -> cpu."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return DeviceInfo("cpu", "cpu (torch unavailable)", None, False)

    def cuda_info(index: int = 0) -> DeviceInfo:
        props = torch.cuda.get_device_properties(index)
        return DeviceInfo(
            f"cuda:{index}" if index else "cuda",
            props.name,
            round(props.total_memory / 1024**3, 2),
            True,
        )

    if requested.isdigit():
        idx = int(requested)
        if torch.cuda.is_available() and idx < torch.cuda.device_count():
            return cuda_info(idx)
        return DeviceInfo("cpu", _cpu_name(), None, False)

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return cuda_info(0)
    if requested == "cuda":
        # Explicitly asked for CUDA and it is not there: fall back but be loud
        # about it rather than crashing a 6-hour run at epoch 0.
        return DeviceInfo("cpu", _cpu_name() + " (cuda requested but unavailable)", None, False)

    mps = getattr(torch.backends, "mps", None)
    if requested in {"auto", "mps"} and mps is not None and mps.is_available():
        return DeviceInfo("mps", "Apple Metal (MPS)", None, True)

    return DeviceInfo("cpu", _cpu_name(), None, False)


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "cpu"


def _version(module: str) -> str:
    try:
        mod = __import__(module)
    except Exception:
        return "not installed"
    return getattr(mod, "__version__", "unknown")


def environment_report() -> dict[str, Any]:
    """Everything needed to reproduce or discount a benchmark number."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "torch": _version("torch"),
        "opencv": _version("cv2"),
        "ultralytics": _version("ultralytics"),
        "numpy": _version("numpy"),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda or "n/a"
        info["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        info["cuda_available"] = False
        info["cuda_version"] = "n/a"
        info["cuda_device_count"] = 0

    dev = resolve_device("auto")
    info["resolved_device"] = dev.device
    info["device_name"] = dev.name
    info["device_memory_gb"] = dev.total_memory_gb
    info["nvidia_smi"] = _nvidia_smi()
    return info


def _nvidia_smi() -> str:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "not found"
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "no devices"
    except Exception:
        return "query failed"


def seed_everything(seed: int) -> None:
    """PRD NFR-10: fixed seeds so two runs of the same config agree."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
