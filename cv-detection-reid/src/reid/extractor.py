"""Appearance embedding extraction -- the input to the D1 differentiator.

PRD 9.2: a 512-d L2-normalised embedding per detection crop, all crops in a
frame batched into a single forward pass, crops expanded 5%, resized to
256x128, ImageNet-normalised, and crops under 16x32 px skipped as too degraded
to embed meaningfully.

**Backbone honesty.** The PRD specifies `torchreid` OSNet, which is the right
model: it is trained with a ReID objective, so its embedding space is shaped by
"same identity, different view" rather than by "same category". Three backends
are supported and the active one is recorded in every report, because a ReID
number means something different depending on which produced it:

  `osnet`      -- torchreid OSNet. The PRD choice; used when installed.
  `resnet18`   -- ImageNet-pretrained torchvision ResNet18, global-pooled to
                  512-d. **Not** a ReID model: it was trained to collapse
                  intra-class variation, which is the opposite of what ReID
                  needs, so it separates a person from a bus far better than
                  it separates two people. It is a working, honest fallback,
                  and its weaker discrimination is visible in the calibration
                  curve rather than hidden.
  `colour`     -- an HSV colour-histogram descriptor with no learned weights.
                  Dependency-free, surprisingly serviceable for short
                  occlusions where clothing colour is the dominant cue, and it
                  guarantees the pipeline runs anywhere.

Whichever is active, the contract is identical: L2-normalised vectors where
cosine similarity is a dot product, so the tracker and the gallery never learn
which backbone produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from ..config import Config
from ..utils.device import resolve_device
from ..utils.logging import get_logger

log = get_logger("reid.extractor")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# PRD 9.2: boxes are expanded 5% before cropping, so a slightly tight detection
# box does not clip the shoulders or wheels that carry the appearance signal.
CROP_EXPAND = 0.05


@dataclass
class CropSpec:
    height: int = 256
    width: int = 128
    min_w: int = 16
    min_h: int = 32


def expand_and_crop(frame: np.ndarray, xyxy: Sequence[float], spec: CropSpec) -> np.ndarray | None:
    """Expand the box 5%, clip to the frame, and return the crop (or None)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    x1 -= bw * CROP_EXPAND / 2
    x2 += bw * CROP_EXPAND / 2
    y1 -= bh * CROP_EXPAND / 2
    y2 += bh * CROP_EXPAND / 2
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if xi2 - xi1 < spec.min_w or yi2 - yi1 < spec.min_h:
        # Too small to embed meaningfully. The caller falls back to motion-only
        # association -- a confident embedding from an 8-pixel smear is worse
        # than no embedding, because the tracker would trust it.
        return None
    crop = frame[yi1:yi2, xi1:xi2]
    return crop if crop.size else None


class ReidExtractor:
    """Batched appearance embeddings with a documented backbone fallback chain."""

    def __init__(self, cfg: Config, backend: str | None = None):
        self.cfg = cfg
        self.dim = cfg.reid.embedding_dim
        self.spec = CropSpec(
            height=cfg.reid.crop_size[0], width=cfg.reid.crop_size[1],
            min_w=cfg.reid.min_crop_wh[0], min_h=cfg.reid.min_crop_wh[1],
        )
        self.device = resolve_device(cfg.detection.device)
        self.backend = backend or "auto"
        self._model = None
        self._resolve_backend()

    # -- backend selection -------------------------------------------------
    def _resolve_backend(self) -> None:
        wanted = self.backend
        if wanted in ("auto", "osnet"):
            if self._try_osnet():
                return
            if wanted == "osnet":
                log.warning(
                    "torchreid is not installed; falling back. Install it with "
                    "`pip install torchreid` to use the PRD's OSNet backbone.",
                    extra={"event": "osnet_unavailable"},
                )
        if wanted in ("auto", "osnet", "resnet18"):
            if self._try_resnet():
                return
        self.backend = "colour"
        self.dim = 96
        log.info(
            "ReID backend: colour histogram (no learned weights). Reported "
            "ReID numbers name this backbone.",
            extra={"event": "reid_backend", "backend": "colour"},
        )

    def _try_osnet(self) -> bool:
        try:
            from torchreid.utils import FeatureExtractor
        except Exception:
            return False
        try:
            self._model = FeatureExtractor(
                model_name=self.cfg.reid.model,
                device=self.device.device,
            )
            self.backend = "osnet"
            log.info(f"ReID backend: torchreid {self.cfg.reid.model}",
                     extra={"event": "reid_backend", "backend": "osnet"})
            return True
        except Exception as exc:
            log.warning(f"torchreid present but failed to load: {exc}")
            return False

    def _try_resnet(self) -> bool:
        try:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception:
            return False
        try:
            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            net.fc = torch.nn.Identity()          # 512-d global-pooled features
            net.eval().to(self.device.device)
            self._model = net
            self._torch = torch
            self.backend = "resnet18"
            self.dim = 512
            log.info(
                "ReID backend: torchvision ResNet18 (ImageNet). Not a ReID-"
                "trained model -- see module docstring; the calibration curve "
                "shows what that costs.",
                extra={"event": "reid_backend", "backend": "resnet18"},
            )
            return True
        except Exception as exc:
            log.warning(f"torchvision resnet18 unavailable: {exc}")
            return False

    # -- inference ---------------------------------------------------------
    def extract(self, frame: np.ndarray, boxes: Sequence[Sequence[float]]) -> dict[int, np.ndarray]:
        """Embed every usable crop in one batched pass.

        Returns `{index into boxes -> embedding}`. Indices whose crop was too
        small or unusable are simply absent, which the tracker reads as "no
        appearance evidence" rather than as a zero vector.
        """
        if not len(boxes):
            return {}
        crops: list[np.ndarray] = []
        index: list[int] = []
        for i, xyxy in enumerate(boxes):
            crop = expand_and_crop(frame, xyxy, self.spec)
            if crop is not None:
                crops.append(crop)
                index.append(i)
        if not crops:
            return {}

        if self.backend == "colour":
            feats = np.stack([self._colour_descriptor(c) for c in crops])
        elif self.backend == "osnet":
            feats = np.asarray(self._model(crops))
        else:
            feats = self._resnet_forward(crops)

        feats = l2_normalise(feats)
        return {index[i]: feats[i] for i in range(len(index))}

    def _resnet_forward(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = np.stack([self._preprocess(c) for c in crops])
        tensor = torch.from_numpy(batch).to(self.device.device)
        with torch.no_grad():
            out = self._model(tensor)
        return out.cpu().numpy()

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (self.spec.width, self.spec.height), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return img.transpose(2, 0, 1)             # HWC -> CHW

    def _colour_descriptor(self, crop: np.ndarray) -> np.ndarray:
        """Spatially-banded HSV histogram.

        Three horizontal bands (roughly head / torso / legs for a person, or
        roof / body / wheels for a vehicle) times a 4x4x2 HSV histogram = 96
        dimensions. The banding is what makes it more than a colour average: a
        person in a white shirt and dark trousers is distinguishable from one
        in dark shirt and white trousers, which a whole-crop histogram cannot
        tell apart at all.
        """
        img = cv2.resize(crop, (self.spec.width, self.spec.height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        bands = np.array_split(hsv, 3, axis=0)
        parts = []
        for band in bands:
            hist = cv2.calcHist([band], [0, 1, 2], None, [4, 4, 2],
                                [0, 180, 0, 256, 0, 256]).flatten()
            total = hist.sum()
            parts.append(hist / total if total > 0 else hist)
        return np.concatenate(parts).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            "reid_backend": self.backend,
            "reid_model": self.cfg.reid.model if self.backend == "osnet" else self.backend,
            "embedding_dim": self.dim,
            "device": self.device.device,
            "crop_size": f"{self.spec.height}x{self.spec.width}",
        }


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Unit-normalise rows so cosine similarity is a plain dot product."""
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return x / norms
