"""Per-frame difficulty attributes.

PRD 7.6: the manifest carries `lighting`, `blur_score`, `occlusion_level`, ...
because "these attribute columns are what make the difficulty-sliced
evaluation possible -- they must be captured at annotation time, not
reconstructed later."

That is the whole point of this module. An averaged mAP hides exactly the
failures that matter operationally (PRD 13.3, differentiator D4); slicing needs
a per-frame attribute, and a per-frame attribute is cheap to compute at sample
time and expensive to recover afterwards.

Every measure here is deliberately simple and deterministic -- these are
*stratification labels*, not model inputs. A reviewer must be able to read the
definition and agree the slice means what it says.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from ..config import AttributesConfig

LIGHTING_LEVELS = ("night", "dusk", "day")
BLUR_LEVELS = ("blurred", "sharp")


@dataclass(frozen=True)
class FrameAttributes:
    """Difficulty descriptors for a single sampled frame."""

    brightness: float       # mean HSV V channel, 0-255
    contrast: float         # std-dev of the grayscale image
    blur_score: float       # variance of the Laplacian; low = blurred
    lighting: str           # night | dusk | day
    blur_level: str         # blurred | sharp
    motion_score: float     # mean abs-diff vs the previous sampled frame, 0-255

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Standard focus measure: the variance of the Laplacian response.

    A sharp image has strong second derivatives at edges and therefore a high
    variance; a motion-blurred or defocused image has its high frequencies
    suppressed and scores low. Reported in absolute units so the threshold in
    the config is auditable rather than adaptive-and-mysterious.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def classify_lighting(brightness: float, cfg: AttributesConfig) -> str:
    if brightness < cfg.lighting_night_below:
        return "night"
    if brightness < cfg.lighting_dusk_below:
        return "dusk"
    return "day"


def motion_score(prev_gray: np.ndarray | None, gray: np.ndarray) -> float:
    """Mean absolute difference against the previous *sampled* frame.

    A cheap proxy for "how much did the scene or the camera move", used for the
    camera-motion slice (PRD 13.3) where GMC earns its place. It is not a
    calibrated ego-motion estimate and is not presented as one -- the real GMC
    lands in Phase 2.3.
    """
    if prev_gray is None:
        return 0.0
    if prev_gray.shape != gray.shape:
        prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))
    return float(np.mean(cv2.absdiff(prev_gray, gray)))


def compute_attributes(
    frame_bgr: np.ndarray,
    cfg: AttributesConfig,
    prev_gray: np.ndarray | None = None,
) -> tuple[FrameAttributes, np.ndarray]:
    """Return the frame's attributes plus its grayscale image for reuse."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("compute_attributes received an empty frame")

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Brightness from HSV-V rather than grayscale luma: V is max(R,G,B), which
    # tracks perceived scene illumination and is far less sensitive to a single
    # saturated colour (a red bus filling the frame) than a luma average.
    value = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]

    brightness = float(np.mean(value))
    blur = variance_of_laplacian(gray)
    attrs = FrameAttributes(
        brightness=round(brightness, 2),
        contrast=round(float(np.std(gray)), 2),
        blur_score=round(blur, 2),
        lighting=classify_lighting(brightness, cfg),
        blur_level="blurred" if blur < cfg.blur_hard_below else "sharp",
        motion_score=round(motion_score(prev_gray, gray), 2),
    )
    return attrs, gray
