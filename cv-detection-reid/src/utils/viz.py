"""Drawing and annotated-video output.

PRD 9.6. Phase 1 needs boxes, class, confidence and an FPS badge; the track ID,
motion trail and unique-object counters slot into the same functions in Phase
2-3, which is why the signature already carries `track_id`.

Colour is `hash(id) -> hue`, deterministic across runs, so the same object is
the same colour in two different renders of the same clip -- a small thing that
makes side-by-side ablation videos actually comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_for(key: int | str) -> tuple[int, int, int]:
    """Deterministic BGR colour for a class id or track id."""
    h = (hash(str(key)) * 2654435761) % 180        # OpenCV hue range is 0-179
    hsv = np.uint8([[[h, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_box(
    frame: np.ndarray,
    xyxy: Sequence[float],
    label: str,
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    if not label:
        return
    (tw, th), base = cv2.getTextSize(label, _FONT, 0.45, 1)
    # Keep the caption inside the frame when the box is at the top edge --
    # otherwise every truncated object at y=0 loses its label.
    top = y1 - th - base - 2
    if top < 0:
        top = y1 + 2
    cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + base + 2), colour, -1)
    cv2.putText(frame, label, (x1 + 2, top + th + 1), _FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    boxes: Iterable,
    class_names: Sequence[str],
    show_conf: bool = True,
) -> np.ndarray:
    """Render `PredBox`/`GTBox`-shaped records. Colour follows track id if present."""
    out = frame.copy()
    for b in boxes:
        cls_id = int(getattr(b, "cls_id", 0))
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        track_id = getattr(b, "track_id", None)
        conf = getattr(b, "conf", None)
        parts = [name]
        if show_conf and conf is not None:
            parts.append(f"{conf:.2f}")
        if track_id is not None:
            parts.append(f"#{track_id}")
        draw_box(out, b.xyxy, " ".join(parts), colour_for(track_id if track_id is not None else cls_id))
    return out


def draw_hud(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Translucent top-left HUD: FPS, frame index, counters."""
    if not lines:
        return frame
    pad, lh = 8, 20
    w = max(cv2.getTextSize(t, _FONT, 0.55, 1)[0][0] for t in lines) + 2 * pad
    h = lh * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * i + 12), _FONT, 0.55, (0, 255, 200), 1, cv2.LINE_AA)
    return frame


class VideoWriter:
    """Lazy `cv2.VideoWriter` that infers frame size from the first frame written."""

    def __init__(self, path: Path, fps: float = 30.0, fourcc: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps and fps > 0 else 30.0
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: cv2.VideoWriter | None = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise IOError(f"cannot open video writer for {self.path}")
        self._writer.write(frame)
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
