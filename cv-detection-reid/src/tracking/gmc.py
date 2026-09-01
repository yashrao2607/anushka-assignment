"""Global / camera-motion compensation (GMC).

PRD 9.3: "sparse optical flow / ECC between consecutive frames estimates global
camera motion and warps predicted track positions before association.
**Essential** for dashcam/handheld footage; without it, every track's
prediction is wrong whenever the camera pans, and ID switches explode."

The failure it prevents is specific. A Kalman filter predicts where an object
will be *in the world*; association happens *in the image*. When the camera
pans 20 px between frames, every prediction is 20 px stale in the same
direction, IoU with the true detection collapses, and the tracker births new
ids for objects it was tracking perfectly a frame ago. Estimating the frame-to-
frame homography and warping the predictions through it removes that bias.

`sparseOptFlow` is the default because it costs ~2 ms on a 960x540 frame; ECC
is more accurate on low-texture scenes and ~10x slower, so it is offered but
not default. `none` exists so EXP-3 can measure what GMC actually bought.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

METHODS = ("sparseOptFlow", "ecc", "none")


class GMC:
    """Estimates a 2x3 affine warp from the previous frame to the current one."""

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2):
        if method not in METHODS:
            raise ValueError(f"gmc method must be one of {METHODS}, got {method!r}")
        self.method = method
        self.downscale = max(1, int(downscale))
        self._prev_gray: np.ndarray | None = None
        self._prev_points: np.ndarray | None = None
        self.last_warp = np.eye(2, 3, dtype=np.float32)

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_points = None
        self.last_warp = np.eye(2, 3, dtype=np.float32)

    def apply(self, frame: np.ndarray | None) -> np.ndarray:
        """Return the 2x3 affine warp taking the previous frame to this one."""
        warp = np.eye(2, 3, dtype=np.float32)
        if self.method == "none" or frame is None or cv2 is None:
            self.last_warp = warp
            return warp

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.downscale > 1:
            gray = cv2.resize(gray, (gray.shape[1] // self.downscale, gray.shape[0] // self.downscale))

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect(gray)
            self.last_warp = warp
            return warp

        try:
            warp = self._estimate(gray)
        except cv2.error:
            warp = np.eye(2, 3, dtype=np.float32)

        self._prev_gray = gray
        self._prev_points = self._detect(gray)
        self.last_warp = warp
        return warp

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _detect(gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=3
        )

    def _estimate(self, gray: np.ndarray) -> np.ndarray:
        if self.method == "ecc":
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp = cv2.findTransformECC(
                self._prev_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 1
            )
            return self._rescale(warp)

        if self._prev_points is None or len(self._prev_points) < 8:
            return np.eye(2, 3, dtype=np.float32)

        pts, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, self._prev_points, None)
        if pts is None or status is None:
            return np.eye(2, 3, dtype=np.float32)
        ok = status.reshape(-1) == 1
        prev_ok = self._prev_points.reshape(-1, 2)[ok]
        curr_ok = pts.reshape(-1, 2)[ok]
        if len(prev_ok) < 8:
            return np.eye(2, 3, dtype=np.float32)

        # RANSAC, not least squares: the moving *objects* are outliers to the
        # camera's motion, and a plain fit would partly track them instead.
        warp, _ = cv2.estimateAffinePartial2D(prev_ok, curr_ok, method=cv2.RANSAC)
        if warp is None:
            return np.eye(2, 3, dtype=np.float32)
        return self._rescale(warp.astype(np.float32))

    def _rescale(self, warp: np.ndarray) -> np.ndarray:
        """Translation was estimated on the downscaled image; scale it back up."""
        if self.downscale > 1:
            warp = warp.copy()
            warp[0, 2] *= self.downscale
            warp[1, 2] *= self.downscale
        return warp


def apply_warp_to_xyxy(xyxy, warp: np.ndarray) -> tuple[float, float, float, float]:
    """Push an xyxy box through a 2x3 affine warp."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    corners = np.array([[x1, y1, 1.0], [x2, y2, 1.0]], dtype=float)
    out = corners @ warp.T
    return float(out[0, 0]), float(out[0, 1]), float(out[1, 0]), float(out[1, 1])
