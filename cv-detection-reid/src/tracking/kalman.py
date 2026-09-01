"""Kalman filter for bounding-box tracking.

PRD 9.3: "Kalman filter with a constant-velocity model on (x, y, aspect,
height) plus velocities."

State is 8-dimensional: `[cx, cy, a, h, vcx, vcy, va, vh]` where `a = w / h`.
Parameterising by aspect ratio and height rather than width and height is the
SORT convention and it matters: an object's apparent height is a smooth proxy
for its distance, while its aspect ratio is near-constant for a rigid object.
Predicting those two independently is far better conditioned than predicting
width and height, which are strongly correlated.

Process and measurement noise both scale with the box height, so a distant
50-px car is not held to the same positional tolerance as a near 300-px one.
That single detail is most of the difference between a tracker that survives
perspective and one that does not.
"""

from __future__ import annotations

import numpy as np

NDIM = 4
DT = 1.0


class KalmanFilterXYAH:
    """Constant-velocity Kalman filter over (centre-x, centre-y, aspect, height)."""

    def __init__(self, std_weight_position: float = 1.0 / 20, std_weight_velocity: float = 1.0 / 160):
        self._motion_mat = np.eye(2 * NDIM, 2 * NDIM)
        for i in range(NDIM):
            self._motion_mat[i, NDIM + i] = DT
        self._update_mat = np.eye(NDIM, 2 * NDIM)
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

    # -- lifecycle ---------------------------------------------------------
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track from an unassociated measurement `[cx, cy, a, h]`."""
        mean = np.r_[np.asarray(measurement, dtype=float), np.zeros(NDIM)]
        h = float(measurement[3])
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))
        return self._update_mat @ mean, self._update_mat @ covariance @ self._update_mat.T + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        projected_mean, projected_cov = self.project(mean, covariance)
        # Solved rather than inverted: `projected_cov` is small but can be
        # ill-conditioned when a track has been coasting through a long
        # occlusion, and an explicit inverse amplifies exactly that.
        kalman_gain = np.linalg.solve(
            projected_cov.T, (covariance @ self._update_mat.T).T
        ).T
        innovation = np.asarray(measurement, dtype=float) - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_cov

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Squared Mahalanobis distance from a track to each measurement."""
        projected_mean, projected_cov = self.project(mean, covariance)
        d = np.atleast_2d(measurements) - projected_mean
        chol = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(chol, d.T)
        return np.sum(z * z, axis=0)


# ---------------------------------------------------------------------------
# Box conversions -- kept next to the filter so the state convention has
# exactly one definition.
# ---------------------------------------------------------------------------


def xyxy_to_xyah(xyxy) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=float)


def xyah_to_xyxy(xyah) -> tuple[float, float, float, float]:
    cx, cy, a, h = (float(v) for v in xyah[:4])
    w = a * h
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
