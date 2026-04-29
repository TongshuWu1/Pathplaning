"""Simple EKF-like pose estimator.

This is intentionally lightweight.  It is not the behavior planner; it only
provides estimated pose and covariance so LiDAR scans can be inserted into the
map with an appropriate confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import MotionNoiseConfig
from .geometry import Pose, Point, wrap_angle, distance
from .world import Landmark


@dataclass
class PoseBelief:
    pose: np.ndarray  # x, y, theta
    covariance: np.ndarray  # 3x3

    @property
    def xy(self) -> Point:
        return (float(self.pose[0]), float(self.pose[1]))

    @property
    def theta(self) -> float:
        return float(self.pose[2])

    @property
    def cov_trace_xy(self) -> float:
        return float(np.trace(self.covariance[:2, :2]))

    def as_pose(self) -> Pose:
        return (float(self.pose[0]), float(self.pose[1]), float(self.pose[2]))


class PoseEstimator:
    def __init__(self, initial_pose: Pose, cfg: MotionNoiseConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.belief = PoseBelief(
            pose=np.array(initial_pose, dtype=float),
            covariance=np.diag([0.05, 0.05, 0.02]),
        )

    def predict_from_command(self, v: float, omega: float, dt: float) -> None:
        x, y, th = self.belief.pose
        # Odometry has drift/noise.  This is the robot's internal guess.
        v_hat = v + self.rng.normal(0.0, abs(v) * self.cfg.xy_std_per_m + 0.002)
        omega_hat = omega + self.rng.normal(0.0, abs(omega) * self.cfg.theta_std_per_rad + 0.003)
        x += math.cos(th) * v_hat * dt
        y += math.sin(th) * v_hat * dt
        th = wrap_angle(th + omega_hat * dt)
        self.belief.pose[:] = [x, y, th]
        qx = self.cfg.process_xy + abs(v) * dt * self.cfg.xy_std_per_m
        qt = self.cfg.process_theta + abs(omega) * dt * self.cfg.theta_std_per_rad
        self.belief.covariance += np.diag([qx * qx, qx * qx, qt * qt])
        self.belief.covariance = 0.5 * (self.belief.covariance + self.belief.covariance.T)

    def update_with_landmarks(self, visible: list[Landmark], detection_range: float) -> None:
        if not visible:
            return
        # Simple correction toward triangulated landmark-derived pose residual.
        # In real EKF this would be a measurement update; here it is a stable
        # simulation approximation.
        x, y, th = self.belief.pose
        correction = np.zeros(2, dtype=float)
        total_w = 0.0
        for lm in visible:
            d = max(distance((x, y), lm.xy), 1e-3)
            w = max(0.05, 1.0 - d / max(detection_range, 1e-6))
            # Landmarks shrink drift but do not reveal absolute perfect pose.
            direction = np.array([lm.xy[0] - x, lm.xy[1] - y], dtype=float) / d
            # Tiny randomized pseudo-residual; enough to stabilize covariance.
            correction += w * direction * self.rng.normal(0.0, 0.015)
            total_w += w
        if total_w > 0:
            correction /= total_w
            self.belief.pose[0:2] += self.cfg.landmark_xy_gain * correction
        shrink = max(0.15, self.cfg.landmark_cov_shrink ** max(1, len(visible)))
        if any(lm.is_home for lm in visible):
            shrink *= 0.55
        self.belief.covariance[:2, :2] *= shrink
        self.belief.covariance[2, 2] *= max(0.35, shrink)
        self.belief.covariance = 0.5 * (self.belief.covariance + self.belief.covariance.T)

    def quality(self) -> float:
        # Map update confidence derived from pose covariance.  High covariance
        # weakens mapping/certification, but does not pick the robot's target.
        tr = self.belief.cov_trace_xy
        return float(np.clip(math.exp(-1.25 * tr), 0.05, 1.0))
