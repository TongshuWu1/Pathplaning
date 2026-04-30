"""Lightweight EKF pose estimator used by the robots.

The estimator keeps the planning pose in the robot's local belief state.  The
simulator may generate noisy measurements from the hidden truth state, but the
estimator only receives those noisy range/bearing measurements and known
landmark locations.
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
        self.last_landmark_count = 0
        self.last_landmark_residual = 0.0
        self.last_lidar_match_confidence = 0.0
        self.last_teammate_update_count = 0
        self.last_teammate_residual = 0.0

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
        self._regularize_covariance()

    def update_with_landmarks(
        self,
        visible: list[Landmark],
        detection_range: float,
        sensor_pose: Pose | None = None,
    ) -> None:
        """Run a real EKF range/bearing update from visible known landmarks.

        `sensor_pose` is used only by the simulator to synthesize noisy sensor
        readings.  If omitted, measurements are generated around the current
        belief for compatibility with tests that do not have hidden truth.
        """
        self.last_landmark_count = len(visible)
        self.last_landmark_residual = 0.0
        if not visible:
            return

        # Fixed known landmarks: z = [range, bearing].  Measurements are noisy,
        # so the update corrects pose without perfect truth teleportation.
        base_range_std = float(self.cfg.landmark_range_std_m)
        bearing_std = math.radians(float(self.cfg.landmark_bearing_std_deg))
        max_xy_step = float(self.cfg.landmark_max_xy_correction_m)
        max_th_step = math.radians(float(self.cfg.landmark_max_theta_correction_deg))
        I = np.eye(3)
        residual_norms: list[float] = []

        # Update closest / most useful landmarks first.  HOME is an anchor but
        # should not collapse covariance unrealistically, so it uses same EKF math.
        sx, sy, sth = sensor_pose if sensor_pose is not None else self.belief.as_pose()
        ordered = sorted(visible, key=lambda lm: distance((sx, sy), lm.xy))
        for lm in ordered:
            # Simulated measurement from the sensor frame.
            true_dx = lm.xy[0] - sx
            true_dy = lm.xy[1] - sy
            true_r = max(1e-6, math.hypot(true_dx, true_dy))
            if true_r > detection_range + 1e-6:
                continue
            z_r = true_r + self.rng.normal(0.0, base_range_std + 0.012 * true_r)
            z_b = wrap_angle(math.atan2(true_dy, true_dx) - sth + self.rng.normal(0.0, bearing_std))

            x, y, th = self.belief.pose
            dx = lm.xy[0] - x
            dy = lm.xy[1] - y
            q = max(dx * dx + dy * dy, 1e-8)
            pred_r = max(math.sqrt(q), 1e-6)
            pred_b = wrap_angle(math.atan2(dy, dx) - th)
            residual = np.array([z_r - pred_r, wrap_angle(z_b - pred_b)], dtype=float)

            H = np.array(
                [
                    [-dx / pred_r, -dy / pred_r, 0.0],
                    [dy / q, -dx / q, -1.0],
                ],
                dtype=float,
            )
            range_std = base_range_std + 0.012 * pred_r
            R = np.diag([range_std * range_std, bearing_std * bearing_std])

            P = self.belief.covariance
            S = H @ P @ H.T + R
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

            # Robustly down-weight very inconsistent readings rather than
            # letting one bad association create a large pose jump.
            nis = float(residual.T @ Sinv @ residual)
            if nis > 9.0:
                scale = min(8.0, nis / 9.0)
                R = R * scale
                S = H @ P @ H.T + R
                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    Sinv = np.linalg.pinv(S)

            K = P @ H.T @ Sinv
            delta = K @ residual
            xy_norm = float(np.linalg.norm(delta[:2]))
            if xy_norm > max_xy_step:
                delta[:2] *= max_xy_step / max(xy_norm, 1e-9)
            delta[2] = float(np.clip(delta[2], -max_th_step, max_th_step))

            self.belief.pose += delta
            self.belief.pose[2] = wrap_angle(float(self.belief.pose[2]))
            # Joseph form keeps covariance symmetric/positive in long runs.
            KH = K @ H
            self.belief.covariance = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
            self._regularize_covariance()
            residual_norms.append(float(abs(residual[0]) + abs(residual[1]) * max(1.0, pred_r)))

        if residual_norms:
            self.last_landmark_residual = float(np.median(residual_norms))

    def update_with_teammate_pose(
        self,
        teammate_est_pose: Pose,
        teammate_cov_trace: float,
        measured_range: float,
        measured_bearing: float,
    ) -> bool:
        """Fuse a teammate-based relative position observation.

        The measurement model is intentionally conservative: the robot observes
        a noisy relative range/bearing to a teammate, combines that with the
        teammate's reported global pose and covariance, and updates only its own
        x/y belief. Heading still comes from odometry, landmarks, and scan match.
        """
        self.last_teammate_update_count = 0
        self.last_teammate_residual = 0.0
        r = float(measured_range)
        if r <= 1e-6 or not np.isfinite(r):
            return False

        x, y, th = self.belief.pose
        bearing = float(measured_bearing)
        rel_angle = wrap_angle(float(th) + bearing)
        rel_world = np.array([math.cos(rel_angle) * r, math.sin(rel_angle) * r], dtype=float)
        z = np.array([float(teammate_est_pose[0]), float(teammate_est_pose[1])], dtype=float) - rel_world

        teammate_sigma = math.sqrt(max(0.0, float(teammate_cov_trace)) * 0.5) * float(self.cfg.teammate_covariance_scale)
        bearing_std = math.radians(float(self.cfg.teammate_bearing_std_deg))
        rel_sigma = float(self.cfg.teammate_range_std_m) + 0.015 * r + abs(r * bearing_std)
        sigma = max(0.05, rel_sigma + teammate_sigma)
        R = np.diag([sigma * sigma, sigma * sigma])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        P = self.belief.covariance
        residual = z - self.belief.pose[:2]
        S = H @ P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)

        nis = float(residual.T @ Sinv @ residual)
        if nis > 9.0:
            scale = min(10.0, nis / 9.0)
            R = R * scale
            S = H @ P @ H.T + R
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

        K = P @ H.T @ Sinv
        delta = K @ residual
        xy_norm = float(np.linalg.norm(delta[:2]))
        max_xy_step = float(self.cfg.teammate_max_xy_correction_m)
        if xy_norm > max_xy_step:
            delta[:2] *= max_xy_step / max(xy_norm, 1e-9)
        delta[2] = 0.0

        self.belief.pose += delta
        self.belief.pose[2] = wrap_angle(float(self.belief.pose[2]))
        I = np.eye(3)
        KH = K @ H
        self.belief.covariance = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
        self._regularize_covariance()
        self.last_teammate_update_count = 1
        self.last_teammate_residual = float(np.linalg.norm(residual))
        return True

    def apply_lidar_correction(self, dx: float, dy: float, dtheta: float, confidence: float) -> None:
        """Apply bounded correlative scan-matching correction."""
        c = float(np.clip(confidence, 0.0, 1.0))
        self.last_lidar_match_confidence = c
        if c <= 0.0:
            return
        self.belief.pose[0] += self.cfg.lidar_xy_gain * c * float(dx)
        self.belief.pose[1] += self.cfg.lidar_xy_gain * c * float(dy)
        self.belief.pose[2] = wrap_angle(self.belief.pose[2] + self.cfg.lidar_theta_gain * c * float(dtheta))

        # LiDAR scan matching helps, but map alignment can be biased by past
        # drift, so covariance is only reduced moderately.
        shrink_xy = max(0.86, 1.0 - 0.10 * c)
        shrink_th = max(0.90, 1.0 - 0.07 * c)
        self.belief.covariance[:2, :2] *= shrink_xy
        self.belief.covariance[2, 2] *= shrink_th
        self._regularize_covariance()

    def quality(self, scan_consistency: float | None = None, landmark_count: int | None = None) -> float:
        """Pose quality for map insertion, not for goal selection.

        Covariance alone can become overconfident.  Therefore quality combines
        covariance, current scan-map agreement, and whether the estimate has a
        recent fixed landmark anchor.
        """
        # Map quality is driven by estimated pose uncertainty at the moment
        # the LiDAR cells are inserted.  Higher position/heading uncertainty
        # makes the mapped cells less trustworthy in the fused quality overlay.
        pos_sigma = math.sqrt(max(0.0, self.belief.cov_trace_xy))
        theta_sigma = math.sqrt(max(0.0, float(self.belief.covariance[2, 2])))
        pos_q = math.exp(-pos_sigma / 0.85)
        theta_q = math.exp(-theta_sigma / math.radians(18.0))
        cov_q = float(np.clip(pos_q * theta_q, 0.05, 1.0))
        if scan_consistency is None:
            scan_q = 0.78
        else:
            scan_q = float(np.clip(0.35 + 0.65 * scan_consistency, 0.15, 1.0))
        n_lm = self.last_landmark_count if landmark_count is None else int(landmark_count)
        landmark_q = 0.72 + 0.28 * min(1.0, n_lm / 2.0)
        lidar_q = 0.88 + 0.12 * float(np.clip(self.last_lidar_match_confidence, 0.0, 1.0))
        return float(np.clip(cov_q * scan_q * landmark_q * lidar_q, 0.05, 1.0))

    def _regularize_covariance(self) -> None:
        P = 0.5 * (self.belief.covariance + self.belief.covariance.T)
        floor_xy = float(self.cfg.covariance_floor_xy)
        floor_theta = float(self.cfg.covariance_floor_theta)
        max_xy = float(self.cfg.covariance_max_xy)
        max_theta = float(self.cfg.covariance_max_theta)
        # Keep diagonal bounded while preserving off-diagonal information.
        P[0, 0] = float(np.clip(P[0, 0], floor_xy, max_xy))
        P[1, 1] = float(np.clip(P[1, 1], floor_xy, max_xy))
        P[2, 2] = float(np.clip(P[2, 2], floor_theta, max_theta))
        # If numerical coupling is too large after clipping, damp it.
        for a, b in ((0, 1), (0, 2), (1, 2)):
            limit = 0.95 * math.sqrt(max(P[a, a], 1e-12) * max(P[b, b], 1e-12))
            P[a, b] = P[b, a] = float(np.clip(P[a, b], -limit, limit))
        self.belief.covariance = P
