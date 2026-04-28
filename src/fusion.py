from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple
import math

import numpy as np

from .mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGrid


@dataclass
class FusionStats:
    """Small summary used by the UI/status panel."""

    contributing_robots: int = 0
    known_cells: int = 0
    coverage_pct: float = 0.0
    mean_confidence: float = 0.0
    high_confidence_pct: float = 0.0


class TeamFusedMap:
    """
    Communication-respecting team belief map.

    The fused map is intentionally separate from the global truth map.  It stores
    what the team currently believes, plus how much trust we have in each cell.

    Fusion rule, v1:
        winner-take-best confidence fusion.

    A robot can only update the team map when its information is currently
    connected to the home/team communication component.  Old shared information
    stays in the fused map, but confidence slowly decays so stale areas become
    visibly lower quality.
    """

    def __init__(
        self,
        width_m: float,
        height_m: float,
        res: float,
        *,
        min_confidence: float = 0.03,
        stale_decay_s: float = 90.0,
        pose_cov_gain: float = 0.85,
        range_gain: float = 1.15,
    ) -> None:
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.res = float(res)
        self.nx = int(np.ceil(self.width_m / self.res))
        self.ny = int(np.ceil(self.height_m / self.res))

        self.occ = np.full((self.ny, self.nx), UNKNOWN, dtype=np.int8)
        self.conf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.source = np.full((self.ny, self.nx), -1, dtype=np.int16)
        self.updated_at = np.full((self.ny, self.nx), -np.inf, dtype=np.float32)

        self.min_confidence = float(min_confidence)
        self.stale_decay_s = max(1e-6, float(stale_decay_s))
        self.pose_cov_gain = float(pose_cov_gain)
        self.range_gain = max(1e-6, float(range_gain))
        self.last_decay_time = 0.0
        self.stats = FusionStats()

    def reset(self) -> None:
        self.occ.fill(UNKNOWN)
        self.conf.fill(0.0)
        self.source.fill(-1)
        self.updated_at.fill(-np.inf)
        self.last_decay_time = 0.0
        self.stats = FusionStats()

    def update_from_robots(self, robots: Iterable[object], now: float, cfg, *, require_home_connected: bool = True) -> None:
        """Fuse currently shareable robot maps into the team belief map."""
        self._decay_confidence(now)
        contributing = 0
        for robot in robots:
            if require_home_connected and not bool(getattr(robot, 'home_connected', False)):
                continue
            self.update_from_robot(robot, now, cfg)
            contributing += 1
        self.stats = self._compute_stats(contributing)

    def update_from_robot(self, robot: object, now: float, cfg) -> None:
        local_grid: OccupancyGrid = robot.local_map
        local = local_grid.data
        known = local != UNKNOWN
        if not np.any(known):
            return

        yy, xx = np.nonzero(known)
        local_values = local[yy, xx]
        new_conf = self._confidence_for_cells(robot, xx, yy, local_values, cfg)

        # Winner-take-best: only replace the fused cell if this robot gives the
        # most confident currently available estimate.  Also let the current
        # source refresh its own cells when its confidence improves again.
        old_conf = self.conf[yy, xx]
        replace = (new_conf > old_conf) | (self.occ[yy, xx] == UNKNOWN)
        if not np.any(replace):
            return

        ryy = yy[replace]
        rxx = xx[replace]
        self.occ[ryy, rxx] = local_values[replace]
        self.conf[ryy, rxx] = new_conf[replace].astype(np.float32)
        self.source[ryy, rxx] = int(robot.robot_id)
        self.updated_at[ryy, rxx] = float(now)

    def occupancy_for_display(self) -> np.ndarray:
        """Return occupancy, hiding cells whose confidence decayed too far."""
        out = np.array(self.occ, copy=True)
        out[self.conf < self.min_confidence] = UNKNOWN
        return out

    def confidence_rgba(self) -> np.ndarray:
        """
        RGBA image for confidence/uncertainty visualization.

        Unknown cells are gray.  Known cells interpolate:
            red   = low confidence / high uncertainty
            yellow= medium confidence
            green = high confidence / low uncertainty
        """
        conf = np.clip(self.conf, 0.0, 1.0)
        known = (self.occ != UNKNOWN) & (conf >= self.min_confidence)
        rgba = np.zeros((self.ny, self.nx, 4), dtype=np.float32)
        rgba[..., 0] = 0.84
        rgba[..., 1] = 0.86
        rgba[..., 2] = 0.89
        rgba[..., 3] = 1.0

        if not np.any(known):
            return rgba

        c = conf[known]
        low = np.array([0.90, 0.18, 0.16], dtype=np.float32)
        mid = np.array([0.95, 0.74, 0.16], dtype=np.float32)
        high = np.array([0.12, 0.68, 0.32], dtype=np.float32)
        rgb = np.empty((len(c), 3), dtype=np.float32)
        lower = c < 0.5
        t_low = np.clip(c[lower] / 0.5, 0.0, 1.0)[:, None]
        t_high = np.clip((c[~lower] - 0.5) / 0.5, 0.0, 1.0)[:, None]
        rgb[lower] = low * (1.0 - t_low) + mid * t_low
        rgb[~lower] = mid * (1.0 - t_high) + high * t_high
        rgba[known, :3] = rgb
        return rgba

    def _decay_confidence(self, now: float) -> None:
        dt = max(0.0, float(now) - float(self.last_decay_time))
        if dt <= 0.0:
            return
        self.conf *= math.exp(-dt / self.stale_decay_s)
        self.last_decay_time = float(now)

    def _confidence_for_cells(self, robot: object, xx: np.ndarray, yy: np.ndarray, local_values: np.ndarray, cfg) -> np.ndarray:
        wx = (xx.astype(np.float32) + 0.5) * self.res
        wy = (yy.astype(np.float32) + 0.5) * self.res
        rx, ry = float(robot.x_est), float(robot.y_est)
        dist = np.hypot(wx - rx, wy - ry)

        # Pose confidence: large localization covariance means this robot's map
        # alignment is less trustworthy.
        pose_trace = max(0.0, float(robot.covariance_trace()))
        pose_conf = math.exp(-self.pose_cov_gain * pose_trace)

        # Range confidence: local cells close to the robot are more reliable.
        max_range = max(1e-6, float(cfg.lidar_range) * self.range_gain)
        range_conf = np.clip(1.0 - dist / max_range, 0.12, 1.0) ** 1.35

        # Occupied hits are usually stronger evidence than free cells inferred
        # from rays.  Free cells are still useful, just a little less certain.
        sensor_conf = np.where(local_values == OCCUPIED, 0.98, 0.80).astype(np.float32)

        # Longer relay chains are still accepted, but slightly less trusted.
        hops = getattr(robot, 'home_hops', None)
        hop_conf = 1.0 if hops is None else 1.0 / (1.0 + 0.06 * max(0, int(hops)))
        direct_home_bonus = 1.03 if bool(getattr(robot, 'direct_home_link', False)) else 1.0

        conf = sensor_conf * float(pose_conf) * range_conf * float(hop_conf) * float(direct_home_bonus)
        return np.clip(conf, self.min_confidence, 1.0).astype(np.float32)

    def _compute_stats(self, contributing_robots: int) -> FusionStats:
        known = (self.occ != UNKNOWN) & (self.conf >= self.min_confidence)
        known_cells = int(np.count_nonzero(known))
        total = max(1, self.nx * self.ny)
        if known_cells == 0:
            return FusionStats(contributing_robots=contributing_robots)
        known_conf = self.conf[known]
        high = float(np.mean(known_conf >= 0.70)) * 100.0
        return FusionStats(
            contributing_robots=int(contributing_robots),
            known_cells=known_cells,
            coverage_pct=100.0 * known_cells / total,
            mean_confidence=float(np.mean(known_conf)),
            high_confidence_pct=high,
        )
