"""LiDAR sensing models."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import LidarConfig
from .geometry import Pose
from .world import World


@dataclass
class LidarScan:
    angles: np.ndarray
    ranges: np.ndarray
    hit: np.ndarray
    points_world_true: np.ndarray


class LidarSensor:
    def __init__(self, cfg: LidarConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.angles = np.linspace(-math.pi, math.pi, cfg.rays, endpoint=False)

    def sense(self, world: World, true_pose: Pose) -> LidarScan:
        ranges = np.zeros(self.cfg.rays, dtype=float)
        hit = np.zeros(self.cfg.rays, dtype=bool)
        pts = np.zeros((self.cfg.rays, 2), dtype=float)
        th = true_pose[2]
        for k, a in enumerate(self.angles):
            r, _p, h = world.raycast(true_pose, float(a), self.cfg.range, step=self.cfg.raycast_step_m)
            if h and self.rng.random() < self.cfg.dropout_probability:
                h = False
                r = self.cfg.range
            if h:
                sigma = self.cfg.noise_std + self.cfg.range_noise_std_per_m * max(0.0, float(r))
                rn = float(np.clip(r + self.rng.normal(0.0, sigma), 0.02, self.cfg.range))
            else:
                rn = float(np.clip(self.cfg.range + self.rng.normal(0.0, self.cfg.max_range_noise_std), 0.02, self.cfg.range))
            ranges[k] = rn
            hit[k] = h and rn < self.cfg.range - self.cfg.hit_threshold
            pts[k] = [true_pose[0] + math.cos(th + a) * rn, true_pose[1] + math.sin(th + a) * rn]
        return LidarScan(self.angles.copy(), ranges, hit, pts)
