"""Finite hidden-truth world used only by the simulator.

The generator intentionally follows the older simulator style: a 30x30 finite
world, a rectangular home base in the lower-left corner, fewer large rectangular
obstacles with spacing margins, a protected spawn region, landmark beacons, and
one truth-only target near the far side of the map.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from .config import WorldConfig
from .geometry import Point, Pose, Rect, distance, segment_intersects_rect, unit_from_angle


@dataclass(frozen=True)
class Landmark:
    id: int
    xy: Point
    name: str = ""
    is_home: bool = False


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.width = cfg.width
        self.height = cfg.height
        self.home_base = Rect(0.0, 0.0, cfg.home_base_size, cfg.home_base_size).normalized()
        self.home = self.home_base.center
        self.home_marker = Landmark(-1, self.home, name="Home", is_home=True)
        self.obstacles: list[Rect] = []
        self.landmarks: list[Landmark] = []
        self.target: Point = self._default_target()
        self.truth_res = 0.08
        self.truth_mask = np.zeros((1, 1), dtype=bool)
        self._generate()
        self._build_truth_mask()

    @property
    def all_landmarks(self) -> list[Landmark]:
        return [self.home_marker, *self.landmarks]

    def _default_target(self) -> Point:
        tx = self.cfg.target_x if self.cfg.target_x is not None else self.width - self.cfg.world_margin - 1.0
        ty = self.cfg.target_y if self.cfg.target_y is not None else self.height - self.cfg.world_margin - 1.0
        return (float(tx), float(ty))

    def _generate(self) -> None:
        self.target = self._default_target()
        spawn_center = (self.cfg.home_base_padding + 0.8, self.cfg.home_base_padding + 0.8)

        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.cfg.obstacle_count and attempts < 10000:
            attempts += 1
            w = self.rng.uniform(self.cfg.obstacle_min_size, self.cfg.obstacle_max_size)
            h = self.rng.uniform(self.cfg.obstacle_min_size, self.cfg.obstacle_max_size)
            cx = self.rng.uniform(self.cfg.world_margin + w * 0.5, self.width - self.cfg.world_margin - w * 0.5)
            cy = self.rng.uniform(self.cfg.world_margin + h * 0.5, self.height - self.cfg.world_margin - h * 0.5)
            rect = Rect(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5).normalized()
            if _rects_overlap(rect, self.home_base, margin=self.cfg.home_base_padding + 0.35):
                continue
            if distance(rect.center, spawn_center) < self.cfg.spawn_clear_radius:
                continue
            if rect.contains(self.target, margin=self.cfg.target_clear_radius):
                continue
            if any(_rects_overlap(rect, other, margin=self.cfg.obstacle_gap_margin) for other in self.obstacles):
                continue
            self.obstacles.append(rect)

        self.landmarks = []
        attempts = 0
        while len(self.landmarks) < self.cfg.landmark_count and attempts < 10000:
            attempts += 1
            p = (
                self.rng.uniform(self.cfg.world_margin, self.width - self.cfg.world_margin),
                self.rng.uniform(self.cfg.world_margin, self.height - self.cfg.world_margin),
            )
            if self.home_base.contains(p, margin=0.75):
                continue
            if distance(p, spawn_center) < self.cfg.spawn_clear_radius * 0.70:
                continue
            if distance(p, self.target) < self.cfg.target_clear_radius:
                continue
            if not self.is_free(p, margin=0.5):
                continue
            if any(distance(p, lm.xy) < 1.5 for lm in self.landmarks):
                continue
            self.landmarks.append(Landmark(len(self.landmarks), p, name=f"L{len(self.landmarks) + 1}"))

    def _build_truth_mask(self) -> None:
        nx = int(math.ceil(self.width / self.truth_res))
        ny = int(math.ceil(self.height / self.truth_res))
        mask = np.zeros((ny, nx), dtype=bool)
        for obs in self.obstacles:
            ix0 = max(0, int(math.floor(obs.x0 / self.truth_res)))
            ix1 = min(nx, int(math.ceil(obs.x1 / self.truth_res)))
            iy0 = max(0, int(math.floor(obs.y0 / self.truth_res)))
            iy1 = min(ny, int(math.ceil(obs.y1 / self.truth_res)))
            mask[iy0:iy1, ix0:ix1] = True
        self.truth_mask = mask

    def in_bounds(self, p: Point, margin: float = 0.0) -> bool:
        return margin <= p[0] <= self.width - margin and margin <= p[1] <= self.height - margin

    def _truth_occupied(self, p: Point) -> bool:
        i = int(p[0] / self.truth_res)
        j = int(p[1] / self.truth_res)
        if not (0 <= j < self.truth_mask.shape[0] and 0 <= i < self.truth_mask.shape[1]):
            return True
        return bool(self.truth_mask[j, i])

    def is_free(self, p: Point, margin: float = 0.0) -> bool:
        if not self.in_bounds(p, margin=margin):
            return False
        return not any(obs.contains(p, margin=margin) for obs in self.obstacles)

    def segment_free(self, a: Point, b: Point, margin: float = 0.0) -> bool:
        if not self.in_bounds(a, margin=margin) or not self.in_bounds(b, margin=margin):
            return False
        for obs in self.obstacles:
            if segment_intersects_rect(a, b, obs, margin=margin):
                return False
        return True

    def raycast(self, pose: Pose, rel_angle: float, max_range: float, step: float = 0.12) -> tuple[float, Point, bool]:
        x, y, th = pose
        dx, dy = unit_from_angle(th + rel_angle)
        r = 0.0
        last = (x, y)
        while r < max_range:
            r += step
            p = (x + dx * r, y + dy * r)
            if not self.in_bounds(p):
                return min(r, max_range), p, True
            if self._truth_occupied(p):
                return min(r, max_range), p, True
            last = p
        return max_range, last, False

    def visible_landmarks(self, pose: Pose, max_range: float) -> list[Landmark]:
        p = (pose[0], pose[1])
        out: list[Landmark] = []
        for lm in self.all_landmarks:
            if distance(p, lm.xy) <= max_range and self.segment_free(p, lm.xy, margin=0.02):
                out.append(lm)
        return out

    def target_visible(self, pose: Pose, max_range: float) -> bool:
        p = (pose[0], pose[1])
        return distance(p, self.target) <= max_range and self.segment_free(p, self.target, margin=0.02)

    def raster_obstacle_mask(self, resolution: float) -> np.ndarray:
        nx = int(math.ceil(self.width / resolution))
        ny = int(math.ceil(self.height / resolution))
        mask = np.zeros((ny, nx), dtype=bool)
        for obs in self.obstacles:
            ix0 = max(0, int(math.floor(obs.x0 / resolution)))
            ix1 = min(nx, int(math.ceil(obs.x1 / resolution)))
            iy0 = max(0, int(math.floor(obs.y0 / resolution)))
            iy1 = min(ny, int(math.ceil(obs.y1 / resolution)))
            mask[iy0:iy1, ix0:ix1] = True
        return mask


def _rects_overlap(a: Rect, b: Rect, margin: float = 0.0) -> bool:
    return not (
        a.x1 + margin < b.x0 - margin
        or a.x0 - margin > b.x1 + margin
        or a.y1 + margin < b.y0 - margin
        or a.y0 - margin > b.y1 + margin
    )
