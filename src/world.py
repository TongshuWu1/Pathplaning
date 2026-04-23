from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

from .config import SimConfig
from .geometry import RectObstacle, dist, circle_intersects_rect


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    name: str = ''
    is_home: bool = False


@dataclass
class World:
    cfg: SimConfig
    obstacles: List[RectObstacle]
    landmarks: List[Landmark]
    home_base: RectObstacle
    home_marker: Landmark

    @property
    def all_landmarks(self) -> List[Landmark]:
        return [self.home_marker, *self.landmarks]

    @classmethod
    def generate(cls, cfg: SimConfig) -> 'World':
        rng = random.Random(cfg.seed)
        home_base = RectObstacle(
            cx=cfg.home_base_size / 2.0,
            cy=cfg.home_base_size / 2.0,
            w=cfg.home_base_size,
            h=cfg.home_base_size,
        )
        home_marker = Landmark(x=home_base.cx, y=home_base.cy, name='Home', is_home=True)
        spawn_center = (
            cfg.home_base_padding + 0.8,
            cfg.home_base_padding + 0.8,
        )

        obstacles: List[RectObstacle] = []
        tries = 0
        while len(obstacles) < cfg.obstacle_count and tries < 10000:
            tries += 1
            w = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max)
            h = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max)
            cx = rng.uniform(cfg.world_margin + w / 2.0, cfg.world_w - cfg.world_margin - w / 2.0)
            cy = rng.uniform(cfg.world_margin + h / 2.0, cfg.world_h - cfg.world_margin - h / 2.0)
            rect = RectObstacle(cx=cx, cy=cy, w=w, h=h)
            if _rects_overlap(rect, home_base, margin=cfg.home_base_padding + 0.35):
                continue
            if dist((cx, cy), spawn_center) < cfg.spawn_clear_radius:
                continue
            if any(_rects_overlap(rect, other, margin=cfg.obstacle_gap_margin) for other in obstacles):
                continue
            obstacles.append(rect)

        landmarks: List[Landmark] = []
        tries = 0
        while len(landmarks) < cfg.landmark_count and tries < 10000:
            tries += 1
            x = rng.uniform(cfg.world_margin, cfg.world_w - cfg.world_margin)
            y = rng.uniform(cfg.world_margin, cfg.world_h - cfg.world_margin)
            if _point_in_or_near_rect(x, y, home_base, 0.75):
                continue
            if dist((x, y), spawn_center) < cfg.spawn_clear_radius * 0.70:
                continue
            if any(circle_intersects_rect(x, y, 0.5, obs) for obs in obstacles):
                continue
            if any(dist((x, y), (lm.x, lm.y)) < 1.5 for lm in landmarks):
                continue
            landmarks.append(Landmark(x=x, y=y, name=f'L{len(landmarks) + 1}'))
        return cls(cfg=cfg, obstacles=obstacles, landmarks=landmarks, home_base=home_base, home_marker=home_marker)


def _rects_overlap(a: RectObstacle, b: RectObstacle, margin: float = 0.0) -> bool:
    return not (
        a.xmax + margin < b.xmin - margin or
        a.xmin - margin > b.xmax + margin or
        a.ymax + margin < b.ymin - margin or
        a.ymin - margin > b.ymax + margin
    )


def _point_in_or_near_rect(x: float, y: float, rect: RectObstacle, margin: float) -> bool:
    return (
        rect.xmin - margin <= x <= rect.xmax + margin and
        rect.ymin - margin <= y <= rect.ymax + margin
    )
