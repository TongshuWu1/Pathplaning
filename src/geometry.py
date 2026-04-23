from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class RectObstacle:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def xmin(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def xmax(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def ymin(self) -> float:
        return self.cy - self.h / 2.0

    @property
    def ymax(self) -> float:
        return self.cy + self.h / 2.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def circle_intersects_rect(x: float, y: float, radius: float, rect: RectObstacle) -> bool:
    qx = clamp(x, rect.xmin, rect.xmax)
    qy = clamp(y, rect.ymin, rect.ymax)
    return (x - qx) ** 2 + (y - qy) ** 2 <= radius ** 2


def point_in_rect(x: float, y: float, rect: RectObstacle) -> bool:
    return rect.xmin <= x <= rect.xmax and rect.ymin <= y <= rect.ymax


def ray_rect_distance(x0: float, y0: float, dx: float, dy: float, rect: RectObstacle, max_dist: float) -> Optional[float]:
    eps = 1e-9
    tmin = -float('inf')
    tmax = float('inf')

    if abs(dx) < eps:
        if x0 < rect.xmin or x0 > rect.xmax:
            return None
    else:
        tx1 = (rect.xmin - x0) / dx
        tx2 = (rect.xmax - x0) / dx
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))

    if abs(dy) < eps:
        if y0 < rect.ymin or y0 > rect.ymax:
            return None
    else:
        ty1 = (rect.ymin - y0) / dy
        ty2 = (rect.ymax - y0) / dy
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))

    if tmax < max(tmin, 0.0):
        return None
    hit_t = tmin if tmin >= 0.0 else tmax
    if hit_t < 0.0 or hit_t > max_dist:
        return None
    return hit_t


def segment_hits_rect(p0: Tuple[float, float], p1: Tuple[float, float], rect: RectObstacle) -> bool:
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-9:
        return point_in_rect(x0, y0, rect)
    return ray_rect_distance(x0, y0, dx / d, dy / d, rect, d) is not None


def line_of_sight(p0: Tuple[float, float], p1: Tuple[float, float], obstacles: Iterable[RectObstacle]) -> bool:
    return not any(segment_hits_rect(p0, p1, obs) for obs in obstacles)
