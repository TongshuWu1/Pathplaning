"""Small geometry helpers used by the baseline simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


Pose = tuple[float, float, float]
Point = tuple[float, float]


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def distance(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def unit_from_angle(theta: float) -> tuple[float, float]:
    return math.cos(theta), math.sin(theta)


def angle_to(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def segment_length(points: Iterable[Point]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    return sum(distance(a, b) for a, b in zip(pts[:-1], pts[1:]))


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "Rect":
        return Rect(min(self.x0, self.x1), min(self.y0, self.y1), max(self.x0, self.x1), max(self.y0, self.y1))

    @property
    def center(self) -> Point:
        r = self.normalized()
        return ((r.x0 + r.x1) * 0.5, (r.y0 + r.y1) * 0.5)

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        # Rectangles are normalized at construction in the world generator.
        return (self.x0 - margin <= p[0] <= self.x1 + margin) and (self.y0 - margin <= p[1] <= self.y1 + margin)

    def corners(self) -> list[Point]:
        r = self.normalized()
        return [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]


def ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def segment_intersects_rect(a: Point, b: Point, rect: Rect, margin: float = 0.0) -> bool:
    r = Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin).normalized()
    if r.contains(a) or r.contains(b):
        return True
    corners = r.corners()
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(segments_intersect(a, b, c, d) for c, d in edges)


def covariance_ellipse(cov_xy: np.ndarray, scale: float = 2.0, samples: int = 40) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov_xy, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return np.array([]), np.array([])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    theta = np.linspace(0.0, 2.0 * math.pi, samples)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = vecs @ np.diag(np.sqrt(vals) * scale) @ circle
    return ellipse[0], ellipse[1]
