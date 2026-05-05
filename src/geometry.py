"""Small geometry primitives for the fresh environment editor."""
from __future__ import annotations

import math
from dataclasses import dataclass


Point = tuple[float, float]
Pose = tuple[float, float, float]


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "Rect":
        return Rect(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )

    @property
    def center(self) -> Point:
        r = self.normalized()
        return ((r.x0 + r.x1) * 0.5, (r.y0 + r.y1) * 0.5)

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        r = self.normalized()
        return (
            r.x0 - margin <= point[0] <= r.x1 + margin
            and r.y0 - margin <= point[1] <= r.y1 + margin
        )

    def corners(self) -> list[Point]:
        r = self.normalized()
        return [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]


def unit_from_angle(theta: float) -> Point:
    return math.cos(theta), math.sin(theta)


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    eps = 1e-9
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if abs(o1) <= eps and _on_segment(a, c, b):
        return True
    if abs(o2) <= eps and _on_segment(a, d, b):
        return True
    if abs(o3) <= eps and _on_segment(c, a, d):
        return True
    if abs(o4) <= eps and _on_segment(c, b, d):
        return True
    return (o1 > 0.0) != (o2 > 0.0) and (o3 > 0.0) != (o4 > 0.0)


def segment_intersects_rect(a: Point, b: Point, rect: Rect, margin: float = 0.0) -> bool:
    r = Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin).normalized()
    if r.contains(a) or r.contains(b):
        return True
    corners = r.corners()
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(segments_intersect(a, b, c, d) for c, d in edges)
