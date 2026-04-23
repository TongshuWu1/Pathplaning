from __future__ import annotations

from typing import List, Optional, Sequence
import math
import numpy as np

from .packets import RoutePoint


def _wrap_rad(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def downsample_route_points(points: Sequence[RoutePoint], max_points: int) -> List[RoutePoint]:
    if max_points <= 0:
        return []
    pts = list(points)
    if len(pts) <= max_points:
        return pts
    anchors = []
    if pts:
        anchors.append(pts[0])
    anchors.extend(pt for pt in pts[1:-1] if pt.point_type not in {"ROUTE", "RECENT"})
    if len(anchors) >= max_points:
        # Too many semantic anchors: keep them spread in time.
        idxs = np.linspace(0, len(anchors) - 1, num=max_points, dtype=int)
        return [anchors[int(i)] for i in idxs]
    need = max_points - len(anchors)
    regular = [pt for pt in pts[1:-1] if pt.point_type in {"ROUTE", "RECENT"}]
    picked: List[RoutePoint] = []
    if regular and need > 1:
        idxs = np.linspace(0, len(regular) - 1, num=max(0, need - 1), dtype=int)
        seen_idx = set()
        for idx in idxs:
            idx = int(idx)
            if idx in seen_idx:
                continue
            picked.append(regular[idx])
            seen_idx.add(idx)
    combined = [*anchors, *picked]
    if pts:
        combined.append(pts[-1])
    combined = sorted(combined, key=lambda pt: (pt.t, pt.point_type != "HOME"))
    dedup: List[RoutePoint] = []
    seen = set()
    for pt in combined:
        key = (round(pt.x, 3), round(pt.y, 3), pt.point_type, round(pt.t, 2))
        if key in seen:
            continue
        dedup.append(pt)
        seen.add(key)
    return dedup[-max_points:]


class ExecutedRouteMemory:
    def __init__(
        self,
        keep_dist: float,
        recent_keep_dist: float,
        turn_keep_deg: float,
        semantic_merge_dist: float,
        max_route_points: int,
        max_recent_points: int,
    ) -> None:
        self.keep_dist = keep_dist
        self.recent_keep_dist = recent_keep_dist
        self.turn_keep_rad = math.radians(turn_keep_deg)
        self.semantic_merge_dist = semantic_merge_dist
        self.max_route_points = max_route_points
        self.max_recent_points = max_recent_points
        self.route_points: List[RoutePoint] = []
        self.recent_trail: List[RoutePoint] = []
        self.semantic_points: List[RoutePoint] = []
        self._last_heading_rad: Optional[float] = None

    def initialize(self, x: float, y: float, t: float) -> None:
        home = RoutePoint(float(x), float(y), float(t), point_type="HOME", note="spawn/home")
        self.route_points = [home]
        self.recent_trail = [home]
        self.semantic_points = [home]

    def record_motion(self, x: float, y: float, heading_deg: float, t: float) -> None:
        heading_rad = math.radians(heading_deg)
        if not self.route_points:
            self.initialize(x, y, t)
            self._last_heading_rad = heading_rad
            return

        if self._should_keep_recent(x, y, heading_rad):
            self._append_recent(RoutePoint(float(x), float(y), float(t), point_type="RECENT", note="recent_trail"))

        last = self.route_points[-1]
        dist = math.hypot(x - last.x, y - last.y)
        heading_change = 0.0 if self._last_heading_rad is None else abs(_wrap_rad(heading_rad - self._last_heading_rad))
        keep = False
        point_type = "ROUTE"
        note = ""
        if dist >= self.keep_dist:
            keep = True
            note = "distance"
        if heading_change >= self.turn_keep_rad and dist >= 0.35 * self.keep_dist:
            keep = True
            point_type = "TURN"
            note = f"turn={math.degrees(heading_change):.1f}deg"
        if keep:
            pt = RoutePoint(float(x), float(y), float(t), point_type=point_type, note=note)
            self.route_points.append(pt)
            self.route_points = downsample_route_points(self.route_points, self.max_route_points)
            if point_type != "ROUTE":
                self._append_semantic(pt)
        self._last_heading_rad = heading_rad

    def record_event(self, x: float, y: float, t: float, point_type: str, note: str = "") -> None:
        pt = RoutePoint(float(x), float(y), float(t), point_type=point_type, note=note)
        self._append_semantic(pt)
        # Important events become part of the persistent route summary too.
        if not self.route_points or math.hypot(pt.x - self.route_points[-1].x, pt.y - self.route_points[-1].y) >= 0.25 * self.keep_dist:
            self.route_points.append(pt)
            self.route_points = downsample_route_points(self.route_points, self.max_route_points)

    def snapshot_route_points(self, max_points: int) -> List[RoutePoint]:
        return downsample_route_points(self.route_points, max_points)

    def snapshot_recent_trail(self, max_points: int) -> List[RoutePoint]:
        return downsample_route_points(self.recent_trail, max_points)

    def snapshot_semantic_points(self, max_points: int) -> List[RoutePoint]:
        return downsample_route_points(self.semantic_points, max_points)

    def _should_keep_recent(self, x: float, y: float, heading_rad: float) -> bool:
        if not self.recent_trail:
            return True
        last = self.recent_trail[-1]
        dist = math.hypot(x - last.x, y - last.y)
        heading_change = 0.0 if self._last_heading_rad is None else abs(_wrap_rad(heading_rad - self._last_heading_rad))
        return dist >= self.recent_keep_dist or heading_change >= 0.5 * self.turn_keep_rad

    def _append_recent(self, pt: RoutePoint) -> None:
        if self.recent_trail:
            last = self.recent_trail[-1]
            if math.hypot(pt.x - last.x, pt.y - last.y) < 1e-6:
                return
        self.recent_trail.append(pt)
        if len(self.recent_trail) > self.max_recent_points:
            self.recent_trail = self.recent_trail[-self.max_recent_points:]

    def _append_semantic(self, pt: RoutePoint) -> None:
        for old in reversed(self.semantic_points[-8:]):
            if old.point_type == pt.point_type and math.hypot(pt.x - old.x, pt.y - old.y) < self.semantic_merge_dist:
                return
        self.semantic_points.append(pt)
        self.semantic_points = downsample_route_points(self.semantic_points, self.max_route_points)
