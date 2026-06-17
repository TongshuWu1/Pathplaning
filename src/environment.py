"""Editable environment model for the next simulator generation."""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Point, Pose, Rect, segment_intersects_rect, unit_from_angle


ENV_SCHEMA_VERSION = 2


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class EnvironmentRect:
    id: str
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    name: str = ""
    group: str | None = None

    def normalized(self) -> "EnvironmentRect":
        return EnvironmentRect(
            id=self.id,
            kind=self.kind,
            x0=min(float(self.x0), float(self.x1)),
            y0=min(float(self.y0), float(self.y1)),
            x1=max(float(self.x0), float(self.x1)),
            y1=max(float(self.y0), float(self.y1)),
            name=self.name,
            group=self.group,
        )

    @property
    def rect(self) -> Rect:
        r = self.normalized()
        return Rect(r.x0, r.y0, r.x1, r.y1)

    @property
    def center(self) -> Point:
        return self.rect.center

    def contains(self, point: Point, margin: float = 0.0) -> bool:
        return self.rect.contains(point, margin=margin)

    def moved(self, dx: float, dy: float) -> "EnvironmentRect":
        return EnvironmentRect(
            id=self.id,
            kind=self.kind,
            x0=self.x0 + dx,
            y0=self.y0 + dy,
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            name=self.name,
            group=self.group,
        )

    def clamped(self, width: float, height: float) -> "EnvironmentRect":
        r = self.normalized()
        rect_w = r.x1 - r.x0
        rect_h = r.y1 - r.y0
        x0 = min(max(0.0, r.x0), max(0.0, width - rect_w))
        y0 = min(max(0.0, r.y0), max(0.0, height - rect_h))
        return EnvironmentRect(self.id, self.kind, x0, y0, x0 + rect_w, y0 + rect_h, self.name, self.group)

    def to_dict(self) -> dict[str, Any]:
        r = self.normalized()
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "rect": [round(r.x0, 4), round(r.y0, 4), round(r.x1, 4), round(r.y1, 4)],
        }
        if self.name:
            data["name"] = self.name
        if self.group:
            data["group"] = self.group
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EnvironmentRect":
        x0, y0, x1, y1 = data["rect"]
        return EnvironmentRect(
            id=str(data.get("id") or _new_id(str(data.get("kind", "rect")))),
            kind=str(data.get("kind", "obstacle")),
            x0=float(x0),
            y0=float(y0),
            x1=float(x1),
            y1=float(y1),
            name=str(data.get("name", "")),
            group=str(data["group"]) if data.get("group") else None,
        ).normalized()


@dataclass
class EnvironmentPoint:
    id: str
    kind: str
    x: float
    y: float
    name: str = ""

    @property
    def xy(self) -> Point:
        return (float(self.x), float(self.y))

    def contains(self, point: Point, margin: float = 0.25) -> bool:
        return math.hypot(float(point[0]) - self.x, float(point[1]) - self.y) <= margin

    def moved(self, dx: float, dy: float) -> "EnvironmentPoint":
        return EnvironmentPoint(
            id=self.id,
            kind=self.kind,
            x=self.x + dx,
            y=self.y + dy,
            name=self.name,
        )

    def clamped(self, width: float, height: float) -> "EnvironmentPoint":
        return EnvironmentPoint(
            id=self.id,
            kind=self.kind,
            x=min(max(0.0, float(self.x)), float(width)),
            y=min(max(0.0, float(self.y)), float(height)),
            name=self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "xy": [round(float(self.x), 4), round(float(self.y), 4)],
        }
        if self.name:
            data["name"] = self.name
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EnvironmentPoint":
        x, y = data["xy"]
        return EnvironmentPoint(
            id=str(data.get("id") or _new_id(str(data.get("kind", "point")))),
            kind=str(data.get("kind", "landmark")),
            x=float(x),
            y=float(y),
            name=str(data.get("name", "")),
        )


@dataclass
class EditableEnvironment:
    name: str = "untitled"
    width: float = 30.0
    height: float = 30.0
    grid_size: float = 0.25
    wall_thickness: float = 0.25
    rects: list[EnvironmentRect] = field(default_factory=list)
    landmarks: list[EnvironmentPoint] = field(default_factory=list)
    home: EnvironmentRect = field(
        default_factory=lambda: EnvironmentRect("home", "home", 1.0, 1.0, 4.0, 4.0, "HOME")
    )
    target: Point = (27.0, 27.0)
    robot_length: float = 0.72
    robot_width: float = 0.42
    robot_safety_margin: float = 0.14

    def _grid(self) -> float:
        return max(0.05, float(self.grid_size))

    def snap_value(self, value: float) -> float:
        g = self._grid()
        return round(float(value) / g) * g

    def snap_point(self, point: Point) -> Point:
        return (
            min(max(0.0, self.snap_value(point[0])), self.width),
            min(max(0.0, self.snap_value(point[1])), self.height),
        )

    def snap_rect(self, rect: Rect, min_size: float | None = None) -> Rect:
        g = self._grid()
        min_extent = max(g, float(min_size if min_size is not None else g))
        r = rect.normalized()
        x0 = min(max(0.0, self.snap_value(r.x0)), self.width)
        y0 = min(max(0.0, self.snap_value(r.y0)), self.height)
        x1 = min(max(0.0, self.snap_value(r.x1)), self.width)
        y1 = min(max(0.0, self.snap_value(r.y1)), self.height)
        if x1 - x0 < min_extent:
            if x0 + min_extent <= self.width:
                x1 = x0 + min_extent
            else:
                x0 = max(0.0, x1 - min_extent)
        if y1 - y0 < min_extent:
            if y0 + min_extent <= self.height:
                y1 = y0 + min_extent
            else:
                y0 = max(0.0, y1 - min_extent)
        return Rect(x0, y0, x1, y1).normalized()

    def snap_all(self) -> None:
        self.grid_size = self._grid()
        self.wall_thickness = max(self._grid(), self.snap_value(self.wall_thickness))
        self.width = max(8.0, self.snap_value(self.width))
        self.height = max(8.0, self.snap_value(self.height))
        self.home = self._snap_rect_item(self.home)
        self.target = self.snap_point(self.target)
        self.rects = [self._snap_rect_item(item) for item in self.rects]
        self.landmarks = [self._snap_point_item(item) for item in self.landmarks]

    def _snap_rect_item(self, item: EnvironmentRect) -> EnvironmentRect:
        rect = self.snap_rect(item.rect)
        return EnvironmentRect(item.id, item.kind, rect.x0, rect.y0, rect.x1, rect.y1, item.name, item.group)

    def _snap_point_item(self, item: EnvironmentPoint) -> EnvironmentPoint:
        x, y = self.snap_point(item.xy)
        return EnvironmentPoint(item.id, item.kind, x, y, item.name)

    def add_rect(
        self,
        kind: str,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        name: str = "",
        group: str | None = None,
    ) -> EnvironmentRect | None:
        rect = self.snap_rect(Rect(x0, y0, x1, y1))
        item = EnvironmentRect(_new_id(kind), kind, rect.x0, rect.y0, rect.x1, rect.y1, name=name, group=group).normalized()
        if (item.x1 - item.x0) < 0.05 or (item.y1 - item.y0) < 0.05:
            return None
        item = self._snap_rect_item(item.clamped(self.width, self.height))
        self.rects.append(item)
        return item

    def add_landmark(self, x: float, y: float, name: str = "") -> EnvironmentPoint:
        idx = len(self.landmarks) + 1
        px, py = self.snap_point((x, y))
        point = EnvironmentPoint(
            id=_new_id("landmark"),
            kind="landmark",
            x=px,
            y=py,
            name=name or f"L{idx}",
        ).clamped(self.width, self.height)
        self.landmarks.append(point)
        return point

    def set_home(self, rect: Rect) -> None:
        item = EnvironmentRect("home", "home", rect.x0, rect.y0, rect.x1, rect.y1, "HOME")
        self.home = self._snap_rect_item(item.normalized().clamped(self.width, self.height))

    def set_target(self, point: Point) -> None:
        self.target = self.snap_point(point)

    def add_room(self, bounds: Rect, door_side: str = "bottom", door_width: float = 2.2) -> list[EnvironmentRect]:
        r = self.snap_rect(bounds).normalized()
        if r.x1 - r.x0 < 2.0 or r.y1 - r.y0 < 2.0:
            return []
        t = min(max(0.08, float(self.wall_thickness)), min(r.x1 - r.x0, r.y1 - r.y0) * 0.22)
        door_width = self.snap_value(min(max(0.6, float(door_width)), max(0.6, (r.x1 - r.x0) - 2.0 * t)))
        group = _new_id("room")
        items: list[EnvironmentRect] = []

        def add_wall(x0: float, y0: float, x1: float, y1: float, suffix: str) -> None:
            wall = self.add_rect("wall", x0, y0, x1, y1, name=f"room_{suffix}", group=group)
            if wall is not None:
                items.append(wall)

        mid_x = (r.x0 + r.x1) * 0.5
        mid_y = (r.y0 + r.y1) * 0.5
        half_door = door_width * 0.5
        if door_side == "bottom":
            add_wall(r.x0, r.y0, mid_x - half_door, r.y0 + t, "bottom_l")
            add_wall(mid_x + half_door, r.y0, r.x1, r.y0 + t, "bottom_r")
        else:
            add_wall(r.x0, r.y0, r.x1, r.y0 + t, "bottom")
        if door_side == "top":
            add_wall(r.x0, r.y1 - t, mid_x - half_door, r.y1, "top_l")
            add_wall(mid_x + half_door, r.y1 - t, r.x1, r.y1, "top_r")
        else:
            add_wall(r.x0, r.y1 - t, r.x1, r.y1, "top")
        if door_side == "left":
            add_wall(r.x0, r.y0, r.x0 + t, mid_y - half_door, "left_b")
            add_wall(r.x0, mid_y + half_door, r.x0 + t, r.y1, "left_t")
        else:
            add_wall(r.x0, r.y0, r.x0 + t, r.y1, "left")
        if door_side == "right":
            add_wall(r.x1 - t, r.y0, r.x1, mid_y - half_door, "right_b")
            add_wall(r.x1 - t, mid_y + half_door, r.x1, r.y1, "right_t")
        else:
            add_wall(r.x1 - t, r.y0, r.x1, r.y1, "right")
        return items

    def item_at(self, point: Point, margin: float = 0.08) -> EnvironmentRect | EnvironmentPoint | None:
        for landmark in reversed(self.landmarks):
            if landmark.contains(point, margin=max(0.22, margin)):
                return landmark
        for item in reversed(self.rects):
            if item.contains(point, margin=margin):
                return item
        if self.home.contains(point, margin=margin):
            return self.home
        return None

    def remove_item(self, item_id: str) -> bool:
        before = len(self.rects)
        self.rects = [item for item in self.rects if item.id != item_id]
        rect_removed = len(self.rects) != before
        before_landmarks = len(self.landmarks)
        self.landmarks = [item for item in self.landmarks if item.id != item_id]
        return rect_removed or len(self.landmarks) != before_landmarks

    def replace_item(self, item: EnvironmentRect | EnvironmentPoint) -> None:
        if item.id == self.home.id:
            if isinstance(item, EnvironmentRect):
                self.home = self._snap_rect_item(item.normalized().clamped(self.width, self.height))
            return
        if isinstance(item, EnvironmentPoint):
            for idx, old in enumerate(self.landmarks):
                if old.id == item.id:
                    self.landmarks[idx] = self._snap_point_item(item.clamped(self.width, self.height))
                    return
            return
        for idx, old in enumerate(self.rects):
            if old.id == item.id:
                self.rects[idx] = self._snap_rect_item(item.normalized().clamped(self.width, self.height))
                return

    @property
    def obstacle_rects(self) -> list[Rect]:
        return [item.rect for item in self.rects if item.kind in {"obstacle", "wall"}]

    def in_bounds(self, point: Point, margin: float = 0.0) -> bool:
        return margin <= point[0] <= self.width - margin and margin <= point[1] <= self.height - margin

    def is_free(self, point: Point, margin: float = 0.0) -> bool:
        if not self.in_bounds(point, margin=margin):
            return False
        return not any(rect.contains(point, margin=margin) for rect in self.obstacle_rects)

    def segment_free(self, a: Point, b: Point, margin: float = 0.0) -> bool:
        if not self.in_bounds(a, margin=margin) or not self.in_bounds(b, margin=margin):
            return False
        return not any(segment_intersects_rect(a, b, rect, margin=margin) for rect in self.obstacle_rects)

    def raycast(self, pose: Pose, rel_angle: float, max_range: float, step: float = 0.08) -> tuple[float, Point, bool]:
        x, y, theta = pose
        dx, dy = unit_from_angle(theta + rel_angle)
        r = 0.0
        last = (x, y)
        while r < max_range:
            r += step
            p = (x + dx * r, y + dy * r)
            if not self.in_bounds(p):
                return min(r, max_range), p, True
            if not self.is_free(p):
                return min(r, max_range), p, True
            last = p
        return max_range, last, False

    def raster_obstacle_mask(self, resolution: float) -> np.ndarray:
        nx = int(math.ceil(self.width / resolution))
        ny = int(math.ceil(self.height / resolution))
        mask = np.zeros((ny, nx), dtype=bool)
        for rect in self.obstacle_rects:
            ix0 = max(0, int(math.floor(rect.x0 / resolution)))
            ix1 = min(nx, int(math.ceil(rect.x1 / resolution)))
            iy0 = max(0, int(math.floor(rect.y0 / resolution)))
            iy1 = min(ny, int(math.ceil(rect.y1 / resolution)))
            mask[iy0:iy1, ix0:ix1] = True
        return mask

    def resize(self, width: float, height: float) -> None:
        self.width = max(8.0, self.snap_value(float(width)))
        self.height = max(8.0, self.snap_value(float(height)))
        t = min(max(self._grid(), self.snap_value(self.wall_thickness)), min(self.width, self.height) * 0.10)
        self.wall_thickness = t
        resized: list[EnvironmentRect] = []
        for item in self.rects:
            if item.name == "boundary_bottom":
                resized.append(EnvironmentRect(item.id, item.kind, 0.0, 0.0, self.width, t, item.name, item.group))
            elif item.name == "boundary_top":
                resized.append(EnvironmentRect(item.id, item.kind, 0.0, self.height - t, self.width, self.height, item.name, item.group))
            elif item.name == "boundary_left":
                resized.append(EnvironmentRect(item.id, item.kind, 0.0, 0.0, t, self.height, item.name, item.group))
            elif item.name == "boundary_right":
                resized.append(EnvironmentRect(item.id, item.kind, self.width - t, 0.0, self.width, self.height, item.name, item.group))
            else:
                resized.append(item.clamped(self.width, self.height))
        self.rects = [self._snap_rect_item(item) for item in resized]
        self.landmarks = [self._snap_point_item(item.clamped(self.width, self.height)) for item in self.landmarks]
        self.home = self._snap_rect_item(self.home.clamped(self.width, self.height))
        self.target = self.snap_point(self.target)
        self.robot_length = max(0.2, float(self.robot_length))
        self.robot_width = min(max(0.15, float(self.robot_width)), self.robot_length)
        self.robot_safety_margin = max(0.0, float(self.robot_safety_margin))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pathplaning.environment",
            "version": ENV_SCHEMA_VERSION,
            "name": self.name,
            "size": {"width": self.width, "height": self.height},
            "grid_size": self.grid_size,
            "wall_thickness": self.wall_thickness,
            "home": self.home.to_dict(),
            "target": {"xy": [round(self.target[0], 4), round(self.target[1], 4)]},
            "robot": {
                "length": round(float(self.robot_length), 4),
                "width": round(float(self.robot_width), 4),
                "safety_margin": round(float(self.robot_safety_margin), 4),
            },
            "landmarks": [item.to_dict() for item in self.landmarks],
            "rects": [item.to_dict() for item in self.rects],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EditableEnvironment":
        size = data.get("size", {})
        target_xy = data.get("target", {}).get("xy", [27.0, 27.0])
        robot = data.get("robot", {})
        env = EditableEnvironment(
            name=str(data.get("name", "untitled")),
            width=float(size.get("width", data.get("width", 30.0))),
            height=float(size.get("height", data.get("height", 30.0))),
            grid_size=float(data.get("grid_size", 0.25)),
            wall_thickness=float(data.get("wall_thickness", 0.25)),
            rects=[EnvironmentRect.from_dict(item) for item in data.get("rects", [])],
            landmarks=[EnvironmentPoint.from_dict(item) for item in data.get("landmarks", [])],
            home=EnvironmentRect.from_dict(data.get("home", {"id": "home", "kind": "home", "rect": [1.0, 1.0, 4.0, 4.0]})),
            target=(float(target_xy[0]), float(target_xy[1])),
            robot_length=float(robot.get("length", 0.72)),
            robot_width=float(robot.get("width", 0.42)),
            robot_safety_margin=float(robot.get("safety_margin", 0.14)),
        )
        env.home.id = "home"
        env.home.kind = "home"
        env.resize(env.width, env.height)
        return env


def make_default_environment() -> EditableEnvironment:
    env = EditableEnvironment(name="environment_01")
    t = env.wall_thickness
    env.add_rect("wall", 0.0, 0.0, env.width, t, "boundary_bottom")
    env.add_rect("wall", 0.0, env.height - t, env.width, env.height, "boundary_top")
    env.add_rect("wall", 0.0, 0.0, t, env.height, "boundary_left")
    env.add_rect("wall", env.width - t, 0.0, env.width, env.height, "boundary_right")
    env.add_landmark(6.0, 6.0)
    env.add_landmark(24.0, 7.0)
    env.add_landmark(9.0, 24.0)
    return env


def save_environment(env: EditableEnvironment, path: str | Path) -> Path:
    out = Path(path)
    if out.suffix.lower() != ".json":
        out = out.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    env.snap_all()
    out.write_text(json.dumps(env.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out


def load_environment(path: str | Path) -> EditableEnvironment:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EditableEnvironment.from_dict(data)
