from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mapping import OCCUPIED, OccupancyGrid
from .geometry import RectObstacle, circle_intersects_rect


@dataclass
class GridPlanner:
    grid: OccupancyGrid
    robot_radius: float
    inflation_margin: float
    world_obstacles: List[RectObstacle]
    _inflate_offsets: List[Tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        inflate = self.robot_radius + self.inflation_margin
        cells = int(math.ceil(inflate / self.grid.res))
        offsets: List[Tuple[int, int]] = []
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                if (dx * self.grid.res) ** 2 + (dy * self.grid.res) ** 2 <= inflate ** 2:
                    offsets.append((dx, dy))
        self._inflate_offsets = offsets

    def inflated_mask(self, local_map: np.ndarray) -> np.ndarray:
        occ = local_map == OCCUPIED
        blocked = occ.copy()
        ys, xs = np.where(occ)
        for x, y in zip(xs.tolist(), ys.tolist()):
            for dx, dy in self._inflate_offsets:
                xx = x + dx
                yy = y + dy
                if 0 <= xx < self.grid.nx and 0 <= yy < self.grid.ny:
                    blocked[yy, xx] = True
        blocked |= (local_map < 0)
        return blocked

    def exact_world_collision(self, x: float, y: float) -> bool:
        if x - self.robot_radius < 0 or y - self.robot_radius < 0:
            return True
        if x + self.robot_radius > self.grid.width_m or y + self.robot_radius > self.grid.height_m:
            return True
        return any(circle_intersects_rect(x, y, self.robot_radius, obs) for obs in self.world_obstacles)

    def plan(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], local_map: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        sx, sy = self.grid.world_to_grid(*start_xy)
        gx, gy = self.grid.world_to_grid(*goal_xy)
        blocked = self.inflated_mask(local_map)
        if blocked[sy, sx]:
            blocked[sy, sx] = False
        if blocked[gy, gx]:
            found = self._nearest_free((gx, gy), blocked)
            if found is None:
                return None
            gx, gy = found
        start = (sx, sy)
        goal = (gx, gy)
        pq: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(pq, (0.0, start))
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while pq:
            _, node = heapq.heappop(pq)
            if node == goal:
                break
            cx, cy = node
            for dx, dy in nbrs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.grid.nx and 0 <= ny < self.grid.ny):
                    continue
                if blocked[ny, nx]:
                    continue
                if dx != 0 and dy != 0 and (blocked[cy, nx] or blocked[ny, cx]):
                    continue
                step = math.sqrt(2.0) if dx and dy else 1.0
                cand = g_cost[node] + step
                nxt = (nx, ny)
                if cand < g_cost.get(nxt, float('inf')):
                    g_cost[nxt] = cand
                    parent[nxt] = node
                    f = cand + math.hypot(goal[0] - nx, goal[1] - ny)
                    heapq.heappush(pq, (f, nxt))
        if goal not in g_cost:
            return None
        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        path = [self.grid.grid_to_world(x, y) for x, y in cells]
        return self._compress(path, local_map)

    def _nearest_free(self, start: Tuple[int, int], blocked: np.ndarray) -> Optional[Tuple[int, int]]:
        sx, sy = start
        for radius in range(1, 9):
            for yy in range(max(0, sy - radius), min(self.grid.ny, sy + radius + 1)):
                for xx in range(max(0, sx - radius), min(self.grid.nx, sx + radius + 1)):
                    if not blocked[yy, xx]:
                        return (xx, yy)
        return None

    def _compress(self, path: List[Tuple[float, float]], local_map: np.ndarray) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        anchor = path[0]
        for i in range(2, len(path)):
            if not self._segment_free(anchor, path[i], local_map):
                anchor = path[i - 1]
                out.append(anchor)
        out.append(path[-1])
        return out

    def _segment_free(self, p0: Tuple[float, float], p1: Tuple[float, float], local_map: np.ndarray) -> bool:
        blocked = self.inflated_mask(local_map)
        x0, y0 = p0
        x1, y1 = p1
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(d / max(0.08, self.grid.res / 3))))
        for i in range(n + 1):
            t = i / n
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            gx, gy = self.grid.world_to_grid(x, y)
            if blocked[gy, gx] or self.exact_world_collision(x, y):
                return False
        return True
