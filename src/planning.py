from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mapping import OCCUPIED, OccupancyGrid
from .geometry import RectObstacle, circle_intersects_rect

try:
    from scipy.ndimage import distance_transform_edt
except Exception:  # scipy is optional; the planner has pure-Python fallbacks.
    distance_transform_edt = None


@dataclass
class PlanStats:
    path_length_m: float = 0.0
    min_clearance_m: float = 0.0
    avg_clearance_m: float = 0.0
    narrow_fraction: float = 0.0
    expanded_nodes: int = 0


@dataclass
class GridPlanner:
    grid: OccupancyGrid
    robot_radius: float
    inflation_margin: float
    world_obstacles: List[RectObstacle]
    clearance_weight: float = 2.4
    clearance_floor_m: float = 0.70
    narrow_penalty: float = 1.25
    unknown_edge_penalty: float = 0.30
    use_scipy_distance_transform: bool = True
    max_expansions: int = 12000
    _inflate_offsets: List[Tuple[int, int]] = field(init=False, repr=False)
    last_plan_stats: PlanStats = field(default_factory=PlanStats, init=False)

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
        """
        Return cells that the robot should not plan through.

        Fast path:
            scipy distance transform inflates obstacle cells in vectorized code.
        Fallback:
            old Python offset stamping.
        """
        occ = local_map == OCCUPIED
        unknown = local_map < 0
        inflate = self.robot_radius + self.inflation_margin

        if self.use_scipy_distance_transform and distance_transform_edt is not None and np.any(occ):
            # For every non-obstacle cell, distance to nearest obstacle cell.
            dist_to_occ = distance_transform_edt(~occ) * self.grid.res
            blocked = dist_to_occ <= inflate
            blocked |= unknown
            return blocked

        blocked = occ.copy()
        ys, xs = np.where(occ)
        for x, y in zip(xs.tolist(), ys.tolist()):
            for dx, dy in self._inflate_offsets:
                xx = x + dx
                yy = y + dy
                if 0 <= xx < self.grid.nx and 0 <= yy < self.grid.ny:
                    blocked[yy, xx] = True
        blocked |= unknown
        return blocked

    def exact_world_collision(self, x: float, y: float) -> bool:
        if x - self.robot_radius < 0 or y - self.robot_radius < 0:
            return True
        if x + self.robot_radius > self.grid.width_m or y + self.robot_radius > self.grid.height_m:
            return True
        return any(circle_intersects_rect(x, y, self.robot_radius, obs) for obs in self.world_obstacles)

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        local_map: np.ndarray,
        blocked: Optional[np.ndarray] = None,
        clearance: Optional[np.ndarray] = None,
        unknown_adjacent: Optional[np.ndarray] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        self.last_plan_stats = PlanStats()
        sx, sy = self.grid.world_to_grid(*start_xy)
        gx, gy = self.grid.world_to_grid(*goal_xy)

        if blocked is None:
            blocked = self.inflated_mask(local_map)
        if clearance is None:
            clearance = self._clearance_map(blocked)
        if unknown_adjacent is None:
            unknown_adjacent = self._unknown_adjacent_map(local_map)

        if blocked[sy, sx]:
            blocked = blocked.copy()
            found_start = self._nearest_free((sx, sy), blocked)
            if found_start is not None:
                sx, sy = found_start
            else:
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
        expansions = 0

        while pq:
            _, node = heapq.heappop(pq)
            if node == goal:
                break

            expansions += 1
            if self.max_expansions > 0 and expansions > self.max_expansions:
                self.last_plan_stats = PlanStats(expanded_nodes=expansions)
                return None

            cx, cy = node
            base_g = g_cost[node]
            for dx, dy in nbrs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.grid.nx and 0 <= ny < self.grid.ny):
                    continue
                if blocked[ny, nx]:
                    continue
                # no diagonal corner cutting
                if dx != 0 and dy != 0 and (blocked[cy, nx] or blocked[ny, cx]):
                    continue

                step = math.sqrt(2.0) if dx and dy else 1.0
                traversability = self._cell_traversal_cost(nx, ny, clearance, unknown_adjacent)
                cand = base_g + step * traversability
                nxt = (nx, ny)
                if cand < g_cost.get(nxt, float('inf')):
                    g_cost[nxt] = cand
                    parent[nxt] = node
                    f = cand + math.hypot(goal[0] - nx, goal[1] - ny)
                    heapq.heappush(pq, (f, nxt))

        if goal not in g_cost:
            self.last_plan_stats = PlanStats(expanded_nodes=expansions)
            return None

        cells = [goal]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        path = [self.grid.grid_to_world(x, y) for x, y in cells]
        path = self._compress(path, local_map, blocked)
        self.last_plan_stats = self._path_stats(path, clearance, expanded_nodes=expansions)
        return path

    def _nearest_free(self, start: Tuple[int, int], blocked: np.ndarray) -> Optional[Tuple[int, int]]:
        sx, sy = start
        for radius in range(1, 13):
            for yy in range(max(0, sy - radius), min(self.grid.ny, sy + radius + 1)):
                for xx in range(max(0, sx - radius), min(self.grid.nx, sx + radius + 1)):
                    if not blocked[yy, xx]:
                        return (xx, yy)
        return None

    def _compress(self, path: List[Tuple[float, float]], local_map: np.ndarray, blocked: Optional[np.ndarray] = None) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        anchor = path[0]
        for i in range(2, len(path)):
            if not self._segment_free(anchor, path[i], local_map, blocked):
                anchor = path[i - 1]
                out.append(anchor)
        out.append(path[-1])
        return out

    def _segment_free(self, p0: Tuple[float, float], p1: Tuple[float, float], local_map: np.ndarray,
                      blocked: Optional[np.ndarray] = None) -> bool:
        if blocked is None:
            blocked = self.inflated_mask(local_map)
        x0, y0 = p0
        x1, y1 = p1
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(d / max(0.10, self.grid.res / 2))))
        for i in range(n + 1):
            t = i / n
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            gx, gy = self.grid.world_to_grid(x, y)
            if blocked[gy, gx] or self.exact_world_collision(x, y):
                return False
        return True

    def _clearance_map(self, blocked: np.ndarray) -> np.ndarray:
        """
        Distance from each cell to the nearest blocked cell in meters.
        """
        if self.use_scipy_distance_transform and distance_transform_edt is not None and np.any(blocked):
            return distance_transform_edt(~blocked) * self.grid.res

        clearance = np.full((self.grid.ny, self.grid.nx), np.inf, dtype=float)
        pq: List[Tuple[float, Tuple[int, int]]] = []
        for y, x in zip(*np.where(blocked)):
            clearance[y, x] = 0.0
            heapq.heappush(pq, (0.0, (x, y)))
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while pq:
            cur_d, (cx, cy) = heapq.heappop(pq)
            if cur_d > clearance[cy, cx]:
                continue
            for dx, dy in nbrs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.grid.nx and 0 <= ny < self.grid.ny):
                    continue
                step = math.sqrt(2.0) if dx and dy else 1.0
                nd = cur_d + step
                if nd < clearance[ny, nx]:
                    clearance[ny, nx] = nd
                    heapq.heappush(pq, (nd, (nx, ny)))
        return clearance * self.grid.res

    def _unknown_adjacent_map(self, local_map: np.ndarray) -> np.ndarray:
        unknown = local_map < 0
        adj = unknown.copy()
        adj[1:, :] |= unknown[:-1, :]
        adj[:-1, :] |= unknown[1:, :]
        adj[:, 1:] |= unknown[:, :-1]
        adj[:, :-1] |= unknown[:, 1:]
        adj[1:, 1:] |= unknown[:-1, :-1]
        adj[1:, :-1] |= unknown[:-1, 1:]
        adj[:-1, 1:] |= unknown[1:, :-1]
        adj[:-1, :-1] |= unknown[1:, 1:]
        return adj

    def _cell_traversal_cost(self, gx: int, gy: int, clearance: np.ndarray, unknown_adjacent: np.ndarray) -> float:
        c = float(clearance[gy, gx])
        eps = max(0.05, 0.25 * self.grid.res)
        wall_penalty = self.clearance_weight / max(c + eps, eps)
        if c >= self.clearance_floor_m:
            wall_penalty *= 0.25
        narrow_pen = self.narrow_penalty if c < self.clearance_floor_m else 0.0
        unknown_adj_pen = self.unknown_edge_penalty if bool(unknown_adjacent[gy, gx]) else 0.0
        return 1.0 + wall_penalty + narrow_pen + unknown_adj_pen

    def _path_stats(self, path: List[Tuple[float, float]], clearance: np.ndarray, expanded_nodes: int = 0) -> PlanStats:
        if not path:
            return PlanStats(expanded_nodes=expanded_nodes)
        clear_vals: List[float] = []
        path_len = 0.0
        for i, (x, y) in enumerate(path):
            gx, gy = self.grid.world_to_grid(x, y)
            clear_vals.append(float(clearance[gy, gx]))
            if i > 0:
                x0, y0 = path[i - 1]
                path_len += math.hypot(x - x0, y - y0)
        if not clear_vals:
            return PlanStats(path_length_m=path_len, expanded_nodes=expanded_nodes)
        thresh = self.clearance_floor_m
        narrow = sum(1 for v in clear_vals if v < thresh) / max(1, len(clear_vals))
        return PlanStats(
            path_length_m=path_len,
            min_clearance_m=min(clear_vals),
            avg_clearance_m=sum(clear_vals) / len(clear_vals),
            narrow_fraction=narrow,
            expanded_nodes=expanded_nodes,
        )
