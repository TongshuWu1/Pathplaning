from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import heapq
import math
import numpy as np

from .mapping import FREE, UNKNOWN, OccupancyGrid, connected_components, frontier_mask
from .robots.packets import TeammatePacket


@dataclass
class FrontierChoice:
    target_xy: Tuple[float, float]
    score: float
    info_gain: float
    travel_cost: float
    overlap_penalty: float


class LocalFrontierPolicy:
    def __init__(self, grid: OccupancyGrid, sensing_radius: float, trace_radius: float, trace_gain: float,
                 current_pos_gain: float, target_gain: float, decay_s: float):
        self.grid = grid
        self.sensing_radius = sensing_radius
        self.trace_radius = trace_radius
        self.trace_gain = trace_gain
        self.current_pos_gain = current_pos_gain
        self.target_gain = target_gain
        self.decay_s = decay_s

    def choose_target(self, now: float, local_map: np.ndarray, robot_xy: Tuple[float, float],
                      teammate_packets: Sequence[TeammatePacket], planner) -> Optional[FrontierChoice]:
        mask = frontier_mask(local_map)
        comps = connected_components(mask, min_size=5)
        if not comps:
            return None
        blocked = planner.inflated_mask(local_map)
        dist_grid = self._reachable_distances(robot_xy, blocked)
        trace_field = self._trace_penalty_field(now, teammate_packets)
        best: Optional[FrontierChoice] = None
        for comp in comps:
            cx = sum(c[0] for c in comp) / len(comp)
            cy = sum(c[1] for c in comp) / len(comp)
            target = self._find_reachable_standoff((cx, cy), dist_grid, local_map)
            if target is None:
                continue
            tx, ty = target
            info_gain = self._unknown_gain_around(target, local_map)
            gx, gy = self.grid.world_to_grid(tx, ty)
            travel_cost = float(dist_grid[gy, gx]) * self.grid.res
            overlap = float(trace_field[gy, gx])
            score = 1.35 * info_gain - 0.70 * travel_cost - 1.0 * overlap
            choice = FrontierChoice(target_xy=target, score=score, info_gain=info_gain,
                                    travel_cost=travel_cost, overlap_penalty=overlap)
            if best is None or choice.score > best.score:
                best = choice
        return best

    def _reachable_distances(self, robot_xy: Tuple[float, float], blocked: np.ndarray) -> np.ndarray:
        sx, sy = self.grid.world_to_grid(*robot_xy)
        if blocked[sy, sx]:
            blocked = blocked.copy()
            blocked[sy, sx] = False
        dist = np.full((self.grid.ny, self.grid.nx), np.inf, dtype=float)
        dist[sy, sx] = 0.0
        pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, (sx, sy))]
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while pq:
            cur_d, (cx, cy) = heapq.heappop(pq)
            if cur_d > dist[cy, cx]:
                continue
            for dx, dy in nbrs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.grid.nx and 0 <= ny < self.grid.ny):
                    continue
                if blocked[ny, nx]:
                    continue
                if dx != 0 and dy != 0 and (blocked[cy, nx] or blocked[ny, cx]):
                    continue
                step = math.sqrt(2.0) if dx and dy else 1.0
                nd = cur_d + step
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, (nx, ny)))
        return dist

    def _trace_penalty_field(self, now: float, teammate_packets: Sequence[TeammatePacket]) -> np.ndarray:
        field = np.zeros((self.grid.ny, self.grid.nx), dtype=float)
        cells = int(np.ceil(self.trace_radius / self.grid.res))
        for packet in teammate_packets:
            age = max(0.0, now - packet.timestamp)
            decay = math.exp(-age / max(1e-6, self.decay_s))
            self._deposit(field, packet.pose_xy, self.current_pos_gain * decay, cells)
            if packet.target_xy is not None:
                self._deposit(field, packet.target_xy, self.target_gain * decay, cells)
            trimmed = packet.path_xy[-25:]
            for idx, pt in enumerate(trimmed):
                age_factor = (idx + 1) / max(1, len(trimmed))
                self._deposit(field, pt, self.trace_gain * decay * age_factor, cells)
        return field

    def _deposit(self, field: np.ndarray, xy: Tuple[float, float], weight: float, cells: int) -> None:
        gx, gy = self.grid.world_to_grid(*xy)
        for yy in range(max(0, gy - cells), min(self.grid.ny, gy + cells + 1)):
            for xx in range(max(0, gx - cells), min(self.grid.nx, gx + cells + 1)):
                wx, wy = self.grid.grid_to_world(xx, yy)
                d2 = (wx - xy[0]) ** 2 + (wy - xy[1]) ** 2
                sigma2 = max(self.trace_radius, self.grid.res) ** 2
                field[yy, xx] += weight * math.exp(-0.5 * d2 / sigma2)

    def _unknown_gain_around(self, target_xy: Tuple[float, float], local_map: np.ndarray) -> float:
        gx, gy = self.grid.world_to_grid(*target_xy)
        cells = int(np.ceil(self.sensing_radius / self.grid.res))
        count = 0
        for yy in range(max(0, gy - cells), min(self.grid.ny, gy + cells + 1)):
            for xx in range(max(0, gx - cells), min(self.grid.nx, gx + cells + 1)):
                if local_map[yy, xx] == UNKNOWN:
                    wx, wy = self.grid.grid_to_world(xx, yy)
                    if (wx - target_xy[0]) ** 2 + (wy - target_xy[1]) ** 2 <= self.sensing_radius ** 2:
                        count += 1
        return float(count)

    def _find_reachable_standoff(self, frontier_grid_xy: Tuple[float, float], dist_grid: np.ndarray,
                                 local_map: np.ndarray) -> Optional[Tuple[float, float]]:
        fx, fy = frontier_grid_xy
        best = None
        for radius_cells in range(0, 8):
            for yy in range(max(0, int(fy) - radius_cells), min(self.grid.ny, int(fy) + radius_cells + 1)):
                for xx in range(max(0, int(fx) - radius_cells), min(self.grid.nx, int(fx) + radius_cells + 1)):
                    if local_map[yy, xx] != FREE or not np.isfinite(dist_grid[yy, xx]):
                        continue
                    wx, wy = self.grid.grid_to_world(xx, yy)
                    frontier_dist2 = (xx - fx) ** 2 + (yy - fy) ** 2
                    score = frontier_dist2 + 0.05 * float(dist_grid[yy, xx])
                    cand = (wx, wy)
                    if best is None or score < best[0]:
                        best = (score, cand)
            if best is not None:
                return best[1]
        return None
