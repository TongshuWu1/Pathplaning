from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import heapq
import math
import numpy as np

from .mapping import FREE, UNKNOWN, OccupancyGrid, connected_components, frontier_mask
from .robots.packets import TeammatePacket


@dataclass
class FrontierRegion:
    region_id: int
    centroid_xy: Tuple[float, float]
    frontier_size: int
    target_xy: Tuple[float, float]
    travel_cost: float
    info_gain: float
    overlap_penalty: float
    claim_penalty: float
    switch_penalty: float
    stay_bonus: float


@dataclass
class FrontierChoice:
    target_xy: Tuple[float, float]
    score: float
    info_gain: float
    travel_cost: float
    overlap_penalty: float
    region_id: int
    region_center_xy: Tuple[float, float]
    claim_penalty: float = 0.0
    switch_penalty: float = 0.0
    stay_bonus: float = 0.0


class LocalFrontierPolicy:
    def __init__(self, grid: OccupancyGrid, sensing_radius: float, trace_radius: float, trace_gain: float,
                 current_pos_gain: float, target_gain: float, decay_s: float,
                 claim_radius: float = 4.5, claim_penalty: float = 55.0, same_cycle_penalty: float = 75.0,
                 switch_penalty: float = 18.0, stay_bonus: float = 10.0):
        self.grid = grid
        self.sensing_radius = sensing_radius
        self.trace_radius = trace_radius
        self.trace_gain = trace_gain
        self.current_pos_gain = current_pos_gain
        self.target_gain = target_gain
        self.decay_s = decay_s
        self.claim_radius = claim_radius
        self.claim_penalty = claim_penalty
        self.same_cycle_penalty = same_cycle_penalty
        self.switch_penalty_value = switch_penalty
        self.stay_bonus_value = stay_bonus

    def choose_target(self, now: float, local_map: np.ndarray, robot_xy: Tuple[float, float],
                      teammate_packets: Sequence[TeammatePacket], planner,
                      current_region_id: Optional[int] = None, region_hold_active: bool = False,
                      provisional_claims: Optional[Dict[int, Tuple[float, float]]] = None) -> Optional[FrontierChoice]:
        mask = frontier_mask(local_map)
        comps = connected_components(mask, min_size=5)
        if not comps:
            return None
        blocked = planner.inflated_mask(local_map)
        dist_grid = self._reachable_distances(robot_xy, blocked)
        trace_field = self._trace_penalty_field(now, teammate_packets)
        regions = self._build_regions(comps, dist_grid, local_map, trace_field)
        if not regions:
            return None
        claimed = self._collect_claims(teammate_packets, provisional_claims or {})
        best: Optional[FrontierChoice] = None
        for region in regions:
            claim_penalty = self._claim_penalty(region.centroid_xy, region.region_id, claimed, current_region_id)
            switch_penalty = 0.0
            stay_bonus = 0.0
            if current_region_id is not None:
                if region.region_id == current_region_id:
                    stay_bonus = self.stay_bonus_value
                elif region_hold_active:
                    switch_penalty = self.switch_penalty_value
            score = 1.35 * region.info_gain - 0.70 * region.travel_cost - 1.0 * region.overlap_penalty - claim_penalty - switch_penalty + stay_bonus
            choice = FrontierChoice(
                target_xy=region.target_xy,
                score=score,
                info_gain=region.info_gain,
                travel_cost=region.travel_cost,
                overlap_penalty=region.overlap_penalty,
                region_id=region.region_id,
                region_center_xy=region.centroid_xy,
                claim_penalty=claim_penalty,
                switch_penalty=switch_penalty,
                stay_bonus=stay_bonus,
            )
            if best is None or choice.score > best.score:
                best = choice
        return best

    def _build_regions(self, comps, dist_grid, local_map, trace_field) -> List[FrontierRegion]:
        regions: List[FrontierRegion] = []
        for idx, comp in enumerate(comps):
            cx = sum(c[0] for c in comp) / len(comp)
            cy = sum(c[1] for c in comp) / len(comp)
            target = self._find_reachable_standoff((cx, cy), dist_grid, local_map)
            if target is None:
                continue
            tx, ty = target
            info_gain = self._unknown_gain_around(target, local_map) + 0.15 * len(comp)
            gx, gy = self.grid.world_to_grid(tx, ty)
            travel_cost = float(dist_grid[gy, gx]) * self.grid.res
            overlap = float(trace_field[gy, gx])
            centroid_xy = self.grid.grid_to_world(int(round(cx)), int(round(cy)))
            regions.append(FrontierRegion(
                region_id=idx,
                centroid_xy=centroid_xy,
                frontier_size=len(comp),
                target_xy=target,
                travel_cost=travel_cost,
                info_gain=info_gain,
                overlap_penalty=overlap,
                claim_penalty=0.0,
                switch_penalty=0.0,
                stay_bonus=0.0,
            ))
        return regions

    def _collect_claims(self, teammate_packets: Sequence[TeammatePacket], provisional_claims: Dict[int, Tuple[float, float]]):
        claimed: Dict[int, Tuple[Optional[int], Tuple[float, float], str]] = {}
        for rid, center in provisional_claims.items():
            claimed[rid] = (None, center, 'provisional')
        for packet in teammate_packets:
            snap = packet.self_snapshot
            if snap is None or snap.current_region_center_xy is None:
                continue
            claimed[packet.robot_id] = (snap.current_region_id, snap.current_region_center_xy, 'teammate')
        return claimed

    def _claim_penalty(self, region_center_xy, region_id, claimed, current_region_id):
        penalty = 0.0
        for rid, (claimed_region_id, claimed_center_xy, source_kind) in claimed.items():
            if claimed_center_xy is None:
                continue
            d = math.hypot(region_center_xy[0] - claimed_center_xy[0], region_center_xy[1] - claimed_center_xy[1])
            if d > self.claim_radius:
                continue
            if current_region_id is not None and claimed_region_id == current_region_id:
                continue
            w = self.same_cycle_penalty if source_kind == 'provisional' else self.claim_penalty
            penalty += w * max(0.1, 1.0 - d / max(self.claim_radius, 1e-6))
        return penalty

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
