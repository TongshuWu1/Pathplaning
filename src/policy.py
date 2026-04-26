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
class RouteCandidate:
    mode: str
    target_xy: Tuple[float, float]
    path_xy: List[Tuple[float, float]]
    score: float
    info_gain: float
    travel_cost: float
    overlap_penalty: float
    region_id: Optional[int]
    region_center_xy: Optional[Tuple[float, float]]
    region_label: str
    claim_penalty: float = 0.0
    switch_penalty: float = 0.0
    stay_bonus: float = 0.0
    chain_score: float = 0.0
    return_score: float = 0.0
    localization_score: float = 0.0
    risk_penalty: float = 0.0
    min_clearance_m: float = 0.0
    avg_clearance_m: float = 0.0
    narrow_fraction: float = 0.0
    predicted_cov_trace: float = 0.0
    feasible: bool = True
    reject_reason: str = ""


class LocalFrontierPolicy:
    def __init__(
        self,
        grid: OccupancyGrid,
        sensing_radius: float,
        trace_radius: float,
        trace_gain: float,
        current_pos_gain: float,
        target_gain: float,
        decay_s: float,
        claim_radius: float = 4.5,
        claim_penalty: float = 55.0,
        same_cycle_penalty: float = 75.0,
        switch_penalty: float = 18.0,
        stay_bonus: float = 10.0,
        min_route_clearance: float = 0.38,
        max_predicted_cov_trace: float = 2.8,
        covariance_growth_per_m: float = 0.055,
        disconnect_explore_margin: float = 1.2,
        return_path_factor: float = 1.8,
        max_frontier_candidates: int = 10,
    ):
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
        self.min_route_clearance = min_route_clearance
        self.max_predicted_cov_trace = max_predicted_cov_trace
        self.covariance_growth_per_m = covariance_growth_per_m
        self.disconnect_explore_margin = disconnect_explore_margin
        self.return_path_factor = return_path_factor
        self.max_frontier_candidates = max_frontier_candidates

    def choose_route(
        self,
        now: float,
        local_map: np.ndarray,
        robot_xy: Tuple[float, float],
        teammate_packets: Sequence[TeammatePacket],
        planner,
        *,
        robot_cov_trace: float = 0.0,
        home_xy: Optional[Tuple[float, float]] = None,
        home_connected: bool = True,
        home_hops: Optional[int] = None,
        current_mode: str = "idle",
        current_region_id: Optional[int] = None,
        region_hold_active: bool = False,
        provisional_claims: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> Optional[RouteCandidate]:
        blocked = planner.inflated_mask(local_map)
        dist_grid = self._reachable_distances(robot_xy, blocked)
        trace_field = self._trace_penalty_field(now, teammate_packets)
        clearance = planner._clearance_map(blocked)
        claims = self._collect_claims(teammate_packets, provisional_claims or {})
        home_xy = home_xy or robot_xy
        home_dist_now = math.hypot(robot_xy[0] - home_xy[0], robot_xy[1] - home_xy[1])
        candidates: List[RouteCandidate] = []

        frontier_regions = self._top_frontier_regions(local_map, dist_grid, trace_field)
        for region in frontier_regions:
            claim_pen = self._claim_penalty(region.centroid_xy, region.region_id, claims, current_region_id)
            switch_pen = 0.0
            stay_bonus = 0.0
            if current_region_id is not None:
                if region.region_id == current_region_id:
                    stay_bonus = self.stay_bonus_value
                elif region_hold_active:
                    switch_pen = self.switch_penalty_value
            cand = self._make_candidate(
                mode="explore",
                target_xy=region.target_xy,
                planner=planner,
                local_map=local_map,
                clearance=clearance,
                blocked=blocked,
                info_gain=region.info_gain,
                overlap_penalty=region.overlap_penalty,
                claim_penalty=claim_pen,
                switch_penalty=switch_pen,
                stay_bonus=stay_bonus,
                region_id=region.region_id,
                region_center_xy=region.centroid_xy,
                robot_xy=robot_xy,
                home_xy=home_xy,
                home_dist_now=home_dist_now,
                teammate_packets=teammate_packets,
                robot_cov_trace=robot_cov_trace,
                home_connected=home_connected,
                home_hops=home_hops,
            )
            if cand is not None:
                candidates.append(cand)

        need_support = (not home_connected) or ((home_hops is not None) and home_hops >= 2) or (robot_cov_trace >= 0.80 * self.max_predicted_cov_trace)
        if need_support:
            relay_targets = self._relay_targets(robot_xy, home_xy, teammate_packets, home_connected, home_hops)
            for idx, relay_xy in enumerate(relay_targets):
                cand = self._make_candidate(
                    mode="relay",
                    target_xy=relay_xy,
                    planner=planner,
                    local_map=local_map,
                    clearance=clearance,
                    blocked=blocked,
                    info_gain=0.35 * self._unknown_gain_around(relay_xy, local_map),
                    overlap_penalty=float(trace_field[self.grid.world_to_grid(*relay_xy)[1], self.grid.world_to_grid(*relay_xy)[0]]) * 0.6,
                    claim_penalty=0.0,
                    switch_penalty=0.0 if current_mode == "relay" else 0.5 * self.switch_penalty_value,
                    stay_bonus=self.stay_bonus_value if current_mode == "relay" else 0.0,
                    region_id=None,
                    region_center_xy=relay_xy,
                    robot_xy=robot_xy,
                    home_xy=home_xy,
                    home_dist_now=home_dist_now,
                    teammate_packets=teammate_packets,
                    robot_cov_trace=robot_cov_trace,
                    home_connected=home_connected,
                    home_hops=home_hops,
                    route_id=f"relay-{idx}",
                )
                if cand is not None:
                    candidates.append(cand)

        need_return = (not frontier_regions) or (not home_connected) or (robot_cov_trace >= 0.90 * self.max_predicted_cov_trace)
        if need_return:
            return_cand = self._make_candidate(
                mode="return",
                target_xy=home_xy,
                planner=planner,
                local_map=local_map,
                clearance=clearance,
                blocked=blocked,
                info_gain=0.0,
                overlap_penalty=0.0,
                claim_penalty=0.0,
                switch_penalty=0.0 if current_mode == "return" else 0.6 * self.switch_penalty_value,
                stay_bonus=self.stay_bonus_value if current_mode == "return" else 0.0,
                region_id=None,
                region_center_xy=home_xy,
                robot_xy=robot_xy,
                home_xy=home_xy,
                home_dist_now=home_dist_now,
                teammate_packets=teammate_packets,
                robot_cov_trace=robot_cov_trace,
                home_connected=home_connected,
                home_hops=home_hops,
                route_id="return-home",
            )
            if return_cand is not None:
                candidates.append(return_cand)

        feasible = [cand for cand in candidates if cand.feasible]
        if feasible:
            return max(feasible, key=lambda cand: cand.score)
        return None

    def choose_target(self, *args, **kwargs) -> Optional[RouteCandidate]:
        return self.choose_route(*args, **kwargs)

    def _top_frontier_regions(self, local_map: np.ndarray, dist_grid: np.ndarray, trace_field: np.ndarray) -> List[FrontierRegion]:
        mask = frontier_mask(local_map)
        comps = connected_components(mask, min_size=5)
        if not comps:
            return []
        regions = self._build_regions(comps, dist_grid, local_map, trace_field)
        regions.sort(key=lambda r: (1.15 * r.info_gain) - (0.45 * r.travel_cost) - (0.9 * r.overlap_penalty), reverse=True)
        return regions[: self.max_frontier_candidates]

    def _make_candidate(
        self,
        *,
        mode: str,
        target_xy: Tuple[float, float],
        planner,
        local_map: np.ndarray,
        clearance: np.ndarray,
        blocked: np.ndarray,
        info_gain: float,
        overlap_penalty: float,
        claim_penalty: float,
        switch_penalty: float,
        stay_bonus: float,
        region_id: Optional[int],
        region_center_xy: Optional[Tuple[float, float]],
        robot_xy: Tuple[float, float],
        home_xy: Tuple[float, float],
        home_dist_now: float,
        teammate_packets: Sequence[TeammatePacket],
        robot_cov_trace: float,
        home_connected: bool,
        home_hops: Optional[int],
        route_id: Optional[str] = None,
    ) -> Optional[RouteCandidate]:
        path = planner.plan(robot_xy, target_xy, local_map, blocked=blocked, clearance=clearance)
        if path is None or len(path) < 2:
            return None
        stats = planner.last_plan_stats
        tx, ty = path[-1]
        region_label = self._classify_region((tx, ty), blocked, clearance)
        travel_cost = float(stats.path_length_m)
        min_clear = float(stats.min_clearance_m)
        avg_clear = float(stats.avg_clearance_m)
        narrow_fraction = float(stats.narrow_fraction)
        pred_cov = float(robot_cov_trace + self.covariance_growth_per_m * max(0.0, travel_cost))
        home_dist_goal = math.hypot(tx - home_xy[0], ty - home_xy[1])
        home_progress = home_dist_now - home_dist_goal
        support_near_target = self._teammate_support_at_target(target_xy, teammate_packets)
        localization_score = self._localization_score(home_xy, target_xy, pred_cov, support_near_target)
        chain_score = self._chain_score(mode, home_progress, home_connected, home_hops, support_near_target)
        return_path = planner.plan((tx, ty), home_xy, local_map, blocked=blocked, clearance=clearance)
        return_path_len = float(planner.last_plan_stats.path_length_m) if return_path is not None else float('inf')
        return_score = self._return_score(mode, home_progress, return_path_len, home_dist_goal)
        risk_penalty = self._risk_penalty(pred_cov, min_clear, narrow_fraction, home_progress, mode)
        score = self._utility(
            mode=mode,
            info_gain=info_gain,
            travel_cost=travel_cost,
            overlap_penalty=overlap_penalty,
            claim_penalty=claim_penalty,
            switch_penalty=switch_penalty,
            stay_bonus=stay_bonus,
            chain_score=chain_score,
            return_score=return_score,
            localization_score=localization_score,
            risk_penalty=risk_penalty,
            min_clearance_m=min_clear,
            avg_clearance_m=avg_clear,
        )
        feasible, reason = self._feasibility_screen(
            mode=mode,
            min_clearance_m=min_clear,
            predicted_cov_trace=pred_cov,
            home_progress=home_progress,
            return_path_exists=(return_path is not None),
            return_path_len=return_path_len,
            home_dist_now=home_dist_now,
            home_connected=home_connected,
            support_near_target=support_near_target,
        )
        if not feasible:
            score -= 80.0
        return RouteCandidate(
            mode=mode,
            target_xy=(tx, ty),
            path_xy=path,
            score=score,
            info_gain=info_gain,
            travel_cost=travel_cost,
            overlap_penalty=overlap_penalty,
            region_id=region_id,
            region_center_xy=region_center_xy,
            region_label=region_label,
            claim_penalty=claim_penalty,
            switch_penalty=switch_penalty,
            stay_bonus=stay_bonus,
            chain_score=chain_score,
            return_score=return_score,
            localization_score=localization_score,
            risk_penalty=risk_penalty,
            min_clearance_m=min_clear,
            avg_clearance_m=avg_clear,
            narrow_fraction=narrow_fraction,
            predicted_cov_trace=pred_cov,
            feasible=feasible,
            reject_reason=reason,
        )

    def _feasibility_screen(
        self,
        *,
        mode: str,
        min_clearance_m: float,
        predicted_cov_trace: float,
        home_progress: float,
        return_path_exists: bool,
        return_path_len: float,
        home_dist_now: float,
        home_connected: bool,
        support_near_target: float,
    ) -> Tuple[bool, str]:
        if min_clearance_m < self.min_route_clearance:
            return False, 'clearance'
        if mode != 'return' and predicted_cov_trace > self.max_predicted_cov_trace:
            return False, 'covariance'
        if mode == 'explore' and (not home_connected) and home_progress < -self.disconnect_explore_margin:
            return False, 'disconnect-margin'
        if mode != 'return' and not return_path_exists:
            return False, 'no-return'
        if mode != 'return' and return_path_exists:
            max_reasonable = max(4.0, self.return_path_factor * max(home_dist_now, self.sensing_radius * 0.75))
            if return_path_len > max_reasonable:
                return False, 'return-margin'
        if mode == 'relay' and home_progress < -0.6 and support_near_target < 0.4:
            return False, 'weak-relay'
        return True, ''

    def _utility(
        self,
        *,
        mode: str,
        info_gain: float,
        travel_cost: float,
        overlap_penalty: float,
        claim_penalty: float,
        switch_penalty: float,
        stay_bonus: float,
        chain_score: float,
        return_score: float,
        localization_score: float,
        risk_penalty: float,
        min_clearance_m: float,
        avg_clearance_m: float,
    ) -> float:
        clearance_score = 0.80 * min_clearance_m + 0.35 * avg_clearance_m
        if mode == 'explore':
            return (
                1.22 * info_gain
                + 8.0 * clearance_score
                + 18.0 * localization_score
                + 16.0 * chain_score
                + 12.0 * return_score
                - 0.72 * travel_cost
                - 1.0 * overlap_penalty
                - claim_penalty
                - switch_penalty
                - 16.0 * risk_penalty
                + stay_bonus
            )
        if mode == 'relay':
            return (
                0.35 * info_gain
                + 7.0 * clearance_score
                + 24.0 * localization_score
                + 26.0 * chain_score
                + 18.0 * return_score
                - 0.62 * travel_cost
                - 0.55 * overlap_penalty
                - 0.35 * claim_penalty
                - switch_penalty
                - 14.0 * risk_penalty
                + stay_bonus
            )
        return (
            5.0 * clearance_score
            + 22.0 * localization_score
            + 24.0 * chain_score
            + 34.0 * return_score
            - 0.48 * travel_cost
            - 0.30 * overlap_penalty
            - 10.0 * risk_penalty
            + stay_bonus
        )

    def _localization_score(
        self,
        home_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
        predicted_cov_trace: float,
        support_near_target: float,
    ) -> float:
        home_term = 1.0 / (1.0 + 0.25 * math.hypot(target_xy[0] - home_xy[0], target_xy[1] - home_xy[1]))
        cov_term = 1.0 / (1.0 + predicted_cov_trace)
        return 0.55 * cov_term + 0.25 * home_term + 0.20 * min(1.5, support_near_target)

    def _chain_score(
        self,
        mode: str,
        home_progress: float,
        home_connected: bool,
        home_hops: Optional[int],
        support_near_target: float,
    ) -> float:
        hop_term = 0.0 if home_hops is None else max(0.0, 0.2 * (3 - min(3, home_hops)))
        progress_term = home_progress / max(self.sensing_radius, 1.0)
        if mode == 'relay':
            progress_term *= 1.35
        if mode == 'return':
            progress_term *= 1.6
        if mode == 'explore' and not home_connected:
            progress_term *= 1.5
        return progress_term + 0.18 * support_near_target + hop_term

    def _return_score(
        self,
        mode: str,
        home_progress: float,
        return_path_len: float,
        home_dist_goal: float,
    ) -> float:
        margin_term = 0.0 if not np.isfinite(return_path_len) else 1.0 / max(1.0, return_path_len)
        contraction_term = max(-0.5, min(1.8, home_progress / max(self.sensing_radius, 1.0)))
        if mode == 'return':
            contraction_term += 0.45
        if home_dist_goal < self.sensing_radius * 0.75:
            contraction_term += 0.15
        return 0.55 * margin_term + 0.45 * contraction_term

    def _risk_penalty(
        self,
        predicted_cov_trace: float,
        min_clearance_m: float,
        narrow_fraction: float,
        home_progress: float,
        mode: str,
    ) -> float:
        cov_risk = max(0.0, predicted_cov_trace - 0.9) / max(0.5, self.max_predicted_cov_trace)
        clear_risk = max(0.0, self.min_route_clearance - min_clearance_m) / max(self.min_route_clearance, 1e-6)
        disconnect_risk = 0.0
        if mode == 'explore' and home_progress < 0.0:
            disconnect_risk = min(1.5, abs(home_progress) / max(self.sensing_radius, 1.0))
        return 0.40 * cov_risk + 0.35 * narrow_fraction + 0.15 * clear_risk + 0.10 * disconnect_risk

    def _relay_targets(
        self,
        robot_xy: Tuple[float, float],
        home_xy: Tuple[float, float],
        teammate_packets: Sequence[TeammatePacket],
        home_connected: bool,
        home_hops: Optional[int],
    ) -> List[Tuple[float, float]]:
        targets: List[Tuple[float, float]] = []
        if not home_connected or (home_hops is not None and home_hops >= 2):
            for alpha in (0.35, 0.55, 0.75):
                targets.append((
                    robot_xy[0] + alpha * (home_xy[0] - robot_xy[0]),
                    robot_xy[1] + alpha * (home_xy[1] - robot_xy[1]),
                ))
        for packet in teammate_packets:
            snap = packet.self_snapshot
            if snap is None:
                continue
            if snap.home_connected or (snap.home_hops is not None and home_hops is not None and snap.home_hops < home_hops):
                targets.append(packet.pose_xy)
                if packet.target_xy is not None:
                    targets.append(packet.target_xy)
        dedup: List[Tuple[float, float]] = []
        seen = set()
        for x, y in targets:
            key = (round(x, 2), round(y, 2))
            if key in seen:
                continue
            dedup.append((x, y))
            seen.add(key)
        return dedup[:6]

    def _teammate_support_at_target(self, target_xy: Tuple[float, float], teammate_packets: Sequence[TeammatePacket]) -> float:
        support = 0.0
        for packet in teammate_packets:
            d = math.hypot(target_xy[0] - packet.pose_xy[0], target_xy[1] - packet.pose_xy[1])
            if d <= self.sensing_radius * 1.4:
                support += max(0.0, 1.0 - d / max(self.sensing_radius * 1.4, 1e-6))
        return support

    def _classify_region(self, target_xy: Tuple[float, float], blocked: np.ndarray, clearance: np.ndarray) -> str:
        gx, gy = self.grid.world_to_grid(*target_xy)
        c = float(clearance[gy, gx])
        branch = 0
        for yy in range(max(0, gy - 1), min(self.grid.ny, gy + 2)):
            for xx in range(max(0, gx - 1), min(self.grid.nx, gx + 2)):
                if (xx, yy) == (gx, gy):
                    continue
                if not blocked[yy, xx]:
                    branch += 1
        if c < max(self.min_route_clearance * 1.1, 0.45):
            return 'bottleneck'
        if branch >= 6 and c < max(self.min_route_clearance * 2.0, 1.0):
            return 'junction'
        if c < max(self.min_route_clearance * 1.9, 0.95):
            return 'corridor'
        return 'open'

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
            regions.append(
                FrontierRegion(
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
                )
            )
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
