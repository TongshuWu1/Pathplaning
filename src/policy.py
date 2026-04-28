from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import heapq
import math
import numpy as np

from .mapping import FREE, UNKNOWN, OccupancyGrid, connected_components, frontier_mask
from .robots.packets import KnowledgeSnapshot, TeammatePacket


# robot_id, estimated_xy, covariance_trace, home_hops
TeamRobotState = Tuple[int, Tuple[float, float], float, Optional[int]]


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
    voronoi_score: float = 0.0
    outward_score: float = 0.0
    sector_score: float = 0.0
    owner_robot_id: Optional[int] = None
    predicted_coverage: float = 0.0
    novelty_ratio: float = 1.0


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
    voronoi_score: float = 0.0
    outward_score: float = 0.0
    sector_score: float = 0.0
    owner_robot_id: Optional[int] = None
    predicted_coverage: float = 0.0
    novelty_ratio: float = 1.0
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
        check_return_path: bool = False,
        voronoi_bonus: float = 24.0,
        foreign_region_penalty: float = 36.0,
        outward_weight: float = 8.0,
        sector_weight: float = 14.0,
        info_gain_saturation: float = 95.0,
        strict_voronoi_assignment: bool = True,
        team_prediction_enabled: bool = True,
        team_prediction_radius_factor: float = 0.92,
        team_prediction_target_radius_factor: float = 1.05,
        team_prediction_pose_gain: float = 1.15,
        team_prediction_trail_gain: float = 0.75,
        team_prediction_target_gain: float = 1.35,
        team_prediction_path_gain: float = 0.58,
        team_prediction_decay_s: float = 30.0,
        team_prediction_gain_discount: float = 0.82,
        team_prediction_overlap_weight: float = 34.0,
        team_prediction_max_path_points: int = 36,
        team_prediction_min_novelty_ratio: float = 0.18,
        team_prediction_claim_target_gain: float = 1.15,
        max_goal_repair_dist: float = 1.10,
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
        self.check_return_path = check_return_path
        self.voronoi_bonus = voronoi_bonus
        self.foreign_region_penalty = foreign_region_penalty
        self.outward_weight = outward_weight
        self.sector_weight = sector_weight
        self.info_gain_saturation = max(1.0, info_gain_saturation)
        self.strict_voronoi_assignment = bool(strict_voronoi_assignment)
        self.team_prediction_enabled = bool(team_prediction_enabled)
        self.team_prediction_radius_factor = float(team_prediction_radius_factor)
        self.team_prediction_target_radius_factor = float(team_prediction_target_radius_factor)
        self.team_prediction_pose_gain = float(team_prediction_pose_gain)
        self.team_prediction_trail_gain = float(team_prediction_trail_gain)
        self.team_prediction_target_gain = float(team_prediction_target_gain)
        self.team_prediction_path_gain = float(team_prediction_path_gain)
        self.team_prediction_decay_s = float(team_prediction_decay_s)
        self.team_prediction_gain_discount = float(team_prediction_gain_discount)
        self.team_prediction_overlap_weight = float(team_prediction_overlap_weight)
        self.team_prediction_max_path_points = max(1, int(team_prediction_max_path_points))
        self.team_prediction_min_novelty_ratio = max(0.0, min(1.0, float(team_prediction_min_novelty_ratio)))
        self.team_prediction_claim_target_gain = float(team_prediction_claim_target_gain)
        self.max_goal_repair_dist = max(0.0, float(max_goal_repair_dist))

    def choose_route(
        self,
        now: float,
        local_map: np.ndarray,
        robot_xy: Tuple[float, float],
        teammate_packets: Sequence[TeammatePacket],
        planner,
        *,
        robot_cov_trace: float = 0.0,
        robot_id: int = 0,
        robot_count: int = 1,
        home_xy: Optional[Tuple[float, float]] = None,
        home_connected: bool = True,
        home_hops: Optional[int] = None,
        current_mode: str = "idle",
        current_region_id: Optional[int] = None,
        region_hold_active: bool = False,
        provisional_claims: Optional[Dict[int, Tuple[float, float]]] = None,
        team_robot_states: Optional[Sequence[TeamRobotState]] = None,
        teammate_knowledge: Optional[Sequence[KnowledgeSnapshot]] = None,
    ) -> Optional[RouteCandidate]:
        blocked = planner.inflated_mask(local_map)
        dist_grid = self._reachable_distances(robot_xy, blocked)
        trace_field = self._trace_penalty_field(now, teammate_packets)
        claims = self._collect_claims(teammate_packets, provisional_claims or {})
        prediction_field = self._team_prediction_field(
            now,
            robot_id=robot_id,
            teammate_packets=teammate_packets,
            teammate_knowledge=teammate_knowledge,
            provisional_claims=provisional_claims or {},
        )
        clearance = planner._clearance_map(blocked)
        unknown_adjacent = planner._unknown_adjacent_map(local_map)
        home_xy = home_xy or robot_xy
        home_dist_now = math.hypot(robot_xy[0] - home_xy[0], robot_xy[1] - home_xy[1])
        candidates: List[RouteCandidate] = []

        frontier_regions = self._top_frontier_regions(
            local_map,
            dist_grid,
            trace_field,
            prediction_field,
            robot_xy=robot_xy,
            robot_id=robot_id,
            robot_count=robot_count,
            robot_cov_trace=robot_cov_trace,
            home_xy=home_xy,
            home_hops=home_hops,
            teammate_packets=teammate_packets,
            team_robot_states=team_robot_states,
        )
        primary_regions, fallback_regions = self._partition_frontier_regions(
            frontier_regions,
            robot_id=robot_id,
            current_region_id=current_region_id,
        )

        def append_explore_candidates(regions: Sequence[FrontierRegion], *, fallback: bool = False) -> None:
            for region in regions:
                claim_pen = self._claim_penalty(region.centroid_xy, region.region_id, claims, current_region_id)
                # Foreign-region fallback is allowed, but it should be visibly a
                # fallback. This is what prevents clustering: owned regions win
                # unless the robot truly has no reachable owned frontier.
                if fallback and region.owner_robot_id is not None and int(region.owner_robot_id) != int(robot_id):
                    claim_pen += 0.60 * self.foreign_region_penalty
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
                    unknown_adjacent=unknown_adjacent,
                    info_gain=region.info_gain,
                    overlap_penalty=region.overlap_penalty,
                    claim_penalty=claim_pen,
                    switch_penalty=switch_pen,
                    stay_bonus=stay_bonus,
                    voronoi_score=region.voronoi_score,
                    outward_score=region.outward_score,
                    sector_score=region.sector_score,
                    owner_robot_id=region.owner_robot_id,
                    predicted_coverage=region.predicted_coverage,
                    novelty_ratio=region.novelty_ratio,
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

        append_explore_candidates(primary_regions, fallback=False)
        has_feasible_owned_explore = any(c.mode == 'explore' and c.feasible for c in candidates)
        if (not has_feasible_owned_explore) and fallback_regions:
            append_explore_candidates(fallback_regions, fallback=True)

        # Do not turn every temporary communication loss into a relay/return
        # behavior; that is what made the team collapse back into a cluster.
        # Support behavior is now reserved for genuinely weak localization or
        # long multi-hop chains.
        need_support = (
            (not frontier_regions)
            or ((not home_connected) and robot_cov_trace >= 0.60 * self.max_predicted_cov_trace)
            or ((home_hops is not None) and home_hops >= 3)
            or (robot_cov_trace >= 0.80 * self.max_predicted_cov_trace)
        )
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
                    unknown_adjacent=unknown_adjacent,
                    info_gain=0.35 * self._unknown_gain_around(relay_xy, local_map),
                    overlap_penalty=float(trace_field[self.grid.world_to_grid(*relay_xy)[1], self.grid.world_to_grid(*relay_xy)[0]]) * 0.6,
                    claim_penalty=0.0,
                    switch_penalty=0.0 if current_mode == "relay" else 0.5 * self.switch_penalty_value,
                    stay_bonus=self.stay_bonus_value if current_mode == "relay" else 0.0,
                    voronoi_score=0.0,
                    outward_score=0.0,
                    sector_score=0.0,
                    owner_robot_id=None,
                    predicted_coverage=0.0,
                    novelty_ratio=1.0,
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

        # During exploration, losing the home communication chain should not make
        # a robot randomly abandon its assigned frontier.  It should first try
        # relay/support behaviors; return is reserved for exhausted frontiers or
        # genuinely unsafe localization uncertainty.  The simulator-level mission
        # phase forces all robots home only after team exploration is complete.
        # Individual robots should not randomly return home just because their
        # local frontier list is temporarily empty. The simulator mission phase
        # handles the final all-robot return after team exploration is complete.
        need_return = (robot_cov_trace >= 0.90 * self.max_predicted_cov_trace)
        if need_return:
            return_cand = self._make_candidate(
                mode="return",
                target_xy=home_xy,
                planner=planner,
                local_map=local_map,
                clearance=clearance,
                blocked=blocked,
                unknown_adjacent=unknown_adjacent,
                info_gain=0.0,
                overlap_penalty=0.0,
                claim_penalty=0.0,
                switch_penalty=0.0 if current_mode == "return" else 0.6 * self.switch_penalty_value,
                stay_bonus=self.stay_bonus_value if current_mode == "return" else 0.0,
                voronoi_score=0.0,
                outward_score=0.0,
                sector_score=0.0,
                owner_robot_id=None,
                predicted_coverage=0.0,
                novelty_ratio=1.0,
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

        # If every option failed a soft mission heuristic, keep the robot moving
        # instead of letting it idle in place.  Hard clearance failures still
        # remain blocked.
        soft_failed = [
            cand for cand in candidates
            if cand.reject_reason not in {'clearance', 'no-path'}
        ]
        if soft_failed:
            best = max(soft_failed, key=lambda cand: cand.score)
            best.reject_reason = best.reject_reason or 'soft-override'
            return best
        return None

    def choose_target(self, *args, **kwargs) -> Optional[RouteCandidate]:
        return self.choose_route(*args, **kwargs)

    def _top_frontier_regions(
        self,
        local_map: np.ndarray,
        dist_grid: np.ndarray,
        trace_field: np.ndarray,
        prediction_field: Optional[np.ndarray],
        *,
        robot_xy: Tuple[float, float],
        robot_id: int,
        robot_count: int,
        robot_cov_trace: float,
        home_xy: Tuple[float, float],
        home_hops: Optional[int],
        teammate_packets: Sequence[TeammatePacket],
        team_robot_states: Optional[Sequence[TeamRobotState]] = None,
    ) -> List[FrontierRegion]:
        mask = frontier_mask(local_map)
        comps = connected_components(mask, min_size=5)
        if not comps:
            return []
        regions = self._build_regions(
            comps,
            dist_grid,
            local_map,
            trace_field,
            prediction_field,
            robot_xy=robot_xy,
            robot_id=robot_id,
            robot_count=robot_count,
            robot_cov_trace=robot_cov_trace,
            home_xy=home_xy,
            home_hops=home_hops,
            teammate_packets=teammate_packets,
            team_robot_states=team_robot_states,
        )
        def base_score(r: FrontierRegion) -> float:
            return (
                1.00 * r.info_gain
                - 0.42 * r.travel_cost
                - 0.85 * r.overlap_penalty
                + r.voronoi_score
                + r.outward_score
                + r.sector_score
                - 0.55 * self.team_prediction_overlap_weight * r.predicted_coverage
                + 18.0 * max(0.0, r.novelty_ratio - self.team_prediction_min_novelty_ratio)
            )

        owned = [r for r in regions if r.owner_robot_id is None or int(r.owner_robot_id) == int(robot_id)]
        foreign = [r for r in regions if r not in owned]
        owned.sort(key=base_score, reverse=True)
        foreign.sort(key=base_score, reverse=True)
        # Keep enough fallback candidates so a robot does not idle when its
        # assigned region is temporarily blocked, but present owned regions first.
        return owned[: self.max_frontier_candidates] + foreign[: self.max_frontier_candidates]

    def _partition_frontier_regions(
        self,
        regions: Sequence[FrontierRegion],
        *,
        robot_id: int,
        current_region_id: Optional[int],
    ) -> Tuple[List[FrontierRegion], List[FrontierRegion]]:
        if not self.strict_voronoi_assignment:
            return list(regions), []
        primary: List[FrontierRegion] = []
        fallback: List[FrontierRegion] = []
        for region in regions:
            keep_current = current_region_id is not None and region.region_id == current_region_id
            owned = region.owner_robot_id is None or int(region.owner_robot_id) == int(robot_id)
            if owned or keep_current:
                primary.append(region)
            else:
                fallback.append(region)
        return primary, fallback

    def _make_candidate(
        self,
        *,
        mode: str,
        target_xy: Tuple[float, float],
        planner,
        local_map: np.ndarray,
        clearance: np.ndarray,
        blocked: np.ndarray,
        unknown_adjacent: np.ndarray,
        info_gain: float,
        overlap_penalty: float,
        claim_penalty: float,
        switch_penalty: float,
        stay_bonus: float,
        voronoi_score: float,
        outward_score: float,
        sector_score: float,
        owner_robot_id: Optional[int],
        predicted_coverage: float,
        novelty_ratio: float,
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
        path = planner.plan(robot_xy, target_xy, local_map, blocked=blocked, clearance=clearance, unknown_adjacent=unknown_adjacent)
        if path is None or len(path) < 2:
            return None
        stats = planner.last_plan_stats
        tx, ty = path[-1]
        goal_repair_dist = math.hypot(tx - target_xy[0], ty - target_xy[1])
        if mode == "explore" and goal_repair_dist > self.max_goal_repair_dist:
            # A* can legally snap a blocked/unknown target to the nearest free
            # cell.  That is useful for tiny numerical errors, but a large snap
            # means the selected Voronoi/frontier target is no longer the same
            # assignment.  Reject it so the robot can auction/replan a fresh
            # region instead of committing to a stale obstacle-side target.
            return None
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
        if self.check_return_path and mode != "return":
            return_path = planner.plan((tx, ty), home_xy, local_map, blocked=blocked, clearance=clearance, unknown_adjacent=unknown_adjacent)
            return_path_len = float(planner.last_plan_stats.path_length_m) if return_path is not None else float('inf')
            return_path_exists = return_path is not None
        else:
            # Fast interactive mode: avoid a second A* for every candidate.
            # Use the already-computed distance-to-home as a conservative score proxy.
            return_path_len = float(home_dist_goal)
            return_path_exists = True
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
            voronoi_score=voronoi_score,
            outward_score=outward_score,
            sector_score=sector_score,
            predicted_coverage=predicted_coverage,
            novelty_ratio=novelty_ratio,
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
            return_path_exists=return_path_exists,
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
            voronoi_score=voronoi_score,
            outward_score=outward_score,
            sector_score=sector_score,
            owner_robot_id=owner_robot_id,
            predicted_coverage=float(predicted_coverage),
            novelty_ratio=float(novelty_ratio),
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
        voronoi_score: float,
        outward_score: float,
        sector_score: float,
        predicted_coverage: float,
        novelty_ratio: float,
        chain_score: float,
        return_score: float,
        localization_score: float,
        risk_penalty: float,
        min_clearance_m: float,
        avg_clearance_m: float,
    ) -> float:
        clearance_score = 0.80 * min_clearance_m + 0.35 * avg_clearance_m
        saturated_info = self.info_gain_saturation * (1.0 - math.exp(-max(0.0, info_gain) / self.info_gain_saturation))
        if mode == 'explore':
            return (
                1.22 * saturated_info
                + 8.0 * clearance_score
                + voronoi_score
                + outward_score
                + sector_score
                + 18.0 * max(0.0, novelty_ratio - self.team_prediction_min_novelty_ratio)
                + 18.0 * localization_score
                + 16.0 * chain_score
                + 12.0 * return_score
                - 0.72 * travel_cost
                - 1.0 * overlap_penalty
                - self.team_prediction_overlap_weight * predicted_coverage
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

    def _build_regions(
        self,
        comps,
        dist_grid,
        local_map,
        trace_field,
        prediction_field,
        *,
        robot_xy: Tuple[float, float],
        robot_id: int,
        robot_count: int,
        robot_cov_trace: float,
        home_xy: Tuple[float, float],
        home_hops: Optional[int],
        teammate_packets: Sequence[TeammatePacket],
        team_robot_states: Optional[Sequence[TeamRobotState]] = None,
    ) -> List[FrontierRegion]:
        regions: List[FrontierRegion] = []
        home_dist_now = math.hypot(robot_xy[0] - home_xy[0], robot_xy[1] - home_xy[1])
        for idx, comp in enumerate(comps):
            cx = sum(c[0] for c in comp) / len(comp)
            cy = sum(c[1] for c in comp) / len(comp)
            target = self._find_reachable_standoff((cx, cy), dist_grid, local_map)
            if target is None:
                continue
            tx, ty = target
            raw_unknown = self._unknown_gain_around(target, local_map, prediction_field=prediction_field)
            predicted_cov = self._prediction_coverage_around(target, prediction_field)
            novelty_ratio = max(0.0, min(1.0, 1.0 - predicted_cov))
            raw_gain = raw_unknown + 0.15 * len(comp) * novelty_ratio
            info_gain = self.info_gain_saturation * (1.0 - math.exp(-max(0.0, raw_gain) / self.info_gain_saturation))
            gx, gy = self.grid.world_to_grid(tx, ty)
            travel_cost = float(dist_grid[gy, gx]) * self.grid.res
            overlap = float(trace_field[gy, gx]) + self.team_prediction_overlap_weight * predicted_cov
            centroid_xy = self.grid.grid_to_world(int(round(cx)), int(round(cy)))
            stable_region_id = self._stable_region_id(centroid_xy)
            owner_id, voronoi_score = self._soft_voronoi_score(
                target,
                robot_xy=robot_xy,
                robot_id=robot_id,
                robot_cov_trace=robot_cov_trace,
                home_hops=home_hops,
                teammate_packets=teammate_packets,
                team_robot_states=team_robot_states,
            )
            outward_score = self._outward_score(target, robot_xy, home_xy, home_dist_now)
            sector_score = self._sector_score(target, home_xy, robot_id, robot_count)
            regions.append(
                FrontierRegion(
                    region_id=stable_region_id,
                    centroid_xy=centroid_xy,
                    frontier_size=len(comp),
                    target_xy=target,
                    travel_cost=travel_cost,
                    info_gain=info_gain,
                    overlap_penalty=overlap,
                    claim_penalty=0.0,
                    switch_penalty=0.0,
                    stay_bonus=0.0,
                    voronoi_score=voronoi_score,
                    outward_score=outward_score,
                    sector_score=sector_score,
                    owner_robot_id=owner_id,
                    predicted_coverage=predicted_cov,
                    novelty_ratio=novelty_ratio,
                )
            )
        return regions

    def _soft_voronoi_score(
        self,
        target_xy: Tuple[float, float],
        *,
        robot_xy: Tuple[float, float],
        robot_id: int,
        robot_cov_trace: float,
        home_hops: Optional[int],
        teammate_packets: Sequence[TeammatePacket],
        team_robot_states: Optional[Sequence[TeamRobotState]] = None,
    ) -> Tuple[Optional[int], float]:
        """Soft weighted Voronoi ownership for frontier assignment.

        The old version only used packets currently received by the robot, so
        disconnected robots could all pick the same high-gain frontier.  When
        the simulator provides team states, use those estimated poses for a
        stable assignment; otherwise fall back to packet-only decentralized
        information.
        """
        my_weight = self._robot_voronoi_weight(robot_cov_trace, home_hops)
        my_cost = math.hypot(target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]) * my_weight

        comparison: List[Tuple[int, Tuple[float, float], float, Optional[int]]] = []
        if team_robot_states:
            comparison.extend(team_robot_states)
        else:
            for packet in teammate_packets:
                snap = packet.self_snapshot
                if snap is None:
                    continue
                cov = np.asarray(snap.pose_cov, dtype=float)
                cov_trace = float(np.trace(cov[:2, :2])) if cov.ndim == 2 and cov.shape[0] >= 2 else 0.0
                comparison.append((packet.robot_id, snap.pose_xy, cov_trace, snap.home_hops))

        best_other_cost = float('inf')
        best_other_id: Optional[int] = None
        for other_id, other_xy, other_cov_trace, other_home_hops in comparison:
            if int(other_id) == int(robot_id):
                continue
            other_weight = self._robot_voronoi_weight(float(other_cov_trace), other_home_hops)
            other_cost = math.hypot(target_xy[0] - other_xy[0], target_xy[1] - other_xy[1]) * other_weight
            if other_cost < best_other_cost:
                best_other_cost = other_cost
                best_other_id = int(other_id)

        if not np.isfinite(best_other_cost):
            return robot_id, 0.35 * self.voronoi_bonus
        margin = best_other_cost - my_cost
        norm_margin = max(-1.0, min(1.0, margin / max(self.sensing_radius, 1.0)))
        if margin >= 0.0:
            return robot_id, self.voronoi_bonus * (0.35 + 0.65 * norm_margin)
        return best_other_id, -self.foreign_region_penalty * (0.35 + 0.65 * min(1.0, abs(norm_margin)))

    def _robot_voronoi_weight(self, cov_trace: float, home_hops: Optional[int]) -> float:
        cov_term = min(1.0, max(0.0, cov_trace) / max(self.max_predicted_cov_trace, 1e-6))
        hop_term = 0.0 if home_hops is None else max(0.0, min(4.0, float(home_hops))) / 4.0
        return 1.0 + 0.20 * cov_term + 0.08 * hop_term

    def _outward_score(
        self,
        target_xy: Tuple[float, float],
        robot_xy: Tuple[float, float],
        home_xy: Tuple[float, float],
        home_dist_now: float,
    ) -> float:
        target_home_dist = math.hypot(target_xy[0] - home_xy[0], target_xy[1] - home_xy[1])
        progress = target_home_dist - home_dist_now
        return self.outward_weight * max(-0.8, min(1.8, progress / max(self.sensing_radius, 1.0)))

    def _sector_score(
        self,
        target_xy: Tuple[float, float],
        home_xy: Tuple[float, float],
        robot_id: int,
        robot_count: int,
    ) -> float:
        if robot_count <= 1:
            return 0.0
        vx = target_xy[0] - home_xy[0]
        vy = target_xy[1] - home_xy[1]
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return 0.0
        # Fan the robots into the navigable world instead of around the full
        # circle.  This is important when the home base is near a corner: a full
        # circular sector would assign some robots to directions outside the map.
        center_x = 0.5 * self.grid.width_m
        center_y = 0.5 * self.grid.height_m
        base_angle = math.atan2(center_y - home_xy[1], center_x - home_xy[0])
        if robot_count <= 1:
            preferred = base_angle
        else:
            spread = 0.85 * math.pi
            frac = (robot_id % robot_count) / max(1, robot_count - 1)
            preferred = base_angle + (frac - 0.5) * spread
        ux, uy = math.cos(preferred), math.sin(preferred)
        alignment = (vx / norm) * ux + (vy / norm) * uy
        return self.sector_weight * max(-0.75, min(1.0, alignment))
    def _stable_region_id(self, centroid_xy: Tuple[float, float]) -> int:
        """Return a stable coarse grid ID for a frontier cluster.

        Connected-component indices change whenever the frontier image changes,
        which made robots think they had switched regions even when the target
        moved only slightly.  A coarse spatial key gives the hold/switch logic a
        persistent identity.
        """
        coarse_m = max(1.0, 1.75 * self.sensing_radius)
        ix = int(math.floor(centroid_xy[0] / coarse_m))
        iy = int(math.floor(centroid_xy[1] / coarse_m))
        return int(iy * 10000 + ix)

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

    def _nearest_free_cell(self, start: Tuple[int, int], blocked: np.ndarray) -> Optional[Tuple[int, int]]:
        sx, sy = start
        for radius in range(1, 16):
            best = None
            for yy in range(max(0, sy - radius), min(self.grid.ny, sy + radius + 1)):
                for xx in range(max(0, sx - radius), min(self.grid.nx, sx + radius + 1)):
                    if blocked[yy, xx]:
                        continue
                    d2 = (xx - sx) ** 2 + (yy - sy) ** 2
                    if best is None or d2 < best[0]:
                        best = (d2, (xx, yy))
            if best is not None:
                return best[1]
        return None

    def _reachable_distances(self, robot_xy: Tuple[float, float], blocked: np.ndarray) -> np.ndarray:
        sx, sy = self.grid.world_to_grid(*robot_xy)
        if blocked[sy, sx]:
            found = self._nearest_free_cell((sx, sy), blocked)
            if found is not None:
                sx, sy = found
            else:
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
        y0 = max(0, gy - cells)
        y1 = min(self.grid.ny, gy + cells + 1)
        x0 = max(0, gx - cells)
        x1 = min(self.grid.nx, gx + cells + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wx = (xx + 0.5) * self.grid.res
        wy = (yy + 0.5) * self.grid.res
        d2 = (wx - xy[0]) ** 2 + (wy - xy[1]) ** 2
        sigma2 = max(self.trace_radius, self.grid.res) ** 2
        field[y0:y1, x0:x1] += weight * np.exp(-0.5 * d2 / sigma2)


    def _team_prediction_field(
        self,
        now: float,
        *,
        robot_id: int,
        teammate_packets: Sequence[TeammatePacket],
        teammate_knowledge: Optional[Sequence[KnowledgeSnapshot]],
        provisional_claims: Dict[int, Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """Predict which cells are already/soon covered by teammates.

        This is the main exploration redesign: a robot does not treat every
        locally-unknown cell as equally valuable.  It builds a soft field from
        the teammate information it currently knows: teammate pose, remembered
        trail, current target, current region claim, and same-cycle provisional
        claims.  Frontiers inside these predicted LiDAR footprints lose utility.
        """
        if not self.team_prediction_enabled:
            return None
        field = np.zeros((self.grid.ny, self.grid.nx), dtype=np.float32)
        snapshots = self._collect_known_teammate_snapshots(robot_id, teammate_packets, teammate_knowledge)
        if not snapshots and not provisional_claims:
            return field

        sense_radius = max(self.grid.res, self.sensing_radius * self.team_prediction_radius_factor)
        target_radius = max(self.grid.res, self.sensing_radius * self.team_prediction_target_radius_factor)
        max_path = self.team_prediction_max_path_points

        for snap in snapshots:
            if int(snap.subject_robot_id) == int(robot_id):
                continue
            age = max(0.0, now - float(snap.knowledge_time))
            decay = math.exp(-age / max(1e-6, self.team_prediction_decay_s))
            if getattr(snap, 'is_stale', False):
                decay *= 0.45
            cov = np.asarray(snap.pose_cov, dtype=float)
            cov_trace = float(np.trace(cov[:2, :2])) if cov.ndim == 2 and cov.shape[0] >= 2 else 0.0
            confidence = 1.0 / (1.0 + 0.45 * max(0.0, cov_trace))
            w_base = decay * confidence
            if w_base <= 1e-3:
                continue

            self._deposit_prediction(field, snap.pose_xy, self.team_prediction_pose_gain * w_base, sense_radius)
            if snap.target_xy is not None:
                self._deposit_prediction(field, snap.target_xy, self.team_prediction_target_gain * w_base, target_radius)
            if snap.current_region_center_xy is not None:
                self._deposit_prediction(field, snap.current_region_center_xy, 0.65 * self.team_prediction_target_gain * w_base, target_radius)

            # Past route/recent trail approximates where the teammate already scanned.
            # Downsample to keep the UI responsive.
            path = list(snap.path_xy)
            if len(path) > max_path:
                stride = max(1, int(math.ceil(len(path) / max_path)))
                path = path[::stride][-max_path:]
            for idx, pt in enumerate(path):
                freshness = (idx + 1) / max(1, len(path))
                self._deposit_prediction(field, pt, self.team_prediction_path_gain * w_base * freshness, sense_radius)
            recent = [pt.xy for pt in getattr(snap, 'recent_trail', [])][-max_path:]
            for idx, pt in enumerate(recent):
                freshness = (idx + 1) / max(1, len(recent))
                self._deposit_prediction(field, pt, self.team_prediction_trail_gain * w_base * freshness, sense_radius)

        # Provisional claims are created within the same planning cycle so that
        # robot 2, 3, ... do not immediately choose the exact same region as robot 1.
        for rid, xy in provisional_claims.items():
            if int(rid) == int(robot_id):
                continue
            self._deposit_prediction(field, xy, self.team_prediction_claim_target_gain, target_radius)

        np.clip(field, 0.0, 1.0, out=field)
        return field

    def _collect_known_teammate_snapshots(
        self,
        robot_id: int,
        teammate_packets: Sequence[TeammatePacket],
        teammate_knowledge: Optional[Sequence[KnowledgeSnapshot]],
    ) -> List[KnowledgeSnapshot]:
        latest: Dict[int, KnowledgeSnapshot] = {}

        def consider(snap: Optional[KnowledgeSnapshot]) -> None:
            if snap is None or int(snap.subject_robot_id) == int(robot_id):
                return
            prev = latest.get(int(snap.subject_robot_id))
            if prev is None or float(snap.knowledge_time) >= float(prev.knowledge_time):
                latest[int(snap.subject_robot_id)] = snap

        for snap in teammate_knowledge or []:
            consider(snap)
        for packet in teammate_packets:
            for snap in packet.knowledge:
                consider(snap)
        return list(latest.values())

    def _deposit_prediction(self, field: np.ndarray, xy: Tuple[float, float], weight: float, radius: float) -> None:
        if weight <= 0.0:
            return
        gx, gy = self.grid.world_to_grid(*xy)
        cells = int(np.ceil(radius / self.grid.res))
        y0 = max(0, gy - cells)
        y1 = min(self.grid.ny, gy + cells + 1)
        x0 = max(0, gx - cells)
        x1 = min(self.grid.nx, gx + cells + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wx = (xx + 0.5) * self.grid.res
        wy = (yy + 0.5) * self.grid.res
        d2 = (wx - xy[0]) ** 2 + (wy - xy[1]) ** 2
        r2 = max(radius, self.grid.res) ** 2
        disk = d2 <= r2
        # Gaussian disk: high confidence near the path/target, still nonzero near
        # the edge because LiDAR can see a footprint, not only a single point.
        sigma2 = max((0.55 * radius) ** 2, self.grid.res ** 2)
        add = weight * np.exp(-0.5 * d2 / sigma2) * disk
        field[y0:y1, x0:x1] += add.astype(field.dtype, copy=False)

    def _prediction_coverage_around(self, target_xy: Tuple[float, float], prediction_field: Optional[np.ndarray]) -> float:
        if prediction_field is None:
            return 0.0
        gx, gy = self.grid.world_to_grid(*target_xy)
        radius = max(self.grid.res, 0.65 * self.sensing_radius)
        cells = int(np.ceil(radius / self.grid.res))
        y0 = max(0, gy - cells)
        y1 = min(self.grid.ny, gy + cells + 1)
        x0 = max(0, gx - cells)
        x1 = min(self.grid.nx, gx + cells + 1)
        if y0 >= y1 or x0 >= x1:
            return 0.0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wx = (xx + 0.5) * self.grid.res
        wy = (yy + 0.5) * self.grid.res
        disk = (wx - target_xy[0]) ** 2 + (wy - target_xy[1]) ** 2 <= radius ** 2
        if not np.any(disk):
            return 0.0
        return float(np.mean(np.clip(prediction_field[y0:y1, x0:x1][disk], 0.0, 1.0)))

    def _unknown_gain_around(
        self,
        target_xy: Tuple[float, float],
        local_map: np.ndarray,
        prediction_field: Optional[np.ndarray] = None,
    ) -> float:
        gx, gy = self.grid.world_to_grid(*target_xy)
        cells = int(np.ceil(self.sensing_radius / self.grid.res))
        y0 = max(0, gy - cells)
        y1 = min(self.grid.ny, gy + cells + 1)
        x0 = max(0, gx - cells)
        x1 = min(self.grid.nx, gx + cells + 1)
        if y0 >= y1 or x0 >= x1:
            return 0.0
        patch = local_map[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wx = (xx + 0.5) * self.grid.res
        wy = (yy + 0.5) * self.grid.res
        disk = (wx - target_xy[0]) ** 2 + (wy - target_xy[1]) ** 2 <= self.sensing_radius ** 2
        unknown_cells = (patch == UNKNOWN) & disk
        if prediction_field is None:
            return float(np.count_nonzero(unknown_cells))
        predicted = np.clip(prediction_field[y0:y1, x0:x1], 0.0, 1.0)
        # A cell that is unknown to me but very likely inside a teammate's LiDAR
        # footprint should be discounted, not counted as full new information.
        novelty_weight = 1.0 - self.team_prediction_gain_discount * predicted
        novelty_weight = np.clip(novelty_weight, self.team_prediction_min_novelty_ratio, 1.0)
        return float(np.sum(unknown_cells * novelty_weight))

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
