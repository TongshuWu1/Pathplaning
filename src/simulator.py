from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import SimConfig
from .geometry import clamp, line_of_sight
from .mapping import OccupancyGrid, UNKNOWN
from .planning import GridPlanner
from .policy import LocalFrontierPolicy
from .robot import Robot
from .ui import SimulatorUI
from .world import World


ROBOT_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive']
HOME_NODE = -1


class Simulator:
    def __init__(self, cfg: Optional[SimConfig] = None) -> None:
        self.cfg = cfg or SimConfig()
        self.running = False
        self.world: World
        self.grid: OccupancyGrid
        self.planner: GridPlanner
        self.policy: LocalFrontierPolicy
        self.robots: List[Robot]
        self.time_s = 0.0
        self.step_count = 0
        self.replan_count = 0
        self.robot_comm_edges: List[Tuple[int, int]] = []
        self.home_comm_links: List[Tuple[int, Tuple[float, float]]] = []
        self.reset(self.cfg)
        self.ui = SimulatorUI(self)

    def reset(self, cfg: Optional[SimConfig] = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        self.running = False
        self.world = World.generate(self.cfg)
        self.grid = OccupancyGrid(self.cfg.world_w, self.cfg.world_h, self.cfg.grid_res)
        self.planner = GridPlanner(
            grid=self.grid,
            robot_radius=self.cfg.robot_radius,
            inflation_margin=self.cfg.planner_inflation_margin,
            world_obstacles=self.world.obstacles,
            clearance_weight=self.cfg.planner_clearance_weight,
            clearance_floor_m=self.cfg.planner_clearance_floor_m,
            narrow_penalty=self.cfg.planner_narrow_penalty,
            unknown_edge_penalty=self.cfg.planner_unknown_edge_penalty,
        )
        self.policy = LocalFrontierPolicy(
            grid=self.grid,
            sensing_radius=self.cfg.lidar_range,
            trace_radius=self.cfg.teammate_trace_radius,
            trace_gain=self.cfg.teammate_trace_gain,
            current_pos_gain=self.cfg.teammate_current_pos_gain,
            target_gain=self.cfg.teammate_target_gain,
            decay_s=self.cfg.teammate_trace_decay_s,
            claim_radius=self.cfg.frontier_region_claim_radius,
            claim_penalty=self.cfg.frontier_region_claim_penalty,
            same_cycle_penalty=self.cfg.frontier_region_same_cycle_penalty,
            switch_penalty=self.cfg.frontier_region_switch_penalty,
            stay_bonus=self.cfg.frontier_region_stay_bonus,
            min_route_clearance=self.cfg.decision_min_route_clearance,
            max_predicted_cov_trace=self.cfg.decision_max_predicted_cov_trace,
            covariance_growth_per_m=self.cfg.decision_covariance_growth_per_m,
            disconnect_explore_margin=self.cfg.decision_disconnect_explore_margin,
            return_path_factor=self.cfg.decision_return_path_factor,
            max_frontier_candidates=self.cfg.decision_max_frontier_candidates,
        )
        self.robots = []
        run_dir = self._make_run_dir()
        for i, (x, y) in enumerate(self.spawn_positions(self.cfg.robot_count)):
            local = OccupancyGrid(self.cfg.world_w, self.cfg.world_h, self.cfg.grid_res)
            robot = Robot(
                robot_id=i,
                name=f'Robot {i + 1}',
                color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
                x=x,
                y=y,
                heading=0.0,
                cfg=self.cfg,
                local_map=local,
                rng_seed=self.cfg.seed * 1000 + i * 31 + 7,
                log_dir=run_dir,
            )
            local.reveal_disk(robot.x, robot.y, radius=1.4)
            self.robots.append(robot)
        self.time_s = 0.0
        self.step_count = 0
        self.replan_count = 0
        self.robot_comm_edges = []
        self.home_comm_links = []
        self._update_communication_state()
        for robot in self.robots:
            robot.logger.write_snapshot(robot.knowledge_snapshot(self.time_s))

    def _make_run_dir(self) -> str | None:
        if not self.cfg.logs_enabled:
            return None
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = Path(self.cfg.logs_root) / f'run_{ts}'
        path.mkdir(parents=True, exist_ok=True)
        summary = path / 'summary.txt'
        summary.write_text(f'robot_count={self.cfg.robot_count}\nseed={self.cfg.seed}\n')
        return str(path)

    def spawn_positions(self, count: int) -> List[Tuple[float, float]]:
        base = self.world.home_base
        margin = self.cfg.home_base_padding + self.cfg.robot_radius
        usable_w = max(0.8, base.w - 2.0 * margin)
        usable_h = max(0.8, base.h - 2.0 * margin)
        cols = max(1, int(math.ceil(math.sqrt(count))))
        rows = max(1, int(math.ceil(count / cols)))
        if cols == 1:
            xs = [base.xmin + margin + usable_w * 0.5]
        else:
            xs = [base.xmin + margin + usable_w * (i / (cols - 1)) for i in range(cols)]
        if rows == 1:
            ys = [base.ymin + margin + usable_h * 0.5]
        else:
            ys = [base.ymin + margin + usable_h * (j / (rows - 1)) for j in range(rows)]
        pts: List[Tuple[float, float]] = []
        for j in range(rows):
            for i in range(cols):
                pts.append((xs[i], ys[j]))
        pts = pts[:count]
        pts.sort(key=lambda p: (p[1], p[0]))
        return pts

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def step(self) -> None:
        if not self.running:
            return
        self.time_s += self.cfg.dt
        self.step_count += 1

        self._update_communication_state()
        for robot in self.robots:
            robot.note_connectivity_state(self.time_s)
            robot.ingest_shared_teammate_info(self.time_s)
        robots_by_id = {r.robot_id: r for r in self.robots}
        all_landmarks = self.world.all_landmarks
        for robot in self.robots:
            beams = robot.sense(self.world.obstacles)
            robot.update_localization(all_landmarks, robots_by_id, self.world.obstacles, self.time_s)
            robot.update_map(beams)

        provisional_claims = {}
        for robot in self.robots:
            need_replan = (
                robot.request_replan or
                robot.current_target is None or
                (self.time_s - robot.last_plan_time) >= self.cfg.frontier_replan_period or
                ((self.time_s - robot.last_target_time) >= self.cfg.target_hold_time and not robot.current_path)
            )
            if need_replan:
                choice = self.policy.choose_route(
                    self.time_s,
                    robot.local_map.data,
                    robot.est_pose_xy(),
                    robot.received_packets,
                    self.planner,
                    robot_cov_trace=robot.covariance_trace(),
                    home_xy=(self.world.home_marker.x, self.world.home_marker.y),
                    home_connected=robot.home_connected,
                    home_hops=robot.home_hops,
                    current_mode=robot.current_mode,
                    current_region_id=robot.current_region_id,
                    region_hold_active=(self.time_s < robot.region_hold_until),
                    provisional_claims=provisional_claims,
                )
                if choice is not None and choice.path_xy and len(choice.path_xy) >= 2:
                    prev_region = robot.current_region_id
                    robot.current_target = choice.target_xy
                    robot.current_path = choice.path_xy[1:]
                    robot.current_mode = choice.mode
                    robot.last_plan_time = self.time_s
                    robot.last_target_time = self.time_s
                    robot.request_replan = False
                    robot.current_region_id = choice.region_id
                    robot.current_region_center_xy = choice.region_center_xy
                    robot.region_hold_until = self.time_s + self.cfg.frontier_region_hold_time
                    if prev_region != choice.region_id:
                        robot.last_region_switch_time = self.time_s
                    if choice.region_center_xy is not None:
                        provisional_claims[robot.robot_id] = choice.region_center_xy
                    robot.logger.log(
                        self.time_s,
                        'replan',
                        mode=choice.mode,
                        target_xy=choice.target_xy,
                        score=choice.score,
                        info_gain=choice.info_gain,
                        travel_cost=choice.travel_cost,
                        overlap_penalty=choice.overlap_penalty,
                        chain_score=choice.chain_score,
                        return_score=choice.return_score,
                        localization_score=choice.localization_score,
                        feasible=choice.feasible,
                        reject_reason=choice.reject_reason,
                    )
                    feasibility = 'ok' if choice.feasible else f'rej:{choice.reject_reason}'
                    robot.last_choice_debug = (
                        f'{choice.mode} {choice.region_label} {feasibility}\n'
                        f'g={choice.info_gain:4.0f} c={choice.travel_cost:4.1f} clr={choice.min_clearance_m:3.2f} sc={choice.score:5.1f}'
                    )
                    self.replan_count += 1
                elif robot.request_replan:
                    robot.current_mode = 'idle'
                    robot.last_choice_debug = 'replan requested\nno feasible route selected'
                    robot.request_replan = False
            robot.follow_path(self.cfg.dt, self.world.obstacles, now=self.time_s)

        self._update_communication_state()
        for robot in self.robots:
            robot.logger.write_snapshot(robot.knowledge_snapshot(self.time_s))

    def estimated_coverage(self) -> float:
        masks = [(r.local_map.data != UNKNOWN) for r in self.robots]
        if not masks:
            return 0.0
        fused = np.logical_or.reduce(masks)
        return 100.0 * float(np.mean(fused))

    def connected_robot_count(self) -> int:
        return sum(1 for r in self.robots if r.home_connected)

    def mean_covariance_trace(self) -> float:
        if not self.robots:
            return 0.0
        return float(np.mean([r.covariance_trace() for r in self.robots]))

    def mean_localization_error(self) -> float:
        if not self.robots:
            return 0.0
        return float(np.mean([math.hypot(r.x - r.x_est, r.y - r.y_est) for r in self.robots]))

    def max_localization_error(self) -> float:
        if not self.robots:
            return 0.0
        return float(np.max([math.hypot(r.x - r.x_est, r.y - r.y_est) for r in self.robots]))

    def run(self) -> None:
        self.ui.build()
        plt.show()

    def _update_communication_state(self) -> None:
        packets = {r.robot_id: r.make_packet(self.time_s) for r in self.robots}
        adjacency: Dict[int, set[int]] = {HOME_NODE: set()}
        for robot in self.robots:
            adjacency[robot.robot_id] = set()
            robot.direct_neighbors = []
            robot.reachable_peer_ids = []
            robot.received_packets = []
            robot.home_connected = False
            robot.home_hops = None
            robot.direct_home_link = False

        self.robot_comm_edges = []
        self.home_comm_links = []
        for i, a in enumerate(self.robots):
            for b in self.robots[i + 1:]:
                if self._robot_link(a.pose_xy(), b.pose_xy()):
                    adjacency[a.robot_id].add(b.robot_id)
                    adjacency[b.robot_id].add(a.robot_id)
                    a.direct_neighbors.append(b.robot_id)
                    b.direct_neighbors.append(a.robot_id)
                    self.robot_comm_edges.append((a.robot_id, b.robot_id))

        for robot in self.robots:
            anchor = self._home_link_anchor(robot.pose_xy())
            if anchor is not None:
                adjacency[HOME_NODE].add(robot.robot_id)
                adjacency[robot.robot_id].add(HOME_NODE)
                robot.direct_home_link = True
                self.home_comm_links.append((robot.robot_id, anchor))

        hops_from_home = self._bfs_hops(adjacency, HOME_NODE)
        components = self._connected_components(adjacency)
        for robot in self.robots:
            if robot.robot_id in hops_from_home:
                robot.home_connected = True
                robot.home_hops = max(0, hops_from_home[robot.robot_id] - 1)
            component = components.get(robot.robot_id, set())
            robot.reachable_peer_ids = sorted(rid for rid in component if rid not in (HOME_NODE, robot.robot_id))
            robot.received_packets = [packets[rid] for rid in robot.reachable_peer_ids]

    def _robot_link(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1]) <= self.cfg.comm_radius and line_of_sight(p0, p1, self.world.obstacles)

    def _home_link_anchor(self, robot_xy: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        best = None
        for pt in self._home_anchor_points():
            d = math.hypot(pt[0] - robot_xy[0], pt[1] - robot_xy[1])
            if d <= self.cfg.comm_radius and line_of_sight(robot_xy, pt, self.world.obstacles):
                if best is None or d < best[0]:
                    best = (d, pt)
        return None if best is None else best[1]

    def _home_anchor_points(self) -> List[Tuple[float, float]]:
        base = self.world.home_base
        pad = min(0.35, 0.18 * min(base.w, base.h))
        xs = [base.xmin + pad, base.cx, base.xmax - pad]
        ys = [base.ymin + pad, base.cy, base.ymax - pad]
        points = []
        for x in xs:
            for y in ys:
                points.append((clamp(x, base.xmin, base.xmax), clamp(y, base.ymin, base.ymax)))
        uniq = []
        seen = set()
        for pt in points:
            key = (round(pt[0], 3), round(pt[1], 3))
            if key not in seen:
                uniq.append(pt)
                seen.add(key)
        return uniq

    def _bfs_hops(self, adjacency: Dict[int, set[int]], start: int) -> Dict[int, int]:
        hops = {start: 0}
        q = deque([start])
        while q:
            node = q.popleft()
            for nxt in adjacency.get(node, ()):  # pragma: no branch
                if nxt in hops:
                    continue
                hops[nxt] = hops[node] + 1
                q.append(nxt)
        return hops

    def _connected_components(self, adjacency: Dict[int, set[int]]) -> Dict[int, set[int]]:
        comps: Dict[int, set[int]] = {}
        seen: set[int] = set()
        for node in adjacency:
            if node in seen:
                continue
            comp = set()
            q = deque([node])
            seen.add(node)
            while q:
                cur = q.popleft()
                comp.add(cur)
                for nxt in adjacency.get(cur, ()):  # pragma: no branch
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    q.append(nxt)
            for member in comp:
                comps[member] = comp
        return comps
