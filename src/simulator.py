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
from .mapping import FREE, OCCUPIED, OccupancyGrid, UNKNOWN, frontier_mask
from .planning import GridPlanner
from .policy import LocalFrontierPolicy
from .robot import Robot
from .fusion import TeamFusedMap
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
        self.team_fused_map: TeamFusedMap
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
            use_scipy_distance_transform=self.cfg.planner_use_scipy_distance_transform,
            max_expansions=self.cfg.planner_max_expansions,
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
            check_return_path=self.cfg.decision_check_return_path,
            voronoi_bonus=self.cfg.decision_voronoi_bonus,
            foreign_region_penalty=self.cfg.decision_foreign_region_penalty,
            outward_weight=self.cfg.decision_outward_weight,
            sector_weight=self.cfg.decision_sector_weight,
            info_gain_saturation=self.cfg.decision_info_gain_saturation,
            strict_voronoi_assignment=self.cfg.decision_strict_voronoi_assignment,
            team_prediction_enabled=self.cfg.team_prediction_enabled,
            team_prediction_radius_factor=self.cfg.team_prediction_radius_factor,
            team_prediction_target_radius_factor=self.cfg.team_prediction_target_radius_factor,
            team_prediction_pose_gain=self.cfg.team_prediction_pose_gain,
            team_prediction_trail_gain=self.cfg.team_prediction_trail_gain,
            team_prediction_target_gain=self.cfg.team_prediction_target_gain,
            team_prediction_path_gain=self.cfg.team_prediction_path_gain,
            team_prediction_decay_s=self.cfg.team_prediction_decay_s,
            team_prediction_gain_discount=self.cfg.team_prediction_gain_discount,
            team_prediction_overlap_weight=self.cfg.team_prediction_overlap_weight,
            team_prediction_max_path_points=self.cfg.team_prediction_max_path_points,
            team_prediction_min_novelty_ratio=self.cfg.team_prediction_min_novelty_ratio,
            team_prediction_claim_target_gain=self.cfg.team_prediction_claim_target_gain,
            max_goal_repair_dist=self.cfg.goal_repair_max_dist_m,
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
        self.home_anchor_points = self._home_anchor_points()
        self.last_voronoi_owner_grid: Optional[np.ndarray] = None
        self.mission_phase = 'explore'
        self.exploration_complete_since: Optional[float] = None
        self.return_home_started_at: Optional[float] = None
        self.mission_completed_at: Optional[float] = None
        self.last_exploration_frontier_cells = 0
        self.last_exploration_coverage_pct = 0.0
        self.last_exploration_complete_reason = ''
        self.team_fused_map = TeamFusedMap(
            self.cfg.world_w,
            self.cfg.world_h,
            self.cfg.grid_res,
            min_confidence=self.cfg.fusion_min_confidence,
            stale_decay_s=self.cfg.fusion_stale_decay_s,
            pose_cov_gain=self.cfg.fusion_pose_cov_gain,
            range_gain=self.cfg.fusion_range_gain,
        )
        self._update_communication_state()
        self.team_fused_map.update_from_robots(
            self.robots,
            self.time_s,
            self.cfg,
            require_home_connected=self.cfg.fusion_require_home_connection,
        )
        for robot in self.robots:
            robot.logger.write_snapshot(robot.knowledge_snapshot(self.time_s), now=self.time_s, min_period_s=self.cfg.log_snapshot_period_s)

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

        # Use the previous communication graph for this tick's received packets.
        # The graph is rebuilt once at the end of the step after motion.  This
        # avoids doing the same LOS + packet construction work twice per step.
        for robot in self.robots:
            robot.note_connectivity_state(self.time_s)
            robot.ingest_shared_teammate_info(self.time_s)
        robots_by_id = {r.robot_id: r for r in self.robots}
        all_landmarks = self.world.all_landmarks
        for robot in self.robots:
            beams = robot.sense(self.world.obstacles)
            robot.update_localization(all_landmarks, robots_by_id, self.world.obstacles, self.time_s)
            robot.update_map(beams)

        # A selected frontier can become invalid after the newest LiDAR scan
        # turns previously-unknown cells into occupied/free cells.  Break target
        # commitment immediately when the active endpoint or route no longer
        # agrees with the current local map.
        for robot in self.robots:
            self._validate_active_exploration_route(robot)

        # Team fused belief is for visualization; update it on a short period
        # instead of every simulator step.
        fusion_period = max(1, int(getattr(self.cfg, 'fusion_update_period_steps', 1)))
        if self.step_count % fusion_period == 0:
            self.team_fused_map.update_from_robots(
                self.robots,
                self.time_s,
                self.cfg,
                require_home_connected=self.cfg.fusion_require_home_connection,
            )

        self._update_mission_phase_before_planning()
        provisional_claims = {}
        team_robot_states = self._team_robot_states()
        for robot in self.robots:
            if self.mission_phase == 'complete':
                self._hold_robot_at_home(robot)
                continue

            if self.mission_phase == 'return_home':
                # Return-home is still active, but if localization becomes
                # unsafe the return planner uses the same recovery map and
                # conservative motion model below.
                self._update_return_home_behavior(robot)
                robot.follow_path(self.cfg.dt, self.world.obstacles, now=self.time_s)
                continue

            if self._update_localization_recovery_behavior(robot):
                robot.follow_path(self.cfg.dt, self.world.obstacles, now=self.time_s)
                continue

            has_active_path = bool(robot.current_path)
            target_age = self.time_s - robot.last_target_time
            target_dist = float('inf')
            if robot.current_target is not None:
                ex, ey = robot.est_pose_xy()
                target_dist = math.hypot(ex - robot.current_target[0], ey - robot.current_target[1])
            near_target = target_dist <= float(self.cfg.target_replan_near_target_radius)
            committed_to_target = (
                robot.current_target is not None
                and has_active_path
                and not robot.request_replan
                and target_age < float(self.cfg.target_commit_min_s)
            )
            target_hold_active = (
                robot.current_target is not None
                and self.time_s < robot.region_hold_until
                and has_active_path
                and not robot.request_replan
            )
            # Do not re-auction targets while a robot is still travelling.
            # Replanning now happens when the robot is blocked, has arrived, has
            # no target, or is already close enough to its current target.
            if robot.request_replan or robot.current_target is None:
                need_replan = True
            elif committed_to_target:
                need_replan = False
            elif not has_active_path:
                need_replan = target_age >= float(self.cfg.target_hold_time)
            else:
                need_replan = (
                    (not target_hold_active)
                    and near_target
                    and (self.time_s - robot.last_plan_time) >= float(self.cfg.frontier_replan_period)
                )
            if need_replan:
                planning_map = self._make_exploration_planning_map(robot)
                choice = self.policy.choose_route(
                    self.time_s,
                    planning_map,
                    robot.est_pose_xy(),
                    robot.received_packets,
                    self.planner,
                    robot_cov_trace=robot.covariance_trace(),
                    robot_id=robot.robot_id,
                    robot_count=len(self.robots),
                    home_xy=(self.world.home_marker.x, self.world.home_marker.y),
                    home_connected=robot.home_connected,
                    home_hops=robot.home_hops,
                    current_mode=robot.current_mode,
                    current_region_id=robot.current_region_id,
                    region_hold_active=(self.time_s < robot.region_hold_until),
                    provisional_claims=provisional_claims,
                    team_robot_states=team_robot_states,
                    teammate_knowledge=robot.memory_store.export_knowledge() if self.cfg.team_prediction_use_stale_memory else None,
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
                        voronoi_score=choice.voronoi_score,
                        outward_score=choice.outward_score,
                        sector_score=choice.sector_score,
                        owner_robot_id=choice.owner_robot_id,
                        predicted_coverage=choice.predicted_coverage,
                        novelty_ratio=choice.novelty_ratio,
                        feasible=choice.feasible,
                        reject_reason=choice.reject_reason,
                    )
                    feasibility = 'ok' if choice.feasible else f'rej:{choice.reject_reason}'
                    robot.last_choice_debug = (
                        f'{choice.mode} {choice.region_label} {feasibility} owner={choice.owner_robot_id}\n'
                        f'g={choice.info_gain:4.0f} c={choice.travel_cost:4.1f} '
                        f'nov={choice.novelty_ratio:3.2f} cov={choice.predicted_coverage:3.2f}\n'
                        f'v={choice.voronoi_score:4.1f} out={choice.outward_score:4.1f} sc={choice.score:5.1f}'
                    )
                    self.replan_count += 1
                elif robot.request_replan:
                    robot.current_mode = 'idle'
                    robot.last_choice_debug = 'replan requested\nno feasible route selected'
                    robot.request_replan = False
            robot.follow_path(self.cfg.dt, self.world.obstacles, now=self.time_s)

        self._update_communication_state()
        self._update_mission_phase_after_motion()
        for robot in self.robots:
            robot.logger.write_snapshot(robot.knowledge_snapshot(self.time_s), now=self.time_s, min_period_s=self.cfg.log_snapshot_period_s)

    def _localization_safety_level(self, robot: Robot) -> str:
        return robot.localization_state(self.time_s)

    def _make_exploration_planning_map(self, robot: Robot) -> np.ndarray:
        data = np.array(robot.local_map.data, copy=True)
        return self._apply_temporary_route_blocks(robot, data, protect_home=True)


    def _validate_active_exploration_route(self, robot: Robot) -> None:
        """Invalidate stale exploration targets after the newest local map update.

        Frontier targets are selected next to unknown cells.  After a LiDAR scan,
        the same cells may become occupied, may become too close to an obstacle,
        or may simply no longer contain useful unknown information.  In those
        cases we should break the target hold and let the policy choose again
        instead of blindly following the old Voronoi/frontier assignment.
        """
        if not bool(getattr(self.cfg, 'active_goal_revalidation_enabled', True)):
            return
        if self.mission_phase != 'explore':
            return
        if robot.current_mode not in {'explore', 'relay'}:
            return
        if robot.current_target is None or not robot.current_path:
            return
        if self.time_s - float(robot.last_plan_time) < float(getattr(self.cfg, 'active_goal_replan_cooldown_s', 0.35)):
            return

        planning_map = self._make_exploration_planning_map(robot)
        blocked = self.planner.inflated_mask(planning_map)
        target_xy = robot.current_target
        gx, gy = self.grid.world_to_grid(*target_xy)

        reason = ''
        block_xy: Optional[Tuple[float, float]] = None
        hard_invalid = False

        if bool(blocked[gy, gx]):
            reason = 'target became blocked after sensing'
            block_xy = target_xy
            hard_invalid = True
        elif robot.current_mode == 'explore':
            unknown_cells = self._unknown_cells_near_target(robot.local_map.data, target_xy)
            if unknown_cells < int(getattr(self.cfg, 'active_goal_min_unknown_cells', 4)):
                reason = f'target is no longer a frontier ({unknown_cells} unknown cells nearby)'
                hard_invalid = True

        if not reason:
            block_xy = self._first_blocked_active_path_point(robot, planning_map, blocked)
            if block_xy is not None:
                reason = 'path became blocked after sensing'

        if not reason:
            return

        if block_xy is not None:
            robot.record_route_block(block_xy[0], block_xy[1], self.time_s)

        old_target = robot.current_target
        robot.current_path = []
        robot.request_replan = True
        robot.motion_state = 'route-replan'
        robot.region_hold_until = min(robot.region_hold_until, self.time_s)

        if hard_invalid:
            # If the endpoint itself is now bad or no longer informative, release
            # the region claim too.  Otherwise a path-only failure may replan to
            # the same still-useful frontier using a different route.
            robot.current_target = None
            robot.current_region_id = None
            robot.current_region_center_xy = None

        msg = f'active target invalid: {reason}'
        if old_target is not None:
            msg += f'\nold=({old_target[0]:.1f},{old_target[1]:.1f})'
        robot.last_choice_debug = msg
        robot.logger.log(
            self.time_s,
            'active-route-invalid',
            reason=reason,
            old_target_xy=old_target,
            block_xy=block_xy,
            mode=robot.current_mode,
        )

    def _unknown_cells_near_target(self, local_map: np.ndarray, target_xy: Tuple[float, float]) -> int:
        radius = max(
            self.grid.res,
            float(getattr(self.cfg, 'active_goal_check_radius_factor', 0.75)) * float(self.cfg.lidar_range),
        )
        gx, gy = self.grid.world_to_grid(*target_xy)
        cells = int(math.ceil(radius / self.grid.res))
        y0 = max(0, gy - cells)
        y1 = min(self.grid.ny, gy + cells + 1)
        x0 = max(0, gx - cells)
        x1 = min(self.grid.nx, gx + cells + 1)
        if y0 >= y1 or x0 >= x1:
            return 0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        wx = (xx + 0.5) * self.grid.res
        wy = (yy + 0.5) * self.grid.res
        disk = (wx - target_xy[0]) ** 2 + (wy - target_xy[1]) ** 2 <= radius ** 2
        return int(np.count_nonzero((local_map[y0:y1, x0:x1] == UNKNOWN) & disk))

    def _first_blocked_active_path_point(
        self,
        robot: Robot,
        planning_map: np.ndarray,
        blocked: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        max_points = max(1, int(getattr(self.cfg, 'active_goal_path_check_points', 8)))
        waypoints = list(robot.current_path[:max_points])
        if robot.current_target is not None and robot.current_target not in waypoints:
            waypoints.append(robot.current_target)
        if not waypoints:
            return None

        prev = robot.est_pose_xy()
        # Protect the robot's current cell: it can temporarily sit in a cell that
        # is marked unknown/blocked because of localization drift or a short-lived
        # route block.  The validation should check the route ahead, not reject
        # just because the start cell is imperfect.
        psx, psy = self.grid.world_to_grid(*prev)
        if 0 <= psx < self.grid.nx and 0 <= psy < self.grid.ny:
            blocked = blocked.copy()
            blocked[psy, psx] = False

        for wp in waypoints:
            gx, gy = self.grid.world_to_grid(*wp)
            if bool(blocked[gy, gx]):
                return wp
            if not self.planner._segment_free(prev, wp, planning_map, blocked):
                return wp
            prev = wp
        return None



    def _apply_temporary_route_blocks(
        self,
        robot: Robot,
        data: np.ndarray,
        protect_points: Optional[List[Tuple[float, float]]] = None,
        protect_home: bool = False,
    ) -> np.ndarray:
        """Inject short-lived virtual blocks for failed approaches.

        These route blocks are not map observations.  They only bias the next
        planning query away from the place where the true body got stuck, which
        makes A* search for a different corridor instead of repeating the same
        bad local route.
        """
        if not bool(getattr(self.cfg, 'stuck_route_replan_enabled', True)):
            return data
        zones = robot.active_route_blocks(self.time_s)
        if not zones:
            return data

        out = np.array(data, copy=True)
        radius = float(getattr(self.cfg, 'stuck_route_block_radius', 0.78))
        clear_r = float(getattr(self.cfg, 'stuck_route_current_clear_radius', 0.72))
        protected: List[Tuple[float, float]] = [(robot.x_est, robot.y_est), (robot.x, robot.y)]
        if protect_points:
            protected.extend((float(x), float(y)) for x, y in protect_points)
        if protect_home:
            hx, hy = self._home_xy()
            protected.append((hx, hy))
            # Keep the home base navigable even if a robot gets stuck at the
            # entrance.  The failed entrance is still remembered outside this
            # small protected disk.
            home_protect = float(getattr(self.cfg, 'stuck_route_protect_home_radius', 1.15))
        else:
            home_protect = 0.0

        def near_protected(wx: float, wy: float) -> bool:
            for px, py in protected:
                local_clear = clear_r
                if protect_home and math.hypot(px - self._home_xy()[0], py - self._home_xy()[1]) < 1e-6:
                    local_clear = max(local_clear, home_protect)
                if (wx - px) ** 2 + (wy - py) ** 2 <= local_clear ** 2:
                    return True
            return False

        cells = int(math.ceil(radius / self.grid.res))
        for bx, by, _expire in zones:
            gx, gy = self.grid.world_to_grid(float(bx), float(by))
            for yy in range(max(0, gy - cells), min(self.grid.ny, gy + cells + 1)):
                for xx in range(max(0, gx - cells), min(self.grid.nx, gx + cells + 1)):
                    wx, wy = self.grid.grid_to_world(xx, yy)
                    if (wx - bx) ** 2 + (wy - by) ** 2 > radius ** 2:
                        continue
                    if near_protected(wx, wy):
                        continue
                    out[yy, xx] = OCCUPIED
        return out

    def _update_localization_recovery_behavior(self, robot: Robot) -> bool:
        """Anchor-seeking behavior used when pose uncertainty becomes unsafe.

        This is not the final return-home phase.  It is a temporary recovery
        mode: move slowly toward a known landmark, home marker, or reliable
        teammate so the EKF can get absolute corrections again.  Once covariance
        drops, normal exploration resumes.
        """
        level = self._localization_safety_level(robot)
        if level not in {'recover', 'critical'}:
            if robot.current_mode == 'localize':
                robot.current_path = []
                robot.current_target = None
                robot.current_mode = 'idle'
                robot.request_replan = True
            return False

        # Do not keep driving blindly if the robot has been blocked.  Force a
        # new anchor path immediately.
        replan_due = (self.time_s - robot.last_plan_time) >= float(self.cfg.localization_recovery_replan_period)
        near_target = False
        if robot.current_target is not None:
            near_target = math.hypot(robot.x_est - robot.current_target[0], robot.y_est - robot.current_target[1]) <= float(
                self.cfg.localization_recovery_anchor_accept_radius
            )
        if (
            robot.current_mode == 'localize'
            and robot.current_path
            and not robot.request_replan
            and not near_target
            and not replan_due
        ):
            robot.last_choice_debug = (
                f'LOCALIZE continue  state={level}\n'
                f'cov={robot.covariance_trace():4.2f} abs_age={robot.absolute_update_age(self.time_s):3.1f}s'
            )
            return True

        anchors = self._localization_anchor_candidates(robot)
        recovery_map = self._make_recovery_map(robot, anchors)
        best = None
        for anchor_xy, label, priority in anchors:
            target_xy = self._anchor_standoff_target(robot, anchor_xy)
            path = self.planner.plan(robot.est_pose_xy(), target_xy, recovery_map)
            if path is None or len(path) < 2:
                continue
            path_len = float(self.planner.last_plan_stats.path_length_m)
            # Low priority value is better.  Prefer non-home landmarks and
            # reliable teammates; use home as a universal fallback.
            score = path_len + 2.0 * float(priority)
            if best is None or score < best[0]:
                best = (score, path, target_xy, label, path_len)

        if best is None:
            # Final fallback: reverse the true executed trail.  This is still
            # better than continuing an exploration target with a bad pose.
            home_xy = self._home_xy()
            path = self._fallback_reverse_history_path(robot, home_xy)
            if path is None or len(path) < 2:
                robot.current_path = []
                robot.current_mode = 'localize'
                robot.current_target = home_xy
                robot.request_replan = True
                robot.last_choice_debug = (
                    f'LOCALIZE blocked  state={level}\n'
                    f'no anchor path; cov={robot.covariance_trace():4.2f} '
                    f'abs_age={robot.absolute_update_age(self.time_s):3.1f}s'
                )
                return True
            best = (999.0, path, home_xy, 'reverse-history-home', self._path_length(path))

        _, path, target_xy, label, path_len = best
        robot.current_path = path[1:]
        robot.current_target = target_xy
        robot.current_mode = 'localize'
        robot.current_region_id = None
        robot.current_region_center_xy = None
        robot.region_hold_until = -1e9
        robot.last_plan_time = self.time_s
        robot.last_target_time = self.time_s
        robot.request_replan = False
        block_n = len(robot.active_route_blocks(self.time_s))
        block_txt = f' blocks={block_n}' if block_n else ''
        robot.last_choice_debug = (
            f'LOCALIZE seek {label}  state={level}{block_txt}\n'
            f'cov={robot.covariance_trace():4.2f} abs_age={robot.absolute_update_age(self.time_s):3.1f}s '
            f'len={path_len:4.1f}m'
        )
        robot.logger.log(
            self.time_s,
            'localization-recovery',
            state=level,
            target_xy=target_xy,
            anchor_label=label,
            cov_trace=robot.covariance_trace(),
            abs_age=robot.absolute_update_age(self.time_s),
            path_len=path_len,
        )
        return True

    def _localization_anchor_candidates(self, robot: Robot) -> List[Tuple[Tuple[float, float], str, float]]:
        anchors: List[Tuple[Tuple[float, float], str, float]] = []
        seen = set()

        def add(xy: Tuple[float, float], label: str, priority: float) -> None:
            x, y = float(xy[0]), float(xy[1])
            if not (0.0 <= x <= self.cfg.world_w and 0.0 <= y <= self.cfg.world_h):
                return
            key = (round(x, 2), round(y, 2), label)
            if key in seen:
                return
            seen.add(key)
            anchors.append(((x, y), label, float(priority)))

        for name, info in robot.known_landmark_beliefs().items():
            if not bool(info.get('discovered', False)):
                continue
            xy = tuple(info.get('xy', self._home_xy()))
            if bool(info.get('is_home', False)):
                add((float(xy[0]), float(xy[1])), 'home-landmark', 1.0)
            else:
                add((float(xy[0]), float(xy[1])), f'landmark-{name}', 0.15)

        # Home is always a valid conservative anchor, but it should not dominate
        # known non-home landmarks unless they are unavailable.
        add(self._home_xy(), 'home-marker', 1.15)

        for other in self.robots:
            if other.robot_id == robot.robot_id:
                continue
            if other.robot_id in robot.direct_neighbors or other.robot_id in robot.reachable_peer_ids:
                cov = other.covariance_trace()
                priority = 0.45 + min(1.0, cov / max(float(self.cfg.localization_critical_cov_trace), 1e-6))
                add(other.est_pose_xy(), f'teammate-{other.robot_id + 1}', priority)

        # If only home exists, this returns home.  Sort so landmarks/nearby
        # reachable teammates are attempted first.
        anchors.sort(key=lambda item: (item[2], math.hypot(item[0][0] - robot.x_est, item[0][1] - robot.y_est)))
        return anchors[:10]

    def _anchor_standoff_target(self, robot: Robot, anchor_xy: Tuple[float, float]) -> Tuple[float, float]:
        # Stop near the anchor instead of exactly on top of it.  This keeps the
        # robot in a useful observation range while avoiding marker clutter.
        ax, ay = anchor_xy
        rx, ry = robot.est_pose_xy()
        dx, dy = rx - ax, ry - ay
        d = math.hypot(dx, dy)
        standoff = float(self.cfg.localization_recovery_standoff_m)
        if d <= standoff or d < 1e-6:
            return (ax, ay)
        return (ax + standoff * dx / d, ay + standoff * dy / d)

    def _make_recovery_map(
        self,
        robot: Robot,
        anchors: List[Tuple[Tuple[float, float], str, float]],
    ) -> np.ndarray:
        data = self._make_return_map(robot)

        def mark_disk_free(x: float, y: float, radius: float) -> None:
            gx, gy = self.grid.world_to_grid(x, y)
            cells = int(math.ceil(radius / self.grid.res))
            for yy in range(max(0, gy - cells), min(self.grid.ny, gy + cells + 1)):
                for xx in range(max(0, gx - cells), min(self.grid.nx, gx + cells + 1)):
                    wx, wy = self.grid.grid_to_world(xx, yy)
                    if (wx - x) ** 2 + (wy - y) ** 2 <= radius ** 2 and data[yy, xx] != OCCUPIED:
                        data[yy, xx] = FREE

        radius = max(self.cfg.robot_radius * 2.4, self.grid.res * 2.5)
        mark_disk_free(robot.x_est, robot.y_est, radius)
        for anchor_xy, _label, _priority in anchors:
            mark_disk_free(float(anchor_xy[0]), float(anchor_xy[1]), radius)
            standoff = self._anchor_standoff_target(robot, anchor_xy)
            mark_disk_free(float(standoff[0]), float(standoff[1]), radius)
        return data

    def _home_xy(self) -> Tuple[float, float]:
        return (self.world.home_marker.x, self.world.home_marker.y)

    def _robot_is_home(self, robot: Robot) -> bool:
        """Robust home confirmation.

        The simulator knows the true body pose, but the behavior should not
        declare success just because the noisy estimated pose reaches the home
        target.  Require true home contact *and* an estimator-side confirmation:
        estimated pose near home plus either low covariance, a recent Home
        landmark observation, or a direct home LOS link.
        """
        base = self.world.home_base
        tol = max(float(self.cfg.mission_home_arrival_radius), self.cfg.robot_radius * 1.5)
        true_in_base = (
            base.xmin - tol <= robot.x <= base.xmax + tol and
            base.ymin - tol <= robot.y <= base.ymax + tol
        )
        true_near_marker = math.hypot(robot.x - self.world.home_marker.x, robot.y - self.world.home_marker.y) <= max(
            tol, 0.35 * self.cfg.home_base_size
        )
        est_in_base = (
            base.xmin - tol <= robot.x_est <= base.xmax + tol and
            base.ymin - tol <= robot.y_est <= base.ymax + tol
        )
        est_near_marker = math.hypot(robot.x_est - self.world.home_marker.x, robot.y_est - self.world.home_marker.y) <= max(
            tol, 0.35 * self.cfg.home_base_size
        )
        estimator_confirmed = (
            robot.covariance_trace() <= float(self.cfg.localization_home_confirm_cov_trace)
            or robot.recent_home_seen(self.time_s)
            or bool(robot.direct_home_link)
        )
        return bool((true_in_base or true_near_marker) and (est_in_base or est_near_marker) and estimator_confirmed)

    def _count_team_frontier_cells(self) -> int:
        """Count frontier cells on a team-level union map.

        Summing each robot's local frontier count double-counts shared boundary
        cells and can keep the mission in exploration forever.  The return-home
        trigger should reason about what the team has collectively explored.
        """
        if not self.robots:
            return 0
        occ_masks = [(r.local_map.data == OCCUPIED) for r in self.robots]
        free_masks = [(r.local_map.data == FREE) for r in self.robots]
        team = np.full_like(self.robots[0].local_map.data, UNKNOWN)
        occ = np.logical_or.reduce(occ_masks)
        free = np.logical_or.reduce(free_masks) & (~occ)
        team[free] = FREE
        team[occ] = OCCUPIED
        return int(np.count_nonzero(frontier_mask(team)))

    def _exploration_complete_condition(self) -> Tuple[bool, str, float, int]:
        # Use the larger of the shareable fused map and the union of all local
        # maps.  The fused map may intentionally ignore disconnected robots, but
        # mission completion should reflect what the whole team has actually
        # explored.
        coverage = max(float(self.team_fused_coverage()), float(self.estimated_coverage()))
        frontier_cells = self._count_team_frontier_cells()
        self.last_exploration_coverage_pct = coverage
        self.last_exploration_frontier_cells = frontier_cells

        if self.time_s < float(getattr(self.cfg, 'mission_min_explore_s', 0.0)):
            return False, '', coverage, frontier_cells

        frontier_done = (
            frontier_cells <= int(self.cfg.mission_frontier_stop_cells)
            and coverage >= float(getattr(self.cfg, 'mission_frontier_stop_requires_coverage_pct', 0.0))
        )
        coverage_done = coverage >= float(self.cfg.mission_coverage_goal_pct)
        if coverage_done:
            return True, f'coverage goal reached ({coverage:4.1f}%)', coverage, frontier_cells
        if frontier_done:
            return True, f'frontiers exhausted ({frontier_cells} cells, coverage {coverage:4.1f}%)', coverage, frontier_cells
        return False, '', coverage, frontier_cells

    def _update_mission_phase_before_planning(self) -> None:
        if not bool(getattr(self.cfg, 'mission_auto_return_when_explored', True)):
            return
        if self.mission_phase != 'explore':
            return

        complete_now, reason, _, _ = self._exploration_complete_condition()
        if complete_now:
            if self.exploration_complete_since is None:
                self.exploration_complete_since = self.time_s
                self.last_exploration_complete_reason = reason
            elif (self.time_s - self.exploration_complete_since) >= float(self.cfg.mission_completion_confirm_s):
                self._start_return_home(reason or self.last_exploration_complete_reason)
        else:
            self.exploration_complete_since = None
            self.last_exploration_complete_reason = ''

    def _start_return_home(self, reason: str) -> None:
        if self.mission_phase == 'return_home':
            return
        self.mission_phase = 'return_home'
        self.return_home_started_at = self.time_s
        self.last_exploration_complete_reason = reason
        home_xy = self._home_xy()
        for robot in self.robots:
            robot.current_mode = 'return'
            robot.current_target = home_xy
            robot.current_region_id = None
            robot.current_region_center_xy = None
            robot.region_hold_until = -1e9
            robot.current_path = []
            robot.request_replan = True
            robot.last_choice_debug = f'mission return-home\n{reason}'
            robot.logger.log(self.time_s, 'mission-return-home', reason=reason, target_xy=home_xy)

    def _hold_robot_at_home(self, robot: Robot) -> None:
        robot.current_path = []
        robot.current_target = self._home_xy()
        robot.current_mode = 'home'
        robot.request_replan = False
        robot.blocked_steps = 0
        robot.motion_state = 'home'
        robot.last_choice_debug = 'mission complete\nwaiting at home'

    def _update_return_home_behavior(self, robot: Robot) -> None:
        home_xy = self._home_xy()
        if self._robot_is_home(robot):
            robot.current_path = []
            robot.current_target = home_xy
            robot.current_mode = 'home'
            robot.request_replan = False
            robot.blocked_steps = 0
            robot.motion_state = 'home'
            robot.last_choice_debug = 'return-home complete\ninside home base'
            return

        has_path = bool(robot.current_path)
        replan_due = (self.time_s - robot.last_plan_time) >= float(self.cfg.mission_return_replan_period)
        need_plan = robot.request_replan or (not has_path) or robot.current_mode != 'return' or replan_due
        if not need_plan:
            return

        return_map = self._make_return_map(robot)
        path = self.planner.plan(robot.est_pose_xy(), home_xy, return_map)
        source = 'planner'
        if path is None or len(path) < 2:
            path = self._fallback_reverse_history_path(robot, home_xy)
            source = 'reverse-history'

        robot.last_plan_time = self.time_s
        robot.last_target_time = self.time_s
        robot.request_replan = False
        robot.current_mode = 'return'
        robot.current_target = home_xy
        robot.current_region_id = None
        robot.current_region_center_xy = None
        robot.region_hold_until = -1e9

        if path is not None and len(path) >= 2:
            robot.current_path = path[1:]
            block_n = len(robot.active_route_blocks(self.time_s))
            block_txt = f'  blocks={block_n}' if block_n else ''
            robot.last_choice_debug = (
                f'mission return-home {source}{block_txt}\n'
                f'to home, len={self._path_length(path):4.1f}m'
            )
            robot.logger.log(self.time_s, 'return-home-plan', source=source, target_xy=home_xy, path_len=self._path_length(path))
        else:
            robot.current_path = []
            robot.last_choice_debug = 'mission return-home\nno path to home yet'
            robot.logger.log(self.time_s, 'return-home-plan-failed', target_xy=home_xy)

    def _make_return_map(self, robot: Robot) -> np.ndarray:
        """
        Return-home planning map.

        It keeps known occupied cells blocked, but treats the robot's executed
        trail and the home base as known-free. This makes mission return robust
        even when an individual local map still has small unknown gaps along the
        already-traversed corridor.
        """
        data = np.array(robot.local_map.data, copy=True)

        def mark_disk_free(x: float, y: float, radius: float) -> None:
            gx, gy = self.grid.world_to_grid(x, y)
            cells = int(math.ceil(radius / self.grid.res))
            for yy in range(max(0, gy - cells), min(self.grid.ny, gy + cells + 1)):
                for xx in range(max(0, gx - cells), min(self.grid.nx, gx + cells + 1)):
                    wx, wy = self.grid.grid_to_world(xx, yy)
                    if (wx - x) ** 2 + (wy - y) ** 2 <= radius ** 2 and data[yy, xx] != OCCUPIED:
                        data[yy, xx] = FREE

        corridor_radius = max(self.cfg.robot_radius * 2.2, self.grid.res * 2.0)
        for x, y in robot.path_history[-900:]:
            mark_disk_free(float(x), float(y), corridor_radius)
        for x, y in robot.est_path_history[-900:]:
            mark_disk_free(float(x), float(y), corridor_radius)

        base = self.world.home_base
        pad = max(self.cfg.robot_radius, 0.2)
        xs = np.arange(base.xmin + pad, base.xmax - pad + 1e-6, max(self.grid.res, 0.35))
        ys = np.arange(base.ymin + pad, base.ymax - pad + 1e-6, max(self.grid.res, 0.35))
        for x in xs:
            for y in ys:
                mark_disk_free(float(x), float(y), corridor_radius)
        mark_disk_free(robot.x_est, robot.y_est, corridor_radius)
        mark_disk_free(self.world.home_marker.x, self.world.home_marker.y, corridor_radius)
        data = self._apply_temporary_route_blocks(
            robot,
            data,
            protect_points=[self._home_xy(), (robot.x_est, robot.y_est), (robot.x, robot.y)],
            protect_home=True,
        )
        return data

    def _fallback_reverse_history_path(self, robot: Robot, home_xy: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        pts = list(robot.path_history)
        if len(pts) < 2:
            return [robot.est_pose_xy(), home_xy]
        out: List[Tuple[float, float]] = [robot.est_pose_xy()]
        last = out[0]
        for x, y in reversed(pts[:-1]):
            if math.hypot(x - last[0], y - last[1]) >= 0.75:
                out.append((float(x), float(y)))
                last = out[-1]
            if math.hypot(last[0] - home_xy[0], last[1] - home_xy[1]) <= max(0.8, 0.35 * self.cfg.home_base_size):
                break
        if math.hypot(out[-1][0] - home_xy[0], out[-1][1] - home_xy[1]) > 0.25:
            out.append(home_xy)
        return out if len(out) >= 2 else None

    def _path_length(self, path: List[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])))

    def _update_mission_phase_after_motion(self) -> None:
        if self.mission_phase != 'return_home':
            return
        if not self.robots:
            return
        if all(self._robot_is_home(robot) for robot in self.robots):
            self.mission_phase = 'complete'
            self.mission_completed_at = self.time_s
            for robot in self.robots:
                self._hold_robot_at_home(robot)
                robot.logger.log(self.time_s, 'mission-complete', reason='all robots home')
            if bool(getattr(self.cfg, 'mission_stop_when_all_home', True)):
                self.running = False

    def _team_robot_states(self) -> List[Tuple[int, Tuple[float, float], float, Optional[int]]]:
        return [
            (r.robot_id, r.est_pose_xy(), r.covariance_trace(), r.home_hops)
            for r in self.robots
        ]

    def voronoi_owner_grid(self, *, known_only: bool = False) -> np.ndarray:
        """Weighted Voronoi owner per grid cell using estimated robot poses.

        This is used by the UI overlay and by the simulator-level cooperative
        assignment.  Obstacles are hidden from the overlay; unknown cells can be
        hidden by passing known_only=True.
        """
        if not self.robots:
            return np.full((self.grid.ny, self.grid.nx), -1, dtype=np.int16)

        yy, xx = np.mgrid[0:self.grid.ny, 0:self.grid.nx]
        wx = (xx + 0.5) * self.grid.res
        wy = (yy + 0.5) * self.grid.res
        best_cost = np.full((self.grid.ny, self.grid.nx), np.inf, dtype=float)
        owner = np.full((self.grid.ny, self.grid.nx), -1, dtype=np.int16)

        max_cov = max(float(getattr(self.cfg, 'decision_max_predicted_cov_trace', 2.8)), 1e-6)
        for robot in self.robots:
            cov_term = min(1.0, max(0.0, robot.covariance_trace()) / max_cov)
            hop_term = 0.0 if robot.home_hops is None else max(0.0, min(4.0, float(robot.home_hops))) / 4.0
            weight = 1.0 + 0.20 * cov_term + 0.08 * hop_term
            cost = np.hypot(wx - robot.x_est, wy - robot.y_est) * weight
            take = cost < best_cost
            best_cost[take] = cost[take]
            owner[take] = int(robot.robot_id)

        # Hide true obstacles to keep the overlay meaningful and readable.
        for obs in self.world.obstacles:
            x0, y0 = self.grid.world_to_grid(obs.xmin, obs.ymin)
            x1, y1 = self.grid.world_to_grid(obs.xmax, obs.ymax)
            owner[max(0, y0):min(self.grid.ny, y1 + 1), max(0, x0):min(self.grid.nx, x1 + 1)] = -1

        if known_only:
            masks = [(r.local_map.data != UNKNOWN) for r in self.robots]
            if masks:
                known = np.logical_or.reduce(masks)
                owner[~known] = -1

        self.last_voronoi_owner_grid = owner
        return owner

    def estimated_coverage(self) -> float:
        """Legacy coverage: any local robot map knows the cell."""
        masks = [(r.local_map.data != UNKNOWN) for r in self.robots]
        if not masks:
            return 0.0
        fused = np.logical_or.reduce(masks)
        return 100.0 * float(np.mean(fused))

    def team_fused_coverage(self) -> float:
        return float(self.team_fused_map.stats.coverage_pct)

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
        for pt in getattr(self, 'home_anchor_points', self._home_anchor_points()):
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
