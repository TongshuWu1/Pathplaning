"""Search-CAGE baseline simulator orchestrator."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cage_graph import RouteGraph
from .communication import CommunicationManager, HomeMemory, CommunicationState
from .config import AppConfig
from .mapping import OccupancyGrid
from .planner import GridPlanner
from .robot import RobotAgent
from .world import World


@dataclass
class MissionStatus:
    phase: str = "SEARCH_TARGET"
    success: bool = False
    message: str = "Searching for hidden target"





@dataclass
class PassageStatus:
    known: bool = False
    safe: bool = False
    score: float = 0.0
    length: float = 0.0
    mean_quality: float = 0.0
    min_clearance: float = 0.0
    unknown_fraction: float = 1.0
    message: str = "No target passage yet"


class Simulator:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(self.cfg.world.seed)
        self.world = World(self.cfg.world)
        self.time_s = 0.0
        self.step_count = 0
        self.running = True
        self.robots: list[RobotAgent] = []
        self.home_memory = HomeMemory(
            map=OccupancyGrid(self.world.width, self.world.height, self.cfg.mapping),
            graph=RouteGraph(self.cfg.cage.edge_merge_distance),
        )
        self.home_memory.graph.add_node(self.world.home, kind="home", confidence=1.0, allow_merge=False)
        self.communication = CommunicationManager(self.cfg.communication, self.world, self.home_memory, self.cfg.target_reporting)
        self.comm_state = CommunicationState()
        self.mission = MissionStatus()
        self.passage_status = PassageStatus()
        self.exploration_return_requested = False
        self._exploration_complete_counter = 0
        self._spawn_robots()

    def _spawn_robots(self) -> None:
        self.robots = []
        hx, hy = self.world.home
        n = self.cfg.robot.count
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            r = self.cfg.robot.spawn_spacing
            pose = (hx + math.cos(angle) * r, hy + math.sin(angle) * r, angle)
            if not self.world.is_free((pose[0], pose[1]), margin=self.cfg.robot.radius):
                pose = (hx, hy, angle)
            robot_rng = np.random.default_rng(self.cfg.world.seed + 101 * (i + 1))
            self.robots.append(RobotAgent(i, pose, self.cfg, self.world, robot_rng))

    def reset(self, cfg: AppConfig | None = None) -> None:
        self.__init__(cfg or self.cfg)

    def step(self) -> None:
        dt = self.cfg.dt
        self.time_s += dt
        self.step_count += 1

        for robot in self.robots:
            peer_poses=[tuple(other.true_pose) for other in self.robots if other.id!=robot.id]
            robot.step_predict_and_move(self.world, dt, peer_poses=peer_poses)
        for robot in self.robots:
            robot.sense_update_map_and_belief(self.world, self.time_s)

        self.comm_state = self.communication.update(self.robots, self.time_s)
        robot_by_id = {r.id: r for r in self.robots}
        for a_id, b_id in self.comm_state.direct_robot_edges:
            a = robot_by_id[a_id]
            b = robot_by_id[b_id]
            a.update_localization_from_teammate(b, self.world, self.time_s)
            b.update_localization_from_teammate(a, self.world, self.time_s)
        self._update_target_roundtrip_flags()
        self._update_exploration_return_flags()

        reserved_goals: dict[int, tuple[float, float]] = {}
        reserved_frontiers: dict[int, tuple[float, float]] = {}
        for robot in self.robots:
            robot.choose_task_and_plan(self.time_s, reserved_goals=reserved_goals, reserved_frontiers=reserved_frontiers)
            if robot.current_goal is not None and robot.current_task in {"SEARCH_HIER_NBV", "SEARCH_NBV", "SEARCH_FRONTIER", "SEARCH_OPEN_SECTOR", "DEPLOY_FROM_HOME", "EXPLORE_TOWARD_TARGET", "GO_TO_TARGET"}:
                reserved_goals[robot.id] = robot.current_goal
            if robot.current_goal is not None and robot.current_task in {"SEARCH_HIER_NBV", "SEARCH_NBV", "SEARCH_FRONTIER", "DEPLOY_FROM_HOME", "SEARCH_OPEN_SECTOR"}:
                reserved_frontiers[robot.id] = robot.current_goal
        for robot in self.robots:
            robot.compute_control()

        self._update_mission_status()


    def _required_roundtrip_count(self) -> int:
        if self.cfg.cage.require_all_robots_target_roundtrip:
            return len(self.robots)
        return max(1, min(len(self.robots), int(self.cfg.cage.min_robots_completed_roundtrip)))

    def _update_target_roundtrip_flags(self) -> None:
        # Target-roundtrip progression is evaluated by the simulator because it
        # can check physical target visibility and HOME arrival without leaking
        # truth into robot planning. Robots still plan only from their estimated
        # pose and communication-limited maps.
        target_known = self.home_memory.target.detected or any(r.target.detected for r in self.robots)
        if not target_known:
            return
        arrival_radius = max(self.cfg.world.target_radius, float(self.cfg.cage.target_arrival_radius_m))
        for robot in self.robots:
            if not robot.target.detected or robot.completed_target_roundtrip:
                continue
            true_xy = (float(robot.true_pose[0]), float(robot.true_pose[1]))
            if not robot.target_reached:
                est_close = robot.target.xy is not None and math.hypot(robot.est_xy[0]-robot.target.xy[0], robot.est_xy[1]-robot.target.xy[1]) <= arrival_radius
                target_visible = self.world.target_visible(tuple(robot.true_pose), self.cfg.lidar.range)
                if target_visible:
                    robot.mark_target_reached(self.time_s)
                    uploaded = self.communication.upload_robot_to_home(
                        robot,
                        self.time_s,
                        full=True,
                        robots=self.robots,
                        require_connection=self.cfg.target_reporting.require_home_connection_for_target_report,
                    )
                    if uploaded:
                        robot.last_home_full_upload_time = self.time_s
                elif est_close:
                    robot.status.note = "estimated_target_close_without_visual_confirmation"
            if robot.target_reached and self.world.home_base.contains(true_xy):
                robot.mark_target_roundtrip_complete(self.time_s)
                uploaded = self.communication.upload_robot_to_home(
                    robot,
                    self.time_s,
                    full=True,
                    robots=self.robots,
                    require_connection=True,
                )
                if uploaded:
                    robot.last_home_full_upload_time = self.time_s

    def _update_exploration_return_flags(self) -> None:
        # Target workflow has priority over generic exploration completion.
        if self.home_memory.target.detected or any(r.target.detected for r in self.robots):
            return
        if self.exploration_return_requested:
            for robot in self.robots:
                robot.force_return_home = True
            return
        # This check is intentionally throttled because frontier extraction is
        # one of the expensive operations on long runs.
        if self.step_count < 120 or self.step_count % 10 != 0:
            return
        known = self.home_memory.map.known_mask()
        for robot in self.robots:
            known = known | robot.map.known_mask()
        known_ratio = float(np.count_nonzero(known)) / float(max(1, known.size))
        frontier_count = 0
        max_count = max(1, self.cfg.cage.exploration_complete_max_frontiers_per_robot * len(self.robots))
        for robot in self.robots:
            frontiers = robot.map.find_frontiers(
                self.cfg.planning.frontier_min_cluster_size,
                self.cfg.planning.frontier_info_radius_m,
            )
            frontier_count += min(len(frontiers), max_count + 1)
        no_useful_frontiers = frontier_count <= max_count
        enough_known = known_ratio >= self.cfg.cage.exploration_complete_min_known_ratio
        complete_now = enough_known and no_useful_frontiers
        if complete_now:
            self._exploration_complete_counter += 10
        else:
            self._exploration_complete_counter = max(0, self._exploration_complete_counter - 5)
        if self._exploration_complete_counter >= self.cfg.cage.exploration_complete_stable_steps:
            self.exploration_return_requested = True
            for robot in self.robots:
                robot.force_return_home = True

    def _update_mission_status(self) -> None:
        home_target = self.home_memory.target.detected
        local_target = any(r.target.detected for r in self.robots)
        target_xy = self._target_xy_for_passage()
        routes = self.home_memory.graph.top_routes(k=max(1, self.cfg.cage.desired_route_count))
        self.home_memory.best_routes = routes
        returned = [
            r.id
            for r in self.robots
            if self.world.home_base.contains((float(r.true_pose[0]), float(r.true_pose[1])))
        ]
        # A robot that physically returns HOME still performs a full knowledge
        # upload, but target detection alone no longer forces the team home.
        for r in self.robots:
            if r.id in returned and self.time_s - r.last_home_full_upload_time >= self.cfg.communication.packet_period_s:
                uploaded = self.communication.upload_robot_to_home(
                    r,
                    self.time_s,
                    full=True,
                    robots=self.robots,
                    require_connection=True,
                )
                if uploaded:
                    r.last_home_full_upload_time = self.time_s
        all_returned_home = len(returned) == len(self.robots)
        returned_msg = f"{len(returned)}/{len(self.robots)} robots at HOME"

        self._evaluate_passage(target_xy)
        completed = [r.id for r in self.robots if r.completed_target_roundtrip]
        reached = [r.id for r in self.robots if r.target_reached]
        required = self._required_roundtrip_count()
        enough_roundtrips = len(completed) >= required
        roundtrip_msg = f"roundtrip {len(completed)}/{required} complete, reached target {reached or '-'}"
        if target_xy is not None and enough_roundtrips and self.passage_status.safe:
            self.mission = MissionStatus(
                "MISSION_COMPLETE",
                True,
                f"Target roundtrip complete and safe HOME-target passage known: score {self.passage_status.score:.2f}, "
                f"clear {self.passage_status.min_clearance:.2f} m, unknown {self.passage_status.unknown_fraction:.2f}; {roundtrip_msg}",
            )
        elif target_xy is not None and enough_roundtrips:
            self.mission = MissionStatus(
                "ROUNDTRIP_COMPLETE",
                True,
                f"Required robots reached target and returned HOME; passage evaluator says: {self.passage_status.message}; {roundtrip_msg}",
            )
        elif target_xy is not None and self.passage_status.safe:
            self.mission = MissionStatus(
                "SAFE_PASSAGE_KNOWN",
                False,
                f"Safe passage candidate known, but robots still need target roundtrip. {roundtrip_msg}",
            )
        elif home_target:
            self.mission = MissionStatus(
                "TARGET_ROUNDTRIP",
                False,
                f"HOME knows target; every robot goes to target then returns HOME. {self.passage_status.message}; {roundtrip_msg}",
            )
        elif local_target:
            self.mission = MissionStatus(
                "TARGET_REPORTED",
                False,
                "Target found by robot; sharing target and starting target-guided route attempts",
            )
        elif self.exploration_return_requested and all_returned_home:
            self.mission = MissionStatus("COMPLETE", True, "Exploration complete and all robots returned HOME")
        elif self.exploration_return_requested:
            self.mission = MissionStatus("RETURN_HOME_EXPLORATION_COMPLETE", False, f"Exploration complete; returning team to HOME ({returned_msg})")
        else:
            self.mission = MissionStatus("SEARCH_TARGET", False, "Searching for hidden target")

    def _target_xy_for_passage(self):
        if self.home_memory.target.detected and self.home_memory.target.xy is not None:
            return self.home_memory.target.xy
        best = None
        best_conf = -1.0
        for r in self.robots:
            if r.target.detected and r.target.xy is not None and r.target.confidence > best_conf:
                best = r.target.xy
                best_conf = r.target.confidence
        return best

    def _evaluate_passage(self, target_xy) -> None:
        if target_xy is None:
            self.passage_status = PassageStatus(False, False, 0.0, 0.0, 0.0, 0.0, 1.0, "No target passage yet")
            return
        grid = self.home_memory.map
        planner = GridPlanner(self.cfg.planning)
        passage = grid.passage_quality(self.cfg.passage_quality, robot_radius_m=self.cfg.robot.radius)
        result = planner.plan(grid, self.world.home, target_xy, passage_quality=passage)
        if not result.success or len(result.path) < 2:
            self.passage_status = PassageStatus(True, False, 0.0, 0.0, 0.0, 0.0, 1.0, "No connected candidate passage in HOME map")
            return
        cells = []
        seen = set()
        for a, b in zip(result.path[:-1], result.path[1:]):
            ca = grid.world_to_cell(a)
            cb = grid.world_to_cell(b)
            if ca is None or cb is None:
                continue
            for c in grid._bresenham(ca, cb):
                if c not in seen:
                    seen.add(c)
                    cells.append(c)
        if not cells:
            self.passage_status = PassageStatus(True, False, 0.0, 0.0, 0.0, 0.0, 1.0, "Candidate passage had no valid cells")
            return
        known = grid.known_mask()
        clearance = grid.clearance_map(max_radius_m=max(3.0, self.cfg.passage_quality.good_clearance_m * 3.0))
        q_vals = []
        cl_vals = []
        p_vals = []
        unknown = 0
        for i, j in cells:
            if not known[j, i]:
                unknown += 1
            q_vals.append(float(grid.quality[j, i]))
            cl_vals.append(float(clearance[j, i]))
            p_vals.append(float(passage[j, i]))
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(result.path[:-1], result.path[1:]))
        mean_quality = float(np.mean(q_vals)) if q_vals else 0.0
        min_clearance = float(np.min(cl_vals)) if cl_vals else 0.0
        unknown_fraction = float(unknown) / float(max(1, len(cells)))
        passage_score = float(np.mean(p_vals)) if p_vals else 0.0
        safe = (
            passage_score >= self.cfg.cage.safe_passage_score_threshold
            and min_clearance >= self.cfg.cage.safe_passage_min_clearance_m
            and unknown_fraction <= self.cfg.cage.safe_passage_max_unknown_fraction
        )
        msg = f"passage safety {passage_score:.2f}, clear {min_clearance:.2f} m, unknown {unknown_fraction:.2f}"
        self.passage_status = PassageStatus(True, safe, passage_score, length, mean_quality, min_clearance, unknown_fraction, msg)

    def run_headless(self, steps: int = 400) -> MissionStatus:
        for _ in range(steps):
            self.step()
            if self.mission.success:
                break
        return self.mission
