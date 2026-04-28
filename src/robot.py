"""Robot agent for the clean Search-CAGE baseline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .cage_graph import RouteGraph, RouteCandidate
from .config import AppConfig
from .geometry import Point, Pose, angle_to, distance, segment_length, wrap_angle
from .lidar_assessment import LidarAssessment, assess_lidar
from .localization import PoseEstimator
from .mapping import OccupancyGrid, FrontierCluster
from .planner import GridPlanner, PlanResult
from .sensors import LidarScan, LidarSensor
from .world import World


@dataclass
class TargetReport:
    detected: bool = False
    xy: Point | None = None
    confidence: float = 0.0
    source_robot: int = -1
    time_s: float = 0.0
    reported_home: bool = False


@dataclass
class RobotPacket:
    sender_id: int
    time_s: float
    map_digest: dict
    graph_digest: dict
    target_report: dict | None
    task: str
    estimated_pose: tuple[float, float, float]
    pose_cov_trace: float


@dataclass
class RobotStatus:
    task: str = "INIT"
    planning_source: str = "LiDAR local map"
    note: str = ""
    goal: Point | None = None
    last_plan_success: bool = False
    last_plan_reason: str = ""


class RobotAgent:
    def __init__(self, robot_id: int, initial_pose: Pose, cfg: AppConfig, world: World, rng: np.random.Generator):
        self.id = robot_id
        self.cfg = cfg
        self.rng = rng
        self.true_pose = np.array(initial_pose, dtype=float)
        self.true_path: list[Point] = [(float(initial_pose[0]), float(initial_pose[1]))]

        self.estimator = PoseEstimator(initial_pose, cfg.motion, rng)
        self.lidar = LidarSensor(cfg.lidar, rng)
        self.map = OccupancyGrid(world.width, world.height, cfg.mapping)
        self.graph = RouteGraph(cfg.cage.edge_merge_distance)
        self.home_node = self.graph.add_node(world.home, kind="home", confidence=1.0, allow_merge=False)
        self.last_graph_node = self.home_node
        self.last_keypoint_xy = world.home

        self.scan: LidarScan | None = None
        self.assessment = LidarAssessment()
        self.planner = GridPlanner(cfg.planning)
        self.path: list[Point] = []
        self.path_index: int = 0
        self.last_replan_time = -999.0
        self.current_goal: Point | None = None
        self.current_task = "SEARCH"
        self.status = RobotStatus(task="SEARCH")
        self.target = TargetReport()
        self.known_teammate_goals: dict[int, Point] = {}
        self.known_teammate_pose: dict[int, tuple[float, float, float]] = {}
        self.last_command: tuple[float, float] = (0.0, 0.0)
        self.last_pose_quality: float = 1.0
        self.best_routes: list[RouteCandidate] = []
        self.received_packets: int = 0
        self.blocked_events: int = 0

    @property
    def est_pose(self) -> Pose:
        return self.estimator.belief.as_pose()

    @property
    def est_xy(self) -> Point:
        return self.estimator.belief.xy

    @property
    def cov_trace(self) -> float:
        return self.estimator.belief.cov_trace_xy

    def step_predict_and_move(self, world: World, dt: float) -> None:
        v, omega = self.last_command
        x, y, th = self.true_pose
        # True motion follows command; collision checked against hidden world.
        new_th = wrap_angle(th + omega * dt)
        candidate = (float(x + math.cos(new_th) * v * dt), float(y + math.sin(new_th) * v * dt))
        if world.is_free(candidate, margin=self.cfg.robot.radius):
            self.true_pose[:] = [candidate[0], candidate[1], new_th]
        else:
            self.blocked_events += 1
            self.path = []
            self.status.note = "true_collision_prevented_by_sim"
            self.last_command = (0.0, 0.0)
        self.true_path.append((float(self.true_pose[0]), float(self.true_pose[1])))
        self.estimator.predict_from_command(v, omega, dt)

    def sense_update_map_and_belief(self, world: World, time_s: float) -> None:
        visible_landmarks = world.visible_landmarks(tuple(self.true_pose), self.cfg.world.landmark_detection_range)
        self.estimator.update_with_landmarks(visible_landmarks, self.cfg.world.landmark_detection_range)
        self.scan = self.lidar.sense(world, tuple(self.true_pose))
        pose_quality = self.estimator.quality()
        self.last_pose_quality = pose_quality
        prev = None if self.assessment.decision_note == "init" else self.assessment.consistency
        self.map.update_from_lidar(self.est_pose, self.scan, pose_quality, self.id, time_s)
        self.assessment = assess_lidar(self.map, self.est_pose, self.scan, self.cfg.lidar, self.cfg.assessment, prev)
        self._detect_target(world, time_s)
        self._update_route_graph(time_s)

    def _detect_target(self, world: World, time_s: float) -> None:
        if self.target.detected:
            return
        if not world.target_visible(tuple(self.true_pose), self.cfg.lidar.range):
            return
        # Estimate target position from measured relative range/bearing using the robot's estimated pose.
        true_xy = (float(self.true_pose[0]), float(self.true_pose[1]))
        true_bearing = angle_to(true_xy, world.target) - float(self.true_pose[2])
        true_range = distance(true_xy, world.target)
        r = max(0.05, true_range + self.rng.normal(0.0, 0.06))
        b = true_bearing + self.rng.normal(0.0, math.radians(2.0))
        ex, ey, eth = self.est_pose
        est_target = (float(ex + math.cos(eth + b) * r), float(ey + math.sin(eth + b) * r))
        conf = float(np.clip(self.assessment.consistency * self.last_pose_quality, 0.1, 1.0))
        self.target = TargetReport(True, est_target, conf, self.id, time_s, False)
        tid = self.graph.add_node(est_target, kind="target", confidence=conf, allow_merge=True)
        self.graph.target_id = tid
        clearance = max(0.05, min(self.assessment.front_clearance, self.assessment.left_clearance, self.assessment.right_clearance))
        self.graph.add_or_update_edge(self.last_graph_node, tid, clearance=clearance, consistency=max(0.05, self.assessment.consistency), pose_quality=self.last_pose_quality, robot_id=self.id, time_s=time_s, success=True)
        self.status.note = f"target_detected_by_R{self.id}"

    def _update_route_graph(self, time_s: float) -> None:
        xy = self.est_xy
        if distance(xy, self.last_keypoint_xy) < self.cfg.robot.keypoint_spacing:
            return
        node_kind = "anchor" if self.assessment.consistency > 0.68 and self.last_pose_quality > 0.45 else "keypoint"
        node = self.graph.add_node(xy, kind=node_kind, confidence=max(self.assessment.consistency, self.last_pose_quality))
        clearance = max(0.05, min(self.assessment.front_clearance, self.assessment.left_clearance, self.assessment.right_clearance))
        self.graph.add_or_update_edge(
            self.last_graph_node,
            node,
            clearance=clearance,
            consistency=max(0.05, self.assessment.consistency),
            pose_quality=self.last_pose_quality,
            robot_id=self.id,
            time_s=time_s,
            success=not self.assessment.blocked_forward,
        )
        self.last_graph_node = node
        self.last_keypoint_xy = xy

    def receive_packet(self, packet: RobotPacket) -> None:
        if packet.sender_id == self.id:
            return
        self.received_packets += 1
        self.map.merge_from_digest(packet.map_digest)
        self.graph.merge_from_digest(packet.graph_digest)
        self.known_teammate_pose[packet.sender_id] = packet.estimated_pose
        if packet.target_report:
            tr = packet.target_report
            if tr.get("detected"):
                conf = float(tr.get("confidence", 0.0))
                if not self.target.detected or conf > self.target.confidence:
                    xy = tuple(tr["xy"])
                    self.target = TargetReport(True, (float(xy[0]), float(xy[1])), conf, int(tr.get("source_robot", packet.sender_id)), float(tr.get("time_s", packet.time_s)), bool(tr.get("reported_home", False)))
                    tid = self.graph.add_node(self.target.xy, kind="target", confidence=conf, allow_merge=True)
                    self.graph.target_id = tid
        if packet.task and packet.estimated_pose:
            # Track teammate intent coarsely for duplicate frontier penalty.
            pass

    def make_packet(self, time_s: float) -> RobotPacket:
        target_dict = None
        if self.target.detected and self.target.xy is not None:
            target_dict = {
                "detected": True,
                "xy": [float(self.target.xy[0]), float(self.target.xy[1])],
                "confidence": float(self.target.confidence),
                "source_robot": int(self.target.source_robot),
                "time_s": float(self.target.time_s),
                "reported_home": bool(self.target.reported_home),
            }
        return RobotPacket(
            sender_id=self.id,
            time_s=float(time_s),
            map_digest=self.map.make_digest(self.id, time_s),
            graph_digest=self.graph.make_digest(self.id, time_s),
            target_report=target_dict,
            task=self.current_task,
            estimated_pose=self.est_pose,
            pose_cov_trace=self.cov_trace,
        )

    def choose_task_and_plan(self, time_s: float, team_goals: dict[int, Point]) -> None:
        # LiDAR immediate safety dominates the current path, not EKF covariance.
        if self.assessment.blocked_forward and self.path:
            self.path = []
            self.path_index = 0
            self.status.note = "path_invalidated_by_lidar_block"
        if time_s - self.last_replan_time < self.cfg.robot.path_replan_period_s and self.path_index < len(self.path):
            return

        goal, task, reason = self._select_goal_from_lidar_map(team_goals)
        self.current_goal = goal
        self.current_task = task
        self.status.task = task
        self.status.goal = goal
        self.status.planning_source = "LiDAR map + CAGE route evidence"
        self.status.note = reason

        if goal is None:
            self.path = []
            self.path_index = 0
            self.status.last_plan_success = False
            self.status.last_plan_reason = "no_goal_available"
            self.last_replan_time = time_s
            return
        result = self.planner.plan(self.map, self.est_xy, goal)
        if result.success and len(result.path) >= 2:
            self.path = self._downsample_path(result.path, spacing=0.45)
            self.path_index = 0
        else:
            self.path = []
            self.path_index = 0
        self.status.last_plan_success = result.success
        self.status.last_plan_reason = result.reason
        self.last_replan_time = time_s
        self.best_routes = self.graph.top_routes(k=4)

    def _select_goal_from_lidar_map(self, team_goals: dict[int, Point]) -> tuple[Point | None, str, str]:
        # If target is known, focus on route certification/reporting and route-to-target progress.
        if self.target.detected and self.target.xy is not None:
            # Try to connect current graph to target if close enough and target is map-reachable.
            dtarget = distance(self.est_xy, self.target.xy)
            if dtarget > self.cfg.robot.goal_tolerance:
                return self.target.xy, "ADVANCE_TO_TARGET", "target_known_plan_to_detected_target"
            return self.target.xy, "CERTIFY_TARGET_EDGE", "near_detected_target_certifying_edge"

        # LiDAR consistency is a LiDAR/map belief issue. Reanchor to graph if the scan disagrees with map.
        if self.assessment.consistency < self.cfg.cage.reanchor_consistency_threshold:
            anchor = self._nearest_anchor()
            if anchor is not None and distance(anchor, self.est_xy) > 0.35:
                return anchor, "REANCHOR", "low_scan_map_consistency_reanchor"

        frontiers = self.map.find_frontiers(self.cfg.planning.frontier_min_cluster_size, self.cfg.planning.frontier_info_radius_m)
        if not frontiers:
            # If no frontier is visible, move into best open LiDAR sector.
            gx = self.est_xy[0] + math.cos(self.est_pose[2] + self.assessment.best_open_angle) * min(1.4, self.cfg.lidar.range * 0.35)
            gy = self.est_xy[1] + math.sin(self.est_pose[2] + self.assessment.best_open_angle) * min(1.4, self.cfg.lidar.range * 0.35)
            return (gx, gy), "SEARCH_OPEN_SECTOR", "no_frontier_use_best_lidar_open_sector"

        candidates = frontiers[: self.cfg.planning.frontier_sample_count]
        best: tuple[float, FrontierCluster] | None = None
        for fr in candidates:
            d = max(0.1, distance(self.est_xy, fr.centroid_world))
            clearance = self.map.clearance_at(fr.centroid_world)
            dup = 0.0
            for rid, g in team_goals.items():
                if rid == self.id or g is None:
                    continue
                dup += math.exp(-distance(g, fr.centroid_world) / 1.8)
            # If target is unknown, useful search is information + clearance + spatial spread.
            score = (
                self.cfg.planning.information_weight * math.log1p(fr.information_gain)
                + self.cfg.planning.clearance_weight * min(2.0, clearance)
                - self.cfg.planning.distance_weight * d
                - self.cfg.planning.duplicate_penalty_weight * dup
            )
            if best is None or score > best[0]:
                best = (score, fr)
        if best is None:
            return None, "WAIT", "no_good_frontier"
        return best[1].centroid_world, "SEARCH_FRONTIER", "lidar_frontier_selected"

    def _nearest_anchor(self) -> Point | None:
        anchors = [n.xy for n in self.graph.nodes.values() if n.kind in {"home", "anchor"}]
        if not anchors:
            return None
        return min(anchors, key=lambda p: distance(self.est_xy, p))

    def _downsample_path(self, path: list[Point], spacing: float) -> list[Point]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        last = path[0]
        for p in path[1:-1]:
            if distance(last, p) >= spacing:
                out.append(p)
                last = p
        out.append(path[-1])
        return out

    def compute_control(self) -> tuple[float, float]:
        if not self.path or self.path_index >= len(self.path):
            self.last_command = (0.0, 0.0)
            return self.last_command
        pos = self.est_xy
        target = self.path[self.path_index]
        if distance(pos, target) < self.cfg.robot.waypoint_tolerance:
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.last_command = (0.0, 0.0)
                return self.last_command
            target = self.path[self.path_index]
        desired = angle_to(pos, target)
        err = wrap_angle(desired - self.est_pose[2])
        omega = float(np.clip(self.cfg.robot.turn_gain * err, -1.4, 1.4))
        # Slow down only as confidence/safety modulation.  EKF does not select the goal.
        consistency_scale = 0.45 if self.assessment.consistency < self.cfg.assessment.caution_consistency else 1.0
        front_scale = np.clip((self.assessment.front_clearance - self.cfg.lidar.blocked_forward_distance) / 1.0, 0.0, 1.0)
        v = self.cfg.robot.max_speed * consistency_scale * max(0.15, float(front_scale)) * max(0.25, math.cos(err))
        if self.assessment.blocked_forward:
            v = 0.0
        self.last_command = (float(v), omega)
        return self.last_command
