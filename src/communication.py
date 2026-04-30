"""LOS/team packet communication and HOME memory."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .cage_graph import RouteGraph, RouteCandidate
from .config import CommunicationConfig, TargetReportingConfig
from .geometry import Point, distance
from .mapping import OccupancyGrid
from .robot import RobotAgent, RobotPacket, TargetReport
from .world import World


@dataclass
class CommunicationState:
    direct_robot_edges: list[tuple[int, int]] = field(default_factory=list)
    robot_segments: list[tuple[Point, Point]] = field(default_factory=list)
    home_connected: dict[int, bool] = field(default_factory=dict)
    home_segments: list[tuple[Point, Point]] = field(default_factory=list)
    delivered_packets: int = 0


@dataclass
class HomeMemory:
    map: OccupancyGrid
    graph: RouteGraph
    target: TargetReport = field(default_factory=TargetReport)
    received_packets: int = 0
    best_routes: list[RouteCandidate] = field(default_factory=list)
    known_robot_pose: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    known_robot_goal: dict[int, Point | None] = field(default_factory=dict)
    known_robot_task: dict[int, str] = field(default_factory=dict)
    known_robot_paths: dict[int, list[Point]] = field(default_factory=dict)
    known_robot_visits: dict[int, list[Point]] = field(default_factory=dict)
    known_robot_trajectories: dict[int, list[Point]] = field(default_factory=dict)
    known_robot_regions: dict[int, dict] = field(default_factory=dict)
    known_robot_time: dict[int, float] = field(default_factory=dict)
    # Per-robot target roundtrip route evidence uploaded to HOME.
    # Values are serializable dictionaries produced by RobotAgent.target_route_summary().
    route_candidates: dict[int, dict] = field(default_factory=dict)


class CommunicationManager:
    def __init__(self, cfg: CommunicationConfig, world: World, home_memory: HomeMemory, target_cfg: TargetReportingConfig | None = None):
        self.cfg = cfg
        self.target_cfg = target_cfg or TargetReportingConfig()
        self.world = world
        self.home = world.home
        self.home_memory = home_memory
        self.state = CommunicationState()
        self._last_packet_time = -1.0e9

    def update(self, robots: list[RobotAgent], time_s: float) -> CommunicationState:
        """Update LOS links, robot knowledge exchange, and HOME uploads.

        Robot local panels display each robot's communication-limited
        knowledge map. Therefore direct robot LOS packets carry a partial
        knowledge-map digest. HOME receives only each connected robot's
        self-map upload, so teammate-derived knowledge is not relayed into
        HOME fused belief.
        """
        state = CommunicationState(home_connected={r.id: False for r in robots})
        allow_packets = (time_s - self._last_packet_time) >= self.cfg.packet_period_s
        packets: dict[tuple[int, str], RobotPacket] = {}
        robot_by_id = {r.id: r for r in robots}

        def packet_for(robot: RobotAgent, mode: str) -> RobotPacket:
            # mode: "partial" for robot-to-robot bandwidth-limited knowledge,
            # "full" for HOME upload, "empty" for intent-only if ever needed.
            key = (robot.id, mode)
            if key not in packets:
                if mode == "full":
                    packets[key] = robot.make_full_self_packet(time_s)
                elif mode == "partial":
                    packets[key] = robot.make_packet(time_s, include_map_digest=True, max_map_cells=650, map_source="knowledge")
                else:
                    packets[key] = robot.make_packet(time_s, include_map_digest=False)
            return packets[key]

        # Build LOS communication graph containing HOME as node -1.
        adjacency = self._build_los_adjacency(robots, state)

        # Robots connected to HOME through direct or multi-hop LOS can upload to HOME.
        connected_to_home = self._connected_to_home_ids(adjacency)
        for rid in connected_to_home:
            state.home_connected[rid] = True

        if allow_packets:
            # Direct robot LOS exchange: each robot receives the other's knowledge map.
            for a_id, b_id in state.direct_robot_edges:
                a = robot_by_id[a_id]
                b = robot_by_id[b_id]
                a.receive_packet(packet_for(b, "partial"))
                b.receive_packet(packet_for(a, "partial"))
                state.delivered_packets += 2

            # HOME receives each connected robot's own LiDAR map only.
            for rid in sorted(connected_to_home):
                r = robot_by_id[rid]
                self.upload_robot_to_home(r, time_s, full=True, require_connection=False)
                r.last_home_full_upload_time = time_s
                r.receive_packet(self._make_home_packet(time_s))
                state.delivered_packets += 2

            self._last_packet_time = time_s

        self.home_memory.best_routes = self.home_memory.graph.top_routes(k=4)
        self.state = state
        return state

    def _can_communicate(self, a: Point, b: Point) -> bool:
        return distance(a, b) <= self.cfg.radius and self.world.segment_free(a, b, margin=0.03)

    def _build_los_adjacency(self, robots: list[RobotAgent], state: CommunicationState | None = None) -> dict[int, set[int]]:
        adjacency: dict[int, set[int]] = {-1: set()}
        for r in robots:
            adjacency[r.id] = set()
            if self._can_communicate(r.est_xy, self.home):
                adjacency[-1].add(r.id)
                adjacency[r.id].add(-1)
                if state is not None:
                    state.home_segments.append((r.est_xy, self.home))

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                a = robots[i]
                b = robots[j]
                if self._can_communicate(a.est_xy, b.est_xy):
                    adjacency[a.id].add(b.id)
                    adjacency[b.id].add(a.id)
                    if state is not None:
                        state.direct_robot_edges.append((a.id, b.id))
                        state.robot_segments.append((a.est_xy, b.est_xy))
        return adjacency

    def _connected_to_home_ids(self, adjacency: dict[int, set[int]]) -> set[int]:
        connected: set[int] = set()
        q: deque[int] = deque([-1])
        seen = {-1}
        while q:
            node = q.popleft()
            for nb in adjacency.get(node, set()):
                if nb in seen:
                    continue
                seen.add(nb)
                q.append(nb)
                if nb >= 0:
                    connected.add(nb)
        return connected

    def can_upload_to_home(self, robot: RobotAgent, robots: list[RobotAgent] | None = None) -> bool:
        if robots is None:
            return self._can_communicate(robot.est_xy, self.home)
        return robot.id in self._connected_to_home_ids(self._build_los_adjacency(robots))

    def upload_robot_to_home(
        self,
        robot: RobotAgent,
        time_s: float,
        full: bool = True,
        robots: list[RobotAgent] | None = None,
        require_connection: bool = True,
    ) -> bool:
        if require_connection and not self.can_upload_to_home(robot, robots):
            return False
        packet = robot.make_full_self_packet(time_s) if full else robot.make_partial_self_packet(time_s, max_map_cells=650)
        target_accepted = self._deliver_to_home(packet)
        summary = robot.target_route_summary()
        if summary is not None:
            self.home_memory.route_candidates[int(robot.id)] = summary
            if summary.get("roundtrip_complete"):
                robot.route_candidate_uploaded = True
        if target_accepted and robot.target.detected:
            robot.target.reported_home = True
        return True

    def _deliver_to_home(self, packet: RobotPacket) -> bool:
        self.home_memory.received_packets += 1
        target_accepted = False
        if packet.sender_id >= 0:
            self.home_memory.known_robot_pose[packet.sender_id] = packet.estimated_pose
            self.home_memory.known_robot_goal[packet.sender_id] = packet.current_goal
            self.home_memory.known_robot_task[packet.sender_id] = packet.task
            self.home_memory.known_robot_paths[packet.sender_id] = list(packet.current_path_digest)
            self.home_memory.known_robot_visits[packet.sender_id] = list(packet.visited_digest)
            self.home_memory.known_robot_trajectories[packet.sender_id] = list(packet.trajectory_digest)
            if packet.assigned_region_center is not None and packet.assigned_region_id is not None:
                self.home_memory.known_robot_regions[packet.sender_id] = {
                    "region_id": tuple(packet.assigned_region_id),
                    "center": tuple(packet.assigned_region_center),
                    "radius": float(packet.assigned_region_radius),
                    "score": float(packet.assigned_region_score),
                    "time_s": float(packet.time_s),
                }
            else:
                self.home_memory.known_robot_regions.pop(packet.sender_id, None)
            self.home_memory.known_robot_time[packet.sender_id] = float(packet.time_s)
        if packet.map_digest:
            self.home_memory.map.merge_from_digest(packet.map_digest, combine_sources=True)
        self.home_memory.graph.merge_from_digest(packet.graph_digest)
        self.home_memory.graph.mark_all_reported_home()
        if packet.target_report and packet.target_report.get("detected"):
            tr = packet.target_report
            conf = float(tr.get("confidence", 0.0))
            source_robot = int(tr.get("source_robot", packet.sender_id))
            direct_source_report = packet.sender_id == source_robot
            relay_allowed = bool(self.target_cfg.allow_relayed_target_to_home)
            if direct_source_report or relay_allowed:
                if not self.home_memory.target.detected or conf > self.home_memory.target.confidence:
                    xy = tuple(tr["xy"])
                    self.home_memory.target = TargetReport(True, (float(xy[0]), float(xy[1])), conf, source_robot, float(tr.get("time_s", packet.time_s)), True)
                    tid = self.home_memory.graph.add_node(self.home_memory.target.xy, kind="target", confidence=conf, allow_merge=True)
                    self.home_memory.graph.target_id = tid
                target_accepted = True
        return target_accepted

    def _make_home_packet(self, time_s: float) -> RobotPacket:
        target_dict = None
        if self.home_memory.target.detected and self.home_memory.target.xy is not None:
            target_dict = {
                "detected": True,
                "xy": [float(self.home_memory.target.xy[0]), float(self.home_memory.target.xy[1])],
                "confidence": float(self.home_memory.target.confidence),
                "source_robot": int(self.home_memory.target.source_robot),
                "time_s": float(self.home_memory.target.time_s),
                "reported_home": True,
            }
        return RobotPacket(
            sender_id=-1,
            time_s=float(time_s),
            map_digest={},
            graph_digest=self.home_memory.graph.make_digest(-1, time_s),
            target_report=target_dict,
            task="HOME_REPORT",
            current_goal=None,
            current_path_digest=[],
            visited_digest=[],
            trajectory_digest=[],
            estimated_pose=(self.home[0], self.home[1], 0.0),
            pose_cov_trace=0.0,
        )
