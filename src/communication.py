"""Direct LOS packet communication and home-base memory."""
from __future__ import annotations

from dataclasses import dataclass, field

from .cage_graph import RouteGraph, RouteCandidate
from .config import CommunicationConfig
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
    known_robot_time: dict[int, float] = field(default_factory=dict)


class CommunicationManager:
    def __init__(self, cfg: CommunicationConfig, world: World, home_memory: HomeMemory):
        self.cfg = cfg
        self.world = world
        self.home = world.home
        self.home_memory = home_memory
        self.state = CommunicationState()
        self._last_packet_time = -1.0e9

    def update(self, robots: list[RobotAgent], time_s: float) -> CommunicationState:
        state = CommunicationState(home_connected={r.id: False for r in robots})
        allow_packets = (time_s - self._last_packet_time) >= self.cfg.packet_period_s
        packets: dict[int, RobotPacket] = {}

        def packet_for(robot: RobotAgent) -> RobotPacket:
            if robot.id not in packets:
                packets[robot.id] = robot.make_packet(time_s)
            return packets[robot.id]

        for r in robots:
            if self._can_communicate(r.est_xy, self.home):
                state.home_connected[r.id] = True
                state.home_segments.append((r.est_xy, self.home))
                if allow_packets:
                    self._deliver_to_home(packet_for(r))
                    r.receive_packet(self._make_home_packet(time_s))
                    state.delivered_packets += 1

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                a = robots[i]
                b = robots[j]
                if self._can_communicate(a.est_xy, b.est_xy):
                    state.direct_robot_edges.append((a.id, b.id))
                    state.robot_segments.append((a.est_xy, b.est_xy))
                    if allow_packets:
                        a.receive_packet(packet_for(b))
                        b.receive_packet(packet_for(a))
                        state.delivered_packets += 2

        if allow_packets:
            self._last_packet_time = time_s
        self.home_memory.best_routes = self.home_memory.graph.top_routes(k=4)
        self.state = state
        return state

    def _can_communicate(self, a: Point, b: Point) -> bool:
        return distance(a, b) <= self.cfg.radius and self.world.segment_free(a, b, margin=0.03)

    def _deliver_to_home(self, packet: RobotPacket) -> None:
        self.home_memory.received_packets += 1
        if packet.sender_id >= 0:
            self.home_memory.known_robot_pose[packet.sender_id] = packet.estimated_pose
            self.home_memory.known_robot_goal[packet.sender_id] = packet.current_goal
            self.home_memory.known_robot_task[packet.sender_id] = packet.task
            self.home_memory.known_robot_paths[packet.sender_id] = list(packet.current_path_digest)
            self.home_memory.known_robot_visits[packet.sender_id] = list(packet.visited_digest)
            self.home_memory.known_robot_time[packet.sender_id] = float(packet.time_s)
        self.home_memory.map.merge_from_digest(packet.map_digest, combine_sources=True)
        self.home_memory.graph.merge_from_digest(packet.graph_digest)
        self.home_memory.graph.mark_all_reported_home()
        if packet.target_report and packet.target_report.get("detected"):
            tr = packet.target_report
            conf = float(tr.get("confidence", 0.0))
            if not self.home_memory.target.detected or conf > self.home_memory.target.confidence:
                xy = tuple(tr["xy"])
                self.home_memory.target = TargetReport(True, (float(xy[0]), float(xy[1])), conf, int(tr.get("source_robot", packet.sender_id)), float(tr.get("time_s", packet.time_s)), True)
                tid = self.home_memory.graph.add_node(self.home_memory.target.xy, kind="target", confidence=conf, allow_merge=True)
                self.home_memory.graph.target_id = tid

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
            map_digest=self.home_memory.map.make_digest(-1, time_s, max_cells=400),
            graph_digest=self.home_memory.graph.make_digest(-1, time_s),
            target_report=target_dict,
            task="HOME_REPORT",
            current_goal=None,
            current_path_digest=[],
            visited_digest=[],
            estimated_pose=(self.home[0], self.home[1], 0.0),
            pose_cov_trace=0.0,
        )
