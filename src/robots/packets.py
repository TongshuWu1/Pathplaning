from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class RoutePoint:
    x: float
    y: float
    t: float
    point_type: str = "ROUTE"
    note: str = ""

    @property
    def xy(self) -> Tuple[float, float]:
        return (float(self.x), float(self.y))


@dataclass
class KnowledgeSnapshot:
    subject_robot_id: int
    source_robot_id: int
    knowledge_time: float
    pose_xy: Tuple[float, float]
    pose_cov: np.ndarray
    target_xy: Optional[Tuple[float, float]]
    current_region_id: Optional[int] = None
    current_region_center_xy: Optional[Tuple[float, float]] = None
    help_request_active: bool = False
    help_request_xy: Optional[Tuple[float, float]] = None
    help_request_time: float = 0.0
    help_request_reason: str = ''
    help_assigned_helper_id: Optional[int] = None
    home_connected: bool = False
    home_hops: Optional[int] = None
    direct_neighbors: List[int] = field(default_factory=list)
    reachable_peer_ids: List[int] = field(default_factory=list)
    landmark_beliefs: Dict[str, dict] = field(default_factory=dict)
    route_points: List[RoutePoint] = field(default_factory=list)
    recent_trail: List[RoutePoint] = field(default_factory=list)
    semantic_points: List[RoutePoint] = field(default_factory=list)
    is_stale: bool = False
    stale_age_s: float = 0.0
    is_self: bool = False

    @property
    def path_xy(self) -> List[Tuple[float, float]]:
        ordered = sorted([*self.route_points, *self.recent_trail], key=lambda pt: (pt.t, pt.point_type != "HOME"))
        out: List[Tuple[float, float]] = []
        seen = set()
        for pt in ordered:
            key = (round(pt.x, 3), round(pt.y, 3))
            if key in seen:
                continue
            out.append(pt.xy)
            seen.add(key)
        return out

    @property
    def keypoints_xy(self) -> List[Tuple[float, float]]:
        return [pt.xy for pt in self.semantic_points]


@dataclass
class TeammatePacket:
    robot_id: int
    timestamp: float = 0.0
    seq_num: int = 0
    knowledge: List[KnowledgeSnapshot] = field(default_factory=list)

    def subject_snapshot(self, subject_robot_id: int) -> Optional[KnowledgeSnapshot]:
        for snap in self.knowledge:
            if snap.subject_robot_id == subject_robot_id:
                return snap
        return None

    @property
    def self_snapshot(self) -> Optional[KnowledgeSnapshot]:
        return self.subject_snapshot(self.robot_id)

    @property
    def pose_xy(self) -> Tuple[float, float]:
        snap = self.self_snapshot
        return (0.0, 0.0) if snap is None else snap.pose_xy

    @property
    def pose_cov(self) -> np.ndarray:
        snap = self.self_snapshot
        return np.eye(3) if snap is None else np.array(snap.pose_cov, copy=True)

    @property
    def target_xy(self) -> Optional[Tuple[float, float]]:
        snap = self.self_snapshot
        return None if snap is None else snap.target_xy

    @property
    def route_points(self) -> List[RoutePoint]:
        snap = self.self_snapshot
        return [] if snap is None else list(snap.route_points)

    @property
    def recent_trail(self) -> List[RoutePoint]:
        snap = self.self_snapshot
        return [] if snap is None else list(snap.recent_trail)

    @property
    def semantic_points(self) -> List[RoutePoint]:
        snap = self.self_snapshot
        return [] if snap is None else list(snap.semantic_points)

    @property
    def path_xy(self) -> List[Tuple[float, float]]:
        snap = self.self_snapshot
        return [] if snap is None else snap.path_xy

    @property
    def keypoints_xy(self) -> List[Tuple[float, float]]:
        snap = self.self_snapshot
        return [] if snap is None else snap.keypoints_xy
