from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .nav import downsample_route_points
from .packets import KnowledgeSnapshot, RoutePoint, TeammatePacket


@dataclass
class TeammateMemoryRecord:
    robot_id: int
    source_robot_id: Optional[int] = None
    last_update_time: float = 0.0
    last_received_time: float = 0.0
    pose_xy: Tuple[float, float] = (0.0, 0.0)
    pose_cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    target_xy: Optional[Tuple[float, float]] = None
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


class TeammateMemoryStore:
    def __init__(
        self,
        self_robot_id: int,
        max_route_points: int,
        max_recent_points: int,
        display_path_points: int,
        persist_s: float,
    ) -> None:
        self.self_robot_id = self_robot_id
        self.max_route_points = max_route_points
        self.max_recent_points = max_recent_points
        self.display_path_points = display_path_points
        self.persist_s = persist_s
        self.records: Dict[int, TeammateMemoryRecord] = {}

    def merge_packet(self, packet: TeammatePacket, now: float) -> List[TeammateMemoryRecord]:
        updated: List[TeammateMemoryRecord] = []
        for snap in packet.knowledge:
            if snap.subject_robot_id == self.self_robot_id:
                continue
            rec = self._merge_snapshot(snap, now)
            if rec is not None:
                updated.append(rec)
        return updated

    def _merge_snapshot(self, snap: KnowledgeSnapshot, now: float) -> Optional[TeammateMemoryRecord]:
        rec = self.records.get(snap.subject_robot_id)
        if rec is None:
            rec = TeammateMemoryRecord(robot_id=snap.subject_robot_id)
            self.records[snap.subject_robot_id] = rec

        should_accept = False
        if snap.knowledge_time > rec.last_update_time + 1e-9:
            should_accept = True
        elif abs(snap.knowledge_time - rec.last_update_time) <= 1e-9:
            if rec.source_robot_id is None or snap.source_robot_id == snap.subject_robot_id:
                should_accept = True

        if not should_accept and rec.last_received_time > 0.0:
            return None

        rec.source_robot_id = int(snap.source_robot_id)
        rec.last_update_time = float(snap.knowledge_time)
        rec.last_received_time = float(now)
        rec.pose_xy = (float(snap.pose_xy[0]), float(snap.pose_xy[1]))
        rec.pose_cov = np.array(snap.pose_cov, copy=True)
        rec.target_xy = None if snap.target_xy is None else (float(snap.target_xy[0]), float(snap.target_xy[1]))
        rec.home_connected = bool(snap.home_connected)
        rec.home_hops = None if snap.home_hops is None else int(snap.home_hops)
        rec.direct_neighbors = [int(v) for v in snap.direct_neighbors]
        rec.reachable_peer_ids = [int(v) for v in snap.reachable_peer_ids]
        rec.landmark_beliefs = self._merge_landmarks(rec.landmark_beliefs, snap.landmark_beliefs)
        rec.route_points = self._merge_points(rec.route_points, snap.route_points, self.max_route_points)
        rec.recent_trail = self._merge_recent_trail(rec.recent_trail, snap.recent_trail)
        rec.semantic_points = self._merge_points(rec.semantic_points, snap.semantic_points, self.max_route_points)
        rec.is_stale = False
        rec.stale_age_s = 0.0
        return rec

    def prune(self, now: float) -> None:
        for rec in self.records.values():
            rec.stale_age_s = max(0.0, float(now - rec.last_update_time))
            rec.is_stale = rec.stale_age_s > self.persist_s

    def export_knowledge(self) -> List[KnowledgeSnapshot]:
        out: List[KnowledgeSnapshot] = []
        for rec in sorted(self.records.values(), key=lambda r: (r.robot_id, r.last_update_time)):
            out.append(
                KnowledgeSnapshot(
                    subject_robot_id=rec.robot_id,
                    source_robot_id=self.self_robot_id,
                    knowledge_time=float(rec.last_update_time),
                    pose_xy=(float(rec.pose_xy[0]), float(rec.pose_xy[1])),
                    pose_cov=np.array(rec.pose_cov, copy=True),
                    target_xy=rec.target_xy,
                    home_connected=rec.home_connected,
                    home_hops=rec.home_hops,
                    direct_neighbors=list(rec.direct_neighbors),
                    reachable_peer_ids=list(rec.reachable_peer_ids),
                    landmark_beliefs=self._copy_landmarks(rec.landmark_beliefs),
                    route_points=downsample_route_points(rec.route_points, self.max_route_points),
                    recent_trail=downsample_route_points(rec.recent_trail, self.max_recent_points),
                    semantic_points=downsample_route_points(rec.semantic_points, self.max_route_points),
                    is_stale=rec.is_stale,
                    stale_age_s=float(rec.stale_age_s),
                    is_self=False,
                )
            )
        return out

    def _merge_points(self, existing: List[RoutePoint], incoming: List[RoutePoint], max_points: int) -> List[RoutePoint]:
        if not incoming:
            return downsample_route_points(existing, max_points)
        out: List[RoutePoint] = list(existing)
        seen = {(round(pt.x, 3), round(pt.y, 3), pt.point_type, round(pt.t, 2)) for pt in out}
        for pt in incoming:
            key = (round(pt.x, 3), round(pt.y, 3), pt.point_type, round(pt.t, 2))
            if key in seen:
                continue
            out.append(pt)
            seen.add(key)
        out = sorted(out, key=lambda pt: (pt.t, pt.point_type != "HOME"))
        return downsample_route_points(out, max_points)


    def _copy_landmarks(self, beliefs: Dict[str, dict]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for key, info in beliefs.items():
            if not bool(info.get('discovered', False)):
                continue
            out[str(key)] = {
                'xy': [float(info['xy'][0]), float(info['xy'][1])],
                'knowledge_time': float(info.get('knowledge_time', 0.0)),
                'source_robot_id': None if info.get('source_robot_id') is None else int(info['source_robot_id']),
                'observed_count': int(info.get('observed_count', 1)),
                'is_home': bool(info.get('is_home', False)),
                'discovered': True,
            }
        return out

    def _merge_landmarks(self, existing: Dict[str, dict], incoming: Dict[str, dict]) -> Dict[str, dict]:
        out = self._copy_landmarks(existing)
        for key, info in incoming.items():
            if not bool(info.get('discovered', False)):
                continue
            t_new = float(info.get('knowledge_time', 0.0))
            prev = out.get(str(key))
            if prev is None or t_new >= float(prev.get('knowledge_time', 0.0)):
                out[str(key)] = {
                    'xy': [float(info['xy'][0]), float(info['xy'][1])],
                    'knowledge_time': t_new,
                    'source_robot_id': None if info.get('source_robot_id') is None else int(info['source_robot_id']),
                    'observed_count': int(info.get('observed_count', 1)),
                    'is_home': bool(info.get('is_home', False)),
                    'discovered': True,
                }
        return out

    def _merge_recent_trail(self, existing: List[RoutePoint], incoming: List[RoutePoint]) -> List[RoutePoint]:
        if not incoming:
            return existing[-self.max_recent_points:]
        out: List[RoutePoint] = list(existing)
        seen = {(round(pt.x, 3), round(pt.y, 3), round(pt.t, 2)) for pt in out}
        for pt in incoming:
            key = (round(pt.x, 3), round(pt.y, 3), round(pt.t, 2))
            if key in seen:
                continue
            out.append(pt)
            seen.add(key)
        out = sorted(out, key=lambda pt: pt.t)
        return out[-self.max_recent_points:]

    def _combined_path(self, rec: TeammateMemoryRecord) -> List[RoutePoint]:
        pts = [*rec.route_points, *rec.recent_trail]
        pts = sorted(pts, key=lambda pt: (pt.t, pt.point_type != "HOME"))
        dedup: List[RoutePoint] = []
        seen = set()
        for pt in pts:
            key = (round(pt.x, 3), round(pt.y, 3), pt.point_type, round(pt.t, 2))
            if key in seen:
                continue
            dedup.append(pt)
            seen.add(key)
        return downsample_route_points(dedup, self.display_path_points)

    def export_pose_memory(self):
        return {rid: (rec.pose_xy, np.array(rec.pose_cov, copy=True), rec.last_update_time) for rid, rec in self.records.items()}

    def export_target_memory(self):
        return {rid: (rec.target_xy, rec.last_update_time) for rid, rec in self.records.items() if rec.target_xy is not None}

    def export_path_memory(self):
        return {rid: [pt.xy for pt in self._combined_path(rec)] for rid, rec in self.records.items()}

    def export_keypoint_memory(self):
        return {rid: [pt.xy for pt in rec.semantic_points] for rid, rec in self.records.items()}

    def export_stale_memory(self):
        return {
            rid: {
                'is_stale': rec.is_stale,
                'age_s': rec.stale_age_s,
                'knowledge_time': rec.last_update_time,
                'source_robot_id': rec.source_robot_id,
            }
            for rid, rec in self.records.items()
        }
