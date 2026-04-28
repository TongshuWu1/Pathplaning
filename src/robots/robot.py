from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np

from ..config import SimConfig
from ..geometry import RectObstacle, circle_intersects_rect, line_of_sight, ray_rect_distance
from ..mapping import OccupancyGrid, ScanBeam, apply_scan
from ..world import Landmark
from .logger import RobotLogger
from .memory import TeammateMemoryStore
from .nav import ExecutedRouteMemory
from .packets import KnowledgeSnapshot, TeammatePacket


PoseMemory = Tuple[Tuple[float, float], np.ndarray, float]


def _wrap_rad(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


@dataclass
class Robot:
    robot_id: int
    name: str
    color: str
    x: float
    y: float
    heading: float
    cfg: SimConfig
    local_map: OccupancyGrid
    current_target: Optional[Tuple[float, float]] = None
    current_mode: str = 'idle'
    current_region_id: Optional[int] = None
    current_region_center_xy: Optional[Tuple[float, float]] = None
    region_hold_until: float = -1e9
    last_region_switch_time: float = -1e9
    current_path: List[Tuple[float, float]] = field(default_factory=list)
    path_history: List[Tuple[float, float]] = field(default_factory=list)
    est_path_history: List[Tuple[float, float]] = field(default_factory=list)
    last_plan_time: float = -1e9
    last_target_time: float = -1e9
    last_choice_debug: str = ''
    last_scan: List[Tuple[float, float, bool]] = field(default_factory=list)
    last_measurements: List[ScanBeam] = field(default_factory=list)
    received_packets: List[TeammatePacket] = field(default_factory=list)
    blocked_steps: int = 0
    request_replan: bool = False
    motion_state: str = 'idle'
    # Temporary route-level avoidance memory.  Each tuple is
    # (x, y, expire_time_s).  The simulator injects these as virtual blocked
    # disks in planning maps so a stuck robot tries another route instead of
    # repeating the same collision.
    route_block_zones: List[Tuple[float, float, float]] = field(default_factory=list)
    last_route_block_time: float = -1e9
    route_recovery_count: int = 0
    last_recovery_debug: str = ''
    direct_neighbors: List[int] = field(default_factory=list)
    reachable_peer_ids: List[int] = field(default_factory=list)
    home_connected: bool = False
    home_hops: Optional[int] = None
    direct_home_link: bool = False
    x_est: float = 0.0
    y_est: float = 0.0
    heading_est: float = 0.0
    P: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-4)
    rng_seed: int = 0
    last_landmark_updates: int = 0
    last_teammate_updates: int = 0
    last_loc_debug: str = ''
    localization_safety_state: str = 'nominal'
    safety_debug: str = ''
    last_landmark_seen_time: float = 0.0
    last_home_seen_time: float = 0.0
    last_absolute_update_time: float = 0.0
    landmark_ids_used: List[str] = field(default_factory=list)
    shared_pose_memory: dict[int, PoseMemory] = field(default_factory=dict)
    shared_target_memory: dict[int, Tuple[Tuple[float, float], float]] = field(default_factory=dict)
    shared_path_memory: dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    shared_keypoint_memory: dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    shared_memory_state: dict[int, dict] = field(default_factory=dict)
    ray_bearings: List[float] = field(default_factory=list, repr=False)
    odom_scale_bias: float = 1.0
    odom_turn_bias_per_m_rad: float = 0.0
    log_dir: Optional[str] = None
    landmark_memory: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x_est = self.x
        self.y_est = self.y
        self.heading_est = self.heading
        self.P = np.diag([0.10, 0.10, math.radians(6.0) ** 2])
        self.rng = np.random.default_rng(self.rng_seed)
        self.odom_scale_bias = 1.0 + self.rng.normal(0.0, 0.025)
        self.odom_turn_bias_per_m_rad = self.rng.normal(0.0, math.radians(1.2))
        self.est_path_history.append((self.x_est, self.y_est))
        self.path_history.append((self.x, self.y))
        self.ray_bearings = [2.0 * math.pi * i / self.cfg.lidar_rays for i in range(self.cfg.lidar_rays)]
        self.logger = RobotLogger(self.robot_id, self.log_dir, enabled=self.cfg.logs_enabled)
        self.memory_store = TeammateMemoryStore(
            self_robot_id=self.robot_id,
            max_route_points=self.cfg.teammate_memory_max_path_points,
            max_recent_points=self.cfg.teammate_memory_recent_points,
            display_path_points=self.cfg.teammate_memory_display_path_points,
            persist_s=self.cfg.teammate_memory_persist_s,
        )
        self.executed_route_memory = ExecutedRouteMemory(
            keep_dist=self.cfg.executed_route_keep_dist,
            recent_keep_dist=self.cfg.executed_route_recent_keep_dist,
            turn_keep_deg=self.cfg.executed_route_turn_keep_deg,
            semantic_merge_dist=self.cfg.semantic_nav_merge_dist,
            max_route_points=self.cfg.teammate_memory_max_path_points,
            max_recent_points=self.cfg.executed_route_recent_points,
        )
        self.executed_route_memory.initialize(self.x_est, self.y_est, 0.0)
        self.packet_seq = 0
        self.prev_home_connected = self.home_connected
        self.prev_landmark_visible = False
        self.last_landmark_seen_time = 0.0
        self.last_home_seen_time = 0.0
        self.last_absolute_update_time = 0.0
        self.localization_safety_state = 'nominal'
        self.safety_debug = ''
        self.logger.write_snapshot(self.knowledge_snapshot(0.0), now=0.0, force=True)

    @property
    def nav_keypoints_history(self) -> List[Tuple[float, float]]:
        return [pt.xy for pt in self.executed_route_memory.semantic_points]

    def pose_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def est_pose_xy(self) -> Tuple[float, float]:
        return (self.x_est, self.y_est)

    def covariance_trace(self) -> float:
        return float(np.trace(self.P[:2, :2]))

    def prune_route_blocks(self, now: Optional[float]) -> None:
        if now is None:
            return
        self.route_block_zones = [
            (float(x), float(y), float(expire))
            for x, y, expire in self.route_block_zones
            if float(expire) > float(now)
        ]

    def active_route_blocks(self, now: Optional[float]) -> List[Tuple[float, float, float]]:
        self.prune_route_blocks(now)
        return list(self.route_block_zones)

    def record_route_block(self, x: float, y: float, now: Optional[float]) -> bool:
        """Remember a failed local approach for temporary replanning.

        This is intentionally not written into the occupancy grid.  It is a
        short-lived planning memory that says: do not immediately try this same
        corridor again.
        """
        if not bool(getattr(self.cfg, 'stuck_route_replan_enabled', True)) or now is None:
            return False
        now_f = float(now)
        if now_f - float(self.last_route_block_time) < float(getattr(self.cfg, 'stuck_route_replan_cooldown_s', 0.0)):
            return False
        x = max(0.0, min(float(getattr(self.cfg, 'world_w', x)), float(x)))
        y = max(0.0, min(float(getattr(self.cfg, 'world_h', y)), float(y)))
        self.prune_route_blocks(now_f)
        min_sep = float(getattr(self.cfg, 'stuck_route_min_separation_m', 0.5))
        for bx, by, _expire in self.route_block_zones:
            if math.hypot(bx - x, by - y) < min_sep:
                self.last_route_block_time = now_f
                return False
        expire = now_f + float(getattr(self.cfg, 'stuck_route_block_duration_s', 18.0))
        self.route_block_zones.append((x, y, expire))
        max_blocks = max(1, int(getattr(self.cfg, 'stuck_route_max_blocks', 8)))
        if len(self.route_block_zones) > max_blocks:
            self.route_block_zones = self.route_block_zones[-max_blocks:]
        self.last_route_block_time = now_f
        self.route_recovery_count += 1
        self.last_recovery_debug = f'avoid ({x:.1f},{y:.1f}) for {expire - now_f:.0f}s'
        return True


    def absolute_update_age(self, now: Optional[float]) -> float:
        if now is None:
            return 0.0
        return max(0.0, float(now) - float(self.last_absolute_update_time))

    def recent_home_seen(self, now: Optional[float]) -> bool:
        if now is None:
            return False
        return (float(now) - float(self.last_home_seen_time)) <= float(self.cfg.localization_home_recent_s)

    def localization_state(self, now: Optional[float]) -> str:
        cov = self.covariance_trace()
        age = self.absolute_update_age(now)
        timeout = float(self.cfg.localization_no_correction_timeout_s)
        if (
            cov >= float(self.cfg.localization_critical_cov_trace)
            or age >= 2.0 * timeout
            or self.blocked_steps >= max(1, int(self.cfg.localization_stuck_replan_steps))
        ):
            return 'critical'
        if cov >= float(self.cfg.localization_recovery_cov_trace) or age >= timeout:
            return 'recover'
        if cov >= float(self.cfg.localization_slowdown_cov_trace) or age >= 0.65 * timeout:
            return 'caution'
        return 'nominal'

    def localization_speed_scale(self, now: Optional[float]) -> float:
        cov = self.covariance_trace()
        slow = float(self.cfg.localization_slowdown_cov_trace)
        critical = max(slow + 1e-6, float(self.cfg.localization_critical_cov_trace))
        min_scale = float(self.cfg.localization_min_speed_scale)
        if cov <= slow:
            cov_scale = 1.0
        else:
            alpha = max(0.0, min(1.0, (cov - slow) / (critical - slow)))
            cov_scale = 1.0 - (1.0 - min_scale) * alpha
        age = self.absolute_update_age(now)
        timeout = float(self.cfg.localization_no_correction_timeout_s)
        age_scale = 1.0 if age <= 0.65 * timeout else max(min_scale, 1.0 - 0.55 * min(1.0, (age - 0.65 * timeout) / max(0.35 * timeout, 1e-6)))
        mode_scale = 0.62 if self.current_mode == 'localize' else 1.0
        return max(min_scale, min(cov_scale, age_scale, mode_scale))

    def sense(self, obstacles: Sequence[RectObstacle]) -> List[ScanBeam]:
        beams: List[ScanBeam] = []
        rays_for_ui: List[Tuple[float, float, bool]] = []
        heading_true_rad = math.radians(self.heading)
        for bearing in self.ray_bearings:
            ray_ang = heading_true_rad + bearing
            dx = math.cos(ray_ang)
            dy = math.sin(ray_ang)
            best_d = self.cfg.lidar_range
            hit = False
            for obs in obstacles:
                d = ray_rect_distance(self.x, self.y, dx, dy, obs, self.cfg.lidar_range)
                if d is not None and d < best_d:
                    best_d = d
                    hit = True
            hx = self.x + dx * best_d
            hy = self.y + dy * best_d
            beams.append(ScanBeam(range_m=best_d, bearing_rad=bearing, hit=hit))
            rays_for_ui.append((hx, hy, hit))
        self.last_measurements = beams
        self.last_scan = rays_for_ui
        return beams

    def update_localization(self, landmarks: Sequence[Landmark], robots_by_id: dict[int, 'Robot'],
                            obstacles: Sequence[RectObstacle], now: float) -> None:
        self.last_landmark_updates = 0
        self.last_teammate_updates = 0
        self.landmark_ids_used = []
        for lm in landmarks:
            if not self._point_visible((lm.x, lm.y), self.cfg.landmark_obs_range, obstacles):
                continue
            z = self._measure_point((lm.x, lm.y), self.cfg.landmark_range_noise, self.cfg.landmark_bearing_noise_deg)
            R = np.diag([
                self.cfg.landmark_range_noise ** 2,
                math.radians(self.cfg.landmark_bearing_noise_deg) ** 2,
            ])
            if self._range_bearing_update((lm.x, lm.y), z, R, now):
                self.last_landmark_updates += 1
                self.last_landmark_seen_time = float(now)
                self.last_absolute_update_time = float(now)
                lm_name = 'Home' if lm.is_home else lm.name
                if lm.is_home:
                    self.last_home_seen_time = float(now)
                self.landmark_ids_used.append(lm_name)
                self._update_landmark_memory(lm_name, lm.is_home, z, now)

        for packet in self.received_packets:
            if packet.robot_id not in self.direct_neighbors:
                continue
            teammate = robots_by_id.get(packet.robot_id)
            if teammate is None:
                continue
            if not self._point_visible(teammate.pose_xy(), self.cfg.teammate_obs_range, obstacles):
                continue
            z = self._measure_point(teammate.pose_xy(), self.cfg.teammate_range_noise, self.cfg.teammate_bearing_noise_deg)
            age = max(0.0, now - packet.timestamp)
            teammate_pos_trace = float(packet.pose_cov[0, 0] + packet.pose_cov[1, 1])
            sig_r = self.cfg.teammate_range_noise + self.cfg.teammate_covariance_gain * math.sqrt(max(0.0, teammate_pos_trace)) + self.cfg.teammate_age_noise_gain * age
            sig_b = math.radians(
                self.cfg.teammate_bearing_noise_deg + 10.0 * self.cfg.teammate_covariance_gain * math.sqrt(max(0.0, teammate_pos_trace)) + 3.5 * self.cfg.teammate_age_noise_gain * age
            )
            R = np.diag([sig_r ** 2, sig_b ** 2])
            if self._range_bearing_update(packet.pose_xy, z, R, now):
                self.last_teammate_updates += 1
                self.last_absolute_update_time = float(now)

        landmark_visible = self.last_landmark_updates > 0
        if landmark_visible and not self.prev_landmark_visible:
            self.executed_route_memory.record_event(self.x_est, self.y_est, now, 'LANDMARK_GAIN', note='landmarks visible')
        if (not landmark_visible) and self.prev_landmark_visible:
            self.executed_route_memory.record_event(self.x_est, self.y_est, now, 'LANDMARK_LOSS', note='landmarks lost')
        self.prev_landmark_visible = landmark_visible

        age_abs = max(0.0, float(now) - float(self.last_absolute_update_time))
        self.localization_safety_state = self.localization_state(now)
        self.safety_debug = f'{self.localization_safety_state} abs_age={age_abs:3.1f}s'
        self.last_loc_debug = (
            f'loc tr(Pxy)={self.covariance_trace():4.2f}  lm={self.last_landmark_updates}  '
            f'team={self.last_teammate_updates}  safe={self.localization_safety_state}  abs={age_abs:3.1f}s'
        )

    def update_map(self, beams: Sequence[ScanBeam]) -> None:
        # Keep a small free bubble around the estimated body.  With obstacle
        # inflation enabled, a too-small bubble can isolate the robot inside
        # UNKNOWN/OCCUPIED cells after localization drift, making all frontiers
        # look unreachable and causing the robot to idle.
        body_clear_radius = max(0.85, self.cfg.robot_radius * 3.2)
        self._force_local_disk_free(self.x_est, self.y_est, radius=body_clear_radius)
        apply_scan(self.local_map, (self.x_est, self.y_est), self.heading_est, beams, step=self.cfg.lidar_step)
        self._force_local_disk_free(self.x_est, self.y_est, radius=body_clear_radius)

    def _force_local_disk_free(self, x: float, y: float, radius: float) -> None:
        gx, gy = self.local_map.world_to_grid(x, y)
        cells = int(math.ceil(radius / self.local_map.res))
        r2 = radius * radius
        for yy in range(max(0, gy - cells), min(self.local_map.ny, gy + cells + 1)):
            for xx in range(max(0, gx - cells), min(self.local_map.nx, gx + cells + 1)):
                wx, wy = self.local_map.grid_to_world(xx, yy)
                if (wx - x) ** 2 + (wy - y) ** 2 <= r2:
                    self.local_map.data[yy, xx] = 0


    def _copy_landmark_memory(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for key, info in sorted(self.landmark_memory.items()):
            if not bool(info.get('discovered', False)):
                continue
            out[str(key)] = {
                'xy': [float(info['xy'][0]), float(info['xy'][1])],
                'knowledge_time': float(info.get('knowledge_time', 0.0)),
                'source_robot_id': int(info.get('source_robot_id', self.robot_id)),
                'observed_count': int(info.get('observed_count', 1)),
                'is_home': bool(info.get('is_home', False)),
                'discovered': True,
            }
        return out

    def _update_landmark_memory(self, landmark_name: str, is_home: bool, z: Tuple[float, float], now: float) -> None:
        rng, bearing = float(z[0]), float(z[1])
        ang = math.radians(self.heading_est) + bearing
        lx = self.x_est + rng * math.cos(ang)
        ly = self.y_est + rng * math.sin(ang)
        prev = self.landmark_memory.get(landmark_name)
        count = 1 if prev is None else int(prev.get('observed_count', 1)) + 1
        if prev is None:
            xy = [lx, ly]
        else:
            w_old = max(1, count - 1)
            xy = [
                (prev['xy'][0] * w_old + lx) / count,
                (prev['xy'][1] * w_old + ly) / count,
            ]
        self.landmark_memory[landmark_name] = {
            'xy': xy,
            'knowledge_time': float(now),
            'source_robot_id': self.robot_id,
            'observed_count': count,
            'is_home': bool(is_home),
            'discovered': True,
        }

    def _self_knowledge_snapshot(self, now: float) -> KnowledgeSnapshot:
        route_points = self.executed_route_memory.snapshot_route_points(self.cfg.teammate_packet_path_points)
        recent_trail = self.executed_route_memory.snapshot_recent_trail(self.cfg.teammate_packet_recent_points)
        semantic_points = self.executed_route_memory.snapshot_semantic_points(self.cfg.teammate_packet_keypoints)
        return KnowledgeSnapshot(
            subject_robot_id=self.robot_id,
            source_robot_id=self.robot_id,
            knowledge_time=now,
            pose_xy=(self.x_est, self.y_est),
            pose_cov=self.P.copy(),
            target_xy=self.current_target,
            current_region_id=self.current_region_id,
            current_region_center_xy=self.current_region_center_xy,
            home_connected=bool(self.home_connected),
            home_hops=self.home_hops,
            direct_neighbors=list(self.direct_neighbors),
            reachable_peer_ids=list(self.reachable_peer_ids),
            landmark_beliefs=self._copy_landmark_memory(),
            route_points=route_points,
            recent_trail=recent_trail,
            semantic_points=semantic_points,
            is_stale=False,
            stale_age_s=0.0,
            is_self=True,
        )

    def make_packet(self, now: float) -> TeammatePacket:
        self.packet_seq += 1
        knowledge = [self._self_knowledge_snapshot(now), *self.memory_store.export_knowledge()]
        return TeammatePacket(
            robot_id=self.robot_id,
            timestamp=now,
            seq_num=self.packet_seq,
            knowledge=knowledge,
        )

    def ingest_shared_teammate_info(self, now: float) -> None:
        for packet in self.received_packets:
            self.memory_store.merge_packet(packet, now)
        self.memory_store.prune(now)
        self.shared_pose_memory = self.memory_store.export_pose_memory()
        self.shared_target_memory = self.memory_store.export_target_memory()
        self.shared_path_memory = self.memory_store.export_path_memory()
        self.shared_keypoint_memory = self.memory_store.export_keypoint_memory()
        self.shared_memory_state = self.memory_store.export_stale_memory()
        self.logger.write_snapshot(self.knowledge_snapshot(now), now=now, min_period_s=self.cfg.log_snapshot_period_s)

    def note_connectivity_state(self, now: float) -> None:
        if self.home_connected != self.prev_home_connected:
            event_type = 'LOS_HOME_GAIN' if self.home_connected else 'LOS_HOME_LOSS'
            self.executed_route_memory.record_event(self.x_est, self.y_est, now, event_type, note=f'hops={self.home_hops}')
            self.prev_home_connected = self.home_connected

    def knowledge_snapshot(self, now: float) -> dict:
        teammates = {}
        for rid, rec in sorted(self.memory_store.records.items(), key=lambda item: item[0]):
            teammates[rid] = {
                'pose_xy': [float(rec.pose_xy[0]), float(rec.pose_xy[1])],
                'pose_cov': np.array(rec.pose_cov, copy=True),
                'target_xy': None if rec.target_xy is None else [float(rec.target_xy[0]), float(rec.target_xy[1])],
                'current_region_id': rec.current_region_id,
                'current_region_center_xy': None if rec.current_region_center_xy is None else [float(rec.current_region_center_xy[0]), float(rec.current_region_center_xy[1])],
                'home_connected': bool(rec.home_connected),
                'home_hops': rec.home_hops,
                'direct_neighbors': list(rec.direct_neighbors),
                'reachable_peer_ids': list(rec.reachable_peer_ids),
                'knowledge_time': float(rec.last_update_time),
                'received_time': float(rec.last_received_time),
                'source_robot_id': rec.source_robot_id,
                'is_stale': bool(rec.is_stale),
                'age_s': float(rec.stale_age_s),
                'route_point_count': len(rec.route_points),
                'recent_point_count': len(rec.recent_trail),
                'semantic_point_count': len(rec.semantic_points),
                'landmarks': self.memory_store._copy_landmarks(rec.landmark_beliefs),
                'path_xy': [list(pt.xy) for pt in self.memory_store._combined_path(rec)],
                'semantic_points_xy': [list(pt.xy) for pt in rec.semantic_points],
            }
        return {
            'time_s': float(now),
            'robot_id': self.robot_id,
            'self': {
                'pose_est_xy': [float(self.x_est), float(self.y_est)],
                'heading_est_deg': float(self.heading_est),
                'pose_cov': self.P.copy(),
                'cov_trace': float(self.covariance_trace()),
                'target_xy': None if self.current_target is None else [float(self.current_target[0]), float(self.current_target[1])],
                'current_mode': self.current_mode,
                'current_region_id': self.current_region_id,
                'current_region_center_xy': None if self.current_region_center_xy is None else [float(self.current_region_center_xy[0]), float(self.current_region_center_xy[1])],
                'home_connected': bool(self.home_connected),
                'home_hops': self.home_hops,
                'direct_neighbors': list(self.direct_neighbors),
                'reachable_peer_ids': list(self.reachable_peer_ids),
                'route_point_count': len(self.executed_route_memory.route_points),
                'recent_point_count': len(self.executed_route_memory.recent_trail),
                'semantic_point_count': len(self.executed_route_memory.semantic_points),
                'landmarks': self._copy_landmark_memory(),
                'path_xy': [list(pt.xy) for pt in self.executed_route_memory.snapshot_route_points(self.cfg.teammate_memory_display_path_points)],
                'recent_trail_xy': [list(pt.xy) for pt in self.executed_route_memory.snapshot_recent_trail(self.cfg.teammate_packet_recent_points)],
                'semantic_points_xy': [list(pt.xy) for pt in self.executed_route_memory.snapshot_semantic_points(self.cfg.teammate_packet_keypoints)],
            },
            'teammates': teammates,
        }


    def known_landmark_beliefs(self) -> dict[str, dict]:
        combined = self._copy_landmark_memory()
        for rec in self.memory_store.records.values():
            for key, info in rec.landmark_beliefs.items():
                if not bool(info.get('discovered', False)):
                    continue
                prev = combined.get(str(key))
                if prev is None or float(info.get('knowledge_time', 0.0)) >= float(prev.get('knowledge_time', 0.0)):
                    combined[str(key)] = {
                        'xy': [float(info['xy'][0]), float(info['xy'][1])],
                        'knowledge_time': float(info.get('knowledge_time', 0.0)),
                        'source_robot_id': int(info.get('source_robot_id', rec.subject_robot_id if hasattr(rec, 'subject_robot_id') else rec.source_robot_id)),
                        'observed_count': int(info.get('observed_count', 1)),
                        'is_home': bool(info.get('is_home', False)),
                        'discovered': True,
                    }
        return combined

    def record_navigation_keypoint(self, xy: Tuple[float, float]) -> None:
        # Kept for compatibility; long-term memory now comes only from executed motion/events.
        return None

    def follow_path(self, dt: float, world_obstacles: Sequence[RectObstacle], now: Optional[float] = None) -> None:
        self.request_replan = False
        if not self.current_path:
            self.motion_state = 'idle'
            self.blocked_steps = 0
            return

        nav_x, nav_y = self.x_est, self.y_est
        while self.current_path:
            goal = self.current_path[0]
            if math.hypot(goal[0] - nav_x, goal[1] - nav_y) >= self.cfg.waypoint_reach_tol:
                break
            self.current_path.pop(0)
        if not self.current_path:
            self.motion_state = 'arrived'
            self.blocked_steps = 0
            return

        goal = self.current_path[0]
        dx = goal[0] - nav_x
        dy = goal[1] - nav_y
        d = math.hypot(dx, dy)
        if d < 1e-9:
            self.motion_state = 'arrived'
            self.blocked_steps = 0
            return

        old_heading_est_rad = math.radians(self.heading_est)
        desired = math.atan2(dy, dx)
        cmd_turn = _wrap_rad(desired - old_heading_est_rad)
        max_turn = self.cfg.robot_turn_gain * dt
        cmd_turn = max(-max_turn, min(max_turn, cmd_turn))
        speed_scale = max(0.15, 1.0 - min(1.0, abs(cmd_turn) / 1.2))
        safety_scale = self.localization_speed_scale(now)
        cmd_step = min(self.cfg.robot_max_speed * speed_scale * safety_scale * dt, d)

        old_heading_true_rad = math.radians(self.heading)
        new_heading_true_rad = old_heading_true_rad + cmd_turn
        self.heading = (math.degrees(new_heading_true_rad) + 360.0) % 360.0

        prev_x, prev_y = self.x, self.y
        nx = self.x + math.cos(new_heading_true_rad) * cmd_step
        ny = self.y + math.sin(new_heading_true_rad) * cmd_step
        collided = self.collides(nx, ny, world_obstacles)
        if not collided:
            self.x = nx
            self.y = ny

        actual_moved_dist = math.hypot(self.x - prev_x, self.y - prev_y)
        if collided and bool(self.cfg.localization_use_actual_motion_feedback):
            # A collision means the commanded translation did not happen.  The
            # old code still propagated the EKF with cmd_step, so the robot could
            # believe it was moving through an obstacle.  Keep only a tiny slip
            # fraction and inflate uncertainty instead.
            belief_step = min(cmd_step, max(0.0, actual_moved_dist + float(self.cfg.localization_collision_predict_fraction) * cmd_step))
        else:
            belief_step = cmd_step
        self._predict_belief(belief_step, cmd_turn, now)
        if collided:
            inflate = float(self.cfg.localization_stuck_cov_inflation)
            self.P[0, 0] += inflate
            self.P[1, 1] += inflate
            self.P[2, 2] += math.radians(2.5) ** 2
            self.P = 0.5 * (self.P + self.P.T)

        moved = actual_moved_dist >= self.cfg.progress_epsilon
        if moved:
            self.blocked_steps = 0
            self.motion_state = 'move'
            if now is not None:
                self.executed_route_memory.record_motion(self.x_est, self.y_est, self.heading_est, now)
        else:
            self.blocked_steps += 1
            self.motion_state = 'blocked' if collided else 'stalled'
            if self.blocked_steps >= self.cfg.blocked_waypoint_skip_steps and len(self.current_path) > 1:
                self.current_path.pop(0)
                self.motion_state = 'skip-waypoint'
            route_block_added = False
            if self.blocked_steps >= int(getattr(self.cfg, 'stuck_route_block_steps', 3)):
                # Collision point is the best hint for the failed approach.  If
                # this was a stall without a geometry collision, block the next
                # waypoint/heading direction instead so the new plan avoids
                # reusing the same local corridor.
                if collided:
                    block_x, block_y = nx, ny
                elif self.current_path:
                    block_x, block_y = self.current_path[0]
                else:
                    block_x = self.x + math.cos(new_heading_true_rad) * max(self.cfg.robot_radius * 2.0, cmd_step)
                    block_y = self.y + math.sin(new_heading_true_rad) * max(self.cfg.robot_radius * 2.0, cmd_step)
                route_block_added = self.record_route_block(block_x, block_y, now)

            if self.blocked_steps >= max(int(self.cfg.localization_stuck_replan_steps), int(self.cfg.blocked_waypoint_skip_steps)):
                self.request_replan = True
                self.localization_safety_state = 'critical'
                recovery = f'; {self.last_recovery_debug}' if route_block_added or self.last_recovery_debug else ''
                self.safety_debug = f'blocked recovery requested ({self.blocked_steps} blocked steps){recovery}'
            if self.blocked_steps >= self.cfg.blocked_replan_steps:
                self.current_path = []
                self.current_target = None
                self.blocked_steps = 0
                self.request_replan = True
                self.motion_state = 'route-replan'

        self.path_history.append((self.x, self.y))
        if len(self.path_history) > self.cfg.path_history_max_points:
            self.path_history = self.path_history[-self.cfg.path_history_max_points:]

    def _predict_belief(self, moved_dist: float, dtheta: float, now: Optional[float] = None) -> None:
        ds = moved_dist * self.odom_scale_bias + self.rng.normal(0.0, self.cfg.odom_trans_noise)
        dth = dtheta + self.odom_turn_bias_per_m_rad * abs(ds) + self.rng.normal(0.0, math.radians(self.cfg.odom_rot_noise_deg))
        th = math.radians(self.heading_est)
        self.x_est += ds * math.cos(th + 0.5 * dth)
        self.y_est += ds * math.sin(th + 0.5 * dth)
        self.heading_est = (math.degrees(_wrap_rad(th + dth)) + 360.0) % 360.0

        F = np.eye(3)
        F[0, 2] = -ds * math.sin(th + 0.5 * dth)
        F[1, 2] = ds * math.cos(th + 0.5 * dth)
        q_xy = self.cfg.process_xy_noise + 0.2 * abs(ds)
        q_th = math.radians(self.cfg.process_theta_noise_deg) + 0.2 * abs(dth)
        Q = np.diag([q_xy ** 2, q_xy ** 2, q_th ** 2])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.est_path_history.append((self.x_est, self.y_est))
        if len(self.est_path_history) > self.cfg.path_history_max_points:
            self.est_path_history = self.est_path_history[-self.cfg.path_history_max_points:]

    def _measure_point(self, point_xy: Tuple[float, float], sigma_r: float, sigma_b_deg: float) -> np.ndarray:
        dx = point_xy[0] - self.x
        dy = point_xy[1] - self.y
        r = math.hypot(dx, dy) + self.rng.normal(0.0, sigma_r)
        bearing = _wrap_rad(math.atan2(dy, dx) - math.radians(self.heading) + self.rng.normal(0.0, math.radians(sigma_b_deg)))
        return np.array([r, bearing], dtype=float)

    def _range_bearing_update(self, ref_world_xy: Tuple[float, float], z: np.ndarray, R: np.ndarray, now: Optional[float] = None) -> bool:
        mx, my = ref_world_xy
        x, y, th = self.x_est, self.y_est, math.radians(self.heading_est)
        dx = mx - x
        dy = my - y
        q = dx * dx + dy * dy
        if q < 1e-8:
            return False
        r = math.sqrt(q)
        zhat = np.array([r, _wrap_rad(math.atan2(dy, dx) - th)], dtype=float)
        H = np.array([
            [-dx / r, -dy / r, 0.0],
            [dy / q, -dx / q, -1.0],
        ], dtype=float)
        yk = z - zhat
        yk[1] = _wrap_rad(yk[1])
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        state = np.array([x, y, th], dtype=float) + K @ yk
        self.x_est = float(state[0])
        self.y_est = float(state[1])
        self.heading_est = (math.degrees(_wrap_rad(float(state[2]))) + 360.0) % 360.0
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.est_path_history.append((self.x_est, self.y_est))
        if len(self.est_path_history) > self.cfg.path_history_max_points:
            self.est_path_history = self.est_path_history[-self.cfg.path_history_max_points:]
        self.P = np.clip(self.P, -1e6, 1e6)
        return True

    def _point_visible(self, point_xy: Tuple[float, float], max_range: float, obstacles: Sequence[RectObstacle]) -> bool:
        return math.hypot(point_xy[0] - self.x, point_xy[1] - self.y) <= max_range and line_of_sight((self.x, self.y), point_xy, obstacles)

    def collides(self, x: float, y: float, world_obstacles: Sequence[RectObstacle]) -> bool:
        if x - self.cfg.robot_radius < 0 or y - self.cfg.robot_radius < 0:
            return True
        if x + self.cfg.robot_radius > self.cfg.world_w or y + self.cfg.robot_radius > self.cfg.world_h:
            return True
        return any(circle_intersects_rect(x, y, self.cfg.robot_radius, obs) for obs in world_obstacles)
