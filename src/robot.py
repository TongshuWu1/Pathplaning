from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math

import numpy as np

from .config import SimConfig
from .geometry import RectObstacle, circle_intersects_rect, line_of_sight, ray_rect_distance
from .mapping import OccupancyGrid, apply_scan
from .policy import TeammatePacket
from .world import Landmark


def _wrap_rad(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def _wrap_deg(angle: float) -> float:
    return (math.degrees(_wrap_rad(math.radians(angle))) + 360.0) % 360.0


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
    current_path: List[Tuple[float, float]] = field(default_factory=list)
    path_history: List[Tuple[float, float]] = field(default_factory=list)
    est_path_history: List[Tuple[float, float]] = field(default_factory=list)
    last_plan_time: float = -1e9
    last_target_time: float = -1e9
    last_choice_debug: str = ''
    last_scan: List[Tuple[float, float, bool]] = field(default_factory=list)
    received_packets: List[TeammatePacket] = field(default_factory=list)
    blocked_steps: int = 0
    request_replan: bool = False
    motion_state: str = 'idle'
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
    landmark_ids_used: List[str] = field(default_factory=list)
    shared_pose_memory: dict[int, Tuple[Tuple[float, float], float]] = field(default_factory=dict)
    shared_target_memory: dict[int, Tuple[Tuple[float, float], float]] = field(default_factory=dict)
    shared_path_memory: dict[int, List[Tuple[float, float, float]]] = field(default_factory=dict)
    ray_dirs: List[Tuple[float, float]] = field(default_factory=list, repr=False)
    odom_scale_bias: float = 1.0
    odom_turn_bias_per_m_rad: float = 0.0

    def __post_init__(self) -> None:
        self.x_est = self.x
        self.y_est = self.y
        self.heading_est = self.heading
        self.P = np.diag([0.10, 0.10, math.radians(6.0) ** 2])
        self.rng = np.random.default_rng(self.rng_seed)
        self.odom_scale_bias = 1.0 + self.rng.normal(0.0, 0.025)
        self.odom_turn_bias_per_m_rad = self.rng.normal(0.0, math.radians(1.2))
        self.est_path_history.append((self.x_est, self.y_est))
        self.ray_dirs = [
            (math.cos(2.0 * math.pi * i / self.cfg.lidar_rays), math.sin(2.0 * math.pi * i / self.cfg.lidar_rays))
            for i in range(self.cfg.lidar_rays)
        ]

    def pose_xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def est_pose_xy(self) -> Tuple[float, float]:
        return (self.x_est, self.y_est)

    def covariance_trace(self) -> float:
        return float(np.trace(self.P[:2, :2]))

    def sense(self, obstacles: Sequence[RectObstacle]) -> List[Tuple[float, float, bool]]:
        rays = []
        for dx, dy in self.ray_dirs:
            best_d = self.cfg.lidar_range
            hit = False
            for obs in obstacles:
                d = ray_rect_distance(self.x, self.y, dx, dy, obs, self.cfg.lidar_range)
                if d is not None and d < best_d:
                    best_d = d
                    hit = True
            hx = self.x + dx * best_d
            hy = self.y + dy * best_d
            rays.append((hx, hy, hit))
        self.last_scan = rays
        return rays

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
            if self._range_bearing_update((lm.x, lm.y), z, R):
                self.last_landmark_updates += 1
                self.landmark_ids_used.append('Home' if lm.is_home else lm.name)

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
            if self._range_bearing_update(packet.pose_xy, z, R):
                self.last_teammate_updates += 1

        self.last_loc_debug = (
            f'loc tr(Pxy)={self.covariance_trace():4.2f}  lm={self.last_landmark_updates}  team={self.last_teammate_updates}'
        )

    def update_map(self, rays: Sequence[Tuple[float, float, bool]]) -> None:
        self.local_map.reveal_disk(self.x_est, self.y_est, radius=self.cfg.robot_radius * 1.5)
        apply_scan(self.local_map, (self.x_est, self.y_est), rays, step=self.cfg.lidar_step)

    def make_packet(self, now: float) -> TeammatePacket:
        return TeammatePacket(
            robot_id=self.robot_id,
            pose_xy=(self.x_est, self.y_est),
            pose_cov=self.P.copy(),
            target_xy=self.current_target,
            path_xy=self.path_history[-self.cfg.teammate_packet_path_points:],
            timestamp=now,
        )

    def ingest_shared_teammate_info(self, now: float) -> None:
        for packet in self.received_packets:
            self.shared_pose_memory[packet.robot_id] = (packet.pose_xy, now)
            if packet.target_xy is not None:
                self.shared_target_memory[packet.robot_id] = (packet.target_xy, now)
            trimmed = packet.path_xy[-self.cfg.teammate_memory_max_path_points:]
            if trimmed:
                self.shared_path_memory[packet.robot_id] = [(pt[0], pt[1], now) for pt in trimmed]
        cutoff = now - self.cfg.teammate_memory_persist_s
        self.shared_pose_memory = {rid: item for rid, item in self.shared_pose_memory.items() if item[1] >= cutoff}
        self.shared_target_memory = {rid: item for rid, item in self.shared_target_memory.items() if item[1] >= cutoff}
        kept_paths: dict[int, List[Tuple[float, float, float]]] = {}
        for rid, pts in self.shared_path_memory.items():
            recent = [pt for pt in pts if pt[2] >= cutoff]
            if recent:
                kept_paths[rid] = recent[-self.cfg.teammate_memory_max_path_points:]
        self.shared_path_memory = kept_paths

    def follow_path(self, dt: float, world_obstacles: Sequence[RectObstacle]) -> None:
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
        cmd_step = min(self.cfg.robot_max_speed * speed_scale * dt, d)

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
        self._predict_belief(cmd_step, cmd_turn)

        moved = actual_moved_dist >= self.cfg.progress_epsilon
        if moved:
            self.blocked_steps = 0
            self.motion_state = 'move'
        else:
            self.blocked_steps += 1
            self.motion_state = 'blocked' if collided else 'stalled'
            if self.blocked_steps >= self.cfg.blocked_waypoint_skip_steps and len(self.current_path) > 1:
                self.current_path.pop(0)
                self.motion_state = 'skip-waypoint'
            if self.blocked_steps >= self.cfg.blocked_replan_steps:
                self.current_path = []
                self.current_target = None
                self.blocked_steps = 0
                self.request_replan = True
                self.motion_state = 'replan'

        self.path_history.append((self.x, self.y))
        if len(self.path_history) > 320:
            self.path_history = self.path_history[-320:]

    def _predict_belief(self, moved_dist: float, dtheta: float) -> None:
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
        if len(self.est_path_history) > 320:
            self.est_path_history = self.est_path_history[-320:]

    def _measure_point(self, point_xy: Tuple[float, float], sigma_r: float, sigma_b_deg: float) -> np.ndarray:
        dx = point_xy[0] - self.x
        dy = point_xy[1] - self.y
        r = math.hypot(dx, dy) + self.rng.normal(0.0, sigma_r)
        bearing = _wrap_rad(math.atan2(dy, dx) - math.radians(self.heading) + self.rng.normal(0.0, math.radians(sigma_b_deg)))
        return np.array([r, bearing], dtype=float)

    def _range_bearing_update(self, ref_world_xy: Tuple[float, float], z: np.ndarray, R: np.ndarray) -> bool:
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
        if len(self.est_path_history) > 320:
            self.est_path_history = self.est_path_history[-320:]
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
