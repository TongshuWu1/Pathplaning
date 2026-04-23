from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    world_w: float = 40.0
    world_h: float = 28.0
    dt: float = 0.10
    fps_ms: int = 100
    seed: int = 24

    robot_count: int = 4
    robot_radius: float = 0.28
    robot_max_speed: float = 1.15
    robot_turn_gain: float = 3.5
    waypoint_reach_tol: float = 0.22
    progress_epsilon: float = 0.015
    blocked_waypoint_skip_steps: int = 4
    blocked_replan_steps: int = 10

    lidar_range: float = 7.0
    lidar_rays: int = 72
    lidar_step: float = 0.18

    grid_res: float = 0.25
    planner_inflation_margin: float = 0.16
    planner_clearance_weight: float = 2.4
    planner_clearance_floor_m: float = 0.70
    planner_narrow_penalty: float = 1.25
    planner_unknown_edge_penalty: float = 0.30
    frontier_replan_period: float = 1.2
    target_hold_time: float = 2.0

    frontier_region_claim_radius: float = 4.5
    frontier_region_claim_penalty: float = 55.0
    frontier_region_same_cycle_penalty: float = 75.0
    frontier_region_switch_penalty: float = 18.0
    frontier_region_stay_bonus: float = 10.0
    frontier_region_hold_time: float = 4.0

    comm_radius: float = 9.0
    teammate_packet_path_points: int = 96
    teammate_packet_recent_points: int = 20
    teammate_packet_keypoints: int = 48
    teammate_trace_radius: float = 3.0
    teammate_trace_gain: float = 2.1
    teammate_current_pos_gain: float = 3.3
    teammate_target_gain: float = 1.5
    teammate_trace_decay_s: float = 8.0
    teammate_memory_persist_s: float = 10.0
    teammate_memory_max_path_points: int = 320
    teammate_memory_recent_points: int = 48
    teammate_memory_display_path_points: int = 220
    path_history_max_points: int = 1200
    nav_keypoint_merge_dist: float = 0.50
    executed_route_keep_dist: float = 0.55
    executed_route_recent_keep_dist: float = 0.20
    executed_route_recent_points: int = 40
    executed_route_turn_keep_deg: float = 26.0
    semantic_nav_merge_dist: float = 0.70

    logs_enabled: bool = True
    logs_root: str = 'logs'

    home_base_size: float = 4.4
    home_base_padding: float = 0.55
    start_spacing: float = 1.25

    obstacle_count: int = 15
    landmark_count: int = 18
    obstacle_size_min: float = 1.3
    obstacle_size_max: float = 4.2
    obstacle_gap_margin: float = 0.45
    world_margin: float = 1.0
    spawn_clear_radius: float = 4.8

    landmark_obs_range: float = 8.5
    landmark_range_noise: float = 0.10
    landmark_bearing_noise_deg: float = 3.0
    teammate_obs_range: float = 9.0
    teammate_range_noise: float = 0.18
    teammate_bearing_noise_deg: float = 5.0
    teammate_covariance_gain: float = 0.55
    teammate_age_noise_gain: float = 0.10

    odom_trans_noise: float = 0.03
    odom_rot_noise_deg: float = 2.0
    process_xy_noise: float = 0.02
    process_theta_noise_deg: float = 1.4

    local_view_alpha: float = 0.88
    show_rays: bool = False
