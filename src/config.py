from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    world_w: float = 30.0
    world_h: float = 30.0
    dt: float = 0.10
    # Rendering is the main bottleneck in the Matplotlib UI.
    # Slower frame interval + multiple sim steps per frame keeps the sim moving
    # while avoiding a full dashboard redraw every 0.1 simulated second.
    fps_ms: int = 90
    sim_steps_per_render: int = 3
    ui_blit: bool = True
    ui_max_frontier_points: int = 180
    ui_global_path_history_points: int = 120
    ui_local_path_history_points: int = 90
    ui_fused_map_period: int = 3
    ui_local_map_period: int = 3
    ui_frontier_period: int = 5
    ui_memory_period: int = 3
    ui_text_period: int = 2
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
    lidar_rays: int = 64
    lidar_step: float = 0.22

    grid_res: float = 0.25
    planner_inflation_margin: float = 0.16
    planner_clearance_weight: float = 2.4
    planner_clearance_floor_m: float = 0.70
    planner_narrow_penalty: float = 1.25
    planner_unknown_edge_penalty: float = 0.30
    planner_use_scipy_distance_transform: bool = True
    planner_max_expansions: int = 12000
    frontier_replan_period: float = 3.0
    target_hold_time: float = 7.0

    # Mission phase logic. Exploration is allowed to finish only after the
    # team map has enough known area or the remaining frontiers are essentially
    # exhausted. Then every robot switches into a forced go-home phase.
    mission_auto_return_when_explored: bool = True
    mission_coverage_goal_pct: float = 94.0
    mission_frontier_stop_cells: int = 18
    # Avoid the old behavior where a robot could decide the task was done
    # from a small early frontier set.  The team must explore for at least
    # this long, then satisfy either the coverage goal or a low-frontier +
    # high-coverage condition.
    mission_min_explore_s: float = 35.0
    mission_frontier_stop_requires_coverage_pct: float = 88.0
    mission_completion_confirm_s: float = 4.0
    mission_return_replan_period: float = 1.0
    mission_home_arrival_radius: float = 0.75
    mission_stop_when_all_home: bool = True


    frontier_region_claim_radius: float = 6.0
    frontier_region_claim_penalty: float = 95.0
    frontier_region_same_cycle_penalty: float = 130.0
    frontier_region_switch_penalty: float = 40.0
    frontier_region_stay_bonus: float = 28.0
    frontier_region_hold_time: float = 12.0
    # Do not re-auction while the robot is still travelling to a valid target.
    # The active-goal validator below is allowed to break this commitment when
    # new LiDAR observations make the old target/path invalid.
    target_commit_min_s: float = 16.0
    target_replan_near_target_radius: float = 1.8
    # When A* repairs a blocked frontier target, accept the repair only if the
    # endpoint stays close to the intended target/region.  Large snaps usually
    # mean the Voronoi/frontier assignment was inside an obstacle or behind a
    # newly discovered wall.
    goal_repair_max_dist_m: float = 1.10
    # Active route validation: after each LiDAR/map update, re-check the current
    # committed target and upcoming path against the newly known map.
    active_goal_revalidation_enabled: bool = True
    active_goal_min_unknown_cells: int = 4
    active_goal_check_radius_factor: float = 0.75
    active_goal_path_check_points: int = 8
    active_goal_replan_cooldown_s: float = 0.35

    decision_min_route_clearance: float = 0.24
    decision_max_predicted_cov_trace: float = 2.8
    decision_covariance_growth_per_m: float = 0.055
    decision_disconnect_explore_margin: float = 100.0
    decision_return_path_factor: float = 1.8
    decision_check_return_path: bool = False
    decision_max_frontier_candidates: int = 12
    decision_voronoi_bonus: float = 48.0
    decision_foreign_region_penalty: float = 120.0
    # Each robot first tries frontiers inside its weighted Voronoi cell;
    # foreign frontiers are only used as fallback.
    decision_strict_voronoi_assignment: bool = True
    decision_outward_weight: float = 10.0
    decision_sector_weight: float = 28.0
    decision_info_gain_saturation: float = 95.0


    # Team-aware exploration redesign.  These parameters make frontier utility
    # reason about what teammates probably already observed, not only what this
    # robot personally knows.  A teammate's current pose, recent trail, and
    # claimed target reserve a LiDAR-sized footprint so another robot does not
    # waste time remapping the same place.
    team_prediction_enabled: bool = True
    team_prediction_use_stale_memory: bool = True
    team_prediction_radius_factor: float = 0.92
    team_prediction_target_radius_factor: float = 1.05
    team_prediction_pose_gain: float = 1.15
    team_prediction_trail_gain: float = 0.75
    team_prediction_target_gain: float = 1.35
    team_prediction_path_gain: float = 0.58
    team_prediction_decay_s: float = 30.0
    team_prediction_gain_discount: float = 0.82
    team_prediction_overlap_weight: float = 34.0
    team_prediction_max_path_points: int = 36
    team_prediction_min_novelty_ratio: float = 0.18
    team_prediction_claim_target_gain: float = 1.15
    team_prediction_debug: bool = True

    comm_radius: float = 16.0
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

    # Per-step JSON/TXT snapshots cause heavy disk I/O and make the UI lag.
    # Keep this off for interactive runs; enable only when collecting debug logs.
    logs_enabled: bool = False
    logs_root: str = 'logs'
    log_snapshot_period_s: float = 2.0

    home_base_size: float = 4.4
    home_base_padding: float = 0.55
    start_spacing: float = 1.25

    obstacle_count: int = 7
    landmark_count: int = 7
    obstacle_size_min: float = 1.3
    obstacle_size_max: float = 4.2
    obstacle_gap_margin: float = 1.4
    world_margin: float = 2.0
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

    # Localization safety layer.  Exploration should become conservative when
    # pose uncertainty grows, especially after the robot has not seen landmarks
    # or reliable teammates for a while.  This prevents the robot from thinking
    # it is moving/home while the true body is blocked by an obstacle.
    localization_slowdown_cov_trace: float = 1.05
    localization_recovery_cov_trace: float = 1.85
    localization_critical_cov_trace: float = 3.10
    localization_no_correction_timeout_s: float = 11.0
    localization_absolute_recent_s: float = 3.5
    localization_min_speed_scale: float = 0.28
    localization_recovery_replan_period: float = 0.8
    localization_recovery_anchor_accept_radius: float = 1.1
    localization_recovery_standoff_m: float = 1.6
    localization_home_confirm_cov_trace: float = 1.35
    localization_home_recent_s: float = 4.5
    localization_use_actual_motion_feedback: bool = True
    localization_collision_predict_fraction: float = 0.10
    localization_stuck_cov_inflation: float = 0.065
    localization_stuck_replan_steps: int = 5

    # Stuck-route recovery.  These are temporary planning-only blocks, not
    # permanent map obstacles.  When a robot repeatedly collides/stalls, it
    # marks the failed approach for a short time so A* must try a different
    # corridor or a different target.
    stuck_route_replan_enabled: bool = True
    stuck_route_block_steps: int = 3
    stuck_route_block_radius: float = 0.78
    stuck_route_block_duration_s: float = 18.0
    stuck_route_max_blocks: int = 8
    stuck_route_min_separation_m: float = 0.55
    stuck_route_replan_cooldown_s: float = 0.65
    stuck_route_current_clear_radius: float = 0.72
    stuck_route_protect_home_radius: float = 1.15



    # Local self-recovery is intentionally kept as platform infrastructure.
    # Robots no longer request cooperative rescue; a failed local route is
    # handled by temporary planning blocks, target invalidation, and replanning.

    local_view_alpha: float = 0.88
    show_rays: bool = False

    # Team fused belief map.  This is visualization-first; planning still uses
    # each robot's own local map.
    fusion_require_home_connection: bool = True
    fusion_min_confidence: float = 0.03
    fusion_update_period_steps: int = 3
    fusion_stale_decay_s: float = 90.0
    fusion_pose_cov_gain: float = 0.85
    fusion_range_gain: float = 1.15
