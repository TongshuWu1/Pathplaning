"""Grouped configuration for the Search-CAGE baseline.

Tuned for efficient long runs: coarser LiDAR maps, clearance-aware planning,
reward-based exploration, and packet-only teammate intent.
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class WorldConfig:
    width: float = 30.0; height: float = 30.0; seed: int = 2; world_margin: float = 2.0
    home_base_size: float = 4.4; home_base_padding: float = 0.55
    obstacle_count: int = 7; obstacle_min_size: float = 1.3; obstacle_max_size: float = 4.2
    obstacle_gap_margin: float = 1.4; spawn_clear_radius: float = 4.8
    target_x: float | None = None; target_y: float | None = None; target_radius: float = 0.85; target_clear_radius: float = 1.35
    landmark_count: int = 7; landmark_detection_range: float = 8.5

@dataclass(frozen=True)
class RobotConfig:
    count: int = 4; radius: float = 0.36; body_length: float = 0.95; body_width: float = 0.58
    collision_buffer_m: float = 0.12; collision_avoidance_horizon_m: float = 1.25
    teammate_avoidance_turn_gain: float = 1.25; max_speed: float = 0.62; turn_gain: float = 2.4
    waypoint_tolerance: float = 0.30; goal_tolerance: float = 0.50; spawn_spacing: float = 0.95
    path_replan_period_s: float = 2.2; keypoint_spacing: float = 1.25
    goal_commit_time_s: float = 14.0; goal_switch_score_margin: float = 5.0
    goal_switch_same_goal_radius_m: float = 1.20; goal_finish_commit_radius_m: float = 2.40
    goal_progress_switch_margin: float = 3.0; goal_finish_switch_margin: float = 4.0
    stuck_progress_timeout_s: float = 9.0; failed_goal_memory_size: int = 16
    visit_history_spacing_m: float = 0.75; max_visit_history: int = 220
    true_path_spacing_m: float = 0.30; max_true_path_points: int = 450
    path_digest_spacing_m: float = 1.15; max_path_digest_points: int = 12
    visit_digest_spacing_m: float = 1.8; max_visit_digest_points: int = 14
    # Full estimated trajectory history from HOME, kept downsampled for communication/UI.
    trajectory_history_spacing_m: float = 0.70; max_trajectory_history_points: int = 520
    trajectory_digest_spacing_m: float = 1.00; max_trajectory_digest_points: int = 90

@dataclass(frozen=True)
class MotionNoiseConfig:
    xy_std_per_m: float = 0.035; theta_std_per_rad: float = 0.025
    process_xy: float = 0.018; process_theta: float = 0.014
    # Real EKF landmark update noise/limits. These are noisy sensor
    # measurements, not perfect ground-truth corrections.
    landmark_range_std_m: float = 0.12; landmark_bearing_std_deg: float = 3.0
    landmark_max_xy_correction_m: float = 0.35; landmark_max_theta_correction_deg: float = 8.0
    # Keep covariance from becoming falsely tiny while allowing convergence.
    covariance_floor_xy: float = 0.006; covariance_floor_theta: float = 0.003
    covariance_max_xy: float = 2.5; covariance_max_theta: float = 1.2
    # Wider, stronger but still bounded correlative LiDAR scan matching.
    lidar_match_period_s: float = 1.20; lidar_xy_gain: float = 0.55; lidar_theta_gain: float = 0.45
    lidar_match_max_xy_m: float = 0.25; lidar_match_max_theta_deg: float = 6.0
    # Teammate localization uses a noisy relative observation plus the
    # teammate's reported covariance; it should help, not teleport.
    teammate_localization_range_m: float = 6.5; teammate_range_std_m: float = 0.16
    teammate_bearing_std_deg: float = 5.0; teammate_covariance_scale: float = 1.0
    teammate_max_xy_correction_m: float = 0.22

@dataclass(frozen=True)
class LidarConfig:
    range: float = 5.2; rays: int = 48; noise_std: float = 0.005; hit_threshold: float = 0.1
    front_angle_deg: float = 35.0; side_angle_deg: float = 55.0
    blocked_forward_distance: float = 0.80; open_sector_min_width_deg: float = 18.0
    raycast_step_m: float = 0.08; range_noise_std_per_m: float = 0.003
    max_range_noise_std: float = 0.0; dropout_probability: float = 0.0

@dataclass(frozen=True)
class MappingConfig:
    resolution: float = 0.30
    logodds_free: float = -0.42; logodds_occ: float = 0.85
    logodds_min: float = -4.0; logodds_max: float = 4.0
    prob_free_threshold: float = 0.39; prob_occ_threshold: float = 0.66
    quality_overwrite_margin: float = 0.04; low_quality_update_scale: float = 0.35
    lidar_free_kernel_radius_m: float = 0.36; lidar_hit_kernel_radius_m: float = 0.30
    lidar_kernel_min_weight: float = 0.18
    # Map-to-map fusion keeps the highest-confidence cell, not the newest cell.
    merge_quality_margin: float = 0.03

@dataclass(frozen=True)
class PassageQualityConfig:
    """Configurable cell-wise execution/traversal score for HOME passage planning.

    Meaning: if a later execution robot must plan from HOME to target, how good
    is this cell for traversal? The score is intentionally safety-first:

        passage = occupancy_safety^free_weight
                * clearance_score^clearance_weight
                * reliability_discount^map_confidence_weight

    Clearance and obstacle risk dominate. Mapping confidence only discounts the
    score enough to prefer better-supported corridors when safety is similar.
    """
    # Overlay display.
    show_by_default: bool = True
    overlay_alpha: float = 0.42

    # Labeled factor weights in final passage score.
    free_weight: float = 1.35
    map_confidence_weight: float = 0.30
    clearance_weight: float = 1.85

    # Free-space score behavior. Unknown/occupied cells should not become green.
    unknown_score: float = 0.00
    occupied_score: float = 0.00
    free_score_power: float = 1.00

    # Map-confidence behavior. This is a soft reliability discount, not the
    # definition of passage quality.
    min_map_confidence: float = 0.02
    map_confidence_floor: float = 0.72
    map_confidence_power: float = 1.00

    # Clearance score behavior. Center of corridor/open space should be greener.
    min_clearance_m: float = 0.32
    good_clearance_m: float = 2.40
    clearance_reference_percentile: float = 9.0
    clearance_power: float = 1.15

@dataclass(frozen=True)
class TargetReportingConfig:
    """Rules for when HOME is allowed to believe a target report.

    Robots may still share target position with each other so all robots can go
    toward the target. HOME acceptance is stricter by default: HOME accepts a
    target report only from the robot that originally observed it, unless
    relayed target reporting is explicitly enabled.
    """
    allow_robot_to_robot_target_share: bool = True
    allow_relayed_target_to_home: bool = False
    require_home_connection_for_target_report: bool = True

@dataclass(frozen=True)
class AssessmentConfig:
    scan_consistency_tolerance_m: float = 0.55; low_consistency: float = 0.38
    caution_consistency: float = 0.58; consistency_smoothing: float = 0.45
    sector_clearance_percentile: float = 18.0
    open_sector_range_fraction: float = 0.58; open_sector_depth_percentile: float = 65.0
    open_sector_width_weight: float = 1.0; open_sector_depth_weight: float = 0.8; open_sector_forward_weight: float = 0.35

@dataclass(frozen=True)
class PlanningConfig:
    """Small planning config.

    Keep only the knobs that are useful to tune often. The old exploration
    reward-soup parameters were removed; normal exploration now uses a simple
    next-best-view scan-pose selector with a few LiDAR-scaled radii.
    """
    # A* path planning and safety.
    inflation_radius_m: float = 0.88
    critical_clearance_m: float = 0.70
    desired_clearance_m: float = 1.35
    clearance_cost_weight: float = 6.0
    unknown_penalty: float = 2.0
    max_a_star_expansions: int = 6500

    # Fallback/target-guided frontier tools. Normal exploration does not pick
    # frontier cells directly anymore; it picks scan poses by expected LiDAR gain.
    frontier_min_cluster_size: int = 4
    frontier_info_radius_m: float = 1.45
    frontier_sample_count: int = 28
    safe_approach_search_radius_m: float = 1.8
    safe_approach_min_clearance_m: float = 0.78
    frontier_visibility_rays: int = 32
    frontier_plan_eval_count: int = 10
    frontier_path_clearance_weight: float = 1.1
    frontier_path_unknown_penalty_weight: float = 1.6
    distance_weight: float = 0.42

    # Startup deployment: spread robots before normal exploration begins.
    startup_deployment_enabled: bool = True
    startup_deployment_lidar_fraction: float = 1.00
    startup_deployment_angle_spread_deg: float = 210.0

    # Next-best-view exploration.
    nbv_sample_stride_cells: int = 3
    nbv_max_candidates: int = 140
    nbv_plan_eval_count: int = 14
    nbv_local_unknown_radius_lidar_fraction: float = 0.55
    nbv_teammate_hard_avoid_lidar_fraction: float = 0.50
    nbv_teammate_soft_avoid_lidar_fraction: float = 1.00
    nbv_own_path_avoid_lidar_fraction: float = 0.35
    nbv_reservation_lidar_fraction: float = 0.65

    # Hierarchical coarse-to-fine exploration.  Region size is LiDAR-scaled,
    # so this adds stability without many tuning knobs.
    hierarchical_exploration_enabled: bool = True
    region_size_lidar_fraction: float = 0.50
    region_commit_time_s: float = 18.0
    region_switch_score_ratio: float = 1.35

    # Dynamic obstacles and passage evaluation.
    passage_safety_cost_weight: float = 4.0
    dynamic_obstacle_soft_margin_m: float = 0.95
    dynamic_obstacle_cost_weight: float = 7.0
    dynamic_obstacle_max_cov_extra_m: float = 0.65

@dataclass(frozen=True)
class CommunicationConfig:
    radius: float = 14.0; packet_period_s: float = 0.8; teammate_intent_timeout_s: float = 8.0

@dataclass(frozen=True)
class CageConfig:
    route_cert_threshold: float = 0.62; desired_route_count: int = 2; edge_min_length: float = 0.5
    edge_merge_distance: float = 0.55; edge_confidence_decay: float = 0.002
    unknown_target_search_bias: float = 1.0; report_route_bonus: float = 5.0
    reanchor_consistency_threshold: float = 0.35; reanchor_cov_trace_threshold: float = 1.8
    exploration_complete_min_known_ratio: float = 0.58; exploration_complete_max_frontiers_per_robot: int = 3
    exploration_complete_stable_steps: int = 35
    # After target discovery, keep exploring to build reliable HOME-to-target passage knowledge.
    target_corridor_width_m: float = 3.2
    target_corridor_bonus_weight: float = 5.0
    target_corridor_low_quality_weight: float = 2.2
    # Target-roundtrip mission: once any robot finds the target, every robot
    # tries to reach it from its own position, then returns HOME to upload route evidence.
    target_arrival_radius_m: float = 0.85
    target_known_path_max_unknown_fraction: float = 0.28
    require_all_robots_target_roundtrip: bool = True
    min_robots_completed_roundtrip: int = 1
    safe_passage_score_threshold: float = 0.46
    safe_passage_min_clearance_m: float = 0.52
    safe_passage_max_unknown_fraction: float = 0.22

@dataclass(frozen=True)
class UIConfig:
    interval_ms: int = 180; sim_steps_per_render: int = 4; selected_robot: int = 0
    show_lidar_rays: bool = False; show_route_graph: bool = False; show_truth_target: bool = True; max_status_routes: int = 3
    figure_width: float = 16.5; figure_height: float = 10.2
    draw_lidar_stride: int = 3; max_draw_path_points: int = 450; max_draw_graph_edges: int = 120; max_draw_graph_nodes: int = 180
    max_draw_frontiers: int = 22; max_draw_teammate_visit_points: int = 16; max_draw_teammate_trajectory_points: int = 90
    render_truth_every: int = 3; render_team_every: int = 3; render_local_every: int = 6; render_frontier_every: int = 12

@dataclass(frozen=True)
class AppConfig:
    dt: float = 0.12; max_time_s: float = 900.0
    world: WorldConfig = field(default_factory=WorldConfig); robot: RobotConfig = field(default_factory=RobotConfig)
    motion: MotionNoiseConfig = field(default_factory=MotionNoiseConfig); lidar: LidarConfig = field(default_factory=LidarConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig); passage_quality: PassageQualityConfig = field(default_factory=PassageQualityConfig)
    assessment: AssessmentConfig = field(default_factory=AssessmentConfig); target_reporting: TargetReportingConfig = field(default_factory=TargetReportingConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig); communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    cage: CageConfig = field(default_factory=CageConfig); ui: UIConfig = field(default_factory=UIConfig)
    def validate(self) -> None:
        if self.robot.count < 1: raise ValueError('robot.count must be >= 1')
        if self.robot.radius <= 0: raise ValueError('robot.radius must be positive')
        if self.robot.body_length < self.robot.body_width or self.robot.body_width <= 0: raise ValueError('robot body dimensions must be positive and length >= width')
        if self.robot.collision_buffer_m < 0: raise ValueError('robot.collision_buffer_m must be non-negative')
        if self.robot.goal_commit_time_s < 0: raise ValueError('robot.goal_commit_time_s must be non-negative')
        if self.robot.goal_switch_same_goal_radius_m <= 0: raise ValueError('robot.goal_switch_same_goal_radius_m must be positive')
        if self.robot.goal_finish_commit_radius_m < self.robot.goal_tolerance: raise ValueError('robot.goal_finish_commit_radius_m should be >= goal_tolerance')
        if self.world.width <= 2 or self.world.height <= 2: raise ValueError('world dimensions are too small')
        if self.lidar.rays < 12: raise ValueError('lidar.rays must be at least 12')
        if self.lidar.raycast_step_m <= 0: raise ValueError('lidar.raycast_step_m must be positive')
        if self.lidar.dropout_probability < 0 or self.lidar.dropout_probability > 1: raise ValueError('lidar.dropout_probability must be in [0, 1]')
        if self.motion.teammate_localization_range_m <= 0: raise ValueError('motion.teammate_localization_range_m must be positive')
        if self.motion.teammate_range_std_m <= 0: raise ValueError('motion.teammate_range_std_m must be positive')
        if self.motion.teammate_bearing_std_deg <= 0: raise ValueError('motion.teammate_bearing_std_deg must be positive')
        if self.mapping.resolution <= 0: raise ValueError('mapping.resolution must be positive')
        if self.mapping.lidar_free_kernel_radius_m < 0 or self.mapping.lidar_hit_kernel_radius_m < 0: raise ValueError('LiDAR kernel radii must be non-negative')
        if self.mapping.lidar_kernel_min_weight < 0 or self.mapping.lidar_kernel_min_weight > 1: raise ValueError('mapping.lidar_kernel_min_weight must be in [0, 1]')
        if self.assessment.sector_clearance_percentile < 0 or self.assessment.sector_clearance_percentile > 100: raise ValueError('assessment.sector_clearance_percentile must be in [0, 100]')
        if self.assessment.open_sector_depth_percentile < 0 or self.assessment.open_sector_depth_percentile > 100: raise ValueError('assessment.open_sector_depth_percentile must be in [0, 100]')
        if self.planning.frontier_visibility_rays < 8: raise ValueError('planning.frontier_visibility_rays must be >= 8')
        if self.planning.startup_deployment_lidar_fraction <= 0: raise ValueError('planning.startup_deployment_lidar_fraction must be positive')
        if self.planning.nbv_sample_stride_cells < 1: raise ValueError('planning.nbv_sample_stride_cells must be >= 1')
        if self.planning.nbv_max_candidates < 1: raise ValueError('planning.nbv_max_candidates must be >= 1')
        if self.planning.nbv_plan_eval_count < 1: raise ValueError('planning.nbv_plan_eval_count must be >= 1')
        if self.planning.nbv_teammate_hard_avoid_lidar_fraction < 0: raise ValueError('planning.nbv_teammate_hard_avoid_lidar_fraction must be non-negative')
        if self.planning.nbv_teammate_soft_avoid_lidar_fraction < self.planning.nbv_teammate_hard_avoid_lidar_fraction: raise ValueError('soft teammate avoid fraction should be >= hard avoid fraction')
        if self.planning.nbv_reservation_lidar_fraction < 0: raise ValueError('planning.nbv_reservation_lidar_fraction must be non-negative')
        if self.planning.region_size_lidar_fraction <= 0: raise ValueError('planning.region_size_lidar_fraction must be positive')
        if self.planning.region_commit_time_s < 0: raise ValueError('planning.region_commit_time_s must be non-negative')
        if self.planning.region_switch_score_ratio < 1.0: raise ValueError('planning.region_switch_score_ratio should be >= 1.0')
        if self.planning.dynamic_obstacle_soft_margin_m < 0: raise ValueError('planning.dynamic_obstacle_soft_margin_m must be non-negative')
        if self.communication.radius <= 0: raise ValueError('communication.radius must be positive')
        if self.planning.critical_clearance_m > self.planning.desired_clearance_m: raise ValueError('critical clearance should not exceed desired clearance')
        if self.passage_quality.good_clearance_m <= self.passage_quality.min_clearance_m: raise ValueError('passage_quality.good_clearance_m must exceed min_clearance_m')
        if self.passage_quality.clearance_reference_percentile < 0 or self.passage_quality.clearance_reference_percentile > 100: raise ValueError('passage_quality.clearance_reference_percentile must be in [0, 100]')
        if self.passage_quality.overlay_alpha < 0 or self.passage_quality.overlay_alpha > 1: raise ValueError('passage_quality.overlay_alpha must be in [0, 1]')
        if self.passage_quality.map_confidence_floor < 0 or self.passage_quality.map_confidence_floor > 1: raise ValueError('passage_quality.map_confidence_floor must be in [0, 1]')
