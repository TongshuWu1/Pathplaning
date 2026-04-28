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
    count: int = 4; radius: float = 0.18; max_speed: float = 0.62; turn_gain: float = 2.4
    waypoint_tolerance: float = 0.30; goal_tolerance: float = 0.50; spawn_spacing: float = 0.55
    path_replan_period_s: float = 1.15; keypoint_spacing: float = 1.25
    visit_history_spacing_m: float = 0.75; max_visit_history: int = 220
    path_digest_spacing_m: float = 1.15; max_path_digest_points: int = 12

@dataclass(frozen=True)
class MotionNoiseConfig:
    xy_std_per_m: float = 0.035; theta_std_per_rad: float = 0.025
    process_xy: float = 0.018; process_theta: float = 0.014
    landmark_xy_gain: float = 0.55; landmark_cov_shrink: float = 0.55

@dataclass(frozen=True)
class LidarConfig:
    range: float = 5.2; rays: int = 56; noise_std: float = 0.025; hit_threshold: float = 0.18
    front_angle_deg: float = 35.0; side_angle_deg: float = 55.0
    blocked_forward_distance: float = 0.62; open_sector_min_width_deg: float = 18.0

@dataclass(frozen=True)
class MappingConfig:
    resolution: float = 0.24
    logodds_free: float = -0.42; logodds_occ: float = 0.85
    logodds_min: float = -4.0; logodds_max: float = 4.0
    prob_free_threshold: float = 0.39; prob_occ_threshold: float = 0.66
    quality_overwrite_margin: float = 0.04; low_quality_update_scale: float = 0.35

@dataclass(frozen=True)
class AssessmentConfig:
    scan_consistency_tolerance_m: float = 0.55; low_consistency: float = 0.38
    caution_consistency: float = 0.58; consistency_smoothing: float = 0.45

@dataclass(frozen=True)
class PlanningConfig:
    inflation_radius_m: float = 0.58; critical_clearance_m: float = 0.46; desired_clearance_m: float = 0.95
    clearance_cost_weight: float = 4.5; unknown_penalty: float = 2.0; max_a_star_expansions: int = 8500
    frontier_min_cluster_size: int = 4; frontier_sample_count: int = 28; frontier_info_radius_m: float = 1.45
    safe_approach_search_radius_m: float = 2.2; safe_approach_min_clearance_m: float = 0.60
    target_detection_bonus: float = 10.0; distance_weight: float = 0.42
    information_weight: float = 1.35; clearance_weight: float = 1.15; centerline_weight: float = 1.30
    novelty_weight: float = 0.85; duplicate_penalty_weight: float = 4.0
    teammate_path_penalty_weight: float = 6.5; recent_visit_penalty_weight: float = 2.2
    goal_progress_weight: float = 1.25; route_alternate_weight: float = 0.65

@dataclass(frozen=True)
class CommunicationConfig:
    radius: float = 6.0; packet_period_s: float = 0.5; teammate_intent_timeout_s: float = 8.0

@dataclass(frozen=True)
class CageConfig:
    route_cert_threshold: float = 0.62; desired_route_count: int = 2; edge_min_length: float = 0.5
    edge_merge_distance: float = 0.55; edge_confidence_decay: float = 0.002
    unknown_target_search_bias: float = 1.0; report_route_bonus: float = 5.0
    reanchor_consistency_threshold: float = 0.35; reanchor_cov_trace_threshold: float = 1.8

@dataclass(frozen=True)
class UIConfig:
    interval_ms: int = 100; sim_steps_per_render: int = 4; selected_robot: int = 0
    show_lidar_rays: bool = False; show_truth_target: bool = True; max_status_routes: int = 3
    figure_width: float = 16.5; figure_height: float = 10.2
    draw_lidar_stride: int = 3; max_draw_path_points: int = 450; max_draw_graph_edges: int = 260

@dataclass(frozen=True)
class AppConfig:
    dt: float = 0.12; max_time_s: float = 900.0
    world: WorldConfig = field(default_factory=WorldConfig); robot: RobotConfig = field(default_factory=RobotConfig)
    motion: MotionNoiseConfig = field(default_factory=MotionNoiseConfig); lidar: LidarConfig = field(default_factory=LidarConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig); assessment: AssessmentConfig = field(default_factory=AssessmentConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig); communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    cage: CageConfig = field(default_factory=CageConfig); ui: UIConfig = field(default_factory=UIConfig)
    def validate(self) -> None:
        if self.robot.count < 1: raise ValueError('robot.count must be >= 1')
        if self.world.width <= 2 or self.world.height <= 2: raise ValueError('world dimensions are too small')
        if self.lidar.rays < 12: raise ValueError('lidar.rays must be at least 12')
        if self.mapping.resolution <= 0: raise ValueError('mapping.resolution must be positive')
        if self.communication.radius <= 0: raise ValueError('communication.radius must be positive')
        if self.planning.critical_clearance_m > self.planning.desired_clearance_m: raise ValueError('critical clearance should not exceed desired clearance')
