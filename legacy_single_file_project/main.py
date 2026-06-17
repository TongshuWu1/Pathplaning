from __future__ import annotations

# Single-file legacy Search-CAGE simulator.
# Generated from archives/legacy_baseline_source_2026-05-05 so the old code can
# be opened as a separate project and run without package/import path setup.

import argparse
import heapq
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/pathplaning-matplotlib")))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np

try:
    from numba import njit
except Exception:  # optional acceleration dependency
    njit = None



# ============================================================================
# src / config.py
# ============================================================================

"""Grouped configuration for the Search-CAGE baseline.

Tuned for efficient long runs: coarser LiDAR maps, clearance-aware planning,
reward-based exploration, and packet-only teammate intent.
"""
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
    teammate_avoidance_turn_gain: float = 1.30; max_speed: float = 0.54; turn_gain: float = 2.35
    obstacle_avoidance_turn_gain: float = 0.82; obstacle_slowdown_start_m: float = 1.75
    obstacle_min_speed_scale: float = 0.10; lidar_safety_time_horizon_s: float = 1.25
    lidar_safety_stop_margin_m: float = 0.24; lidar_safety_slow_margin_m: float = 1.15
    lidar_reverse_speed: float = 0.16; local_planner_speed_samples: int = 3
    local_planner_omega_samples: int = 7
    waypoint_tolerance: float = 0.30; goal_tolerance: float = 0.50; spawn_spacing: float = 0.95
    path_replan_period_s: float = 2.2; target_path_replan_period_s: float = 5.0; keypoint_spacing: float = 1.25
    goal_commit_time_s: float = 6.0; goal_switch_score_margin: float = 1.15
    goal_switch_same_goal_radius_m: float = 1.20; goal_finish_commit_radius_m: float = 2.40
    goal_progress_switch_margin: float = 0.85; goal_finish_switch_margin: float = 1.40
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
    range: float = 5.2; rays: int = 72; noise_std: float = 0.004; hit_threshold: float = 0.1
    front_angle_deg: float = 35.0; side_angle_deg: float = 55.0
    blocked_forward_distance: float = 0.92; open_sector_min_width_deg: float = 18.0
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
    clearance_smoothing: float = 0.35; open_angle_smoothing: float = 0.30
    blocked_hysteresis_m: float = 0.12
    sector_clearance_percentile: float = 24.0
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
    inflation_radius_m: float = 1.00
    critical_clearance_m: float = 0.74
    desired_clearance_m: float = 1.45
    clearance_cost_weight: float = 7.4
    unknown_penalty: float = 2.0
    max_a_star_expansions: int = 6500

    # Fallback/target-guided frontier tools. Normal exploration does not pick
    # frontier cells directly anymore; it picks scan poses by expected LiDAR gain.
    frontier_min_cluster_size: int = 4
    frontier_info_radius_m: float = 1.45
    frontier_sample_count: int = 28
    safe_approach_search_radius_m: float = 1.8
    safe_approach_min_clearance_m: float = 0.92
    frontier_visibility_rays: int = 32
    frontier_plan_eval_count: int = 10
    frontier_path_clearance_weight: float = 1.35
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
class BeliefMdpConfig:
    """Compact belief-state macro-action planner.

    This is intentionally not a raw grid-world MDP.  Low-level motion still uses
    A*, while this layer scores high-level actions from the robot's belief state.
    """
    enabled: bool = True
    discount: float = 0.88
    heuristic_score_weight: float = 0.34
    target_discovery_weight: float = 18.0
    target_goal_weight: float = 22.0
    certificate_weight: float = 13.0
    communication_weight: float = 10.0
    information_weight: float = 1.15
    travel_cost_weight: float = 0.34
    risk_weight: float = 4.5
    low_clearance_risk_weight: float = 3.0
    unknown_path_risk_weight: float = 2.0
    target_belief_miss_likelihood: float = 0.10
    target_belief_uniform_mix: float = 0.002
    target_belief_detection_sigma_m: float = 0.85
    target_belief_sensor_fraction: float = 0.95
    target_belief_candidate_stride_cells: int = 4
    target_belief_plan_eval_count: int = 8
    weak_edge_certificate_threshold: float = 0.74
    evidence_relay_min_value: float = 0.55

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
    passage_eval_period_s: float = 6.00

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
    mdp: BeliefMdpConfig = field(default_factory=BeliefMdpConfig)
    cage: CageConfig = field(default_factory=CageConfig); ui: UIConfig = field(default_factory=UIConfig)
    def validate(self) -> None:
        if self.robot.count < 1: raise ValueError('robot.count must be >= 1')
        if self.robot.radius <= 0: raise ValueError('robot.radius must be positive')
        if self.robot.max_speed <= 0: raise ValueError('robot.max_speed must be positive')
        if self.robot.obstacle_avoidance_turn_gain < 0: raise ValueError('robot.obstacle_avoidance_turn_gain must be non-negative')
        if self.robot.obstacle_slowdown_start_m <= 0: raise ValueError('robot.obstacle_slowdown_start_m must be positive')
        if self.robot.obstacle_min_speed_scale < 0 or self.robot.obstacle_min_speed_scale > 1: raise ValueError('robot.obstacle_min_speed_scale must be in [0, 1]')
        if self.robot.lidar_safety_time_horizon_s <= 0: raise ValueError('robot.lidar_safety_time_horizon_s must be positive')
        if self.robot.lidar_safety_stop_margin_m < 0 or self.robot.lidar_safety_slow_margin_m <= self.robot.lidar_safety_stop_margin_m: raise ValueError('lidar safety margins must be ordered and non-negative')
        if self.robot.lidar_reverse_speed < 0: raise ValueError('robot.lidar_reverse_speed must be non-negative')
        if self.robot.local_planner_speed_samples < 3: raise ValueError('robot.local_planner_speed_samples must be >= 3')
        if self.robot.local_planner_omega_samples < 5: raise ValueError('robot.local_planner_omega_samples must be >= 5')
        if self.robot.body_length < self.robot.body_width or self.robot.body_width <= 0: raise ValueError('robot body dimensions must be positive and length >= width')
        if self.robot.collision_buffer_m < 0: raise ValueError('robot.collision_buffer_m must be non-negative')
        if self.robot.goal_commit_time_s < 0: raise ValueError('robot.goal_commit_time_s must be non-negative')
        if self.robot.path_replan_period_s <= 0 or self.robot.target_path_replan_period_s <= 0: raise ValueError('robot replan periods must be positive')
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
        if self.assessment.clearance_smoothing < 0 or self.assessment.clearance_smoothing > 1: raise ValueError('assessment.clearance_smoothing must be in [0, 1]')
        if self.assessment.open_angle_smoothing < 0 or self.assessment.open_angle_smoothing > 1: raise ValueError('assessment.open_angle_smoothing must be in [0, 1]')
        if self.assessment.blocked_hysteresis_m < 0: raise ValueError('assessment.blocked_hysteresis_m must be non-negative')
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
        if self.mdp.discount < 0 or self.mdp.discount > 1: raise ValueError('mdp.discount must be in [0, 1]')
        if self.mdp.target_belief_miss_likelihood <= 0 or self.mdp.target_belief_miss_likelihood > 1: raise ValueError('mdp.target_belief_miss_likelihood must be in (0, 1]')
        if self.mdp.target_belief_uniform_mix < 0 or self.mdp.target_belief_uniform_mix > 0.2: raise ValueError('mdp.target_belief_uniform_mix must be in [0, 0.2]')
        if self.mdp.target_belief_detection_sigma_m <= 0: raise ValueError('mdp.target_belief_detection_sigma_m must be positive')
        if self.mdp.target_belief_sensor_fraction <= 0: raise ValueError('mdp.target_belief_sensor_fraction must be positive')
        if self.mdp.target_belief_candidate_stride_cells < 1: raise ValueError('mdp.target_belief_candidate_stride_cells must be >= 1')
        if self.mdp.target_belief_plan_eval_count < 1: raise ValueError('mdp.target_belief_plan_eval_count must be >= 1')
        if self.mdp.weak_edge_certificate_threshold < 0 or self.mdp.weak_edge_certificate_threshold > 1: raise ValueError('mdp.weak_edge_certificate_threshold must be in [0, 1]')
        if self.cage.passage_eval_period_s <= 0: raise ValueError('cage.passage_eval_period_s must be positive')
        if self.planning.critical_clearance_m > self.planning.desired_clearance_m: raise ValueError('critical clearance should not exceed desired clearance')
        if self.passage_quality.good_clearance_m <= self.passage_quality.min_clearance_m: raise ValueError('passage_quality.good_clearance_m must exceed min_clearance_m')
        if self.passage_quality.clearance_reference_percentile < 0 or self.passage_quality.clearance_reference_percentile > 100: raise ValueError('passage_quality.clearance_reference_percentile must be in [0, 100]')
        if self.passage_quality.overlay_alpha < 0 or self.passage_quality.overlay_alpha > 1: raise ValueError('passage_quality.overlay_alpha must be in [0, 1]')
        if self.passage_quality.map_confidence_floor < 0 or self.passage_quality.map_confidence_floor > 1: raise ValueError('passage_quality.map_confidence_floor must be in [0, 1]')


# ============================================================================
# src / geometry.py
# ============================================================================

"""Small geometry helpers used by the baseline simulator."""
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


Pose = tuple[float, float, float]
Point = tuple[float, float]


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def distance(a: Point, b: Point) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def unit_from_angle(theta: float) -> tuple[float, float]:
    return math.cos(theta), math.sin(theta)


def angle_to(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def segment_length(points: Iterable[Point]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0
    return sum(distance(a, b) for a, b in zip(pts[:-1], pts[1:]))


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "Rect":
        return Rect(min(self.x0, self.x1), min(self.y0, self.y1), max(self.x0, self.x1), max(self.y0, self.y1))

    @property
    def center(self) -> Point:
        r = self.normalized()
        return ((r.x0 + r.x1) * 0.5, (r.y0 + r.y1) * 0.5)

    def contains(self, p: Point, margin: float = 0.0) -> bool:
        # Rectangles are normalized at construction in the world generator.
        return (self.x0 - margin <= p[0] <= self.x1 + margin) and (self.y0 - margin <= p[1] <= self.y1 + margin)

    def corners(self) -> list[Point]:
        r = self.normalized()
        return [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]


def ccw(a: Point, b: Point, c: Point) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def segment_intersects_rect(a: Point, b: Point, rect: Rect, margin: float = 0.0) -> bool:
    r = Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin).normalized()
    if r.contains(a) or r.contains(b):
        return True
    corners = r.corners()
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(segments_intersect(a, b, c, d) for c, d in edges)


def covariance_ellipse(cov_xy: np.ndarray, scale: float = 2.0, samples: int = 40) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov_xy, dtype=float)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return np.array([]), np.array([])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    theta = np.linspace(0.0, 2.0 * math.pi, samples)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = vecs @ np.diag(np.sqrt(vals) * scale) @ circle
    return ellipse[0], ellipse[1]


# ============================================================================
# src / cage_graph.py
# ============================================================================

"""Minimal CAGE route graph and route certificates."""
import heapq
import math
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: int
    xy: Point
    kind: str
    confidence: float = 1.0


@dataclass
class EdgeCertificate:
    confidence: float
    length: float
    min_clearance: float
    mean_consistency: float
    pose_quality: float
    traversal_success: int = 0
    failed_traversal: int = 0
    source_robots: set[int] = field(default_factory=set)
    last_updated: float = 0.0
    reported_home: bool = False

    def update(self, clearance: float, consistency: float, pose_quality: float, robot_id: int, time_s: float, success: bool = True) -> None:
        self.min_clearance = min(self.min_clearance, clearance) if self.min_clearance > 0 else clearance
        self.mean_consistency = 0.7 * self.mean_consistency + 0.3 * consistency
        self.pose_quality = 0.7 * self.pose_quality + 0.3 * pose_quality
        if success:
            self.traversal_success += 1
        else:
            self.failed_traversal += 1
        self.source_robots.add(robot_id)
        self.last_updated = time_s
        self.confidence = compute_edge_confidence(
            min_clearance=self.min_clearance,
            consistency=self.mean_consistency,
            pose_quality=self.pose_quality,
            traversal_success=self.traversal_success,
            failed_traversal=self.failed_traversal,
        )


@dataclass
class GraphEdge:
    id: int
    a: int
    b: int
    cert: EdgeCertificate


@dataclass
class RouteCandidate:
    node_ids: list[int]
    edge_ids: list[int]
    length: float
    min_clearance: float
    certificate: float
    reported_home: bool
    status: str


def compute_edge_confidence(
    min_clearance: float,
    consistency: float,
    pose_quality: float,
    traversal_success: int,
    failed_traversal: int,
) -> float:
    clear_score = max(0.05, min(1.0, (min_clearance - 0.15) / 0.85))
    trav_score = min(1.0, 0.35 + 0.18 * max(0, traversal_success))
    fail_penalty = 0.22 * failed_traversal
    raw = 0.34 * clear_score + 0.25 * consistency + 0.18 * pose_quality + 0.23 * trav_score - fail_penalty
    return float(max(0.0, min(1.0, raw)))


class RouteGraph:
    def __init__(self, merge_distance: float = 0.55):
        self.merge_distance = merge_distance
        self.nodes: dict[int, GraphNode] = {}
        self.edges: dict[int, GraphEdge] = {}
        self._adj: dict[int, dict[int, int]] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self.home_id: int | None = None
        self.target_id: int | None = None
        self._version = 0
        self._route_cache_key: tuple[int, int, bool, int | None, int | None] | None = None
        self._route_cache: list[RouteCandidate] = []

    def _touch(self) -> None:
        self._version += 1
        self._route_cache_key = None
        self._route_cache = []

    def copy(self) -> "RouteGraph":
        other = RouteGraph(self.merge_distance)
        other.nodes = {i: GraphNode(n.id, n.xy, n.kind, n.confidence) for i, n in self.nodes.items()}
        other.edges = {
            i: GraphEdge(e.id, e.a, e.b, EdgeCertificate(
                confidence=e.cert.confidence,
                length=e.cert.length,
                min_clearance=e.cert.min_clearance,
                mean_consistency=e.cert.mean_consistency,
                pose_quality=e.cert.pose_quality,
                traversal_success=e.cert.traversal_success,
                failed_traversal=e.cert.failed_traversal,
                source_robots=set(e.cert.source_robots),
                last_updated=e.cert.last_updated,
                reported_home=e.cert.reported_home,
            )) for i, e in self.edges.items()
        }
        other._adj = {a: dict(bs) for a, bs in self._adj.items()}
        other._next_node_id = self._next_node_id
        other._next_edge_id = self._next_edge_id
        other.home_id = self.home_id
        other.target_id = self.target_id
        other._version = self._version
        return other

    def add_node(self, xy: Point, kind: str = "keypoint", confidence: float = 1.0, allow_merge: bool = True) -> int:
        if allow_merge:
            merge_distance = max(self.merge_distance, 1.6) if kind == "target" else self.merge_distance
            for nid, node in self.nodes.items():
                if node.kind == kind and distance(node.xy, xy) <= merge_distance:
                    if confidence > node.confidence:
                        if kind == "target":
                            node.xy = (float(xy[0]), float(xy[1]))
                            self._refresh_edge_lengths(nid)
                        node.confidence = confidence
                        self._touch()
                    return nid
            # Keypoints can merge into anchors/home/target if very close.
            for nid, node in self.nodes.items():
                if distance(node.xy, xy) <= self.merge_distance * 0.55:
                    if confidence > node.confidence:
                        node.confidence = confidence
                        self._touch()
                    return nid
        nid = self._next_node_id
        self._next_node_id += 1
        self.nodes[nid] = GraphNode(nid, xy, kind, confidence)
        self._adj.setdefault(nid, {})
        if kind == "home":
            self.home_id = nid
        elif kind == "target":
            self.target_id = nid
        self._touch()
        return nid

    def _refresh_edge_lengths(self, node_id: int) -> None:
        for eid in self._adj.get(node_id, {}).values():
            edge = self.edges.get(eid)
            if edge is None:
                continue
            edge.cert.length = distance(self.nodes[edge.a].xy, self.nodes[edge.b].xy)

    def add_or_update_edge(
        self,
        a: int,
        b: int,
        clearance: float,
        consistency: float,
        pose_quality: float,
        robot_id: int,
        time_s: float,
        success: bool = True,
    ) -> int | None:
        if a == b or a not in self.nodes or b not in self.nodes:
            return None
        if b in self._adj.get(a, {}):
            eid = self._adj[a][b]
            self.edges[eid].cert.update(clearance, consistency, pose_quality, robot_id, time_s, success)
            self._touch()
            return eid
        length = distance(self.nodes[a].xy, self.nodes[b].xy)
        conf = compute_edge_confidence(clearance, consistency, pose_quality, int(success), int(not success))
        cert = EdgeCertificate(
            confidence=conf,
            length=length,
            min_clearance=clearance,
            mean_consistency=consistency,
            pose_quality=pose_quality,
            traversal_success=int(success),
            failed_traversal=int(not success),
            source_robots={robot_id},
            last_updated=time_s,
        )
        eid = self._next_edge_id
        self._next_edge_id += 1
        self.edges[eid] = GraphEdge(eid, a, b, cert)
        self._adj.setdefault(a, {})[b] = eid
        self._adj.setdefault(b, {})[a] = eid
        self._touch()
        return eid

    def merge_from_digest(self, digest: dict) -> None:
        id_map: dict[int, int] = {}
        for nd in digest.get("nodes", []):
            nid_old = int(nd["id"])
            nid_new = self.add_node(tuple(nd["xy"]), str(nd.get("kind", "keypoint")), float(nd.get("confidence", 1.0)))
            id_map[nid_old] = nid_new
        for ed in digest.get("edges", []):
            a = id_map.get(int(ed["a"]))
            b = id_map.get(int(ed["b"]))
            if a is None or b is None:
                continue
            cert_in = ed.get("cert", {})
            eid = self.add_or_update_edge(
                a, b,
                clearance=float(cert_in.get("min_clearance", 0.4)),
                consistency=float(cert_in.get("mean_consistency", 0.5)),
                pose_quality=float(cert_in.get("pose_quality", 0.5)),
                robot_id=int(digest.get("source_robot", -1)),
                time_s=float(digest.get("time_s", 0.0)),
                success=cert_in.get("failed_traversal", 0) == 0,
            )
            if eid is not None:
                edge = self.edges[eid]
                prev_conf = edge.cert.confidence
                prev_reported = edge.cert.reported_home
                edge.cert.confidence = max(edge.cert.confidence, float(cert_in.get("confidence", 0.0)))
                edge.cert.reported_home = edge.cert.reported_home or bool(cert_in.get("reported_home", False))
                if edge.cert.confidence != prev_conf or edge.cert.reported_home != prev_reported:
                    self._touch()

    def make_digest(self, robot_id: int, time_s: float, max_edges: int = 80) -> dict:
        # Send recent/high-confidence route evidence, always preserving route-critical edges.
        required: dict[int, GraphEdge] = {}
        for node_id in (self.home_id, self.target_id):
            if node_id is None:
                continue
            for eid in self._adj.get(node_id, {}).values():
                required[eid] = self.edges[eid]
        for route in self.top_routes(k=1, require_target=False):
            for eid in route.edge_ids:
                if eid in self.edges:
                    required[eid] = self.edges[eid]
        ranked = sorted(self.edges.values(), key=lambda e: (e.cert.confidence, e.cert.last_updated), reverse=True)
        edge_map = dict(required)
        for edge in ranked:
            if len(edge_map) >= max_edges:
                break
            edge_map.setdefault(edge.id, edge)
        edges = list(edge_map.values())
        node_ids = sorted({n for e in edges for n in (e.a, e.b)})
        return {
            "source_robot": int(robot_id),
            "time_s": float(time_s),
            "nodes": [
                {"id": int(nid), "xy": [float(self.nodes[nid].xy[0]), float(self.nodes[nid].xy[1])], "kind": self.nodes[nid].kind, "confidence": float(self.nodes[nid].confidence)}
                for nid in node_ids
            ],
            "edges": [
                {
                    "id": int(e.id),
                    "a": int(e.a),
                    "b": int(e.b),
                    "cert": {
                        "confidence": float(e.cert.confidence),
                        "length": float(e.cert.length),
                        "min_clearance": float(e.cert.min_clearance),
                        "mean_consistency": float(e.cert.mean_consistency),
                        "pose_quality": float(e.cert.pose_quality),
                        "traversal_success": int(e.cert.traversal_success),
                        "failed_traversal": int(e.cert.failed_traversal),
                        "reported_home": bool(e.cert.reported_home),
                    },
                }
                for e in edges
            ],
        }

    def mark_all_reported_home(self) -> None:
        changed = False
        for edge in self.edges.values():
            if not edge.cert.reported_home:
                edge.cert.reported_home = True
                changed = True
        if changed:
            self._touch()

    def top_routes(self, k: int = 3, require_target: bool = True) -> list[RouteCandidate]:
        cache_key = (self._version, k, require_target, self.home_id, self.target_id)
        if self._route_cache_key == cache_key:
            return list(self._route_cache)
        if self.home_id is None or self.home_id not in self.nodes:
            return []
        if require_target and (self.target_id is None or self.target_id not in self.nodes):
            return []

        # Multi-route best-first search.  Unlike the old version, this keeps
        # searching after the first HOME->TARGET path and returns up to k
        # distinct route candidates.  For graph digests before a target exists,
        # require_target=False returns high-confidence exploratory paths from
        # HOME so useful graph evidence is still shared.
        target_id = self.target_id if self.target_id in self.nodes else None
        pq: list[tuple[float, int, int, list[int], list[int], float, float, float, bool]] = []
        counter = 0
        heapq.heappush(pq, (0.0, counter, self.home_id, [self.home_id], [], 0.0, math.inf, 1.0, True))
        routes: list[RouteCandidate] = []
        seen_routes: set[tuple[int, ...]] = set()
        best_scores: dict[int, list[float]] = {self.home_id: [0.0]}
        max_keep_per_node = max(2, min(6, k + 2))
        max_iter = max(3000, len(self.edges) * 60)

        while pq and counter < max_iter and len(routes) < max(1, k):
            score, _, cur, path_nodes, path_edges, length, min_clearance, cert, reported = heapq.heappop(pq)
            reached_target = target_id is not None and cur == target_id and path_edges
            exploratory_route = (not require_target) and cur != self.home_id and path_edges
            if reached_target or exploratory_route:
                sig = tuple(path_nodes)
                if sig not in seen_routes:
                    seen_routes.add(sig)
                    if reached_target:
                        status = "certified" if cert >= 0.62 else "candidate"
                        if not reported:
                            status += "/needs_report"
                    else:
                        status = "exploratory"
                    routes.append(RouteCandidate(
                        list(path_nodes),
                        list(path_edges),
                        float(length),
                        float(min_clearance if math.isfinite(min_clearance) else 0.0),
                        float(cert),
                        bool(reported),
                        status,
                    ))
                    if reached_target and len(routes) >= k:
                        break
                # For exploratory digests, keep expanding to find longer/high
                # confidence alternatives.  For target routes, do not expand
                # through the target node.
                if reached_target:
                    continue

            for nb, eid in self._adj.get(cur, {}).items():
                if nb in path_nodes:
                    continue
                edge = self.edges[eid]
                new_length = length + edge.cert.length
                new_min_clearance = min(min_clearance, edge.cert.min_clearance)
                new_cert = min(cert, edge.cert.confidence)
                new_reported = reported and edge.cert.reported_home
                new_score = new_length / max(new_cert, 0.05) + 3.0 * max(0.0, 0.65 - new_cert)
                kept = best_scores.setdefault(nb, [])
                if len(kept) >= max_keep_per_node and new_score >= max(kept):
                    continue
                kept.append(new_score)
                kept.sort()
                del kept[max_keep_per_node:]
                counter += 1
                heapq.heappush(pq, (
                    new_score,
                    counter,
                    nb,
                    path_nodes + [nb],
                    path_edges + [eid],
                    new_length,
                    new_min_clearance,
                    new_cert,
                    new_reported,
                ))

        routes.sort(key=lambda r: (r.length / max(r.certificate, 0.05), -r.certificate))
        self._route_cache_key = cache_key
        self._route_cache = routes[:max(1, k)]
        return list(self._route_cache)

    def route_points(self, route: RouteCandidate) -> list[Point]:
        return [self.nodes[n].xy for n in route.node_ids if n in self.nodes]


# ============================================================================
# src / world.py
# ============================================================================

"""Finite hidden-truth world used only by the simulator.

The generator intentionally follows the older simulator style: a 30x30 finite
world, a rectangular home base in the lower-left corner, fewer large rectangular
obstacles with spacing margins, a protected spawn region, landmark beacons, and
one truth-only target near the far side of the map.
"""
import math
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Landmark:
    id: int
    xy: Point
    name: str = ""
    is_home: bool = False


class World:
    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.width = cfg.width
        self.height = cfg.height
        self.home_base = Rect(0.0, 0.0, cfg.home_base_size, cfg.home_base_size).normalized()
        self.home = self.home_base.center
        self.home_marker = Landmark(-1, self.home, name="Home", is_home=True)
        self.obstacles: list[Rect] = []
        self.landmarks: list[Landmark] = []
        self.target: Point = self._default_target()
        self.truth_res = 0.08
        self.truth_mask = np.zeros((1, 1), dtype=bool)
        self._generate()
        self._build_truth_mask()

    @property
    def all_landmarks(self) -> list[Landmark]:
        return [self.home_marker, *self.landmarks]

    def _default_target(self) -> Point:
        tx = self.cfg.target_x if self.cfg.target_x is not None else self.width - self.cfg.world_margin - 1.0
        ty = self.cfg.target_y if self.cfg.target_y is not None else self.height - self.cfg.world_margin - 1.0
        return (float(tx), float(ty))

    def _generate(self) -> None:
        self.target = self._default_target()
        spawn_center = (self.cfg.home_base_padding + 0.8, self.cfg.home_base_padding + 0.8)

        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.cfg.obstacle_count and attempts < 10000:
            attempts += 1
            w = self.rng.uniform(self.cfg.obstacle_min_size, self.cfg.obstacle_max_size)
            h = self.rng.uniform(self.cfg.obstacle_min_size, self.cfg.obstacle_max_size)
            cx = self.rng.uniform(self.cfg.world_margin + w * 0.5, self.width - self.cfg.world_margin - w * 0.5)
            cy = self.rng.uniform(self.cfg.world_margin + h * 0.5, self.height - self.cfg.world_margin - h * 0.5)
            rect = Rect(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5).normalized()
            if _rects_overlap(rect, self.home_base, margin=self.cfg.home_base_padding + 0.35):
                continue
            if distance(rect.center, spawn_center) < self.cfg.spawn_clear_radius:
                continue
            if rect.contains(self.target, margin=self.cfg.target_clear_radius):
                continue
            if any(_rects_overlap(rect, other, margin=self.cfg.obstacle_gap_margin) for other in self.obstacles):
                continue
            self.obstacles.append(rect)

        self.landmarks = []
        attempts = 0
        while len(self.landmarks) < self.cfg.landmark_count and attempts < 10000:
            attempts += 1
            p = (
                self.rng.uniform(self.cfg.world_margin, self.width - self.cfg.world_margin),
                self.rng.uniform(self.cfg.world_margin, self.height - self.cfg.world_margin),
            )
            if self.home_base.contains(p, margin=0.75):
                continue
            if distance(p, spawn_center) < self.cfg.spawn_clear_radius * 0.70:
                continue
            if distance(p, self.target) < self.cfg.target_clear_radius:
                continue
            if not self.is_free(p, margin=0.5):
                continue
            if any(distance(p, lm.xy) < 1.5 for lm in self.landmarks):
                continue
            self.landmarks.append(Landmark(len(self.landmarks), p, name=f"L{len(self.landmarks) + 1}"))

    def _build_truth_mask(self) -> None:
        nx = int(math.ceil(self.width / self.truth_res))
        ny = int(math.ceil(self.height / self.truth_res))
        mask = np.zeros((ny, nx), dtype=bool)
        for obs in self.obstacles:
            ix0 = max(0, int(math.floor(obs.x0 / self.truth_res)))
            ix1 = min(nx, int(math.ceil(obs.x1 / self.truth_res)))
            iy0 = max(0, int(math.floor(obs.y0 / self.truth_res)))
            iy1 = min(ny, int(math.ceil(obs.y1 / self.truth_res)))
            mask[iy0:iy1, ix0:ix1] = True
        self.truth_mask = mask

    def in_bounds(self, p: Point, margin: float = 0.0) -> bool:
        return margin <= p[0] <= self.width - margin and margin <= p[1] <= self.height - margin

    def _truth_occupied(self, p: Point) -> bool:
        i = int(p[0] / self.truth_res)
        j = int(p[1] / self.truth_res)
        if not (0 <= j < self.truth_mask.shape[0] and 0 <= i < self.truth_mask.shape[1]):
            return True
        return bool(self.truth_mask[j, i])

    def is_free(self, p: Point, margin: float = 0.0) -> bool:
        if not self.in_bounds(p, margin=margin):
            return False
        return not any(obs.contains(p, margin=margin) for obs in self.obstacles)

    def segment_free(self, a: Point, b: Point, margin: float = 0.0) -> bool:
        if not self.in_bounds(a, margin=margin) or not self.in_bounds(b, margin=margin):
            return False
        for obs in self.obstacles:
            if segment_intersects_rect(a, b, obs, margin=margin):
                return False
        return True

    def raycast(self, pose: Pose, rel_angle: float, max_range: float, step: float = 0.12) -> tuple[float, Point, bool]:
        x, y, th = pose
        dx, dy = unit_from_angle(th + rel_angle)
        r = 0.0
        last = (x, y)
        while r < max_range:
            r += step
            p = (x + dx * r, y + dy * r)
            if not self.in_bounds(p):
                return min(r, max_range), p, True
            if self._truth_occupied(p):
                return min(r, max_range), p, True
            last = p
        return max_range, last, False

    def visible_landmarks(self, pose: Pose, max_range: float) -> list[Landmark]:
        p = (pose[0], pose[1])
        out: list[Landmark] = []
        for lm in self.all_landmarks:
            if distance(p, lm.xy) <= max_range and self.segment_free(p, lm.xy, margin=0.02):
                out.append(lm)
        return out

    def target_visible(self, pose: Pose, max_range: float) -> bool:
        p = (pose[0], pose[1])
        return distance(p, self.target) <= max_range and self.segment_free(p, self.target, margin=0.02)

    def raster_obstacle_mask(self, resolution: float) -> np.ndarray:
        nx = int(math.ceil(self.width / resolution))
        ny = int(math.ceil(self.height / resolution))
        mask = np.zeros((ny, nx), dtype=bool)
        for obs in self.obstacles:
            ix0 = max(0, int(math.floor(obs.x0 / resolution)))
            ix1 = min(nx, int(math.ceil(obs.x1 / resolution)))
            iy0 = max(0, int(math.floor(obs.y0 / resolution)))
            iy1 = min(ny, int(math.ceil(obs.y1 / resolution)))
            mask[iy0:iy1, ix0:ix1] = True
        return mask


def _rects_overlap(a: Rect, b: Rect, margin: float = 0.0) -> bool:
    return not (
        a.x1 + margin < b.x0 - margin
        or a.x0 - margin > b.x1 + margin
        or a.y1 + margin < b.y0 - margin
        or a.y0 - margin > b.y1 + margin
    )


# ============================================================================
# src / sensors.py
# ============================================================================

"""LiDAR sensing models."""
import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None


if njit is not None:
    @njit(cache=True)
    def _raycast_many_jit(
        truth_mask: np.ndarray,
        width: float,
        height: float,
        truth_res: float,
        x: float,
        y: float,
        th: float,
        angles: np.ndarray,
        max_range: float,
        step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.empty(len(angles), dtype=np.float64)
        hits = np.zeros(len(angles), dtype=np.bool_)
        ny, nx = truth_mask.shape
        for k in range(len(angles)):
            theta = th + angles[k]
            dx = math.cos(theta)
            dy = math.sin(theta)
            r = 0.0
            out_r = max_range
            hit = False
            while r < max_range:
                r += step
                px = x + dx * r
                py = y + dy * r
                if px < 0.0 or px > width or py < 0.0 or py > height:
                    out_r = r if r < max_range else max_range
                    hit = True
                    break
                i = int(px / truth_res)
                j = int(py / truth_res)
                if j < 0 or j >= ny or i < 0 or i >= nx:
                    out_r = r if r < max_range else max_range
                    hit = True
                    break
                if truth_mask[j, i]:
                    out_r = r if r < max_range else max_range
                    hit = True
                    break
            ranges[k] = out_r
            hits[k] = hit
        return ranges, hits
else:
    _raycast_many_jit = None


@dataclass
class LidarScan:
    angles: np.ndarray
    ranges: np.ndarray
    hit: np.ndarray
    points_world_true: np.ndarray


class LidarSensor:
    def __init__(self, cfg: LidarConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.angles = np.linspace(-math.pi, math.pi, cfg.rays, endpoint=False)

    def sense(self, world: World, true_pose: Pose) -> LidarScan:
        ranges = np.zeros(self.cfg.rays, dtype=float)
        hit = np.zeros(self.cfg.rays, dtype=bool)
        pts = np.zeros((self.cfg.rays, 2), dtype=float)
        th = true_pose[2]
        if _raycast_many_jit is not None:
            raw_ranges, raw_hits = _raycast_many_jit(
                world.truth_mask,
                float(world.width),
                float(world.height),
                float(world.truth_res),
                float(true_pose[0]),
                float(true_pose[1]),
                float(th),
                self.angles,
                float(self.cfg.range),
                float(self.cfg.raycast_step_m),
            )
        else:
            raw_ranges = np.zeros(self.cfg.rays, dtype=float)
            raw_hits = np.zeros(self.cfg.rays, dtype=bool)
            for k, a in enumerate(self.angles):
                raw_ranges[k], _p, raw_hits[k] = world.raycast(true_pose, float(a), self.cfg.range, step=self.cfg.raycast_step_m)
        for k, a in enumerate(self.angles):
            r = float(raw_ranges[k])
            h = bool(raw_hits[k])
            if h and self.rng.random() < self.cfg.dropout_probability:
                h = False
                r = self.cfg.range
            if h:
                sigma = self.cfg.noise_std + self.cfg.range_noise_std_per_m * max(0.0, float(r))
                rn = float(np.clip(r + self.rng.normal(0.0, sigma), 0.02, self.cfg.range))
            else:
                rn = float(np.clip(self.cfg.range + self.rng.normal(0.0, self.cfg.max_range_noise_std), 0.02, self.cfg.range))
            ranges[k] = rn
            hit[k] = h and rn < self.cfg.range - self.cfg.hit_threshold
            pts[k] = [true_pose[0] + math.cos(th + a) * rn, true_pose[1] + math.sin(th + a) * rn]
        return LidarScan(self.angles.copy(), ranges, hit, pts)


# ============================================================================
# src / mapping.py
# ============================================================================

"""LiDAR occupancy map with per-cell quality.

Cells are updated from LiDAR scans transformed by the estimated pose.  Updates
made with poor pose confidence are weaker and cannot blindly overwrite older
higher-quality evidence.
"""
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None


@dataclass
class FrontierCluster:
    cells: list[tuple[int, int]]
    centroid_world: Point
    information_gain: float


if njit is not None:
    @njit(cache=True)
    def _update_from_lidar_jit(
        logodds: np.ndarray,
        quality: np.ndarray,
        last_seen: np.ndarray,
        source: np.ndarray,
        source_mask: np.ndarray,
        nx: int,
        ny: int,
        res: float,
        est_x: float,
        est_y: float,
        est_th: float,
        angles: np.ndarray,
        ranges: np.ndarray,
        hits: np.ndarray,
        pose_quality: float,
        robot_id: int,
        source_bit: int,
        time_s: float,
        logodds_free: float,
        logodds_occ: float,
        logodds_min: float,
        logodds_max: float,
        quality_margin: float,
        low_quality_scale: float,
        free_oi: np.ndarray,
        free_oj: np.ndarray,
        free_ow: np.ndarray,
        hit_oi: np.ndarray,
        hit_oj: np.ndarray,
        hit_ow: np.ndarray,
    ) -> bool:
        start_i = int(math.floor(est_x / res))
        start_j = int(math.floor(est_y / res))
        if start_i < 0 or start_i >= nx or start_j < 0 or start_j >= ny:
            return False

        obs_q = pose_quality
        if obs_q < 0.0:
            obs_q = 0.0
        elif obs_q > 1.0:
            obs_q = 1.0
        if obs_q <= 0.0:
            return False

        wrote = False
        for ray_idx in range(len(angles)):
            a = angles[ray_idx]
            rng = ranges[ray_idx]
            th = est_th + a
            end_x = est_x + math.cos(th) * rng
            end_y = est_y + math.sin(th) * rng
            end_i = int(math.floor(end_x / res))
            end_j = int(math.floor(end_y / res))
            if end_i < 0 or end_i >= nx or end_j < 0 or end_j >= ny:
                rr = rng - res
                if rr < 0.0:
                    rr = 0.0
                end_x = est_x + math.cos(th) * rr
                end_y = est_y + math.sin(th) * rr
                end_i = int(math.floor(end_x / res))
                end_j = int(math.floor(end_y / res))
                if end_i < 0 or end_i >= nx or end_j < 0 or end_j >= ny:
                    continue

            dx = abs(end_i - start_i)
            dy = -abs(end_j - start_j)
            sx = 1 if start_i < end_i else -1
            sy = 1 if start_j < end_j else -1
            err = dx + dy
            x = start_i
            y = start_j
            hit = bool(hits[ray_idx])

            while True:
                terminal = x == end_i and y == end_j
                if not terminal or not hit:
                    _write_kernel_jit(
                        logodds,
                        quality,
                        last_seen,
                        source,
                        source_mask,
                        nx,
                        ny,
                        x,
                        y,
                        logodds_free,
                        obs_q,
                        robot_id,
                        source_bit,
                        time_s,
                        logodds_min,
                        logodds_max,
                        quality_margin,
                        low_quality_scale,
                        free_oi,
                        free_oj,
                        free_ow,
                    )
                    wrote = True

                if terminal:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

            if hit:
                _write_kernel_jit(
                    logodds,
                    quality,
                    last_seen,
                    source,
                    source_mask,
                    nx,
                    ny,
                    end_i,
                    end_j,
                    logodds_occ,
                    obs_q,
                    robot_id,
                    source_bit,
                    time_s,
                    logodds_min,
                    logodds_max,
                    quality_margin,
                    low_quality_scale,
                    hit_oi,
                    hit_oj,
                    hit_ow,
                )
                wrote = True
        return wrote


    @njit(cache=True)
    def _write_kernel_jit(
        logodds: np.ndarray,
        quality: np.ndarray,
        last_seen: np.ndarray,
        source: np.ndarray,
        source_mask: np.ndarray,
        nx: int,
        ny: int,
        ci: int,
        cj: int,
        delta: float,
        obs_quality: float,
        robot_id: int,
        source_bit: int,
        time_s: float,
        logodds_min: float,
        logodds_max: float,
        quality_margin: float,
        low_quality_scale: float,
        offsets_i: np.ndarray,
        offsets_j: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        for k in range(len(offsets_i)):
            i = ci + offsets_i[k]
            j = cj + offsets_j[k]
            if i < 0 or i >= nx or j < 0 or j >= ny:
                continue
            weighted_quality = obs_quality * weights[k]
            if weighted_quality <= 0.0:
                continue
            current_q = quality[j, i]
            if weighted_quality + quality_margin >= current_q:
                scale = 1.0
            else:
                denom = current_q
                if denom < 1e-6:
                    denom = 1e-6
                ratio = weighted_quality / denom
                if ratio < 0.05:
                    ratio = 0.05
                scale = low_quality_scale * ratio
            lo = logodds[j, i] + delta * weighted_quality * scale
            if lo < logodds_min:
                lo = logodds_min
            elif lo > logodds_max:
                lo = logodds_max
            logodds[j, i] = lo

            decayed = current_q * 0.995
            quality[j, i] = decayed if decayed > weighted_quality else weighted_quality
            last_seen[j, i] = time_s
            source[j, i] = robot_id
            if source_bit != 0:
                source_mask[j, i] = source_mask[j, i] | source_bit
else:
    _update_from_lidar_jit = None


if njit is not None:
    @njit(cache=True)
    def _predict_scan_ranges_jit(
        occupied: np.ndarray,
        nx: int,
        ny: int,
        res: float,
        est_x: float,
        est_y: float,
        est_th: float,
        angles: np.ndarray,
        max_range: float,
        step: float,
    ) -> np.ndarray:
        out = np.empty(len(angles), dtype=np.float64)
        for k in range(len(angles)):
            theta = est_th + angles[k]
            c = math.cos(theta)
            s = math.sin(theta)
            r = 0.0
            val = max_range
            while r < max_range:
                r += step
                px = est_x + c * r
                py = est_y + s * r
                i = int(math.floor(px / res))
                j = int(math.floor(py / res))
                if i < 0 or i >= nx or j < 0 or j >= ny:
                    val = r if r < max_range else max_range
                    break
                if occupied[j, i]:
                    val = r if r < max_range else max_range
                    break
            out[k] = val
        return out
else:
    _predict_scan_ranges_jit = None


class OccupancyGrid:
    def __init__(self, width: float, height: float, cfg: MappingConfig):
        self.width_m = width
        self.height_m = height
        self.cfg = cfg
        self.res = cfg.resolution
        self.nx = int(math.ceil(width / self.res))
        self.ny = int(math.ceil(height / self.res))
        self.logodds = np.zeros((self.ny, self.nx), dtype=float)
        self.quality = np.zeros((self.ny, self.nx), dtype=float)
        self.last_seen = np.full((self.ny, self.nx), -np.inf, dtype=float)
        self.source = np.full((self.ny, self.nx), -1, dtype=int)
        self.source_mask = np.zeros((self.ny, self.nx), dtype=np.int64)
        self._version = 0
        self._clearance_cache: dict[tuple[int, int], np.ndarray] = {}
        self._kernel_cache: dict[tuple[float, float, float], tuple[tuple[int, int, float], ...]] = {}
        self._kernel_array_cache: dict[tuple[float, float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def copy(self) -> "OccupancyGrid":
        other = OccupancyGrid(self.width_m, self.height_m, self.cfg)
        other.logodds = self.logodds.copy()
        other.quality = self.quality.copy()
        other.last_seen = self.last_seen.copy()
        other.source = self.source.copy()
        other.source_mask = self.source_mask.copy()
        other._version = self._version
        return other


    def _source_bit(self, robot_id: int) -> int:
        return 1 << robot_id if 0 <= robot_id < 62 else 0

    def _invalidate_cache(self) -> None:
        self._version += 1
        self._clearance_cache.clear()

    def clearance_map(self, max_radius_m: float = 3.0) -> np.ndarray:
        """Approximate distance to nearest occupied cell in meters.

        Two-pass chamfer transform: fast, dependency-free, and good enough for
        centerline/clearance-aware planning on the coarser grid.
        """
        max_cells = int(math.ceil(max_radius_m / self.res))
        key = (self._version, max_cells)
        cached = self._clearance_cache.get(key)
        if cached is not None:
            return cached
        occ = self.occupied_mask()
        inf = float(max_cells + 4)
        dist = np.full((self.ny, self.nx), inf, dtype=float)
        dist[occ] = 0.0
        diag = math.sqrt(2.0)
        for y in range(self.ny):
            for x in range(self.nx):
                v = dist[y, x]
                if x > 0: v = min(v, dist[y, x - 1] + 1.0)
                if y > 0: v = min(v, dist[y - 1, x] + 1.0)
                if x > 0 and y > 0: v = min(v, dist[y - 1, x - 1] + diag)
                if x + 1 < self.nx and y > 0: v = min(v, dist[y - 1, x + 1] + diag)
                dist[y, x] = v
        for y in range(self.ny - 1, -1, -1):
            for x in range(self.nx - 1, -1, -1):
                v = dist[y, x]
                if x + 1 < self.nx: v = min(v, dist[y, x + 1] + 1.0)
                if y + 1 < self.ny: v = min(v, dist[y + 1, x] + 1.0)
                if x + 1 < self.nx and y + 1 < self.ny: v = min(v, dist[y + 1, x + 1] + diag)
                if x > 0 and y + 1 < self.ny: v = min(v, dist[y + 1, x - 1] + diag)
                dist[y, x] = v
        clearance = np.clip(dist * self.res, 0.0, max_radius_m)
        self._clearance_cache[key] = clearance
        return clearance

    def world_to_cell(self, p: Point) -> tuple[int, int] | None:
        i = int(math.floor(p[0] / self.res))
        j = int(math.floor(p[1] / self.res))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def cell_to_world(self, cell: tuple[int, int]) -> Point:
        i, j = cell
        return ((i + 0.5) * self.res, (j + 0.5) * self.res)

    def probability(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logodds))

    def free_mask(self) -> np.ndarray:
        return self.probability() < self.cfg.prob_free_threshold

    def occupied_mask(self) -> np.ndarray:
        return self.probability() > self.cfg.prob_occ_threshold

    def known_mask(self) -> np.ndarray:
        p = self.probability()
        return (p < self.cfg.prob_free_threshold) | (p > self.cfg.prob_occ_threshold)

    def traversable_mask(self, inflation_m: float) -> np.ndarray:
        occ = self.occupied_mask()
        inflated = self.inflate_mask(occ, inflation_m)
        margin = int(math.ceil(max(0.0, inflation_m) / self.res))
        if margin > 0:
            inflated[:margin, :] = True
            inflated[-margin:, :] = True
            inflated[:, :margin] = True
            inflated[:, -margin:] = True
        # Unknown remains traversable but costly for exploration; occupied/inflated is not.
        return ~inflated

    def inflate_mask(self, mask: np.ndarray, radius_m: float) -> np.ndarray:
        radius = int(math.ceil(radius_m / self.res))
        if radius <= 0:
            return mask.copy()
        out = mask.copy()
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            y0 = max(0, y - radius)
            y1 = min(self.ny, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(self.nx, x + radius + 1)
            out[y0:y1, x0:x1] = True
        return out

    def _bresenham(self, a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        cells = []
        while True:
            if 0 <= x < self.nx and 0 <= y < self.ny:
                cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return cells

    def _write_cell(self, cell: tuple[int, int], delta: float, obs_quality: float, robot_id: int, time_s: float, weight: float = 1.0) -> None:
        i, j = cell
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return
        w = min(1.0, max(0.0, float(weight)))
        if w <= 0.0:
            return
        weighted_quality = min(1.0, max(0.0, float(obs_quality) * w))
        current_q = self.quality[j, i]
        # Higher-quality observations dominate. Lower-quality observations are
        # allowed to nudge, but not overwrite aggressively.
        if weighted_quality + self.cfg.quality_overwrite_margin >= current_q:
            scale = 1.0
        else:
            scale = self.cfg.low_quality_update_scale * max(0.05, weighted_quality / max(current_q, 1e-6))
        self.logodds[j, i] = clamp(
            float(self.logodds[j, i] + delta * weighted_quality * scale),
            self.cfg.logodds_min,
            self.cfg.logodds_max,
        )
        self.quality[j, i] = max(current_q * 0.995, weighted_quality)
        self.last_seen[j, i] = time_s
        self.source[j, i] = robot_id
        self.source_mask[j, i] |= self._source_bit(robot_id)

    def _write_cell_kernel(self, cell: tuple[int, int], delta: float, obs_quality: float, robot_id: int, time_s: float, radius_m: float) -> None:
        radius = max(0.0, float(radius_m))
        if radius <= 1e-9:
            self._write_cell(cell, delta, obs_quality, robot_id, time_s)
            return
        ci, cj = cell
        obs_q = min(1.0, max(0.0, float(obs_quality)))
        if obs_q <= 0.0:
            return
        quality = self.quality
        logodds = self.logodds
        last_seen = self.last_seen
        source = self.source
        source_mask = self.source_mask
        quality_margin = self.cfg.quality_overwrite_margin
        low_quality_scale = self.cfg.low_quality_update_scale
        lo_min = self.cfg.logodds_min
        lo_max = self.cfg.logodds_max
        src_bit = self._source_bit(robot_id)
        for oi, oj, weight in self._kernel_offsets(radius):
            i = ci + oi
            j = cj + oj
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                continue
            weighted_quality = obs_q * weight
            if weighted_quality <= 0.0:
                continue
            current_q = quality[j, i]
            if weighted_quality + quality_margin >= current_q:
                scale = 1.0
            else:
                scale = low_quality_scale * max(0.05, weighted_quality / max(current_q, 1e-6))
            logodds[j, i] = min(lo_max, max(lo_min, float(logodds[j, i] + float(delta) * weighted_quality * scale)))
            quality[j, i] = max(current_q * 0.995, weighted_quality)
            last_seen[j, i] = time_s
            source[j, i] = robot_id
            if src_bit:
                source_mask[j, i] |= src_bit

    def _kernel_offsets(self, radius: float) -> tuple[tuple[int, int, float], ...]:
        min_weight = min(1.0, max(0.0, float(self.cfg.lidar_kernel_min_weight)))
        key = (round(float(radius), 9), round(float(self.res), 9), round(min_weight, 9))
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached

        radius_cells = max(0, int(math.ceil(radius / self.res)))
        sigma = max(self.res * 0.5, radius * 0.62)
        offsets: list[tuple[int, int, float]] = []
        for oj in range(-radius_cells, radius_cells + 1):
            for oi in range(-radius_cells, radius_cells + 1):
                dist_m = math.hypot(oi * self.res, oj * self.res)
                if dist_m > radius + 1e-9:
                    continue
                weight = math.exp(-0.5 * (dist_m / sigma) ** 2)
                if dist_m > 0.0:
                    weight = max(min_weight, weight)
                offsets.append((oi, oj, weight))

        out = tuple(offsets)
        self._kernel_cache[key] = out
        return out

    def update_from_lidar(self, est_pose: Pose, scan: LidarScan, pose_quality: float, robot_id: int, time_s: float) -> None:
        if _update_from_lidar_jit is not None:
            free_oi, free_oj, free_ow = self._kernel_offset_arrays(self.cfg.lidar_free_kernel_radius_m)
            hit_oi, hit_oj, hit_ow = self._kernel_offset_arrays(self.cfg.lidar_hit_kernel_radius_m)
            wrote = _update_from_lidar_jit(
                self.logodds,
                self.quality,
                self.last_seen,
                self.source,
                self.source_mask,
                self.nx,
                self.ny,
                float(self.res),
                float(est_pose[0]),
                float(est_pose[1]),
                float(est_pose[2]),
                scan.angles,
                scan.ranges,
                scan.hit,
                float(pose_quality),
                int(robot_id),
                int(self._source_bit(robot_id)),
                float(time_s),
                float(self.cfg.logodds_free),
                float(self.cfg.logodds_occ),
                float(self.cfg.logodds_min),
                float(self.cfg.logodds_max),
                float(self.cfg.quality_overwrite_margin),
                float(self.cfg.low_quality_update_scale),
                free_oi,
                free_oj,
                free_ow,
                hit_oi,
                hit_oj,
                hit_ow,
            )
            if wrote:
                self._invalidate_cache()
            return

        start = self.world_to_cell((est_pose[0], est_pose[1]))
        if start is None:
            return
        th = est_pose[2]
        wrote = False
        for angle, r, hit in zip(scan.angles, scan.ranges, scan.hit):
            end = (est_pose[0] + math.cos(th + float(angle)) * float(r), est_pose[1] + math.sin(th + float(angle)) * float(r))
            end_cell = self.world_to_cell(end)
            if end_cell is None:
                # Clamp outside endpoint by skipping the terminal occupied update.
                rr = max(0.0, float(r) - self.res)
                end = (est_pose[0] + math.cos(th + float(angle)) * rr, est_pose[1] + math.sin(th + float(angle)) * rr)
                end_cell = self.world_to_cell(end)
                if end_cell is None:
                    continue
            ray_cells = self._bresenham(start, end_cell)
            if not ray_cells:
                continue
            free_cells = ray_cells[:-1] if hit else ray_cells
            for c in free_cells:
                self._write_cell_kernel(c, self.cfg.logodds_free, pose_quality, robot_id, time_s, self.cfg.lidar_free_kernel_radius_m)
                wrote = True
            if hit:
                self._write_cell_kernel(ray_cells[-1], self.cfg.logodds_occ, pose_quality, robot_id, time_s, self.cfg.lidar_hit_kernel_radius_m)
                wrote = True
        if wrote:
            self._invalidate_cache()

    def _kernel_offset_arrays(self, radius_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radius = max(0.0, float(radius_m))
        min_weight = min(1.0, max(0.0, float(self.cfg.lidar_kernel_min_weight)))
        key = (round(radius, 9), round(float(self.res), 9), round(min_weight, 9))
        cached = self._kernel_array_cache.get(key)
        if cached is not None:
            return cached
        offsets = self._kernel_offsets(radius)
        out = (
            np.asarray([oi for oi, _oj, _w in offsets], dtype=np.int64),
            np.asarray([oj for _oi, oj, _w in offsets], dtype=np.int64),
            np.asarray([w for _oi, _oj, w in offsets], dtype=np.float64),
        )
        self._kernel_array_cache[key] = out
        return out

    def predict_scan_ranges(self, est_pose: Pose, angles: np.ndarray, max_range: float) -> np.ndarray:
        occ = self.occupied_mask()
        step = max(0.5 * self.res, 0.05)
        if _predict_scan_ranges_jit is not None:
            return _predict_scan_ranges_jit(
                occ,
                self.nx,
                self.ny,
                float(self.res),
                float(est_pose[0]),
                float(est_pose[1]),
                float(est_pose[2]),
                angles,
                float(max_range),
                float(step),
            )
        out = np.full(len(angles), max_range, dtype=float)
        for k, a in enumerate(angles):
            theta = est_pose[2] + float(a)
            r = 0.0
            while r < max_range:
                r += step
                p = (est_pose[0] + math.cos(theta) * r, est_pose[1] + math.sin(theta) * r)
                cell = self.world_to_cell(p)
                if cell is None:
                    out[k] = min(r, max_range)
                    break
                i, j = cell
                if occ[j, i]:
                    out[k] = min(r, max_range)
                    break
        return out

    def find_frontiers(self, min_cluster_size: int, info_radius_m: float) -> list[FrontierCluster]:
        free = self.free_mask()
        known = self.known_mask()
        unknown = ~known
        frontier = np.zeros_like(free, dtype=bool)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if not free[j, i]:
                    continue
                nb = unknown[j - 1:j + 2, i - 1:i + 2]
                if np.any(nb):
                    frontier[j, i] = True
        visited = np.zeros_like(frontier, dtype=bool)
        clusters: list[FrontierCluster] = []
        radius = max(1, int(round(info_radius_m / self.res)))
        for j in range(self.ny):
            for i in range(self.nx):
                if not frontier[j, i] or visited[j, i]:
                    continue
                q = deque([(i, j)])
                visited[j, i] = True
                cells: list[tuple[int, int]] = []
                while q:
                    ci, cj = q.popleft()
                    cells.append((ci, cj))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = ci + di, cj + dj
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and frontier[nj, ni] and not visited[nj, ni]:
                            visited[nj, ni] = True
                            q.append((ni, nj))
                if len(cells) < min_cluster_size:
                    continue
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                centroid_cell = (int(round(float(np.mean(xs)))), int(round(float(np.mean(ys)))))
                x0 = max(0, centroid_cell[0] - radius)
                x1 = min(self.nx, centroid_cell[0] + radius + 1)
                y0 = max(0, centroid_cell[1] - radius)
                y1 = min(self.ny, centroid_cell[1] + radius + 1)
                gain = float(np.sum(unknown[y0:y1, x0:x1]))
                clusters.append(FrontierCluster(cells, self.cell_to_world(centroid_cell), gain))
        clusters.sort(key=lambda c: c.information_gain, reverse=True)
        return clusters

    def clearance_at(self, p: Point, max_radius_m: float = 3.0) -> float:
        cell = self.world_to_cell(p)
        if cell is None:
            return 0.0
        i, j = cell
        return float(self.clearance_map(max_radius_m)[j, i])

    def safe_approach_point(
        self,
        frontier: FrontierCluster,
        start: Point,
        search_radius_m: float,
        min_clearance_m: float,
        desired_clearance_m: float,
        clearance: np.ndarray | None = None,
        free: np.ndarray | None = None,
        known: np.ndarray | None = None,
    ) -> Point:
        if clearance is None:
            clearance = self.clearance_map(max_radius_m=max(3.0, desired_clearance_m * 2.5))
        if free is None:
            free = self.free_mask()
        if known is None:
            known = self.known_mask()
        centroid = frontier.centroid_world
        ccell = self.world_to_cell(centroid)
        if ccell is None:
            return centroid
        radius = max(1, int(math.ceil(search_radius_m / self.res)))
        ci, cj = ccell
        best_cell = None
        best_score = -math.inf
        for j in range(max(0, cj - radius), min(self.ny, cj + radius + 1)):
            for i in range(max(0, ci - radius), min(self.nx, ci + radius + 1)):
                if not free[j, i] or not known[j, i]:
                    continue
                cl = float(clearance[j, i])
                if cl < min_clearance_m:
                    continue
                p = self.cell_to_world((i, j))
                dc = distance(p, centroid)
                ds = distance(p, start)
                score = 2.2 * min(1.0, cl / max(1e-6, desired_clearance_m)) - 0.40 * dc - 0.03 * ds
                if score > best_score:
                    best_score = score
                    best_cell = (i, j)
        return self.cell_to_world(best_cell) if best_cell is not None else centroid

    def segment_min_clearance(self, a: Point, b: Point, max_radius_m: float = 3.0) -> float:
        ca = self.world_to_cell(a)
        cb = self.world_to_cell(b)
        if ca is None or cb is None:
            return 0.0
        clearance = self.clearance_map(max_radius_m)
        cells = self._bresenham(ca, cb)
        vals = [float(clearance[j, i]) for i, j in cells if 0 <= i < self.nx and 0 <= j < self.ny]
        return min(vals) if vals else 0.0

    def path_min_clearance(self, path: list[Point], max_radius_m: float = 3.0) -> float:
        if len(path) < 2:
            return 0.0
        clearance = self.clearance_map(max_radius_m)
        best = math.inf
        for a, b in zip(path[:-1], path[1:]):
            ca = self.world_to_cell(a)
            cb = self.world_to_cell(b)
            if ca is None or cb is None:
                return 0.0
            vals = [float(clearance[j, i]) for i, j in self._bresenham(ca, cb) if 0 <= i < self.nx and 0 <= j < self.ny]
            if vals:
                best = min(best, min(vals))
        return float(best if math.isfinite(best) else 0.0)

    def merge_from_digest(self, digest: dict, combine_sources: bool = False) -> None:
        """Merge a received map digest by highest confidence, not newest time.

        This method is used for robot knowledge-map fusion and HOME fused-map
        fusion. A newer packet from a poorly localized robot should not
        overwrite an older, more reliable cell. The incoming cell replaces the
        existing cell only when its stored mapping quality is higher by a small
        margin, or when the cell was previously unknown.
        """
        idx = digest.get("cells", [])
        vals = digest.get("logodds", [])
        quals = digest.get("quality", [])
        masks = digest.get("source_mask", [])
        src = int(digest.get("source_robot", -1))
        t = float(digest.get("time_s", 0.0))
        changed = False
        src_bit = self._source_bit(src)
        margin = float(getattr(self.cfg, "merge_quality_margin", 0.03))
        for k, ((i, j), lo, q) in enumerate(zip(idx, vals, quals)):
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                continue
            incoming_lo = clamp(float(lo), self.cfg.logodds_min, self.cfg.logodds_max)
            incoming_q = min(1.0, max(0.0, float(q)))
            if incoming_q <= 0.01:
                continue
            incoming_mask = int(masks[k]) if k < len(masks) else src_bit
            if src_bit:
                incoming_mask |= src_bit
            current_q = float(self.quality[j, i])
            current_mask = int(self.source_mask[j, i])

            accept = current_q <= 0.01 or incoming_q > current_q + margin
            if accept:
                self.logodds[j, i] = incoming_lo
                self.quality[j, i] = incoming_q
                self.source[j, i] = src
                self.source_mask[j, i] = current_mask | incoming_mask
                self.last_seen[j, i] = max(float(self.last_seen[j, i]), t)
                changed = True
            elif incoming_mask & ~current_mask:
                # Preserve provenance without changing the best-confidence cell.
                self.source_mask[j, i] = current_mask | incoming_mask
                self.last_seen[j, i] = max(float(self.last_seen[j, i]), t)
                changed = True
        if changed:
            self._invalidate_cache()

    def passage_quality(
        self,
        cfg: PassageQualityConfig,
        robot_radius_m: float = 0.0,
        max_radius_m: float | None = None,
    ) -> np.ndarray:
        """Return per-cell execution/traversal passage score in [0, 1].

        Passage quality answers: if a later execution robot uses this map to
        drive from HOME to target, how good is this cell for the route?

            occupancy safety × obstacle clearance × soft reliability discount

        Clearance and obstacle risk dominate. Mapping confidence only discounts
        otherwise safe cells so a low-confidence but wide/open corridor is not
        treated as worse than a tight wall-hugging corridor.
        """
        prob = self.probability()
        raw_quality = np.clip(self.quality, 0.0, 1.0)
        known = self.known_mask()
        free = prob < self.cfg.prob_free_threshold
        occupied = prob > self.cfg.prob_occ_threshold

        # 1) Occupancy safety.  The score falls as a cell approaches the
        # occupied threshold, even before it is hard-classified as occupied.
        free_score = np.clip(
            (self.cfg.prob_occ_threshold - prob) / max(1e-6, self.cfg.prob_occ_threshold - 0.05),
            0.0,
            1.0,
        )
        free_score = np.power(free_score, max(0.01, float(cfg.free_score_power)))

        # 2) Soft reliability.  Cell quality comes from the mapper pose at
        # observation time, but it should not define safety by itself.
        confidence_score = np.clip(raw_quality, 0.0, 1.0)
        confidence_score = np.power(confidence_score, max(0.01, float(cfg.map_confidence_power)))
        confidence_floor = min(1.0, max(0.0, float(cfg.map_confidence_floor)))
        reliability_score = confidence_floor + (1.0 - confidence_floor) * confidence_score

        # 3) Clearance score.  The center of a corridor/open area should score
        # higher than cells close to corridor walls. Use a broad/adaptive
        # reference so the score forms a gradient instead of turning green as
        # soon as clearance exceeds the robot radius.
        min_clearance = max(float(cfg.min_clearance_m), float(robot_radius_m))
        good_clearance = max(float(cfg.good_clearance_m), min_clearance + self.res)
        radius = max_radius_m if max_radius_m is not None else max(3.0, good_clearance * 1.8)
        clearance = self.clearance_map(max_radius_m=radius)
        clear_ref = good_clearance
        ref_mask = known & free & np.isfinite(clearance)
        if np.any(ref_mask):
            pct = min(100.0, max(0.0, float(cfg.clearance_reference_percentile)))
            clear_ref = max(clear_ref, float(np.percentile(clearance[ref_mask], pct)))
        clearance_score = np.clip((clearance - min_clearance) / max(1e-6, clear_ref - min_clearance), 0.0, 1.0)
        clearance_score = clearance_score * clearance_score * (3.0 - 2.0 * clearance_score)
        clearance_score = np.power(clearance_score, max(0.01, float(cfg.clearance_power)))

        passage = (
            np.power(free_score, max(0.01, float(cfg.free_weight)))
            * np.power(clearance_score, max(0.01, float(cfg.clearance_weight)))
            * np.power(reliability_score, max(0.01, float(cfg.map_confidence_weight)))
        )

        # Unknown is not low-quality passage; it is not passage evidence yet.
        passage[~(known & free)] = float(cfg.unknown_score)
        passage[occupied] = float(cfg.occupied_score)
        return np.clip(passage, 0.0, 1.0)

    def make_digest(self, robot_id: int, time_s: float, max_cells: int | None = 650) -> dict:
        known = self.known_mask() & (self.quality > 0.05)
        ys, xs = np.nonzero(known)
        if max_cells is not None and len(xs) > max_cells:
            # Prefer recent/high-quality cells for bandwidth-limited packets.
            score = self.quality[ys, xs] + 0.001 * np.maximum(0.0, self.last_seen[ys, xs])
            keep = np.argsort(score)[-int(max_cells):]
            xs = xs[keep]
            ys = ys[keep]
        cells = [(int(i), int(j)) for i, j in zip(xs, ys)]
        return {
            "source_robot": int(robot_id),
            "time_s": float(time_s),
            "cells": cells,
            "logodds": [float(self.logodds[j, i]) for i, j in cells],
            "quality": [float(self.quality[j, i]) for i, j in cells],
            "source_mask": [int(self.source_mask[j, i] | self._source_bit(robot_id)) for i, j in cells],
        }


# ============================================================================
# src / lidar_assessment.py
# ============================================================================

"""LiDAR-first local safety and scan-map consistency assessment."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LidarAssessment:
    consistency: float = 0.0
    mismatch_fraction: float = 1.0
    front_clearance: float = 0.0
    left_clearance: float = 0.0
    right_clearance: float = 0.0
    blocked_forward: bool = True
    open_sector_count: int = 0
    best_open_angle: float = 0.0
    decision_note: str = "init"


def _sector_clearance(scan: LidarScan, center: float, half_width: float, fallback: float, percentile: float) -> float:
    delta = np.angle(np.exp(1j * (scan.angles - center)))
    m = np.abs(delta) <= half_width
    if not np.any(m):
        return fallback
    return float(np.percentile(scan.ranges[m], percentile))


def assess_lidar(
    grid: OccupancyGrid,
    est_pose: Pose,
    scan: LidarScan,
    lidar_cfg: LidarConfig,
    assess_cfg: AssessmentConfig,
    previous_consistency: float | None = None,
) -> LidarAssessment:
    sub = max(1, len(scan.angles) // 32)
    sample_angles = scan.angles[::sub]
    sample_ranges = scan.ranges[::sub]
    pred = grid.predict_scan_ranges(est_pose, sample_angles, lidar_cfg.range)
    valid = (sample_ranges < lidar_cfg.range * 0.98) | (pred < lidar_cfg.range * 0.98)
    if np.any(valid):
        err = np.abs(sample_ranges[valid] - pred[valid])
        norm = np.clip(err / max(assess_cfg.scan_consistency_tolerance_m, 1e-6), 0.0, 1.0)
        raw_consistency = float(1.0 - np.mean(norm))
        mismatch_fraction = float(np.mean(norm > 0.65))
    else:
        raw_consistency = 0.75
        mismatch_fraction = 0.0
    if previous_consistency is None:
        consistency = raw_consistency
    else:
        a = assess_cfg.consistency_smoothing
        consistency = float(a * raw_consistency + (1.0 - a) * previous_consistency)

    front_half = math.radians(lidar_cfg.front_angle_deg)
    side_half = math.radians(lidar_cfg.side_angle_deg * 0.5)
    sector_pct = float(assess_cfg.sector_clearance_percentile)
    front = _sector_clearance(scan, 0.0, front_half, lidar_cfg.range, sector_pct)
    left = _sector_clearance(scan, math.pi / 2.0, side_half, lidar_cfg.range, sector_pct)
    right = _sector_clearance(scan, -math.pi / 2.0, side_half, lidar_cfg.range, sector_pct)
    blocked = front <= lidar_cfg.blocked_forward_distance

    open_mask = scan.ranges > max(lidar_cfg.range * assess_cfg.open_sector_range_fraction, lidar_cfg.blocked_forward_distance * 2.0)
    min_width = max(1, int(round(math.radians(lidar_cfg.open_sector_min_width_deg) / (2 * math.pi) * len(scan.angles))))
    sectors: list[tuple[int, int]] = []
    n = len(open_mask)
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if not open_mask[start] or visited[start]:
            continue
        idxs = []
        k = start
        while open_mask[k] and not visited[k]:
            visited[k] = True
            idxs.append(k)
            k = (k + 1) % n
            if k == start:
                break
        if len(idxs) >= min_width:
            sectors.append((idxs[0], idxs[-1]))
    if sectors:
        def sector_score(ab: tuple[int, int]) -> float:
            start, end = ab
            width = ((end - start) % n) + 1
            idx = [(start + o) % n for o in range(width)]
            mid = (start + (width - 1) / 2.0) % n
            mid_angle = float(scan.angles[int(mid) % n])
            depth = float(np.percentile(scan.ranges[idx], assess_cfg.open_sector_depth_percentile))
            forward = math.cos(mid_angle)
            return (
                assess_cfg.open_sector_width_weight * (width / max(1, n))
                + assess_cfg.open_sector_depth_weight * (depth / max(1e-6, lidar_cfg.range))
                + assess_cfg.open_sector_forward_weight * forward
            )

        best = max(sectors, key=sector_score)
        width = ((best[1] - best[0]) % n) + 1
        mid = (best[0] + (width - 1) / 2.0) % n
        best_open_angle = float(scan.angles[int(mid) % n])
    else:
        best_open_angle = 0.0

    if blocked:
        note = "blocked_forward_by_lidar"
    elif consistency < assess_cfg.low_consistency:
        note = "low_scan_map_consistency"
    elif consistency < assess_cfg.caution_consistency:
        note = "caution_scan_map_consistency"
    else:
        note = "lidar_map_agree"

    return LidarAssessment(
        consistency=float(consistency),
        mismatch_fraction=mismatch_fraction,
        front_clearance=front,
        left_clearance=left,
        right_clearance=right,
        blocked_forward=bool(blocked),
        open_sector_count=len(sectors),
        best_open_angle=best_open_angle,
        decision_note=note,
    )


# ============================================================================
# src / localization.py
# ============================================================================

"""Lightweight EKF pose estimator used by the robots.

The estimator keeps the planning pose in the robot's local belief state.  The
simulator may generate noisy measurements from the hidden truth state, but the
estimator only receives those noisy range/bearing measurements and known
landmark locations.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseBelief:
    pose: np.ndarray  # x, y, theta
    covariance: np.ndarray  # 3x3

    @property
    def xy(self) -> Point:
        return (float(self.pose[0]), float(self.pose[1]))

    @property
    def theta(self) -> float:
        return float(self.pose[2])

    @property
    def cov_trace_xy(self) -> float:
        return float(np.trace(self.covariance[:2, :2]))

    def as_pose(self) -> Pose:
        return (float(self.pose[0]), float(self.pose[1]), float(self.pose[2]))


class PoseEstimator:
    def __init__(self, initial_pose: Pose, cfg: MotionNoiseConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.belief = PoseBelief(
            pose=np.array(initial_pose, dtype=float),
            covariance=np.diag([0.05, 0.05, 0.02]),
        )
        self.last_landmark_count = 0
        self.last_landmark_residual = 0.0
        self.last_lidar_match_confidence = 0.0
        self.last_teammate_update_count = 0
        self.last_teammate_residual = 0.0

    def predict_from_command(self, v: float, omega: float, dt: float) -> None:
        x, y, th = self.belief.pose
        # Odometry has drift/noise.  This is the robot's internal guess.
        v_hat = v + self.rng.normal(0.0, abs(v) * self.cfg.xy_std_per_m + 0.002)
        omega_hat = omega + self.rng.normal(0.0, abs(omega) * self.cfg.theta_std_per_rad + 0.003)
        x += math.cos(th) * v_hat * dt
        y += math.sin(th) * v_hat * dt
        th = wrap_angle(th + omega_hat * dt)
        self.belief.pose[:] = [x, y, th]
        qx = self.cfg.process_xy + abs(v) * dt * self.cfg.xy_std_per_m
        qt = self.cfg.process_theta + abs(omega) * dt * self.cfg.theta_std_per_rad
        self.belief.covariance += np.diag([qx * qx, qx * qx, qt * qt])
        self._regularize_covariance()

    def update_with_landmarks(
        self,
        visible: list[Landmark],
        detection_range: float,
        sensor_pose: Pose | None = None,
    ) -> None:
        """Run a real EKF range/bearing update from visible known landmarks.

        `sensor_pose` is used only by the simulator to synthesize noisy sensor
        readings.  If omitted, measurements are generated around the current
        belief for compatibility with tests that do not have hidden truth.
        """
        self.last_landmark_count = len(visible)
        self.last_landmark_residual = 0.0
        if not visible:
            return

        # Fixed known landmarks: z = [range, bearing].  Measurements are noisy,
        # so the update corrects pose without perfect truth teleportation.
        base_range_std = float(self.cfg.landmark_range_std_m)
        bearing_std = math.radians(float(self.cfg.landmark_bearing_std_deg))
        max_xy_step = float(self.cfg.landmark_max_xy_correction_m)
        max_th_step = math.radians(float(self.cfg.landmark_max_theta_correction_deg))
        I = np.eye(3)
        residual_norms: list[float] = []

        # Update closest / most useful landmarks first.  HOME is an anchor but
        # should not collapse covariance unrealistically, so it uses same EKF math.
        sx, sy, sth = sensor_pose if sensor_pose is not None else self.belief.as_pose()
        ordered = sorted(visible, key=lambda lm: distance((sx, sy), lm.xy))
        for lm in ordered:
            # Simulated measurement from the sensor frame.
            true_dx = lm.xy[0] - sx
            true_dy = lm.xy[1] - sy
            true_r = max(1e-6, math.hypot(true_dx, true_dy))
            if true_r > detection_range + 1e-6:
                continue
            z_r = true_r + self.rng.normal(0.0, base_range_std + 0.012 * true_r)
            z_b = wrap_angle(math.atan2(true_dy, true_dx) - sth + self.rng.normal(0.0, bearing_std))

            x, y, th = self.belief.pose
            dx = lm.xy[0] - x
            dy = lm.xy[1] - y
            q = max(dx * dx + dy * dy, 1e-8)
            pred_r = max(math.sqrt(q), 1e-6)
            pred_b = wrap_angle(math.atan2(dy, dx) - th)
            residual = np.array([z_r - pred_r, wrap_angle(z_b - pred_b)], dtype=float)

            H = np.array(
                [
                    [-dx / pred_r, -dy / pred_r, 0.0],
                    [dy / q, -dx / q, -1.0],
                ],
                dtype=float,
            )
            range_std = base_range_std + 0.012 * pred_r
            R = np.diag([range_std * range_std, bearing_std * bearing_std])

            P = self.belief.covariance
            S = H @ P @ H.T + R
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

            # Robustly down-weight very inconsistent readings rather than
            # letting one bad association create a large pose jump.
            nis = float(residual.T @ Sinv @ residual)
            if nis > 9.0:
                scale = min(8.0, nis / 9.0)
                R = R * scale
                S = H @ P @ H.T + R
                try:
                    Sinv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    Sinv = np.linalg.pinv(S)

            K = P @ H.T @ Sinv
            delta = K @ residual
            xy_norm = float(np.linalg.norm(delta[:2]))
            if xy_norm > max_xy_step:
                delta[:2] *= max_xy_step / max(xy_norm, 1e-9)
            delta[2] = float(np.clip(delta[2], -max_th_step, max_th_step))

            self.belief.pose += delta
            self.belief.pose[2] = wrap_angle(float(self.belief.pose[2]))
            # Joseph form keeps covariance symmetric/positive in long runs.
            KH = K @ H
            self.belief.covariance = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
            self._regularize_covariance()
            residual_norms.append(float(abs(residual[0]) + abs(residual[1]) * max(1.0, pred_r)))

        if residual_norms:
            self.last_landmark_residual = float(np.median(residual_norms))

    def update_with_teammate_pose(
        self,
        teammate_est_pose: Pose,
        teammate_cov_trace: float,
        measured_range: float,
        measured_bearing: float,
    ) -> bool:
        """Fuse a teammate-based relative position observation.

        The measurement model is intentionally conservative: the robot observes
        a noisy relative range/bearing to a teammate, combines that with the
        teammate's reported global pose and covariance, and updates only its own
        x/y belief. Heading still comes from odometry, landmarks, and scan match.
        """
        self.last_teammate_update_count = 0
        self.last_teammate_residual = 0.0
        r = float(measured_range)
        if r <= 1e-6 or not np.isfinite(r):
            return False

        x, y, th = self.belief.pose
        bearing = float(measured_bearing)
        rel_angle = wrap_angle(float(th) + bearing)
        rel_world = np.array([math.cos(rel_angle) * r, math.sin(rel_angle) * r], dtype=float)
        z = np.array([float(teammate_est_pose[0]), float(teammate_est_pose[1])], dtype=float) - rel_world

        teammate_sigma = math.sqrt(max(0.0, float(teammate_cov_trace)) * 0.5) * float(self.cfg.teammate_covariance_scale)
        bearing_std = math.radians(float(self.cfg.teammate_bearing_std_deg))
        rel_sigma = float(self.cfg.teammate_range_std_m) + 0.015 * r + abs(r * bearing_std)
        sigma = max(0.05, rel_sigma + teammate_sigma)
        R = np.diag([sigma * sigma, sigma * sigma])
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        P = self.belief.covariance
        residual = z - self.belief.pose[:2]
        S = H @ P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)

        nis = float(residual.T @ Sinv @ residual)
        if nis > 9.0:
            scale = min(10.0, nis / 9.0)
            R = R * scale
            S = H @ P @ H.T + R
            try:
                Sinv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                Sinv = np.linalg.pinv(S)

        K = P @ H.T @ Sinv
        delta = K @ residual
        xy_norm = float(np.linalg.norm(delta[:2]))
        max_xy_step = float(self.cfg.teammate_max_xy_correction_m)
        if xy_norm > max_xy_step:
            delta[:2] *= max_xy_step / max(xy_norm, 1e-9)
        delta[2] = 0.0

        self.belief.pose += delta
        self.belief.pose[2] = wrap_angle(float(self.belief.pose[2]))
        I = np.eye(3)
        KH = K @ H
        self.belief.covariance = (I - KH) @ P @ (I - KH).T + K @ R @ K.T
        self._regularize_covariance()
        self.last_teammate_update_count = 1
        self.last_teammate_residual = float(np.linalg.norm(residual))
        return True

    def apply_lidar_correction(self, dx: float, dy: float, dtheta: float, confidence: float) -> None:
        """Apply bounded correlative scan-matching correction."""
        c = float(np.clip(confidence, 0.0, 1.0))
        self.last_lidar_match_confidence = c
        if c <= 0.0:
            return
        self.belief.pose[0] += self.cfg.lidar_xy_gain * c * float(dx)
        self.belief.pose[1] += self.cfg.lidar_xy_gain * c * float(dy)
        self.belief.pose[2] = wrap_angle(self.belief.pose[2] + self.cfg.lidar_theta_gain * c * float(dtheta))

        # LiDAR scan matching helps, but map alignment can be biased by past
        # drift, so covariance is only reduced moderately.
        shrink_xy = max(0.86, 1.0 - 0.10 * c)
        shrink_th = max(0.90, 1.0 - 0.07 * c)
        self.belief.covariance[:2, :2] *= shrink_xy
        self.belief.covariance[2, 2] *= shrink_th
        self._regularize_covariance()

    def quality(self, scan_consistency: float | None = None, landmark_count: int | None = None) -> float:
        """Pose quality for map insertion, not for goal selection.

        Covariance alone can become overconfident.  Therefore quality combines
        covariance, current scan-map agreement, and whether the estimate has a
        recent fixed landmark anchor.
        """
        # Map quality is driven by estimated pose uncertainty at the moment
        # the LiDAR cells are inserted.  Higher position/heading uncertainty
        # makes the mapped cells less trustworthy in the fused quality overlay.
        pos_sigma = math.sqrt(max(0.0, self.belief.cov_trace_xy))
        theta_sigma = math.sqrt(max(0.0, float(self.belief.covariance[2, 2])))
        pos_q = math.exp(-pos_sigma / 0.85)
        theta_q = math.exp(-theta_sigma / math.radians(18.0))
        cov_q = float(np.clip(pos_q * theta_q, 0.05, 1.0))
        if scan_consistency is None:
            scan_q = 0.78
        else:
            scan_q = float(np.clip(0.35 + 0.65 * scan_consistency, 0.15, 1.0))
        n_lm = self.last_landmark_count if landmark_count is None else int(landmark_count)
        landmark_q = 0.72 + 0.28 * min(1.0, n_lm / 2.0)
        lidar_q = 0.88 + 0.12 * float(np.clip(self.last_lidar_match_confidence, 0.0, 1.0))
        return float(np.clip(cov_q * scan_q * landmark_q * lidar_q, 0.05, 1.0))

    def _regularize_covariance(self) -> None:
        P = 0.5 * (self.belief.covariance + self.belief.covariance.T)
        floor_xy = float(self.cfg.covariance_floor_xy)
        floor_theta = float(self.cfg.covariance_floor_theta)
        max_xy = float(self.cfg.covariance_max_xy)
        max_theta = float(self.cfg.covariance_max_theta)
        # Keep diagonal bounded while preserving off-diagonal information.
        P[0, 0] = float(np.clip(P[0, 0], floor_xy, max_xy))
        P[1, 1] = float(np.clip(P[1, 1], floor_xy, max_xy))
        P[2, 2] = float(np.clip(P[2, 2], floor_theta, max_theta))
        # If numerical coupling is too large after clipping, damp it.
        for a, b in ((0, 1), (0, 2), (1, 2)):
            limit = 0.95 * math.sqrt(max(P[a, a], 1e-12) * max(P[b, b], 1e-12))
            P[a, b] = P[b, a] = float(np.clip(P[a, b], -limit, limit))
        self.belief.covariance = P


# ============================================================================
# src / planner.py
# ============================================================================

"""Clearance-aware grid A* planner over the robot's local LiDAR map."""
import heapq, math
from dataclasses import dataclass
import numpy as np
@dataclass
class PlanResult:
    path: list[Point]
    success: bool
    cost: float
    reason: str = ""
    min_clearance: float = 0.0

class GridPlanner:
    def __init__(self, cfg: PlanningConfig): self.cfg = cfg

    def plan(
        self,
        grid: OccupancyGrid,
        start: Point,
        goal: Point,
        passage_quality: np.ndarray | None = None,
        dynamic_obstacles: list[tuple[Point, float]] | None = None,
    ) -> PlanResult:
        start_cell, goal_cell = grid.world_to_cell(start), grid.world_to_cell(goal)
        if start_cell is None or goal_cell is None:
            return PlanResult([], False, math.inf, "start_or_goal_outside_map")
        traversable = grid.traversable_mask(self.cfg.inflation_radius_m)
        clearance = grid.clearance_map(max_radius_m=max(3.0, self.cfg.desired_clearance_m * 3.0))
        dynamic_blocked, dynamic_cost = self._dynamic_obstacle_fields(grid, dynamic_obstacles)
        if dynamic_blocked is not None:
            traversable = traversable & ~dynamic_blocked
        goal_cell = self._nearest_good_cell(traversable, clearance, goal_cell) or goal_cell
        if not traversable[start_cell[1], start_cell[0]]:
            start_cell = self._nearest_good_cell(traversable, clearance, start_cell) or start_cell
        if not traversable[start_cell[1], start_cell[0]] or not traversable[goal_cell[1], goal_cell[0]]:
            return PlanResult([], False, math.inf, "no_traversable_start_or_goal")
        prob, known = grid.probability(), grid.known_mask()
        nbrs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),(1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2))]
        heap, counter = [(0.0, 0, start_cell)], 0
        came: dict[tuple[int,int], tuple[int,int]] = {}
        g = {start_cell: 0.0}
        expanded = 0
        while heap and expanded < self.cfg.max_a_star_expansions:
            _, _, cur = heapq.heappop(heap); expanded += 1
            if cur == goal_cell:
                cells=[cur]
                while cur in came:
                    cur=came[cur]; cells.append(cur)
                cells.reverse(); path=[grid.cell_to_world(c) for c in cells]
                return PlanResult(path, True, g[goal_cell], "ok", min(float(clearance[j,i]) for i,j in cells))
            ci,cj=cur
            for di,dj,step in nbrs:
                ni,nj=ci+di,cj+dj
                if not (0 <= ni < grid.nx and 0 <= nj < grid.ny) or not traversable[nj,ni]: continue
                if di and dj and (not traversable[cj,ni] or not traversable[nj,ci]): continue
                unknown_cost = self.cfg.unknown_penalty if not known[nj,ni] else 0.0
                occ_soft = max(0.0, float(prob[nj,ni] - 0.5)) * 2.0
                cl = float(clearance[nj,ni])
                deficit = max(0.0, self.cfg.desired_clearance_m - cl) / max(1e-6, self.cfg.desired_clearance_m)
                clearance_cost = self.cfg.clearance_cost_weight * deficit * deficit + (3.0 if cl < self.cfg.critical_clearance_m else 0.0)
                passage_cost = 0.0
                if passage_quality is not None:
                    passage_score = float(np.clip(passage_quality[nj, ni], 0.0, 1.0))
                    passage_cost = self.cfg.passage_safety_cost_weight * (1.0 - passage_score) ** 2
                dynamic_penalty = float(dynamic_cost[nj, ni]) if dynamic_cost is not None else 0.0
                new_g = g[cur] + step * grid.res * (1.0 + unknown_cost + occ_soft + clearance_cost + passage_cost + dynamic_penalty)
                nb=(ni,nj)
                if new_g < g.get(nb, math.inf):
                    came[nb]=cur; g[nb]=new_g; counter += 1
                    heapq.heappush(heap, (new_g + distance(grid.cell_to_world(nb), grid.cell_to_world(goal_cell)), counter, nb))
        return PlanResult([], False, math.inf, "a_star_failed")

    def _dynamic_obstacle_fields(
        self,
        grid: OccupancyGrid,
        dynamic_obstacles: list[tuple[Point, float]] | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not dynamic_obstacles:
            return None, None
        blocked = np.zeros((grid.ny, grid.nx), dtype=bool)
        cost = np.zeros((grid.ny, grid.nx), dtype=float)
        soft = max(0.0, float(self.cfg.dynamic_obstacle_soft_margin_m))
        for p, radius_m in dynamic_obstacles:
            cell = grid.world_to_cell(p)
            if cell is None:
                continue
            hard = max(0.0, float(radius_m))
            total = hard + soft
            radius_cells = max(1, int(math.ceil(total / grid.res)))
            ci, cj = cell
            y0, y1 = max(0, cj - radius_cells), min(grid.ny, cj + radius_cells + 1)
            x0, x1 = max(0, ci - radius_cells), min(grid.nx, ci + radius_cells + 1)
            for j in range(y0, y1):
                for i in range(x0, x1):
                    q = grid.cell_to_world((i, j))
                    d = distance(p, q)
                    if d <= hard:
                        blocked[j, i] = True
                    elif soft > 1e-9 and d <= total:
                        t = (total - d) / soft
                        cost[j, i] = max(cost[j, i], self.cfg.dynamic_obstacle_cost_weight * t * t)
        return blocked, cost

    def _nearest_good_cell(self, traversable: np.ndarray, clearance: np.ndarray, cell: tuple[int,int], radius: int = 14) -> tuple[int,int] | None:
        ci,cj=cell; ny,nx=traversable.shape
        min_clearance = min(self.cfg.safe_approach_min_clearance_m, self.cfg.desired_clearance_m)
        if 0 <= cj < ny and 0 <= ci < nx and traversable[cj,ci] and float(clearance[cj,ci]) >= min_clearance:
            return cell
        best=None; best_score=-math.inf
        for r in range(1, radius+1):
            y0,y1=max(0,cj-r),min(ny,cj+r+1); x0,x1=max(0,ci-r),min(nx,ci+r+1)
            ys,xs=np.nonzero(traversable[y0:y1,x0:x1])
            for yy,xx in zip(ys,xs):
                c=(x0+int(xx), y0+int(yy)); d2=(c[0]-ci)**2+(c[1]-cj)**2
                if float(clearance[c[1],c[0]]) < min_clearance:
                    continue
                score=float(clearance[c[1],c[0]]) - 0.04*d2
                if score > best_score: best_score=score; best=c
            if best is not None: return best
        return None


# ============================================================================
# src / robot.py
# ============================================================================

"""Robot agent for the clean Search-CAGE baseline.

Every robot plans from its own communication-limited knowledge map, its EKF
pose estimate, and packet-received teammate intent.  Ground truth is used only
by the simulator to produce sensing/collision/evaluation.
"""
import math
from dataclasses import dataclass, field
import numpy as np
@dataclass
class TargetReport:
    detected: bool = False; xy: Point | None = None; confidence: float = 0.0
    source_robot: int = -1; time_s: float = 0.0; reported_home: bool = False

@dataclass
class RobotPacket:
    sender_id: int; time_s: float; map_digest: dict; graph_digest: dict; target_report: dict | None
    task: str; current_goal: Point | None; current_path_digest: list[Point]; visited_digest: list[Point]
    # Full downsampled estimated trajectory from HOME to current position.
    # This is persistent team-history knowledge, unlike the short current path/visit digest.
    trajectory_digest: list[Point]
    estimated_pose: tuple[float, float, float]; pose_cov_trace: float
    assigned_region_id: tuple[int, int] | None = None
    assigned_region_center: Point | None = None
    assigned_region_radius: float = 0.0
    assigned_region_score: float = 0.0

@dataclass
class RobotStatus:
    task: str = "INIT"; planning_source: str = "knowledge map + EKF pose estimate"
    note: str = ""; goal: Point | None = None; last_plan_success: bool = False; last_plan_reason: str = ""
    last_path_min_clearance: float = 0.0; reward_breakdown: dict[str, float] = field(default_factory=dict)

@dataclass
class CoarseRegion:
    region_id: tuple[int, int]
    center: Point
    radius: float
    unknown_cells: int
    known_free_cells: int
    frontier_support: int
    score: float

@dataclass
class MacroActionCandidate:
    goal: Point | None
    task: str
    reason: str
    breakdown: dict[str, float]

@dataclass
class MacroActionScore:
    task: str
    goal: Point | None
    reason: str
    score: float
    terms: dict[str, float]

class RobotAgent:
    def __init__(self, robot_id: int, initial_pose: Pose, cfg: AppConfig, world: World, rng: np.random.Generator):
        self.id=robot_id; self.cfg=cfg; self.rng=rng
        self.true_pose=np.array(initial_pose,dtype=float); self.true_path=[(float(initial_pose[0]),float(initial_pose[1]))]
        self.estimator=PoseEstimator(initial_pose,cfg.motion,rng); self.lidar=LidarSensor(cfg.lidar,rng)
        # self_map contains only this robot's own LiDAR observations.
        # knowledge_map contains everything this robot knows: self_map plus
        # teammate/relay map digests received through communication.  Existing
        # planner/UI code uses robot.map, so keep it as the knowledge map.
        self.self_map=OccupancyGrid(world.width,world.height,cfg.mapping)
        self.knowledge_map=OccupancyGrid(world.width,world.height,cfg.mapping)
        self.map=self.knowledge_map
        self.graph=RouteGraph(cfg.cage.edge_merge_distance)
        self.home_node=self.graph.add_node(world.home,kind="home",confidence=1.0,allow_merge=False)
        self.home_xy=world.home
        self.search_prior_xy=self._sector_prior_point(world)
        self.target_belief=np.ones((self.map.ny,self.map.nx),dtype=float)
        self._normalize_target_belief()
        self.last_graph_node=self.home_node; self.last_keypoint_xy=world.home
        self.scan: LidarScan | None=None; self.assessment=LidarAssessment(); self.planner=GridPlanner(cfg.planning)
        self.path=[]; self.path_index=0; self.last_replan_time=-999.0; self.current_goal=None; self.current_task="SEARCH"
        self.goal_commit_start=-999.0; self.goal_commit_score=-math.inf; self.best_goal_distance=math.inf; self.last_goal_progress_time=0.0
        self.status=RobotStatus(task="SEARCH"); self.target=TargetReport()
        self.force_return_home=False; self.last_scan_match_time=-999.0
        self.known_teammate_goals={}; self.known_teammate_paths={}; self.known_teammate_visits={}; self.known_teammate_tasks={}
        self.known_teammate_pose={}; self.known_teammate_cov={}; self.known_teammate_last_seen={}
        # Persistent full-from-HOME teammate trajectory memory.  These paths are
        # downsampled estimated trajectories and are not deleted with short-term
        # intent expiry; they describe where this robot believes teammates have been.
        self.known_teammate_trajectories={}; self.known_teammate_trajectory_time={}
        # LOS-realistic coarse exploration intent. A robot only knows teammate
        # regions after receiving packets through the existing LOS communication.
        self.assigned_region: CoarseRegion | None = None
        self.assigned_region_start_time: float = -999.0
        self.known_teammate_regions: dict[int, dict] = {}
        self.visit_history=[self.est_xy]
        self.trajectory_from_home=[self.est_xy]
        self.failed_goal_memory=[]
        # Target-roundtrip state.  Once a target report is known, every robot
        # tries to reach the target from its own current location, records the
        # route attempt, then returns HOME and uploads route/map evidence.
        self.target_reached=False
        self.completed_target_roundtrip=False
        self.target_route_trace=[]
        self.return_route_trace=[]
        self.target_reached_time=-999.0
        self.completed_roundtrip_time=-999.0
        self.route_candidate_uploaded=False
        self.last_command=(0.0,0.0); self.last_pose_quality=1.0; self.best_routes=[]; self.received_packets=0; self.blocked_events=0; self.last_home_full_upload_time=-999.0
        self.last_mdp_score=0.0
        self.last_mdp_candidates:list[MacroActionScore]=[]

    @property
    def est_pose(self)->Pose: return self.estimator.belief.as_pose()
    @property
    def est_xy(self)->Point: return self.estimator.belief.xy
    @property
    def cov_trace(self)->float: return self.estimator.belief.cov_trace_xy

    def _sector_prior_point(self, world: World)->Point:
        hx,hy=world.home
        far=(world.width-world.cfg.world_margin, world.height-world.cfg.world_margin)
        base=angle_to(world.home,far)
        n=max(1,self.cfg.robot.count); spread=math.radians(78.0)
        offset=0.0 if n==1 else (self.id/(n-1)-0.5)*spread
        ang=base+offset; dx,dy=math.cos(ang),math.sin(ang)
        margin=max(0.8,world.cfg.world_margin*0.5); ts=[]
        if dx>1e-6: ts.append((world.width-margin-hx)/dx)
        elif dx<-1e-6: ts.append((margin-hx)/dx)
        if dy>1e-6: ts.append((world.height-margin-hy)/dy)
        elif dy<-1e-6: ts.append((margin-hy)/dy)
        t=max(1.0,min(v for v in ts if v>0)) if any(v>0 for v in ts) else max(world.width,world.height)*0.6
        return (float(min(world.width-margin,max(margin,hx+dx*t))),float(min(world.height-margin,max(margin,hy+dy*t))))

    def step_predict_and_move(self, world: World, dt: float, peer_poses: list[Pose] | None = None) -> None:
        v,omega=self.last_command; x,y,th=self.true_pose
        new_th=wrap_angle(th+omega*dt); cand=(float(x+math.cos(new_th)*v*dt), float(y+math.sin(new_th)*v*dt))
        executed_v, executed_omega = v, omega
        collision_free=self._peer_collision_free(cand,peer_poses or [])
        if world.is_free(cand, margin=self.cfg.robot.radius) and collision_free: self.true_pose[:]=[cand[0],cand[1],new_th]
        else:
            self.blocked_events+=1; self.path=[]; self.status.note="true_robot_collision_prevented_by_sim" if not collision_free else "true_collision_prevented_by_sim"; self.last_command=(0.0,0.0)
            executed_v, executed_omega = 0.0, 0.0
        self._append_true_path_sample()
        self.estimator.predict_from_command(executed_v,executed_omega,dt)

    def _peer_collision_free(self,cand:Point,peer_poses:list[Pose])->bool:
        min_sep=2.0*float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m)
        return all(distance(cand,(float(p[0]),float(p[1])))>=min_sep for p in peer_poses)

    def _append_true_path_sample(self)->None:
        xy=(float(self.true_pose[0]),float(self.true_pose[1]))
        if self.true_path and distance(xy,self.true_path[-1])<self.cfg.robot.true_path_spacing_m:
            return
        self.true_path.append(xy)
        if len(self.true_path)>self.cfg.robot.max_true_path_points:
            del self.true_path[:-self.cfg.robot.max_true_path_points]

    def sense_update_map_and_belief(self, world: World, time_s: float) -> None:
        # Landmarks are sensed from hidden truth, but the estimator receives only
        # noisy range/bearing measurements to known beacon locations.
        visible_landmarks = world.visible_landmarks(tuple(self.true_pose), self.cfg.world.landmark_detection_range)
        self.estimator.update_with_landmarks(
            visible_landmarks,
            self.cfg.world.landmark_detection_range,
            sensor_pose=tuple(self.true_pose),
        )
        self.scan = self.lidar.sense(world, tuple(self.true_pose))
        self._maybe_apply_lidar_scan_match(time_s)

        # Assess scan-map agreement on the map BEFORE inserting this scan; this
        # avoids falsely high confidence from scoring a scan against itself.
        prev_assessment = None if self.assessment.decision_note == "init" else self.assessment
        prev = None if prev_assessment is None else prev_assessment.consistency
        pre_assessment = assess_lidar(self.self_map, self.est_pose, self.scan, self.cfg.lidar, self.cfg.assessment, prev)
        self.last_pose_quality = self.estimator.quality(
            scan_consistency=pre_assessment.consistency,
            landmark_count=len(visible_landmarks),
        )
        self.self_map.update_from_lidar(self.est_pose, self.scan, self.last_pose_quality, self.id, time_s)
        self.knowledge_map.update_from_lidar(self.est_pose, self.scan, self.last_pose_quality, self.id, time_s)
        # Navigation/planning uses the updated knowledge map, but pose-quality
        # scoring above used the self map to avoid scoring against just-received
        # teammate cells.
        raw_assessment = assess_lidar(self.knowledge_map, self.est_pose, self.scan, self.cfg.lidar, self.cfg.assessment, pre_assessment.consistency)
        self.assessment = self._smooth_lidar_assessment(raw_assessment, prev_assessment)
        self._update_visit_history()
        self._detect_target(world, time_s)
        self._update_target_belief(time_s)
        self._update_route_graph(time_s)

    def update_localization_from_teammate(self, teammate:"RobotAgent", world:World, time_s:float)->bool:
        if teammate.id==self.id:
            return False
        my_true=(float(self.true_pose[0]),float(self.true_pose[1]))
        other_true=(float(teammate.true_pose[0]),float(teammate.true_pose[1]))
        true_range=distance(my_true,other_true)
        if true_range>float(self.cfg.motion.teammate_localization_range_m):
            return False
        if not world.segment_free(my_true,other_true,margin=min(0.05,float(self.cfg.robot.radius)*0.25)):
            return False
        bearing=wrap_angle(angle_to(my_true,other_true)-float(self.true_pose[2]))
        z_range=max(0.02,true_range+self.rng.normal(0.0,float(self.cfg.motion.teammate_range_std_m)+0.012*true_range))
        z_bearing=wrap_angle(bearing+self.rng.normal(0.0,math.radians(float(self.cfg.motion.teammate_bearing_std_deg))))
        return self.estimator.update_with_teammate_pose(teammate.est_pose,teammate.cov_trace,z_range,z_bearing)

    def _smooth_lidar_assessment(self,new:LidarAssessment,prev:LidarAssessment|None)->LidarAssessment:
        if prev is None:
            return new
        a=float(self.cfg.assessment.clearance_smoothing)
        front=float(a*new.front_clearance+(1.0-a)*prev.front_clearance)
        left=float(a*new.left_clearance+(1.0-a)*prev.left_clearance)
        right=float(a*new.right_clearance+(1.0-a)*prev.right_clearance)
        angle_alpha=float(self.cfg.assessment.open_angle_smoothing)
        raw_delta=wrap_angle(new.best_open_angle-prev.best_open_angle)
        best_open=float(wrap_angle(prev.best_open_angle+angle_alpha*raw_delta))
        block_threshold=float(self.cfg.lidar.blocked_forward_distance)
        release_threshold=block_threshold+float(self.cfg.assessment.blocked_hysteresis_m)
        if prev.blocked_forward:
            blocked=front<=release_threshold
        else:
            blocked=front<=block_threshold
        note=new.decision_note
        if blocked:
            note="blocked_forward_by_lidar"
        return LidarAssessment(
            consistency=float(new.consistency),
            mismatch_fraction=float(new.mismatch_fraction),
            front_clearance=front,
            left_clearance=left,
            right_clearance=right,
            blocked_forward=bool(blocked),
            open_sector_count=int(new.open_sector_count),
            best_open_angle=best_open,
            decision_note=note,
        )

    def _maybe_apply_lidar_scan_match(self, time_s: float) -> None:
        if self.scan is None:
            return
        if time_s - self.last_scan_match_time < self.cfg.motion.lidar_match_period_s:
            return
        self.last_scan_match_time = time_s
        if np.count_nonzero(self.self_map.quality > 0.05) < 35:
            return

        stride = max(4, len(self.scan.angles) // 14)
        angles = self.scan.angles[::stride]
        ranges = self.scan.ranges[::stride]
        hits = self.scan.hit[::stride]
        if len(angles) < 8:
            return

        base_pose = self.est_pose
        max_xy = float(self.cfg.motion.lidar_match_max_xy_m)
        max_th = math.radians(float(self.cfg.motion.lidar_match_max_theta_deg))
        th = base_pose[2]
        forward = np.array([math.cos(th), math.sin(th)], dtype=float)
        lateral = np.array([-math.sin(th), math.cos(th)], dtype=float)

        def scan_error(pose: Pose, regularization: float = 0.0) -> float:
            pred = self.self_map.predict_scan_ranges(pose, angles, self.cfg.lidar.range)
            active = hits | (pred < self.cfg.lidar.range * 0.96)
            if not np.any(active):
                active = np.ones_like(pred, dtype=bool)
            err = np.abs(pred[active] - ranges[active])
            if len(err) == 0:
                return float("inf")
            # Robust metric: median handles single-ray noise, mean catches broad mismatch.
            return float(np.median(err) + 0.30 * np.mean(err) + regularization)

        base_err = scan_error(base_pose)
        best = (base_err, 0.0, 0.0, 0.0)
        lin_steps = (-1.0, -0.5, 0.0, 0.5, 1.0)
        th_steps = (-1.0, 0.0, 1.0)
        for fs in lin_steps:
            for ls in lin_steps:
                delta = forward * (fs * max_xy) + lateral * (ls * max_xy)
                xy_reg = 0.05 * math.hypot(fs, ls)
                for ts in th_steps:
                    dth = ts * max_th
                    pose = (
                        base_pose[0] + float(delta[0]),
                        base_pose[1] + float(delta[1]),
                        wrap_angle(base_pose[2] + dth),
                    )
                    e = scan_error(pose, regularization=xy_reg + 0.035 * abs(ts))
                    if e < best[0]:
                        best = (e, float(delta[0]), float(delta[1]), float(dth))

        improvement = base_err - best[0]
        # Require a meaningful improvement; otherwise the scan matcher could
        # chase noise or reinforce a drifted self-map.
        if improvement > 0.030:
            confidence = float(np.clip(improvement / 0.42, 0.0, 1.0)) * self.estimator.quality()
            self.estimator.apply_lidar_correction(best[1], best[2], best[3], confidence)

    def _update_visit_history(self)->None:
        xy=self.est_xy
        if not self.visit_history or distance(xy,self.visit_history[-1])>=self.cfg.robot.visit_history_spacing_m:
            self.visit_history.append(xy); self.visit_history=self.visit_history[-self.cfg.robot.max_visit_history:]
        if not self.trajectory_from_home or distance(xy,self.trajectory_from_home[-1])>=self.cfg.robot.trajectory_history_spacing_m:
            self.trajectory_from_home.append((float(xy[0]),float(xy[1])))
            if len(self.trajectory_from_home)>self.cfg.robot.max_trajectory_history_points:
                # Preserve the initial HOME-side point and keep the recent history.
                keep=max(2,self.cfg.robot.max_trajectory_history_points)
                self.trajectory_from_home=[self.trajectory_from_home[0]]+self.trajectory_from_home[-(keep-1):]
        self._append_roundtrip_trace(xy)

    def _activate_target_guidance(self,time_s:float)->None:
        # Start a route attempt only once.  A later, higher-confidence target
        # report may update self.target.xy, but the route should still record
        # the robot's path from when target knowledge first became available.
        if not self.target_route_trace and not self.target_reached:
            self.target_route_trace=[(float(self.est_xy[0]),float(self.est_xy[1]))]

    def _append_roundtrip_trace(self,xy:Point)->None:
        if not self.target.detected or self.completed_target_roundtrip:
            return
        trace = self.return_route_trace if self.target_reached else self.target_route_trace
        if not trace or distance(xy,trace[-1])>=self.cfg.robot.visit_history_spacing_m:
            trace.append((float(xy[0]),float(xy[1])))
            max_len=max(80,self.cfg.robot.max_true_path_points)
            if len(trace)>max_len:
                del trace[:-max_len]

    def mark_target_reached(self,time_s:float)->None:
        if self.target_reached:
            return
        self.target_reached=True
        self.target_reached_time=float(time_s)
        self.current_goal=None
        self.path=[]
        self.path_index=0
        if not self.target_route_trace:
            self.target_route_trace=[(float(self.est_xy[0]),float(self.est_xy[1]))]
        self.target_route_trace.append((float(self.est_xy[0]),float(self.est_xy[1])))
        self.return_route_trace=[(float(self.est_xy[0]),float(self.est_xy[1]))]
        self.status.note=f"target_reached_by_R{self.id}_returning_home"

    def mark_target_roundtrip_complete(self,time_s:float)->None:
        if self.completed_target_roundtrip:
            return
        self.completed_target_roundtrip=True
        self.completed_roundtrip_time=float(time_s)
        self.current_goal=None
        self.path=[]
        self.path_index=0
        if self.return_route_trace:
            self.return_route_trace.append((float(self.est_xy[0]),float(self.est_xy[1])))
        self.status.note=f"target_roundtrip_complete_R{self.id}"

    def target_route_summary(self)->dict|None:
        if not self.target_reached or len(self.target_route_trace)<2:
            return None
        path=[(float(x),float(y)) for x,y in self.target_route_trace]
        ret=[(float(x),float(y)) for x,y in self.return_route_trace]
        def path_len(pts:list[Point])->float:
            return float(sum(distance(a,b) for a,b in zip(pts[:-1],pts[1:]))) if len(pts)>=2 else 0.0
        cells=[]; seen=set()
        for pts in (path,ret):
            for a,b in zip(pts[:-1],pts[1:]):
                ca=self.self_map.world_to_cell(a); cb=self.self_map.world_to_cell(b)
                if ca is None or cb is None: continue
                for c in self.self_map._bresenham(ca,cb):
                    if c not in seen:
                        seen.add(c); cells.append(c)
        known=self.self_map.known_mask()
        clearance=self.self_map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*3.0))
        q_vals=[]; cl_vals=[]; unknown=0
        for i,j in cells:
            if not known[j,i]: unknown+=1
            q_vals.append(float(self.self_map.quality[j,i]))
            cl_vals.append(float(clearance[j,i]))
        total=max(1,len(cells))
        return {
            "robot_id":int(self.id),
            "target_reached":bool(self.target_reached),
            "roundtrip_complete":bool(self.completed_target_roundtrip),
            "target_reached_time":float(self.target_reached_time),
            "completed_time":float(self.completed_roundtrip_time),
            "target_xy":[float(self.target.xy[0]),float(self.target.xy[1])] if self.target.xy else None,
            "route_to_target":path,
            "return_route":ret,
            "route_length":path_len(path),
            "return_length":path_len(ret),
            "mean_quality":float(np.mean(q_vals)) if q_vals else 0.0,
            "min_clearance":float(min(cl_vals)) if cl_vals else 0.0,
            "unknown_fraction":float(unknown)/float(total),
        }

    def _target_prior_support(self)->np.ndarray:
        known=self.map.known_mask()
        occupied=self.map.occupied_mask()
        support=np.where(occupied,0.0,0.35+0.65*(~known).astype(float))
        prior_cell=self.map.world_to_cell(self.search_prior_xy)
        if prior_cell is not None:
            jj,ii=np.indices(support.shape)
            wx=(ii+0.5)*self.map.res
            wy=(jj+0.5)*self.map.res
            sigma=max(self.map.width_m,self.map.height_m)*0.55
            dx=wx-float(self.search_prior_xy[0])
            dy=wy-float(self.search_prior_xy[1])
            support*=0.55+0.45*np.exp(-(dx*dx+dy*dy)/(2.0*sigma*sigma))
        return support

    def _normalize_target_belief(self)->None:
        support=self._target_prior_support()
        belief=np.asarray(self.target_belief,dtype=float)
        belief=np.where(support>0.0,np.maximum(0.0,belief),0.0)
        total=float(np.sum(belief))
        if total<=1e-12:
            belief=support
            total=float(np.sum(belief))
        if total<=1e-12:
            belief=np.ones_like(support,dtype=float)
            total=float(np.sum(belief))
        self.target_belief=belief/total

    def _visible_cells_from_scan(self)->set[tuple[int,int]]:
        visible:set[tuple[int,int]]=set()
        if self.scan is None:
            return visible
        start=self.map.world_to_cell(self.est_xy)
        if start is None:
            return visible
        ex,ey,eth=self.est_pose
        max_r=float(self.cfg.lidar.range)*float(self.cfg.mdp.target_belief_sensor_fraction)
        for a_local,rng,hit in zip(self.scan.angles,self.scan.ranges,self.scan.hit):
            rr=min(float(rng),max_r)
            if rr<=self.map.res:
                continue
            a=eth+float(a_local)
            end=(float(ex+math.cos(a)*rr),float(ey+math.sin(a)*rr))
            end_cell=self.map.world_to_cell(end)
            if end_cell is None:
                continue
            cells=self.map._bresenham(start,end_cell)
            usable=cells[1:-1] if bool(hit) and len(cells)>2 else cells[1:]
            for c in usable:
                visible.add(c)
        return visible

    def _update_target_belief(self,time_s:float)->None:
        if not bool(self.cfg.mdp.enabled):
            return
        support=self._target_prior_support()
        if self.target.detected and self.target.xy is not None:
            jj,ii=np.indices(self.target_belief.shape)
            wx=(ii+0.5)*self.map.res
            wy=(jj+0.5)*self.map.res
            sigma=max(self.map.res,float(self.cfg.mdp.target_belief_detection_sigma_m))
            dx=wx-float(self.target.xy[0])
            dy=wy-float(self.target.xy[1])
            detected=np.exp(-(dx*dx+dy*dy)/(2.0*sigma*sigma))*support
            total=float(np.sum(detected))
            if total>1e-12:
                detected=detected/total
                alpha=float(np.clip(0.65+0.30*self.target.confidence,0.65,0.96))
                self.target_belief=(1.0-alpha)*self.target_belief+alpha*detected
            self._normalize_target_belief()
            return

        visible=self._visible_cells_from_scan()
        if visible:
            miss=float(self.cfg.mdp.target_belief_miss_likelihood)
            for i,j in visible:
                self.target_belief[j,i]*=miss
        occupied=self.map.occupied_mask()
        self.target_belief=np.where(occupied,0.0,self.target_belief)
        mix=float(self.cfg.mdp.target_belief_uniform_mix)
        if mix>0.0:
            support_total=float(np.sum(support))
            if support_total>1e-12:
                self.target_belief=(1.0-mix)*self.target_belief+mix*(support/support_total)
        self._normalize_target_belief()

    def target_belief_entropy(self)->float:
        p=self.target_belief[self.target_belief>1e-15]
        if p.size<=1:
            return 0.0
        ent=-float(np.sum(p*np.log(p)))
        return float(np.clip(ent/max(1e-12,math.log(float(self.target_belief.size))),0.0,1.0))

    def _target_belief_gain(self,viewpoint:Point)->float:
        cell=self.map.world_to_cell(viewpoint)
        if cell is None:
            return 0.0
        ci,cj=cell
        radius=max(1,int(math.ceil(float(self.cfg.lidar.range)*float(self.cfg.mdp.target_belief_sensor_fraction)/self.map.res)))
        y0,y1=max(0,cj-radius),min(self.map.ny,cj+radius+1)
        x0,x1=max(0,ci-radius),min(self.map.nx,ci+radius+1)
        sub=self.target_belief[y0:y1,x0:x1]
        if sub.size==0:
            return 0.0
        jj,ii=np.indices(sub.shape)
        dx=(ii+x0-ci)*self.map.res
        dy=(jj+y0-cj)*self.map.res
        dist=np.sqrt(dx*dx+dy*dy)
        mask=dist<=float(self.cfg.lidar.range)*float(self.cfg.mdp.target_belief_sensor_fraction)
        weighted=sub*mask*np.exp(-0.10*dist)
        return float(np.sum(weighted))

    def _detect_target(self, world: World, time_s: float)->None:
        if self.target.detected or not world.target_visible(tuple(self.true_pose), self.cfg.lidar.range): return
        true_xy=(float(self.true_pose[0]),float(self.true_pose[1])); brg=angle_to(true_xy,world.target)-float(self.true_pose[2]); rr=distance(true_xy,world.target)
        r=max(0.05,rr+self.rng.normal(0.0,0.06)); b=brg+self.rng.normal(0.0,math.radians(2.0))
        ex,ey,eth=self.est_pose; est_target=(float(ex+math.cos(eth+b)*r), float(ey+math.sin(eth+b)*r))
        conf=float(np.clip(self.assessment.consistency*self.last_pose_quality,0.1,1.0))
        self.target=TargetReport(True,est_target,conf,self.id,time_s,False)
        self._activate_target_guidance(time_s)
        tid=self.graph.add_node(est_target,kind="target",confidence=conf,allow_merge=True); self.graph.target_id=tid
        clearance=max(0.05,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        self.graph.add_or_update_edge(self.last_graph_node,tid,clearance=clearance,consistency=max(0.05,self.assessment.consistency),pose_quality=self.last_pose_quality,robot_id=self.id,time_s=time_s,success=True)
        self.status.note=f"target_detected_by_R{self.id}"

    def _update_route_graph(self,time_s:float)->None:
        xy=self.est_xy
        if distance(xy,self.last_keypoint_xy)<self.cfg.robot.keypoint_spacing: return
        kind="anchor" if self.assessment.consistency>0.68 and self.last_pose_quality>0.45 else "keypoint"
        node=self.graph.add_node(xy,kind=kind,confidence=max(self.assessment.consistency,self.last_pose_quality))
        clearance=max(0.05,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        self.graph.add_or_update_edge(self.last_graph_node,node,clearance=clearance,consistency=max(0.05,self.assessment.consistency),pose_quality=self.last_pose_quality,robot_id=self.id,time_s=time_s,success=not self.assessment.blocked_forward)
        self.last_graph_node=node; self.last_keypoint_xy=xy

    def receive_packet(self, packet: RobotPacket)->None:
        if packet.sender_id==self.id: return
        self.received_packets+=1
        # Robot-to-robot packets can carry communication-limited knowledge maps.
        # Merge them into knowledge_map only; self_map remains pure own LiDAR.
        if packet.sender_id >= 0 and packet.map_digest:
            self.knowledge_map.merge_from_digest(packet.map_digest, combine_sources=True)
        self.graph.merge_from_digest(packet.graph_digest)
        if packet.sender_id>=0:
            self.known_teammate_pose[packet.sender_id]=packet.estimated_pose; self.known_teammate_cov[packet.sender_id]=float(packet.pose_cov_trace)
            self.known_teammate_last_seen[packet.sender_id]=float(packet.time_s); self.known_teammate_tasks[packet.sender_id]=packet.task
            if packet.current_goal is not None: self.known_teammate_goals[packet.sender_id]=(float(packet.current_goal[0]),float(packet.current_goal[1]))
            else: self.known_teammate_goals.pop(packet.sender_id,None)
            self.known_teammate_paths[packet.sender_id]=[(float(x),float(y)) for x,y in packet.current_path_digest]
            self.known_teammate_visits[packet.sender_id]=[(float(x),float(y)) for x,y in packet.visited_digest]
            if packet.trajectory_digest:
                self.known_teammate_trajectories[packet.sender_id]=[(float(x),float(y)) for x,y in packet.trajectory_digest]
                self.known_teammate_trajectory_time[packet.sender_id]=float(packet.time_s)
            if packet.assigned_region_center is not None and packet.assigned_region_id is not None:
                cx,cy=packet.assigned_region_center
                self.known_teammate_regions[packet.sender_id]={
                    "region_id": tuple(packet.assigned_region_id),
                    "center": (float(cx),float(cy)),
                    "radius": float(packet.assigned_region_radius),
                    "score": float(packet.assigned_region_score),
                    "time_s": float(packet.time_s),
                    "task": str(packet.task),
                }
            else:
                self.known_teammate_regions.pop(packet.sender_id,None)
        target_share_allowed = packet.sender_id == -1 or bool(self.cfg.target_reporting.allow_robot_to_robot_target_share)
        if target_share_allowed and packet.target_report and packet.target_report.get("detected"):
            tr=packet.target_report; conf=float(tr.get("confidence",0.0))
            if not self.target.detected or conf>self.target.confidence:
                xy=tuple(tr["xy"]); self.target=TargetReport(True,(float(xy[0]),float(xy[1])),conf,int(tr.get("source_robot",packet.sender_id)),float(tr.get("time_s",packet.time_s)),bool(tr.get("reported_home",False)))
                self._activate_target_guidance(float(tr.get("time_s",packet.time_s)))
                tid=self.graph.add_node(self.target.xy,kind="target",confidence=conf,allow_merge=True); self.graph.target_id=tid
            elif bool(tr.get("reported_home",False)):
                self.target.reported_home=True
        self._expire_stale_teammate_intent(packet.time_s)

    def make_packet(self,time_s:float,include_map_digest:bool=True,max_map_cells:int|None=650,map_source:str="knowledge")->RobotPacket:
        """Create a packet. Robot-to-robot packets use knowledge_map; HOME uploads use self_map."""
        target_dict=None
        if self.target.detected and self.target.xy is not None:
            target_dict={"detected":True,"xy":[float(self.target.xy[0]),float(self.target.xy[1])],"confidence":float(self.target.confidence),"source_robot":int(self.target.source_robot),"time_s":float(self.target.time_s),"reported_home":bool(self.target.reported_home)}
        map_digest={}
        if include_map_digest:
            if map_source=="self":
                map_obj=self.self_map
            elif map_source=="knowledge":
                map_obj=self.knowledge_map
            else:
                raise ValueError(f"unknown map_source {map_source!r}")
            map_digest=map_obj.make_digest(self.id,time_s,max_cells=max_map_cells)
        region_id=None; region_center=None; region_radius=0.0; region_score=0.0
        if self.assigned_region is not None and self.current_task in {"SEARCH_HIER_NBV","SEARCH_NBV","SEARCH_TARGET_BELIEF","DEPLOY_FROM_HOME"}:
            region_id=tuple(self.assigned_region.region_id)
            region_center=(float(self.assigned_region.center[0]),float(self.assigned_region.center[1]))
            region_radius=float(self.assigned_region.radius)
            region_score=float(self.assigned_region.score)
        return RobotPacket(self.id,float(time_s),map_digest,self.graph.make_digest(self.id,time_s),target_dict,self.current_task,self.current_goal,self._path_digest(),self._visited_digest(),self._trajectory_digest(),self.est_pose,self.cov_trace,region_id,region_center,region_radius,region_score)

    def make_full_knowledge_packet(self,time_s:float)->RobotPacket:
        return self.make_packet(time_s,include_map_digest=True,max_map_cells=None,map_source="knowledge")

    def make_full_self_packet(self,time_s:float)->RobotPacket:
        return self.make_packet(time_s,include_map_digest=True,max_map_cells=None,map_source="self")

    def make_partial_self_packet(self,time_s:float,max_map_cells:int|None=650)->RobotPacket:
        return self.make_packet(time_s,include_map_digest=True,max_map_cells=max_map_cells,map_source="self")

    def _path_digest(self)->list[Point]:
        if not self.path or self.path_index>=len(self.path): return []
        pts=[self.est_xy]+self.path[self.path_index:]; out=[]; last=None
        for p in pts:
            if last is None or distance(last,p)>=self.cfg.robot.path_digest_spacing_m:
                out.append((float(p[0]),float(p[1]))); last=p
            if len(out)>=self.cfg.robot.max_path_digest_points: break
        return out

    def _visited_digest(self)->list[Point]:
        out=[]; last=None
        for p in reversed(self.visit_history):
            if last is None or distance(last,p)>=self.cfg.robot.visit_digest_spacing_m:
                out.append((float(p[0]),float(p[1]))); last=p
            if len(out)>=self.cfg.robot.max_visit_digest_points: break
        out.reverse()
        return out

    def _trajectory_digest(self)->list[Point]:
        """Full downsampled estimated path from HOME to current position."""
        pts=list(self.trajectory_from_home)
        if not pts:
            return []
        if distance(pts[-1],self.est_xy)>0.15:
            pts.append(self.est_xy)
        spacing=max(0.05,float(self.cfg.robot.trajectory_digest_spacing_m))
        out=[]; last=None
        for p in pts:
            if last is None or distance(last,p)>=spacing:
                out.append((float(p[0]),float(p[1]))); last=p
        # Always keep the current endpoint.
        end=(float(pts[-1][0]),float(pts[-1][1]))
        if not out or distance(out[-1],end)>0.15:
            out.append(end)
        max_pts=max(2,int(self.cfg.robot.max_trajectory_digest_points))
        if len(out)>max_pts:
            idx=np.linspace(0,len(out)-1,max_pts).astype(int)
            out=[out[int(i)] for i in idx]
        return out

    def choose_task_and_plan(self,time_s:float,reserved_goals:dict[int,Point]|None=None,reserved_frontiers:dict[int,Point]|None=None)->None:
        self._expire_stale_teammate_intent(time_s); team_goals=self.fresh_teammate_goals(time_s); team_paths=self.fresh_teammate_paths(time_s); team_visits=self.fresh_teammate_visits(time_s); team_trajectories=self.known_teammate_trajectories_snapshot()
        dynamic_obstacles=self._teammate_dynamic_obstacles()
        if reserved_goals:
            for rid,g in reserved_goals.items():
                if rid!=self.id and g is not None:
                    team_goals[-1000-rid]=(float(g[0]),float(g[1]))
        event_replan=False
        if self.assessment.blocked_forward and self.path:
            self._remember_failed_goal(); self.path=[]; self.path_index=0; self.status.note="path_invalidated_by_lidar_block"; event_replan=True
        if self._goal_progress_stalled(time_s):
            self._remember_failed_goal(); self.path=[]; self.path_index=0; self.status.note="goal_progress_stalled"; event_replan=True
        if self._should_keep_committed_goal(time_s,event_replan): return
        target_mode = self.target.detected and not self.completed_target_roundtrip
        force_target_replan = target_mode and self.current_task not in {"GO_TO_TARGET","EXPLORE_TOWARD_TARGET","VERIFY_ROUTE_EVIDENCE","RELAY_EVIDENCE_HOME","RETURN_HOME_AFTER_TARGET","WAIT_AT_HOME_DONE"}
        replan_period = float(self.cfg.robot.target_path_replan_period_s if target_mode else self.cfg.robot.path_replan_period_s)
        if not event_replan and not force_target_replan and time_s-self.last_replan_time<replan_period:
            goal_reached=self.current_goal is not None and distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance
            if self.path_index<len(self.path) or not goal_reached:
                return
        goal,task,reason,breakdown=self._select_goal_from_lidar_map(team_goals,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_frontiers or {},time_s)
        if self._should_reject_goal_switch(goal,breakdown,time_s,event_replan):
            self.status.note="committed_current_goal"
            self.last_replan_time=time_s
            return
        self.current_goal=goal; self.current_task=task; self.status.task=task; self.status.goal=goal
        self.status.planning_source="communication-limited knowledge map + EKF pose estimate"; self.status.note=reason; self.status.reward_breakdown=breakdown
        if goal is None:
            self.path=[]; self.path_index=0; self.status.last_plan_success=False; self.status.last_plan_reason="no_goal_available"; self.status.last_path_min_clearance=0.0; self.last_replan_time=time_s; return
        result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
        if result.success and len(result.path)>=2:
            simplified=self._downsample_path(result.path,spacing=0.45)
            if task in {"GO_TO_TARGET","EXPLORE_TOWARD_TARGET","VERIFY_ROUTE_EVIDENCE","RELAY_EVIDENCE_HOME","RETURN_HOME_AFTER_TARGET"}:
                self.path=simplified; self.path_index=0
                self.status.last_path_min_clearance=max(0.0,float(result.min_clearance))
            else:
                simp_clear=self.map.path_min_clearance(simplified)
                self.path=simplified if simp_clear>=self.cfg.planning.critical_clearance_m else result.path; self.path_index=0
                self.status.last_path_min_clearance=max(0.0,min(result.min_clearance,simp_clear if simplified else result.min_clearance))
        elif task in {"REPORT_TARGET_HOME","RELAY_EVIDENCE_HOME","RETURN_HOME_CERT_ROUTE","RETURN_HOME_AFTER_TARGET","RETURN_HOME_EXPLORATION_COMPLETE"}:
            self.path=self._homing_fallback_path(goal); self.path_index=0
            result.success=bool(self.path); result.reason="homing_fallback" if self.path else result.reason
            self.status.last_path_min_clearance=max(0.0,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        elif task in {"GO_TO_TARGET","EXPLORE_TOWARD_TARGET","VERIFY_ROUTE_EVIDENCE"}:
            self.path=self._target_fallback_path(goal); self.path_index=0
            result.success=bool(self.path); result.reason="target_directed_fallback" if self.path else result.reason
            self.status.last_path_min_clearance=max(0.0,min(self.assessment.front_clearance,self.assessment.left_clearance,self.assessment.right_clearance))
        else:
            self._remember_failed_goal(goal); self.path=[]; self.path_index=0; self.status.last_path_min_clearance=0.0
        self.status.last_plan_success=result.success; self.status.last_plan_reason=result.reason; self.last_replan_time=time_s; self.best_routes=self.graph.top_routes(k=4)
        if result.success and self.path:
            self.current_goal=self.path[-1]; self.status.goal=self.current_goal
        if result.success:
            self.goal_commit_start=time_s; self.goal_commit_score=float(breakdown.get("score",0.0)); self.best_goal_distance=distance(self.est_xy,self.current_goal if self.current_goal is not None else goal); self.last_goal_progress_time=time_s

    def _remember_failed_goal(self, goal: Point | None = None)->None:
        g=goal if goal is not None else self.current_goal
        if g is None: return
        self.failed_goal_memory.append((float(g[0]),float(g[1])))
        self.failed_goal_memory=self.failed_goal_memory[-self.cfg.robot.failed_goal_memory_size:]

    def _goal_progress_stalled(self,time_s:float)->bool:
        if self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        d=distance(self.est_xy,self.current_goal)
        if d+0.35<self.best_goal_distance:
            self.best_goal_distance=d; self.last_goal_progress_time=time_s
            return False
        return time_s-self.last_goal_progress_time>self.cfg.robot.stuck_progress_timeout_s

    def _should_keep_committed_goal(self,time_s:float,event_replan:bool)->bool:
        if event_replan or self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        if self.target.detected: return False
        current_d=distance(self.est_xy,self.current_goal)
        if current_d<=self.cfg.robot.goal_tolerance: return False
        if time_s-self.goal_commit_start<self.cfg.robot.goal_commit_time_s:
            self.status.note="commit_hold"
            return True
        if current_d<=self.cfg.robot.goal_finish_commit_radius_m:
            self.status.note="commit_finish_current_goal"
            return True
        return False

    def _should_reject_goal_switch(self,goal:Point|None,breakdown:dict[str,float],time_s:float,event_replan:bool)->bool:
        if event_replan or goal is None or self.current_goal is None or not self.path or self.path_index>=len(self.path): return False
        if self.target.detected or distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_tolerance: return False
        if distance(goal,self.current_goal)<self.cfg.robot.goal_switch_same_goal_radius_m: return False
        new_score=float(breakdown.get("score",0.0))
        required_gain=float(self.cfg.robot.goal_switch_score_margin)
        if time_s-self.last_goal_progress_time<=self.cfg.robot.stuck_progress_timeout_s:
            required_gain+=float(self.cfg.robot.goal_progress_switch_margin)
        if distance(self.est_xy,self.current_goal)<=self.cfg.robot.goal_finish_commit_radius_m*1.35:
            required_gain+=float(self.cfg.robot.goal_finish_switch_margin)
        return new_score<self.goal_commit_score+required_gain

    def fresh_teammate_goals(self,time_s:float)->dict[int,Point]: self._expire_stale_teammate_intent(time_s); return dict(self.known_teammate_goals)
    def fresh_teammate_paths(self,time_s:float)->dict[int,list[Point]]: self._expire_stale_teammate_intent(time_s); return {rid:list(path) for rid,path in self.known_teammate_paths.items() if path}
    def fresh_teammate_visits(self,time_s:float)->dict[int,list[Point]]: self._expire_stale_teammate_intent(time_s); return {rid:list(path) for rid,path in self.known_teammate_visits.items() if path}
    def known_teammate_trajectories_snapshot(self)->dict[int,list[Point]]:
        return {rid:list(path) for rid,path in self.known_teammate_trajectories.items() if path}
    def _teammate_dynamic_obstacles(self)->list[tuple[Point,float]]:
        out:list[tuple[Point,float]]=[]
        base=2.0*float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m)
        max_extra=float(self.cfg.planning.dynamic_obstacle_max_cov_extra_m)
        for rid,pose in self.known_teammate_pose.items():
            if rid==self.id:
                continue
            cov=float(self.known_teammate_cov.get(rid,0.0))
            cov_extra=min(max_extra,0.65*math.sqrt(max(0.0,cov)*0.5))
            out.append(((float(pose[0]),float(pose[1])),base+cov_extra))
        path_radius=float(self.cfg.robot.radius)+0.5*float(self.cfg.robot.collision_buffer_m)
        for rid,path in self.known_teammate_paths.items():
            if rid==self.id or not path:
                continue
            stride=max(1,len(path)//8)
            for p in path[::stride][:10]:
                out.append(((float(p[0]),float(p[1])),path_radius))
        return out[:48]
    def _expire_stale_teammate_intent(self,time_s:float)->None:
        stale=[rid for rid,stamp in self.known_teammate_last_seen.items() if time_s-stamp>self.cfg.communication.teammate_intent_timeout_s]
        for rid in stale:
            for d in (self.known_teammate_last_seen,self.known_teammate_goals,self.known_teammate_paths,self.known_teammate_visits,self.known_teammate_tasks,self.known_teammate_pose,self.known_teammate_cov,self.known_teammate_regions): d.pop(rid,None)

    def _mdp_candidate(self,goal:Point|None,task:str,reason:str,breakdown:dict[str,float]|None=None)->MacroActionCandidate:
        return MacroActionCandidate(goal,task,reason,dict(breakdown or {}))

    def _choose_mdp_action(self,candidates:list[MacroActionCandidate],time_s:float)->tuple[Point|None,str,str,dict[str,float]]:
        if not candidates:
            self.last_mdp_candidates=[]
            return None,"WAIT","mdp_no_candidate",{}
        if not bool(self.cfg.mdp.enabled):
            best=max(candidates,key=lambda c:float(c.breakdown.get("score",0.0)))
            self.last_mdp_candidates=[MacroActionScore(best.task,best.goal,best.reason,float(best.breakdown.get("score",0.0)),dict(best.breakdown))]
            return best.goal,best.task,best.reason,best.breakdown
        scored:list[MacroActionScore]=[]
        bd_by_task:dict[tuple[str,int],dict[str,float]]={}
        for cand in candidates:
            mdp_score,bd=self._mdp_score_candidate(cand,time_s)
            key=(cand.task,id(cand))
            bd_by_task[key]=bd
            scored.append(MacroActionScore(cand.task,cand.goal,cand.reason,float(mdp_score),self._mdp_trace_terms(bd)))
        scored.sort(key=lambda item:item.score,reverse=True)
        self.last_mdp_candidates=scored[:8]
        best_score=scored[0].score
        best_idx=next(i for i,c in enumerate(candidates) if c.task==scored[0].task and c.goal==scored[0].goal and c.reason==scored[0].reason)
        best=candidates[best_idx]
        bd=dict(bd_by_task[(best.task,id(best))])
        bd.update({
            "score":float(best_score),
            "mdp_score":float(best_score),
            "mdp_enabled":1.0,
            "mdp_candidates":float(len(scored)),
            "belief_entropy":float(self.target_belief_entropy()),
            "mdp_winner_rank":0.0,
        })
        if len(scored)>1:
            bd["mdp_second_score"]=float(scored[1].score)
            bd["mdp_margin"]=float(scored[0].score-scored[1].score)
        for idx,item in enumerate(scored[:4]):
            bd[f"mdp_rank_{idx}_score"]=float(item.score)
            bd[f"mdp_rank_{idx}_task_code"]=float(self._task_code(item.task))
        self.last_mdp_score=float(best_score)
        return best.goal,best.task,"belief_mdp_"+best.reason,bd

    def _mdp_score_candidate(self,cand:MacroActionCandidate,time_s:float)->tuple[float,dict[str,float]]:
        bd=dict(cand.breakdown)
        goal=cand.goal
        dist=0.0 if goal is None else distance(self.est_xy,goal)
        path_len=float(bd.get("planned_path_length",bd.get("target_distance",bd.get("distance_home",dist))))
        raw_clearance=float(bd.get("planned_path_clearance",bd.get("raw_clearance_m",self.assessment.front_clearance)))
        unknown_frac=float(bd.get("planned_path_unknown_fraction",bd.get("target_path_unknown_fraction",0.0)))
        target_belief_raw=float(bd.get("target_belief_gain",self._target_belief_gain(goal) if goal is not None else 0.0))
        target_belief=float(np.clip(target_belief_raw/0.075,0.0,1.0))
        target_goal=float(np.clip(bd.get("target_goal_value",1.0 if cand.task in {"GO_TO_TARGET","EXPLORE_TOWARD_TARGET"} and self.target.detected else 0.0),0.0,1.0))
        cert_gain=float(np.clip(bd.get("certificate_improvement",bd.get("certificate_gap",0.0))/0.35,0.0,1.0))
        comm_value=float(np.clip(bd.get("communication_value",0.0)/1.5,0.0,1.0))
        info_raw=float(bd.get("info",0.0))
        info_raw+=0.45*math.log1p(float(bd.get("expected_lidar_unknown_gain",bd.get("expected_visibility",0.0))))
        info=float(np.clip(info_raw/4.0,0.0,1.0))
        heuristic=float(bd.get("score",0.0))
        heuristic_norm=float(np.clip((heuristic+4.0)/18.0,0.0,1.0))
        clearance_norm=float(np.clip(raw_clearance/max(1e-6,float(self.cfg.planning.desired_clearance_m)),0.0,1.0))
        progress=float(np.clip(bd.get("target_progress",0.0),0.0,1.0))
        clearance_gap=max(0.0,float(self.cfg.planning.critical_clearance_m)-raw_clearance)/max(1e-6,float(self.cfg.planning.critical_clearance_m))
        risk=float(np.clip(float(bd.get("risk",0.0))+clearance_gap+0.65*unknown_frac,0.0,1.0))
        travel=float(np.clip(path_len/max(1e-6,float(self.cfg.lidar.range)*2.8),0.0,1.0))
        teammate_pen=float(np.clip(
            float(bd.get("teammate_pose_penalty",0.0))/8.0+
            float(bd.get("reservation_penalty",0.0))/8.0+
            float(bd.get("path_teammate_penalty",0.0))/8.0,
            0.0,1.0,
        ))
        if cand.task=="DEPLOY_FROM_HOME":
            info=max(info,0.55)
            travel*=0.55
        if cand.task=="SEARCH_OPEN_SECTOR":
            info=max(info,0.25)
        if cand.task=="GO_TO_TARGET":
            target_goal=max(target_goal,1.0)
            info*=0.25
        if cand.task=="RELAY_EVIDENCE_HOME":
            comm_value=max(comm_value,0.65)
        if cand.task=="VERIFY_ROUTE_EVIDENCE":
            cert_gain=max(cert_gain,0.45)
        score=(
            float(self.cfg.mdp.heuristic_score_weight)*heuristic_norm+
            float(self.cfg.mdp.discount)*(
                float(self.cfg.mdp.target_discovery_weight)*target_belief+
                float(self.cfg.mdp.target_goal_weight)*target_goal+
                float(self.cfg.mdp.certificate_weight)*cert_gain+
                float(self.cfg.mdp.communication_weight)*comm_value+
                float(self.cfg.mdp.information_weight)*info+
                2.2*clearance_norm+
                1.4*progress
            )-
            float(self.cfg.mdp.travel_cost_weight)*travel-
            4.0*teammate_pen-
            float(self.cfg.mdp.risk_weight)*risk
        )
        bd.update({
            "mdp_heuristic_component":float(self.cfg.mdp.heuristic_score_weight)*heuristic_norm,
            "mdp_target_belief":float(target_belief),
            "mdp_target_belief_raw":float(target_belief_raw),
            "mdp_target_goal":float(target_goal),
            "mdp_certificate_gain":float(cert_gain),
            "mdp_communication_value":float(comm_value),
            "mdp_information_value":float(info),
            "mdp_information_raw":float(info_raw),
            "mdp_clearance_value":float(clearance_norm),
            "mdp_progress_value":float(progress),
            "mdp_travel_cost":float(self.cfg.mdp.travel_cost_weight)*travel,
            "mdp_teammate_penalty":float(teammate_pen),
            "mdp_risk":float(risk),
            "mdp_action_"+cand.task:1.0,
        })
        return float(score),bd

    def _mdp_trace_terms(self,bd:dict[str,float])->dict[str,float]:
        return {
            "belief":float(bd.get("mdp_target_belief",0.0)),
            "target":float(bd.get("mdp_target_goal",0.0)),
            "cert":float(bd.get("mdp_certificate_gain",0.0)),
            "comm":float(bd.get("mdp_communication_value",0.0)),
            "info":float(bd.get("mdp_information_value",0.0)),
            "clear":float(bd.get("mdp_clearance_value",0.0)),
            "travel":float(bd.get("mdp_travel_cost",0.0)),
            "risk":float(bd.get("mdp_risk",0.0)),
            "team":float(bd.get("mdp_teammate_penalty",0.0)),
        }

    def _task_code(self,task:str)->int:
        codes={
            "DEPLOY_FROM_HOME":1,
            "SEARCH_HIER_NBV":2,
            "SEARCH_TARGET_BELIEF":3,
            "SEARCH_OPEN_SECTOR":4,
            "GO_TO_TARGET":5,
            "EXPLORE_TOWARD_TARGET":6,
            "VERIFY_ROUTE_EVIDENCE":7,
            "RELAY_EVIDENCE_HOME":8,
            "RETURN_HOME_AFTER_TARGET":9,
        }
        return codes.get(task,0)

    def _evidence_relay_value(self,time_s:float)->float:
        value=0.0
        if self.target.detected and not self.target.reported_home:
            value+=1.25 if self.target.source_robot==self.id else 0.65
        if self.target_reached and not self.route_candidate_uploaded:
            value+=1.10
        route=self._best_local_target_route()
        if route is not None:
            if not route.reported_home:
                value+=0.45
            value+=0.55*max(0.0,float(self.cfg.cage.route_cert_threshold)-float(route.certificate))
        if time_s-self.last_home_full_upload_time>24.0 and value>0.0:
            value+=0.20
        return float(value)

    def _select_evidence_relay_candidate(self,time_s:float)->MacroActionCandidate|None:
        value=self._evidence_relay_value(time_s)
        if value<float(self.cfg.mdp.evidence_relay_min_value):
            return None
        home_dist=distance(self.est_xy,self.home_xy)
        if home_dist<=self.cfg.robot.goal_tolerance:
            return None
        return self._mdp_candidate(self.home_xy,"RELAY_EVIDENCE_HOME","unreported_evidence_return_to_home",{
            "communication_value":float(value),
            "distance_home":float(home_dist),
            "score":float(8.0*value-0.22*home_dist),
        })

    def _select_weak_route_evidence_candidate(self)->MacroActionCandidate|None:
        route=self._best_local_target_route()
        if route is None or not route.edge_ids:
            return None
        weakest_id=min(route.edge_ids,key=lambda eid:self.graph.edges[eid].cert.confidence if eid in self.graph.edges else 1.0)
        edge=self.graph.edges.get(weakest_id)
        if edge is None or edge.a not in self.graph.nodes or edge.b not in self.graph.nodes:
            return None
        threshold=max(float(self.cfg.cage.route_cert_threshold),float(self.cfg.mdp.weak_edge_certificate_threshold))
        gap=max(0.0,threshold-float(edge.cert.confidence))
        if gap<=1e-6 and route.reported_home:
            return None
        ax,ay=self.graph.nodes[edge.a].xy
        bx,by=self.graph.nodes[edge.b].xy
        goal=(float((ax+bx)*0.5),float((ay+by)*0.5))
        if distance(goal,self.est_xy)<=self.cfg.robot.goal_tolerance:
            goal=self.graph.nodes[edge.a].xy if distance(self.est_xy,self.graph.nodes[edge.a].xy)>distance(self.est_xy,self.graph.nodes[edge.b].xy) else self.graph.nodes[edge.b].xy
        return self._mdp_candidate(goal,"VERIFY_ROUTE_EVIDENCE","weakest_route_edge_needs_certificate",{
            "certificate_gap":float(gap),
            "certificate_improvement":float(gap+0.20*(0.0 if route.reported_home else 1.0)),
            "weak_edge_cert":float(edge.cert.confidence),
            "route_certificate":float(route.certificate),
            "route_reported_home":float(1.0 if route.reported_home else 0.0),
            "raw_clearance_m":float(edge.cert.min_clearance),
            "score":float(12.0*gap+2.0*(0.0 if route.reported_home else 1.0)-0.12*distance(self.est_xy,goal)),
        })

    def _select_target_belief_goal(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,dynamic_obstacles:list[tuple[Point,float]],reserved_targets:dict[int,Point]|None)->MacroActionCandidate|None:
        stride=max(1,int(self.cfg.mdp.target_belief_candidate_stride_cells))
        desired=max(1e-6,float(self.cfg.planning.desired_clearance_m))
        safe_clearance=max(float(self.cfg.planning.safe_approach_min_clearance_m),float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m))
        reservation_radius=float(self.cfg.planning.nbv_reservation_lidar_fraction)*float(self.cfg.lidar.range)
        min_goal_distance=max(0.75,float(self.cfg.robot.goal_tolerance)*1.5)
        pre:list[tuple[float,Point,dict[str,float]]]=[]
        for j in range((self.id*stride) % stride,self.map.ny,stride):
            for i in range((self.id+j) % stride,self.map.nx,stride):
                if occupied_mask[j,i]:
                    continue
                p=self.map.cell_to_world((i,j))
                d=distance(self.est_xy,p)
                if d<min_goal_distance:
                    continue
                cl=float(clearance_map[j,i])
                if cl<safe_clearance:
                    continue
                reserved_pen=self._reservation_overlap_penalty(p,reserved_targets or {},reservation_radius)
                if reserved_pen>0.0:
                    continue
                belief_gain=self._target_belief_gain(p)
                if belief_gain<0.0025:
                    continue
                local_unknown=self._local_cell_count(unknown_mask,i,j,max(1,int(math.ceil(0.45*self.cfg.lidar.range/self.map.res))))
                known_support=self._local_cell_count(known_mask & (~occupied_mask),i,j,max(1,int(math.ceil(0.30*self.cfg.lidar.range/self.map.res))))
                if known_support<2 and d>float(self.cfg.lidar.range):
                    continue
                clearance_score=min(1.5,cl/desired)
                score=18.0*belief_gain+0.45*math.log1p(float(local_unknown))+1.15*clearance_score-0.22*d
                pre.append((float(score),p,{
                    "score":float(score),
                    "target_belief_gain":float(belief_gain),
                    "local_unknown_cells":float(local_unknown),
                    "raw_clearance_m":float(cl),
                    "clearance_score":float(clearance_score),
                    "distance_cost":float(0.22*d),
                    "belief_search_known_support":float(known_support),
                }))
        if not pre:
            return None
        best=None
        for prelim,goal,bd in sorted(pre,key=lambda x:x[0],reverse=True)[:max(1,int(self.cfg.mdp.target_belief_plan_eval_count))]:
            result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
            if not result.success or len(result.path)<2:
                continue
            path_len=self._path_length(result.path)
            unknown_frac=self._path_unknown_fraction(result.path)
            visibility=self._expected_lidar_visibility_gain(goal,unknown_mask,occupied_mask)
            final=prelim+0.55*math.log1p(visibility)-0.18*path_len-1.8*unknown_frac
            out=dict(bd)
            out.update({
                "score":float(final),
                "pre_plan_score":float(prelim),
                "planned_path_length":float(path_len),
                "planned_path_clearance":float(result.min_clearance),
                "planned_path_unknown_fraction":float(unknown_frac),
                "expected_lidar_unknown_gain":float(visibility),
            })
            if best is None or final>best[0]:
                best=(float(final),goal,out)
        if best is None:
            return None
        return self._mdp_candidate(best[1],"SEARCH_TARGET_BELIEF","target_belief_macro_action_selected",best[2])

    def _select_goal_from_lidar_map(self,team_goals:dict[int,Point],team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]],dynamic_obstacles:list[tuple[Point,float]],reserved_frontiers:dict[int,Point]|None=None,time_s:float=0.0)->tuple[Point|None,str,str,dict[str,float]]:
        """Choose the next task/goal from communication-limited knowledge.

        Normal exploration was intentionally simplified in this version:
        find reachable frontiers, prefer ones that expose unknown cells, and
        avoid regions already covered by teammate paths.  Target and return-home
        workflows remain separate so exploration tuning does not pollute the
        target roundtrip behavior.
        """
        reserved_frontiers=reserved_frontiers or {}
        target_known = self.target.detected and self.target.xy is not None
        if self.completed_target_roundtrip:
            return None,"WAIT_AT_HOME_DONE","target_roundtrip_complete_wait_at_home",{"roundtrip_complete":1.0}
        if self.target_reached:
            home_dist=distance(self.est_xy,self.home_xy)
            if home_dist>self.cfg.robot.goal_tolerance:
                return self.home_xy,"RETURN_HOME_AFTER_TARGET","target_reached_return_home",{"distance_home":home_dist}
            return None,"WAIT_AT_HOME_DONE","target_roundtrip_complete_wait_at_home",{"distance_home":home_dist}
        if self.force_return_home and not target_known:
            home_dist=distance(self.est_xy,self.home_xy)
            if home_dist>self.cfg.robot.goal_tolerance:
                return self.home_xy,"RETURN_HOME_EXPLORATION_COMPLETE","exploration_complete_return_home",{"distance_home":home_dist}
            return None,"WAIT_AT_HOME","exploration_complete_wait_at_home",{"distance_home":home_dist}
        if self.assessment.consistency<self.cfg.cage.reanchor_consistency_threshold:
            anchor=self._nearest_anchor()
            if anchor is not None and distance(anchor,self.est_xy)>0.35:
                return anchor,"REANCHOR","low_scan_map_consistency_reanchor",{"consistency":self.assessment.consistency}

        # Special startup behavior: before the robots have useful history, push
        # each one out from HOME along a different assigned direction.  This
        # prevents all robots from fighting for the same first frontier.
        if (not target_known) and self.cfg.planning.startup_deployment_enabled:
            deploy=self._select_startup_deployment_goal(dynamic_obstacles,reserved_frontiers)
            if deploy is not None:
                goal,bd=deploy
                startup_candidates=[self._mdp_candidate(goal,"DEPLOY_FROM_HOME","startup_deployment_spread_from_home",bd)]
                probe=self._sector_probe_goal()
                if probe is not None:
                    startup_candidates.append(self._mdp_candidate(probe,"SEARCH_OPEN_SECTOR","startup_open_sector_alternative",{"open_sector":1.0,"score":1.0,"target_belief_gain":self._target_belief_gain(probe)}))
                return self._choose_mdp_action(startup_candidates,time_s)

        target_candidates:list[MacroActionCandidate]=[]
        target_goal:Point|None=None
        if target_known:
            target_goal=(float(self.target.xy[0]),float(self.target.xy[1]))
            if distance(self.est_xy,target_goal)<=self.cfg.robot.goal_tolerance:
                probe=self._target_probe_goal(target_goal, allow_beyond_target=True)
                if probe is not None:
                    target_candidates.append(self._mdp_candidate(probe,"EXPLORE_TOWARD_TARGET","estimated_target_close_probe_for_physical_confirmation",{"target_probe":1.0,"target_progress":self._target_progress_reward(probe),"target_goal_value":0.75}))
            direct=self.planner.plan(self.map,self.est_xy,target_goal,dynamic_obstacles=dynamic_obstacles)
            unknown_frac=self._path_unknown_fraction(direct.path) if direct.success else 1.0
            if direct.success and direct.min_clearance>=self.cfg.planning.critical_clearance_m:
                target_candidates.append(self._mdp_candidate(target_goal,"GO_TO_TARGET","target_known_drive_to_target_map_while_moving",{
                    "target_distance":distance(self.est_xy,target_goal),
                    "target_path_unknown_fraction":float(unknown_frac),
                    "planned_path_length":float(self._path_length(direct.path)),
                    "raw_clearance_m":float(direct.min_clearance),
                    "target_goal_value":1.0,
                    "score":25.0+max(0.0,10.0-distance(self.est_xy,target_goal)),
                }))
            if target_candidates:
                return self._choose_mdp_action(target_candidates,time_s)
            probe=self._fast_target_probe_goal(target_goal)
            if probe is not None:
                return self._choose_mdp_action([
                    self._mdp_candidate(probe,"EXPLORE_TOWARD_TARGET","target_directed_fast_lidar_probe",{
                        "target_probe":1.0,
                        "target_progress":self._target_progress_reward(probe),
                        "target_goal_value":0.40,
                        "score":8.0+5.0*self._target_progress_reward(probe),
                    })
                ],time_s)

        clearance_map=self.map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*2.5))
        free_mask=self.map.free_mask(); known_mask=self.map.known_mask(); unknown_mask=~known_mask; occupied_mask=self.map.occupied_mask()

        # Target-guided behavior can still use frontiers as hints near the
        # target corridor.  Normal exploration below does not choose frontier
        # cells directly; it chooses scan poses by expected LiDAR information.
        frontiers=self.map.find_frontiers(self.cfg.planning.frontier_min_cluster_size,self.cfg.planning.frontier_info_radius_m)
        if target_known:
            weak=self._select_weak_route_evidence_candidate()
            if weak is not None:
                target_candidates.append(weak)
            relay=self._select_evidence_relay_candidate(time_s)
            if relay is not None:
                target_candidates.append(relay)
            if frontiers:
                best=self._select_target_directed_frontier_goal(frontiers,clearance_map,free_mask,known_mask,unknown_mask,occupied_mask,dynamic_obstacles)
                if best is not None:
                    bd=dict(best[2])
                    bd.setdefault("target_goal_value",0.55)
                    target_candidates.append(self._mdp_candidate(best[1],"EXPLORE_TOWARD_TARGET","target_directed_frontier_selected",bd))
            probe=self._target_probe_goal((float(self.target.xy[0]),float(self.target.xy[1])))
            if probe is not None:
                target_candidates.append(self._mdp_candidate(probe,"EXPLORE_TOWARD_TARGET","target_directed_probe_after_frontier_filter",{"target_probe":1.0,"target_progress":self._target_progress_reward(probe),"target_goal_value":0.45}))
            if target_candidates:
                return self._choose_mdp_action(target_candidates,time_s)
            return None,"WAIT","target_known_no_reachable_goal",{}

        # Normal exploration: use a TARE-style hierarchy, but keep it LOS-realistic.
        # First choose a coarse unknown region from this robot's communication-limited
        # knowledge map, then run the existing dense NBV scan-pose selector only
        # around that region.
        normal_candidates:list[MacroActionCandidate]=[]
        best=self._select_hierarchical_nbv_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_frontiers,time_s)
        if best is not None:
            normal_candidates.append(self._mdp_candidate(best[1],"SEARCH_HIER_NBV","hierarchical_region_guided_nbv_selected",best[2]))

        belief_goal=self._select_target_belief_goal(clearance_map,known_mask,unknown_mask,occupied_mask,dynamic_obstacles,reserved_frontiers)
        if belief_goal is not None:
            normal_candidates.append(belief_goal)

        # Last fallback: if NBV cannot find a reachable scan pose, move through
        # the best open sector to create new observations.
        probe=self._sector_probe_goal()
        if probe is not None:
            normal_candidates.append(self._mdp_candidate(probe,"SEARCH_OPEN_SECTOR","nbv_no_candidate_use_open_sector_probe",{"open_sector":1.0,"score":1.0,"target_belief_gain":self._target_belief_gain(probe)}))
        if normal_candidates:
            return self._choose_mdp_action(normal_candidates,time_s)
        gx=self.est_xy[0]+math.cos(self.est_pose[2]+self.assessment.best_open_angle)*min(1.8,self.cfg.lidar.range*0.38)
        gy=self.est_xy[1]+math.sin(self.est_pose[2]+self.assessment.best_open_angle)*min(1.8,self.cfg.lidar.range*0.38)
        fallback=(gx,gy)
        return self._choose_mdp_action([self._mdp_candidate(fallback,"SEARCH_OPEN_SECTOR","nbv_no_candidate_use_best_lidar_open_sector",{"open_sector":1.0,"score":0.5,"target_belief_gain":self._target_belief_gain(fallback)})],time_s)

    def _select_startup_deployment_goal(self,dynamic_obstacles:list[tuple[Point,float]],reserved_frontiers:dict[int,Point]|None=None)->tuple[Point,dict[str,float]]|None:
        done_radius=0.85*float(self.cfg.planning.startup_deployment_lidar_fraction)*float(self.cfg.lidar.range)
        home_d=distance(self.est_xy,self.home_xy)
        if home_d>=done_radius:
            return None
        n=max(1,int(self.cfg.robot.count))
        # Center launch directions on the rough HOME-to-far-corner exploration
        # direction, but spread robots broadly so startup is not chaotic.
        far=(self.map.width_m-float(self.cfg.world.world_margin),self.map.height_m-float(self.cfg.world.world_margin))
        base=angle_to(self.home_xy,far)
        spread=math.radians(float(self.cfg.planning.startup_deployment_angle_spread_deg))
        offset=0.0 if n==1 else (self.id/(n-1)-0.5)*spread
        assigned=base+offset
        deploy_dist=max(float(self.cfg.lidar.range)*float(self.cfg.planning.startup_deployment_lidar_fraction),self.cfg.robot.goal_tolerance*3.0)
        samples=4
        step=math.radians(12.0)
        angle_offsets=[0.0]
        for k in range(1,9):
            angle_offsets.extend([k*step,-k*step])
        clearance=self.map.clearance_map(max_radius_m=max(3.0,self.cfg.planning.desired_clearance_m*2.5))
        known=self.map.known_mask(); unknown=~known; occupied=self.map.occupied_mask()
        best=None
        reserved=reserved_frontiers or {}
        reservation_radius=max(0.2,float(self.cfg.planning.nbv_reservation_lidar_fraction)*float(self.cfg.lidar.range))
        for dist_scale in np.linspace(1.0,0.55,samples):
            dist_m=deploy_dist*float(dist_scale)
            for off in angle_offsets:
                a=assigned+float(off)
                goal=(self.home_xy[0]+math.cos(a)*dist_m,self.home_xy[1]+math.sin(a)*dist_m)
                cell=self.map.world_to_cell(goal)
                if cell is None:
                    continue
                if any(distance(goal,g)<reservation_radius for rid,g in reserved.items() if rid!=self.id and g is not None):
                    continue
                cl=float(clearance[cell[1],cell[0]])
                visibility=self._expected_lidar_visibility_gain(goal,unknown,occupied)
                angle_pen=abs(float(off))/max(step,1e-6)
                dist_from_robot=distance(self.est_xy,goal)
                score=3.0*dist_scale+0.70*math.log1p(visibility)+1.1*min(1.5,cl/max(1e-6,self.cfg.planning.desired_clearance_m))-0.22*angle_pen-0.04*dist_from_robot
                bd={
                    "score":float(score),
                    "deployment_home_distance":float(home_d),
                    "deployment_done_radius_m":float(done_radius),
                    "deployment_goal_distance_m":float(dist_m),
                    "deployment_angle_offset_deg":float(math.degrees(off)),
                    "expected_visibility":float(visibility),
                    "raw_clearance_m":float(cl),
                    "deployment_distance_from_robot_m":float(dist_from_robot),
                }
                if best is None or score>best[0]:
                    best=(float(score),goal,bd)
        if best is None:
            return None
        return best[1],best[2]

    def _select_target_directed_frontier_goal(self,frontiers:list[FrontierCluster],clearance_map:np.ndarray,free_mask:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,dynamic_obstacles:list[tuple[Point,float]])->tuple[float,Point,dict[str,float]]|None:
        candidates=[]
        limit=max(1,min(len(frontiers),int(self.cfg.planning.frontier_sample_count)))
        for fr in frontiers[:limit]:
            approach=self._exploration_safe_approach_point(fr,clearance_map,free_mask,known_mask)
            d=max(0.1,distance(self.est_xy,approach))
            if d < max(0.75,self.cfg.robot.goal_tolerance*1.5):
                continue
            target_progress=self._target_progress_reward(approach)
            robot_target_corridor=self._robot_target_corridor_reward(approach)
            if target_progress < -0.05 and robot_target_corridor < 0.35:
                continue
            cell=self.map.world_to_cell(approach)
            clearance=float(clearance_map[cell[1],cell[0]]) if cell is not None else 0.0
            visibility=self._expected_lidar_visibility_gain(approach,unknown_mask,occupied_mask)
            info=math.log1p(fr.information_gain)+0.8*math.log1p(visibility)
            corridor=self._target_corridor_reward(approach)
            corridor_lowq=self._target_corridor_low_quality_reward(approach)
            score=0.45*info+1.2*min(1.5,clearance/max(1e-6,self.cfg.planning.desired_clearance_m))+8.0*target_progress+5.0*robot_target_corridor+self.cfg.cage.target_corridor_bonus_weight*corridor+self.cfg.cage.target_corridor_low_quality_weight*corridor_lowq-0.65*self.cfg.planning.distance_weight*d
            bd={"score":float(score),"info":float(info),"frontier_gain":float(fr.information_gain),"expected_visibility":float(visibility),"raw_clearance_m":float(clearance),"distance_cost":float(0.65*self.cfg.planning.distance_weight*d),"target_progress":float(target_progress),"robot_target_corridor":float(robot_target_corridor),"target_corridor":float(self.cfg.cage.target_corridor_bonus_weight*corridor),"corridor_lowq":float(self.cfg.cage.target_corridor_low_quality_weight*corridor_lowq),"target_mode_ignores_team_path_penalties":1.0}
            candidates.append((score,approach,bd))
        return self._best_planned_frontier_candidate(candidates,dynamic_obstacles)

    def _select_hierarchical_nbv_goal(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]],dynamic_obstacles:list[tuple[Point,float]],reserved_targets:dict[int,Point]|None,time_s:float)->tuple[float,Point,dict[str,float]]|None:
        if not bool(self.cfg.planning.hierarchical_exploration_enabled):
            self.assigned_region=None
            best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=False)
            if best is None:
                best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=True)
            return best

        team_points=self._team_history_points(team_paths,team_visits,team_trajectories)
        hard_radius=float(self.cfg.planning.nbv_teammate_hard_avoid_lidar_fraction)*float(self.cfg.lidar.range)
        soft_radius=max(hard_radius,float(self.cfg.planning.nbv_teammate_soft_avoid_lidar_fraction)*float(self.cfg.lidar.range))
        regions=self._build_coarse_exploration_regions(known_mask,unknown_mask,occupied_mask,clearance_map,team_points,reserved_targets or {},hard_radius,soft_radius)
        ordered=self._ordered_region_choices(regions,time_s)
        tried=0
        for region in ordered[:5]:
            tried+=1
            for relaxed in (False,True):
                best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=relaxed,focus_region=region)
                if best is None:
                    continue
                switched=self.assigned_region is None or self.assigned_region.region_id!=region.region_id
                self.assigned_region=region
                if switched:
                    self.assigned_region_start_time=time_s
                bd=dict(best[2])
                bd.update({
                    "hierarchical_exploration":1.0,
                    "coarse_region_i":float(region.region_id[0]),
                    "coarse_region_j":float(region.region_id[1]),
                    "coarse_region_center_x":float(region.center[0]),
                    "coarse_region_center_y":float(region.center[1]),
                    "coarse_region_radius_m":float(region.radius),
                    "coarse_region_score":float(region.score),
                    "coarse_region_unknown_cells":float(region.unknown_cells),
                    "coarse_region_known_free_cells":float(region.known_free_cells),
                    "coarse_region_frontier_support":float(region.frontier_support),
                    "coarse_regions_available":float(len(regions)),
                    "coarse_regions_tried":float(tried),
                    "coarse_region_relaxed_nbv":float(1.0 if relaxed else 0.0),
                })
                return best[0],best[1],bd

        # Region layer failed to produce a reachable local scan pose.  Fall back to
        # the existing global NBV instead of stopping exploration.
        self.assigned_region=None
        best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=False)
        if best is None:
            best=self._select_next_best_view_goal(clearance_map,known_mask,unknown_mask,occupied_mask,team_paths,team_visits,team_trajectories,dynamic_obstacles,reserved_targets,relaxed=True)
        if best is not None:
            bd=dict(best[2]); bd.update({"hierarchical_exploration":0.5,"hierarchical_fallback_global_nbv":1.0,"coarse_regions_available":float(len(regions))})
            return best[0],best[1],bd
        return None

    def _ordered_region_choices(self,regions:list[CoarseRegion],time_s:float)->list[CoarseRegion]:
        if not regions:
            self.assigned_region=None
            return []
        by_id={r.region_id:r for r in regions}
        current=by_id.get(self.assigned_region.region_id) if self.assigned_region is not None else None
        best=max(regions,key=lambda r:r.score)
        ordered:list[CoarseRegion]=[]
        if current is not None:
            commit_time=float(self.cfg.planning.region_commit_time_s)
            switch_ratio=max(1.0,float(self.cfg.planning.region_switch_score_ratio))
            if time_s-self.assigned_region_start_time<commit_time:
                ordered.append(current)
            elif current.score>0.0 and best.score<current.score*switch_ratio:
                ordered.append(current)
        if best not in ordered:
            ordered.append(best)
        for r in sorted(regions,key=lambda r:r.score,reverse=True):
            if r not in ordered:
                ordered.append(r)
            if len(ordered)>=8:
                break
        return ordered

    def _build_coarse_exploration_regions(self,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,clearance_map:np.ndarray,team_points:list[Point],reserved_targets:dict[int,Point],hard_radius:float,soft_radius:float)->list[CoarseRegion]:
        res=float(self.map.res)
        region_size=max(4.0*res,float(self.cfg.lidar.range)*float(self.cfg.planning.region_size_lidar_fraction))
        block=max(3,int(math.ceil(region_size/res)))
        radius=0.5*math.sqrt(2.0)*block*res
        known_free=known_mask & (~occupied_mask)
        min_unknown=max(4,int(0.08*block*block))
        regions:list[CoarseRegion]=[]
        rid_j=0
        for y0 in range(0,self.map.ny,block):
            y1=min(self.map.ny,y0+block)
            rid_i=0
            for x0 in range(0,self.map.nx,block):
                x1=min(self.map.nx,x0+block)
                sub_unknown=unknown_mask[y0:y1,x0:x1]
                unknown_count=int(np.count_nonzero(sub_unknown))
                if unknown_count<min_unknown:
                    rid_i+=1; continue
                sub_occ=occupied_mask[y0:y1,x0:x1]
                if np.count_nonzero(sub_occ)>0.62*sub_occ.size:
                    rid_i+=1; continue
                cx=(x0+x1)*0.5*res; cy=(y0+y1)*0.5*res
                center=(float(cx),float(cy))
                ci=int(min(self.map.nx-1,max(0,round(cx/res-0.5))))
                cj=int(min(self.map.ny-1,max(0,round(cy/res-0.5))))
                support_r=max(2,block//2+2)
                known_free_count=self._local_cell_count(known_free,ci,cj,support_r)
                if known_free_count<2 and distance(center,self.est_xy)>float(self.cfg.lidar.range)*1.10:
                    rid_i+=1; continue
                frontier_support=min(unknown_count,known_free_count)
                d_robot=distance(self.est_xy,center)
                team_min=self._min_distance_to_points(center,team_points)
                team_pen=self._distance_band_penalty(team_min,hard_radius,soft_radius)
                intent_pen=self._known_teammate_region_penalty(center,radius)
                reserve_pen=self._reservation_overlap_penalty(center,reserved_targets,float(self.cfg.lidar.range)*float(self.cfg.planning.nbv_reservation_lidar_fraction))
                failed=self._failed_goal_penalty(center)
                known_support=min(1.0,known_free_count/max(1.0,0.45*block*block))
                clearance=float(clearance_map[cj,ci]) if 0<=cj<clearance_map.shape[0] and 0<=ci<clearance_map.shape[1] else 0.0
                clearance_score=min(1.5,clearance/max(1e-6,float(self.cfg.planning.desired_clearance_m)))
                score=(
                    2.3*math.log1p(float(unknown_count))+
                    0.85*math.sqrt(float(max(0,frontier_support)))+
                    0.65*known_support+
                    0.45*clearance_score-
                    0.18*d_robot-
                    5.5*team_pen-
                    6.5*intent_pen-
                    4.0*reserve_pen-
                    2.0*failed
                )
                regions.append(CoarseRegion((rid_i,rid_j),center,float(radius),unknown_count,int(known_free_count),int(frontier_support),float(score)))
                rid_i+=1
            rid_j+=1
        return regions

    def _known_teammate_region_penalty(self,center:Point,radius:float)->float:
        if not self.known_teammate_regions:
            return 0.0
        best=0.0
        for info in self.known_teammate_regions.values():
            task=str(info.get("task",""))
            if not (task.startswith("SEARCH") or task=="DEPLOY_FROM_HOME"):
                continue
            other_center=info.get("center")
            if other_center is None:
                continue
            other_radius=float(info.get("radius",0.0))
            overlap_radius=max(0.1,float(radius)+other_radius)
            d=distance(center,(float(other_center[0]),float(other_center[1])))
            if d<overlap_radius:
                best=max(best,1.0-d/overlap_radius)
        return float(best)

    def _region_unknown_mask(self,unknown_mask:np.ndarray,region:CoarseRegion)->np.ndarray:
        out=np.zeros_like(unknown_mask,dtype=bool)
        cell=self.map.world_to_cell(region.center)
        if cell is None:
            return unknown_mask
        ci,cj=cell
        radius=float(region.radius)+0.35*float(self.cfg.lidar.range)
        r_cells=max(1,int(math.ceil(radius/self.map.res)))
        y0,y1=max(0,cj-r_cells),min(self.map.ny,cj+r_cells+1)
        x0,x1=max(0,ci-r_cells),min(self.map.nx,ci+r_cells+1)
        for j in range(y0,y1):
            wy=(j+0.5)*self.map.res
            for i in range(x0,x1):
                wx=(i+0.5)*self.map.res
                if math.hypot(wx-region.center[0],wy-region.center[1])<=radius:
                    out[j,i]=True
        return unknown_mask & out

    def _select_next_best_view_goal(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]],dynamic_obstacles:list[tuple[Point,float]],reserved_targets:dict[int,Point]|None,relaxed:bool,focus_region:CoarseRegion|None=None)->tuple[float,Point,dict[str,float]]|None:
        team_points=self._team_history_points(team_paths,team_visits,team_trajectories)
        own_points=self._downsample_points(self.trajectory_from_home,max_points=90)
        hard_radius=float(self.cfg.planning.nbv_teammate_hard_avoid_lidar_fraction)*float(self.cfg.lidar.range)
        soft_radius=max(hard_radius,float(self.cfg.planning.nbv_teammate_soft_avoid_lidar_fraction)*float(self.cfg.lidar.range))
        own_radius=float(self.cfg.planning.nbv_own_path_avoid_lidar_fraction)*float(self.cfg.lidar.range)
        reservation_radius=float(self.cfg.planning.nbv_reservation_lidar_fraction)*float(self.cfg.lidar.range)
        active_unknown_mask=unknown_mask
        if focus_region is not None:
            active_unknown_mask=self._region_unknown_mask(unknown_mask,focus_region)
        candidates=self._build_nbv_scan_pose_candidates(clearance_map,known_mask,active_unknown_mask,occupied_mask,team_points,own_points,reserved_targets or {},hard_radius,soft_radius,own_radius,reservation_radius,relaxed,focus_region)
        return self._best_planned_nbv_candidate(candidates,dynamic_obstacles,team_points,own_points,hard_radius,soft_radius,own_radius,relaxed)

    def _build_nbv_scan_pose_candidates(self,clearance_map:np.ndarray,known_mask:np.ndarray,unknown_mask:np.ndarray,occupied_mask:np.ndarray,team_points:list[Point],own_points:list[Point],reserved_targets:dict[int,Point],hard_radius:float,soft_radius:float,own_radius:float,reservation_radius:float,relaxed:bool,focus_region:CoarseRegion|None=None)->list[tuple[float,Point,dict[str,float]]]:
        stride=max(1,int(self.cfg.planning.nbv_sample_stride_cells))
        max_candidates=max(1,int(self.cfg.planning.nbv_max_candidates))
        desired=max(1e-6,float(self.cfg.planning.desired_clearance_m))
        safe_clearance=max(float(self.cfg.planning.safe_approach_min_clearance_m),float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m))
        local_radius=max(1,int(math.ceil(float(self.cfg.lidar.range)*float(self.cfg.planning.nbv_local_unknown_radius_lidar_fraction)/self.map.res)))
        min_goal_distance=max(0.75,float(self.cfg.robot.goal_tolerance)*1.5)
        focus_limit=math.inf
        if focus_region is not None:
            focus_limit=float(focus_region.radius)+0.95*float(self.cfg.lidar.range)
        pre:list[tuple[float,Point,dict[str,float]]]=[]
        blocked_by_teammate=0; blocked_by_reservation=0; skipped_low_gain=0; skipped_near=0
        # Robot-specific phase offset prevents every robot from testing exactly
        # the same lattice cells when they start with nearly identical maps.
        phase=(self.id*stride)//max(1,int(self.cfg.robot.count))
        for j in range(phase % stride,self.map.ny,stride):
            for i in range((phase+j) % stride,self.map.nx,stride):
                if occupied_mask[j,i]:
                    continue
                p=self.map.cell_to_world((i,j))
                d=distance(self.est_xy,p)
                region_focus=0.0
                if focus_region is not None:
                    d_region=distance(p,focus_region.center)
                    if d_region>focus_limit:
                        continue
                    region_focus=max(0.0,1.0-max(0.0,d_region-float(focus_region.radius))/max(1e-6,float(self.cfg.lidar.range)))
                if d<min_goal_distance:
                    skipped_near+=1
                    continue
                cl=float(clearance_map[j,i])
                if cl<safe_clearance:
                    continue
                local_unknown=self._local_cell_count(unknown_mask,i,j,local_radius)
                if local_unknown<4:
                    skipped_low_gain+=1
                    continue
                local_known=self._local_cell_count(known_mask & (~occupied_mask),i,j,local_radius)
                if local_known<2 and d>float(self.cfg.lidar.range)*0.85:
                    # This keeps the target inside/near the frontier of the
                    # unknown, not deep random unknown cells across the map.
                    continue
                team_min=self._min_distance_to_points(p,team_points)
                reserved_pen=self._reservation_overlap_penalty(p,reserved_targets,reservation_radius)
                if not relaxed and reserved_pen>0.0:
                    blocked_by_reservation+=1
                    continue
                if not relaxed and team_min<hard_radius:
                    blocked_by_teammate+=1
                    continue
                teammate_soft=self._distance_band_penalty(team_min,hard_radius,soft_radius)
                own_min=self._min_distance_to_points(p,own_points)
                own_soft=self._distance_band_penalty(own_min,0.0,max(0.1,own_radius))
                # Cheap pre-score before expensive LiDAR raycasting.
                unknown_density=min(2.0,local_unknown/45.0)
                known_support=min(1.0,local_known/20.0)
                target_inside_unknown=1.0 if bool(unknown_mask[j,i]) else 0.0
                clearance_score=min(1.5,cl/desired)
                failed=float(self._failed_goal_penalty(p))
                score=(
                    2.0*unknown_density+
                    0.75*target_inside_unknown+
                    0.65*known_support+
                    1.15*clearance_score+
                    0.75*region_focus-
                    0.28*d-
                    7.5*teammate_soft-
                    1.8*own_soft-
                    7.5*reserved_pen-
                    2.0*failed
                )
                bd={
                    "score":float(score),
                    "nbv_mode":float(2.0 if relaxed else 1.0),
                    "target_inside_unknown":float(target_inside_unknown),
                    "local_unknown_cells":float(local_unknown),
                    "local_known_free_cells":float(local_known),
                    "unknown_density_score":float(unknown_density),
                    "raw_clearance_m":float(cl),
                    "clearance_score":float(clearance_score),
                    "region_focus_score":float(region_focus),
                    "region_focus_reward":float(0.75*region_focus),
                    "distance_cost":float(0.28*d),
                    "teammate_min_distance_m":float(team_min if math.isfinite(team_min) else 9999.0),
                    "teammate_hard_avoid_radius_m":float(hard_radius),
                    "teammate_soft_avoid_radius_m":float(soft_radius),
                    "teammate_pose_penalty":float(7.5*teammate_soft),
                    "own_min_distance_m":float(own_min if math.isfinite(own_min) else 9999.0),
                    "own_path_penalty":float(1.8*own_soft),
                    "reservation_penalty":float(7.5*reserved_pen),
                    "failed_goal_penalty":float(2.0*failed),
                    "strict_candidates_blocked_by_teammate":float(blocked_by_teammate),
                    "strict_candidates_blocked_by_reservation":float(blocked_by_reservation),
                    "near_candidates_skipped":float(skipped_near),
                    "low_gain_candidates_skipped":float(skipped_low_gain),
                }
                pre.append((float(score),p,bd))
        if not pre:
            return []
        # Raycast only the best rough candidates; this keeps the GUI responsive.
        out=[]
        for prelim,p,bd in sorted(pre,key=lambda x:x[0],reverse=True)[:max_candidates]:
            expected_gain=self._expected_lidar_visibility_gain(p,unknown_mask,occupied_mask)
            if expected_gain<1.0:
                continue
            gain_score=math.log1p(expected_gain)
            final=prelim+2.6*gain_score
            info=dict(bd)
            info.update({
                "score":float(final),
                "pre_lidar_score":float(prelim),
                "expected_lidar_unknown_gain":float(expected_gain),
                "lidar_gain_score":float(gain_score),
                "lidar_gain_reward":float(2.6*gain_score),
            })
            out.append((float(final),p,info))
        return out

    def _best_planned_nbv_candidate(self,candidates:list[tuple[float,Point,dict[str,float]]],dynamic_obstacles:list[tuple[Point,float]],team_points:list[Point],own_points:list[Point],hard_radius:float,soft_radius:float,own_radius:float,relaxed:bool)->tuple[float,Point,dict[str,float]]|None:
        if not candidates:
            return None
        desired=max(1e-6,float(self.cfg.planning.desired_clearance_m))
        best=None
        eval_count=max(1,int(self.cfg.planning.nbv_plan_eval_count))
        for prelim,goal,bd in sorted(candidates,key=lambda x:x[0],reverse=True)[:eval_count]:
            result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
            if not result.success or len(result.path)<2:
                continue
            path_len=self._path_length(result.path)
            unknown_frac=self._path_unknown_fraction(result.path)
            path_team_overlap=self._path_history_overlap(result.path,team_points,hard_radius,soft_radius)
            path_own_overlap=self._path_history_overlap(result.path,own_points,0.0,max(0.1,own_radius))
            if (not relaxed) and path_team_overlap>0.70:
                continue
            clearance_bonus=1.0*min(1.5,float(result.min_clearance)/desired)
            unknown_path_bonus=0.55*unknown_frac
            detour_cost=0.20*path_len
            path_team_pen=7.0*path_team_overlap
            path_own_pen=1.2*path_own_overlap
            final=prelim+clearance_bonus+unknown_path_bonus-detour_cost-path_team_pen-path_own_pen
            out=dict(bd)
            out.update({
                "score":float(final),
                "pre_plan_score":float(prelim),
                "planned_path_length":float(path_len),
                "planned_path_clearance":float(result.min_clearance),
                "planned_path_unknown_fraction":float(unknown_frac),
                "path_clearance_bonus":float(clearance_bonus),
                "path_unknown_exploration_bonus":float(unknown_path_bonus),
                "path_detour_cost":float(detour_cost),
                "path_teammate_overlap":float(path_team_overlap),
                "path_own_overlap":float(path_own_overlap),
                "path_teammate_penalty":float(path_team_pen),
                "path_own_penalty":float(path_own_pen),
            })
            if best is None or final>best[0]:
                best=(float(final),goal,out)
        return best

    def _local_cell_count(self,mask:np.ndarray,i:int,j:int,radius:int)->int:
        y0=max(0,j-radius); y1=min(mask.shape[0],j+radius+1)
        x0=max(0,i-radius); x1=min(mask.shape[1],i+radius+1)
        return int(np.count_nonzero(mask[y0:y1,x0:x1]))

    def _exploration_safe_approach_point(self,frontier:FrontierCluster,clearance:np.ndarray,free:np.ndarray,known:np.ndarray)->Point:
        centroid=frontier.centroid_world
        ccell=self.map.world_to_cell(centroid)
        if ccell is None:
            return centroid
        radius=max(1,int(math.ceil(self.cfg.planning.safe_approach_search_radius_m/self.map.res)))
        ci,cj=ccell
        best_cell=None; best_score=-math.inf
        desired=max(1e-6,self.cfg.planning.desired_clearance_m)
        for j in range(max(0,cj-radius),min(self.map.ny,cj+radius+1)):
            for i in range(max(0,ci-radius),min(self.map.nx,ci+radius+1)):
                if not free[j,i] or not known[j,i]:
                    continue
                cl=float(clearance[j,i])
                if cl<self.cfg.planning.safe_approach_min_clearance_m:
                    continue
                p=self.map.cell_to_world((i,j))
                dc=distance(p,centroid)
                ds=distance(p,self.est_xy)
                # No sector ownership here.  Just choose a safe point near the
                # frontier boundary with good corridor center clearance.
                score=2.4*min(1.5,cl/desired)-0.38*dc-0.025*ds
                if score>best_score:
                    best_score=score; best_cell=(i,j)
        if best_cell is not None:
            return self.map.cell_to_world(best_cell)
        return self.map.safe_approach_point(frontier,self.est_xy,self.cfg.planning.safe_approach_search_radius_m,self.cfg.planning.safe_approach_min_clearance_m,self.cfg.planning.desired_clearance_m,clearance=clearance,free=free,known=known)

    def _team_history_points(self,team_paths:dict[int,list[Point]],team_visits:dict[int,list[Point]],team_trajectories:dict[int,list[Point]])->list[Point]:
        pts:list[Point]=[]
        for source in (team_paths,team_visits,team_trajectories):
            for rid,path in source.items():
                if rid==self.id or not path:
                    continue
                pts.extend(self._downsample_points(path,max_points=70))
        return self._downsample_points(pts,max_points=180)

    def _downsample_points(self,pts:list[Point],max_points:int)->list[Point]:
        if not pts:
            return []
        if len(pts)<=max_points:
            return [(float(x),float(y)) for x,y in pts]
        idx=np.linspace(0,len(pts)-1,max_points).astype(int)
        return [(float(pts[int(i)][0]),float(pts[int(i)][1])) for i in idx]

    def _min_distance_to_points(self,p:Point,pts:list[Point])->float:
        if not pts:
            return math.inf
        return float(min(distance(p,q) for q in pts))

    def _distance_band_penalty(self,d:float,hard_radius:float,soft_radius:float)->float:
        if not math.isfinite(d):
            return 0.0
        hard=max(0.0,float(hard_radius))
        soft=max(hard+1e-6,float(soft_radius))
        if d<=hard:
            return 1.0
        if d>=soft:
            return 0.0
        t=(soft-d)/(soft-hard)
        return float(max(0.0,min(1.0,t*t)))

    def _reservation_overlap_penalty(self,p:Point,reserved_frontiers:dict[int,Point],radius:float)->float:
        if not reserved_frontiers or radius<=1e-9:
            return 0.0
        best=0.0
        for rid,g in reserved_frontiers.items():
            if rid==self.id or g is None:
                continue
            d=distance(p,g)
            if d<radius:
                best=max(best,1.0-d/max(1e-6,radius))
        return float(best)

    def _path_history_overlap(self,path:list[Point],history:list[Point],hard_radius:float,soft_radius:float)->float:
        if not path or not history:
            return 0.0
        sampled_path=self._downsample_points(path,max_points=45)
        sampled_hist=self._downsample_points(history,max_points=140)
        vals=[]
        for p in sampled_path:
            d=self._min_distance_to_points(p,sampled_hist)
            vals.append(self._distance_band_penalty(d,hard_radius,soft_radius))
        return float(sum(vals)/max(1,len(vals)))

    def _path_unknown_fraction(self,path:list[Point])->float:
        if len(path)<2:
            return 1.0
        known=self.map.known_mask()
        cells=[]; seen=set()
        for a,b in zip(path[:-1],path[1:]):
            ca=self.map.world_to_cell(a); cb=self.map.world_to_cell(b)
            if ca is None or cb is None: continue
            for c in self.map._bresenham(ca,cb):
                if c not in seen:
                    seen.add(c); cells.append(c)
        if not cells:
            return 1.0
        unknown=sum(1 for i,j in cells if not known[j,i])
        return float(unknown)/float(len(cells))

    def _path_length(self,path:list[Point])->float:
        return float(sum(distance(a,b) for a,b in zip(path[:-1],path[1:]))) if len(path)>=2 else 0.0

    def _expected_lidar_visibility_gain(self,viewpoint:Point,unknown:np.ndarray,occupied:np.ndarray)->float:
        start=self.map.world_to_cell(viewpoint)
        if start is None:
            return 0.0
        rays=max(8,int(self.cfg.planning.frontier_visibility_rays))
        max_range=float(self.cfg.lidar.range)
        step_back=max(self.map.res,0.25)
        seen:set[tuple[int,int]]=set()
        gain=0.0
        for a in np.linspace(-math.pi,math.pi,rays,endpoint=False):
            rr=max_range
            end_cell=None
            while rr>self.map.res:
                end=(viewpoint[0]+math.cos(float(a))*rr,viewpoint[1]+math.sin(float(a))*rr)
                end_cell=self.map.world_to_cell(end)
                if end_cell is not None:
                    break
                rr-=step_back
            if end_cell is None:
                continue
            cells=self.map._bresenham(start,end_cell)
            for i,j in cells[1:]:
                if occupied[j,i]:
                    break
                if unknown[j,i] and (i,j) not in seen:
                    seen.add((i,j))
                    d=math.hypot((i-start[0])*self.map.res,(j-start[1])*self.map.res)
                    gain+=math.exp(-0.10*d)
        return float(gain)

    def _best_planned_frontier_candidate(self,candidates:list[tuple[float,Point,dict[str,float]]],dynamic_obstacles:list[tuple[Point,float]])->tuple[float,Point,dict[str,float]]|None:
        if not candidates:
            return None
        desired=max(1e-6,self.cfg.planning.desired_clearance_m)
        best=None
        eval_count=max(1,int(self.cfg.planning.frontier_plan_eval_count))
        for prelim,goal,bd in sorted(candidates,key=lambda x:x[0],reverse=True)[:eval_count]:
            result=self.planner.plan(self.map,self.est_xy,goal,dynamic_obstacles=dynamic_obstacles)
            if not result.success or len(result.path)<2:
                continue
            path_len=self._path_length(result.path)
            unknown_frac=self._path_unknown_fraction(result.path)
            clearance_bonus=self.cfg.planning.frontier_path_clearance_weight*min(1.5,result.min_clearance/desired)
            unknown_penalty=self.cfg.planning.frontier_path_unknown_penalty_weight*unknown_frac
            detour_cost=0.18*self.cfg.planning.distance_weight*path_len
            final=prelim+clearance_bonus-unknown_penalty-detour_cost
            out=dict(bd)
            out.update({
                "score":float(final),
                "pre_plan_score":float(prelim),
                "planned_path_length":float(path_len),
                "planned_path_clearance":float(result.min_clearance),
                "planned_path_unknown_fraction":float(unknown_frac),
                "path_clearance_bonus":float(clearance_bonus),
                "path_unknown_penalty":float(unknown_penalty),
                "path_detour_cost":float(detour_cost),
            })
            if best is None or final>best[0]:
                best=(float(final),goal,out)
        return best

    def _lidar_open_direction_reward(self,p:Point)->float:
        if self.assessment.open_sector_count<=0:
            return 0.0
        rel=wrap_angle(angle_to(self.est_xy,p)-self.est_pose[2])
        err=abs(wrap_angle(rel-self.assessment.best_open_angle))
        clearance_scale=float(np.clip(self.assessment.front_clearance/max(1e-6,self.cfg.lidar.range),0.15,1.0))
        return float(math.exp(-((err/0.75)**2))*clearance_scale)

    def _target_progress_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        cur=distance(self.est_xy,self.target.xy)
        cand=distance(p,self.target.xy)
        return float(np.clip((cur-cand)/max(1.0,self.cfg.lidar.range),-0.5,1.5))

    def _robot_target_corridor_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        sx,sy=self.est_xy; tx,ty=self.target.xy; px,py=p
        vx,vy=tx-sx,ty-sy; vv=vx*vx+vy*vy
        if vv<=1e-9:
            return 0.0
        t=max(0.0,min(1.0,((px-sx)*vx+(py-sy)*vy)/vv))
        cx,cy=sx+t*vx,sy+t*vy
        d=math.hypot(px-cx,py-cy)
        width=max(0.6,float(self.cfg.cage.target_corridor_width_m)*0.65)
        return float(math.exp(-((d/width)**2))*(0.35+0.65*t))

    def _target_probe_goal(self,target_goal:Point, allow_beyond_target:bool=False)->Point|None:
        base=angle_to(self.est_xy,target_goal)
        offsets=[0.0,math.radians(14),-math.radians(14),math.radians(28),-math.radians(28),math.radians(45),-math.radians(45)]
        if self.assessment.blocked_forward or self.assessment.front_clearance<self.cfg.planning.critical_clearance_m:
            offsets=[self.assessment.best_open_angle]+offsets
        best=None; best_score=-math.inf
        max_step=min(self.cfg.lidar.range*0.70,3.4,distance(self.est_xy,target_goal))
        if max_step<=self.cfg.robot.goal_tolerance:
            if not allow_beyond_target:
                return target_goal
            base=self.est_pose[2]+self.assessment.best_open_angle
            max_step=min(self.cfg.lidar.range*0.55,2.4)
        traversable=self.map.traversable_mask(self.cfg.planning.inflation_radius_m)
        for dist_m in np.linspace(max(0.9,self.cfg.robot.goal_tolerance*2.0),max_step,5):
            for off in offsets:
                a=base+off
                p=(self.est_xy[0]+math.cos(a)*float(dist_m),self.est_xy[1]+math.sin(a)*float(dist_m))
                cell=self.map.world_to_cell(p)
                if cell is None:
                    continue
                i,j=cell
                if not traversable[j,i]:
                    continue
                cl=self.map.clearance_at(p,max_radius_m=2.5)
                if cl<self.cfg.planning.critical_clearance_m:
                    continue
                progress=self._target_progress_reward(p)
                corridor=self._robot_target_corridor_reward(p)
                score=6.0*progress+2.2*corridor+min(1.5,cl/max(1e-6,self.cfg.planning.desired_clearance_m))-0.04*distance(self.est_xy,p)
                if score>best_score:
                    best_score=score; best=(float(p[0]),float(p[1]))
        return best

    def _target_corridor_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        hx,hy=self.home_xy; tx,ty=self.target.xy; px,py=p
        vx,vy=tx-hx,ty-hy; vv=vx*vx+vy*vy
        if vv<=1e-9:
            return 0.0
        t=max(0.0,min(1.0,((px-hx)*vx+(py-hy)*vy)/vv))
        cx,cy=hx+t*vx,hy+t*vy
        d=math.hypot(px-cx,py-cy)
        width=max(0.5,float(self.cfg.cage.target_corridor_width_m))
        tube=math.exp(-((d/width)**2))
        along=0.45+0.55*t
        return float(tube*along)

    def _target_corridor_low_quality_reward(self,p:Point)->float:
        if not self.target.detected or self.target.xy is None:
            return 0.0
        base=self._target_corridor_reward(p)
        if base<=1e-6:
            return 0.0
        cell=self.map.world_to_cell(p)
        if cell is None:
            return 0.0
        i,j=cell
        radius=max(1,int(round(1.2/self.map.res)))
        y0,y1=max(0,j-radius),min(self.map.ny,j+radius+1)
        x0,x1=max(0,i-radius),min(self.map.nx,i+radius+1)
        known=self.map.known_mask()[y0:y1,x0:x1]
        quality=np.clip(self.map.quality[y0:y1,x0:x1],0.0,1.0)
        if known.size==0:
            return 0.0
        unknown_frac=1.0-float(np.count_nonzero(known))/float(known.size)
        mean_q=float(np.mean(quality[known])) if np.any(known) else 0.0
        lowq=max(0.0,1.0-mean_q)
        return float(base*(0.65*unknown_frac+0.35*lowq))

    def _fast_target_probe_goal(self,target_goal:Point)->Point|None:
        d_target=distance(self.est_xy,target_goal)
        if d_target<=self.cfg.robot.goal_tolerance:
            return None
        target_angle=angle_to(self.est_xy,target_goal)
        base=target_angle
        if self.assessment.blocked_forward or self.assessment.front_clearance<self.cfg.planning.critical_clearance_m:
            base=self.est_pose[2]+self.assessment.best_open_angle
        offsets=[0.0]
        if abs(wrap_angle(base-target_angle))>0.10:
            offsets.append(wrap_angle(target_angle-base))
        offsets.extend([math.radians(18.0),-math.radians(18.0),math.radians(34.0),-math.radians(34.0)])
        traversable=self.map.traversable_mask(self.cfg.planning.inflation_radius_m)
        step_max=min(float(self.cfg.lidar.range)*0.55,2.6,d_target)
        step_min=min(step_max,max(0.75,float(self.cfg.robot.goal_tolerance)*1.7))
        for dist_m in np.linspace(step_max,step_min,4):
            for off in offsets:
                a=base+float(off)
                p=(float(self.est_xy[0]+math.cos(a)*float(dist_m)),float(self.est_xy[1]+math.sin(a)*float(dist_m)))
                cell=self.map.world_to_cell(p)
                if cell is None:
                    continue
                i,j=cell
                if not traversable[j,i]:
                    continue
                return p
        return None

    def _should_report_target_to_home(self)->bool:
        if self.target.source_robot==self.id: return True
        if any(task=="REPORT_TARGET_HOME" for task in self.known_teammate_tasks.values()): return False
        my_home_d=distance(self.est_xy,self.home_xy)
        teammate_home_distances=[
            distance((float(pose[0]),float(pose[1])),self.home_xy)
            for pose in self.known_teammate_pose.values()
        ]
        return not teammate_home_distances or my_home_d<=min(teammate_home_distances)+1.5
    def _best_local_target_route(self):
        routes=self.graph.top_routes(k=1)
        return routes[0] if routes else None
    def _failed_goal_penalty(self,p:Point)->float:
        return math.exp(-min(distance(p,q) for q in self.failed_goal_memory)/2.2) if self.failed_goal_memory else 0.0
    def _sector_probe_goal(self)->Point|None:
        mission_angle=angle_to(self.home_xy,self.search_prior_xy)
        max_d=min(self.cfg.lidar.range*0.75,3.8)
        traversable=self.map.traversable_mask(self.cfg.planning.inflation_radius_m)
        known=self.map.known_mask()
        for dist_m in np.linspace(max(1.4,self.cfg.robot.goal_tolerance*2.5),max_d,6):
            for off in (0.0,math.radians(18),-math.radians(18),math.radians(36),-math.radians(36)):
                a=mission_angle+off
                p=(self.est_xy[0]+math.cos(a)*float(dist_m),self.est_xy[1]+math.sin(a)*float(dist_m))
                cell=self.map.world_to_cell(p)
                if cell is None:
                    continue
                i,j=cell
                if not traversable[j,i] or not known[j,i]:
                    continue
                cl=self.map.clearance_at(p,max_radius_m=2.5)
                if cl>=self.cfg.planning.critical_clearance_m:
                    return (float(p[0]),float(p[1]))
        return None

    def _nearest_anchor(self)->Point|None:
        anchors=[n.xy for n in self.graph.nodes.values() if n.kind in {"home","anchor"}]
        return min(anchors,key=lambda p:distance(self.est_xy,p)) if anchors else None
    def _downsample_path(self,path:list[Point],spacing:float)->list[Point]:
        if len(path)<=2: return path
        out=[path[0]]; last=path[0]
        for p in path[1:-1]:
            if distance(last,p)>=spacing: out.append(p); last=p
        out.append(path[-1]); return out
    def _target_fallback_path(self,goal:Point)->list[Point]:
        if goal is None:
            return []
        d=distance(self.est_xy,goal)
        if d<=self.cfg.robot.goal_tolerance:
            return []
        probe=self._target_probe_goal(goal)
        if probe is None:
            return []
        return [self.est_xy,probe]

    def _homing_fallback_path(self,goal:Point)->list[Point]:
        d=distance(self.est_xy,goal)
        if d<=self.cfg.robot.goal_tolerance: return []
        desired=angle_to(self.est_xy,goal)
        if self.assessment.blocked_forward or self.assessment.front_clearance<self.cfg.planning.critical_clearance_m:
            desired=self.est_pose[2]+self.assessment.best_open_angle
        step=min(1.6,max(0.6,d))
        p=(self.est_xy[0]+math.cos(desired)*step,self.est_xy[1]+math.sin(desired)*step)
        cell=self.map.world_to_cell(p)
        if cell is None:
            return []
        i,j=cell
        traversable=self.map.traversable_mask(self.cfg.planning.inflation_radius_m)
        if not traversable[j,i]:
            return []
        return [self.est_xy,p]
    def _teammate_avoidance_control(self)->tuple[float,float]:
        turn=0.0
        speed_scale=1.0
        base=2.0*float(self.cfg.robot.radius)+float(self.cfg.robot.collision_buffer_m)
        horizon=max(0.1,float(self.cfg.robot.collision_avoidance_horizon_m))
        for rid,pose in self.known_teammate_pose.items():
            if rid==self.id:
                continue
            mate=(float(pose[0]),float(pose[1]))
            d=distance(self.est_xy,mate)
            cov=float(self.known_teammate_cov.get(rid,0.0))
            cov_extra=min(float(self.cfg.planning.dynamic_obstacle_max_cov_extra_m),0.55*math.sqrt(max(0.0,cov)*0.5))
            hard=base+cov_extra
            slow_radius=hard+horizon
            if d>=slow_radius:
                continue
            rel=wrap_angle(angle_to(self.est_xy,mate)-self.est_pose[2])
            strength=float(np.clip((slow_radius-d)/horizon,0.0,1.0))
            if abs(rel)<math.radians(115.0):
                side=-1.0 if rel>=0.0 else 1.0
                if abs(rel)<0.08:
                    side=-1.0 if rid<self.id else 1.0
                turn+=side*float(self.cfg.robot.teammate_avoidance_turn_gain)*strength
            if abs(rel)<math.radians(70.0):
                clear=max(0.0,d-hard)
                speed_scale=min(speed_scale,float(np.clip(clear/max(0.1,horizon),0.0,1.0)))
        return float(np.clip(speed_scale,0.0,1.0)),float(turn)

    def _lidar_obstacle_points_robot_frame(self)->np.ndarray:
        if self.scan is None:
            return np.zeros((0,2),dtype=float)
        max_range=float(self.cfg.lidar.range)-float(self.cfg.lidar.hit_threshold)
        ranges=np.asarray(self.scan.ranges,dtype=float)
        mask=np.asarray(self.scan.hit,dtype=bool) | (ranges<max_range)
        if not np.any(mask):
            return np.zeros((0,2),dtype=float)
        angles=np.asarray(self.scan.angles,dtype=float)[mask]
        ranges=ranges[mask]
        if len(ranges)>24:
            idx=np.linspace(0,len(ranges)-1,24,dtype=int)
            angles=angles[idx]
            ranges=ranges[idx]
        return np.column_stack((np.cos(angles)*ranges,np.sin(angles)*ranges))

    def _rollout_min_lidar_clearance(self,v:float,omega:float,obstacles:np.ndarray)->tuple[bool,float,float,float,float]:
        if obstacles.size==0:
            horizon=max(float(self.cfg.robot.lidar_safety_time_horizon_s),float(self.cfg.dt))
            if abs(omega)<1e-6:
                return True, float(self.cfg.lidar.range), v*horizon, 0.0, omega*horizon
            radius=v/omega
            th=omega*horizon
            return True, float(self.cfg.lidar.range), radius*math.sin(th), radius*(1.0-math.cos(th)), th
        horizon=max(float(self.cfg.robot.lidar_safety_time_horizon_s),float(self.cfg.dt))
        step=max(0.08,min(0.24,float(self.cfg.dt)*2.0))
        steps=max(4,int(math.ceil(horizon/step)))
        half_l=0.5*float(self.cfg.robot.body_length)+float(self.cfg.robot.collision_buffer_m)+float(self.cfg.robot.lidar_safety_stop_margin_m)
        half_w=0.5*float(self.cfg.robot.body_width)+float(self.cfg.robot.collision_buffer_m)+float(self.cfg.robot.lidar_safety_stop_margin_m)
        min_clear_sq=math.inf
        x=y=th=0.0
        cxs=obstacles[:,0]
        cys=obstacles[:,1]
        for _ in range(steps):
            if abs(omega)<1e-6:
                x+=v*math.cos(th)*step
                y+=v*math.sin(th)*step
            else:
                old=th
                th=wrap_angle(th+omega*step)
                radius=v/omega
                x+=radius*(math.sin(th)-math.sin(old))
                y-=radius*(math.cos(th)-math.cos(old))
            ct=math.cos(th); st=math.sin(th)
            dx=cxs-x; dy=cys-y
            bx=ct*dx+st*dy
            by=-st*dx+ct*dy
            abs_bx=np.abs(bx)
            abs_by=np.abs(by)
            inside=(abs_bx<=half_l)&(abs_by<=half_w)
            if np.any(inside):
                return False, -float(np.max(np.minimum(half_l-abs_bx[inside],half_w-abs_by[inside]))), x, y, th
            ox=np.maximum(abs_bx-half_l,0.0)
            oy=np.maximum(abs_by-half_w,0.0)
            outside_sq=ox*ox+oy*oy
            if outside_sq.size:
                min_clear_sq=min(min_clear_sq,float(np.min(outside_sq)))
        clear=math.sqrt(min_clear_sq) if math.isfinite(min_clear_sq) else float(self.cfg.lidar.range)
        return True,float(clear),x,y,th

    def _lidar_local_planner(self,target:Point,desired_v:float,desired_omega:float)->tuple[float,float]:
        obstacles=self._lidar_obstacle_points_robot_frame()
        max_omega=1.55
        target_dx=target[0]-self.est_xy[0]
        target_dy=target[1]-self.est_xy[1]
        ct=math.cos(-self.est_pose[2]); st=math.sin(-self.est_pose[2])
        target_local=(ct*target_dx-st*target_dy,st*target_dx+ct*target_dy)
        target_dist=max(1e-6,math.hypot(target_local[0],target_local[1]))
        target_unit=(target_local[0]/target_dist,target_local[1]/target_dist)
        slow_margin=max(1e-6,float(self.cfg.robot.lidar_safety_slow_margin_m))
        desired_v=float(np.clip(desired_v,-float(self.cfg.robot.lidar_reverse_speed),float(self.cfg.robot.max_speed)))
        desired_omega=float(np.clip(desired_omega,-max_omega,max_omega))
        desired_safe,desired_clear,_,_,_=self._rollout_min_lidar_clearance(desired_v,desired_omega,obstacles)
        if desired_safe and desired_clear>=slow_margin and not self.assessment.blocked_forward:
            return desired_v,desired_omega
        speed_cap=max(float(desired_v),0.85*float(self.cfg.robot.max_speed))
        speeds=list(np.linspace(0.0,min(float(self.cfg.robot.max_speed),speed_cap),int(self.cfg.robot.local_planner_speed_samples)))
        speeds.append(desired_v)
        if self.assessment.blocked_forward or self.assessment.front_clearance<float(self.cfg.lidar.blocked_forward_distance)+0.18:
            speeds.append(-float(self.cfg.robot.lidar_reverse_speed))
        omegas=list(np.linspace(-max_omega,max_omega,int(self.cfg.robot.local_planner_omega_samples)))
        omegas.extend([desired_omega,float(np.clip(self.cfg.robot.turn_gain*self.assessment.best_open_angle,-max_omega,max_omega))])
        speeds=sorted({round(float(v),3) for v in speeds if -float(self.cfg.robot.lidar_reverse_speed)-1e-6<=v<=float(self.cfg.robot.max_speed)+1e-6})
        omegas=sorted({round(float(w),3) for w in omegas if -max_omega-1e-6<=w<=max_omega+1e-6})
        best=None
        for v in speeds:
            for omega in omegas:
                safe,clearance,end_x,end_y,end_th=self._rollout_min_lidar_clearance(v,omega,obstacles)
                if not safe:
                    continue
                progress=(end_x*target_unit[0]+end_y*target_unit[1])/max(1.0,target_dist)
                heading_err=wrap_angle(math.atan2(target_local[1]-end_y,target_local[0]-end_x)-end_th)
                heading_score=0.5+0.5*math.cos(heading_err)
                clear_score=float(np.clip(clearance/slow_margin,0.0,1.0))
                speed_score=float(np.clip(v/max(1e-6,float(self.cfg.robot.max_speed)),0.0,1.0))
                command_match=1.0-float(np.clip(abs(v-desired_v)/max(1e-6,float(self.cfg.robot.max_speed))+0.35*abs(omega-desired_omega)/max_omega,0.0,1.0))
                reverse_penalty=1.2 if v<0.0 else 0.0
                score=3.0*heading_score+2.6*clear_score+2.0*progress+0.95*speed_score+0.35*command_match-0.18*abs(omega)/max_omega-reverse_penalty
                if best is None or score>best[0]:
                    best=(score,float(v),float(omega),clearance)
        if best is not None:
            return best[1],best[2]
        fallback_turn=float(np.clip(self.cfg.robot.turn_gain*self.assessment.best_open_angle,-max_omega,max_omega))
        if abs(fallback_turn)<0.25:
            fallback_turn=0.55 if self.assessment.right_clearance>=self.assessment.left_clearance else -0.55
        return 0.0,fallback_turn

    def compute_control(self)->tuple[float,float]:
        if not self.path or self.path_index>=len(self.path): self.last_command=(0.0,0.0); return self.last_command
        pos=self.est_xy; target=self.path[self.path_index]
        if distance(pos,target)<self.cfg.robot.waypoint_tolerance:
            self.path_index+=1
            if self.path_index>=len(self.path): self.last_command=(0.0,0.0); return self.last_command
            target=self.path[self.path_index]
        desired=angle_to(pos,target); err=wrap_angle(desired-self.est_pose[2])
        desired_clear=max(1e-6,float(self.cfg.planning.desired_clearance_m))
        side_clear=min(self.assessment.left_clearance,self.assessment.right_clearance)
        front_warn=max(float(self.cfg.robot.obstacle_slowdown_start_m),float(self.cfg.lidar.blocked_forward_distance)+0.10)
        avoid_speed_scale,avoid_turn=self._teammate_avoidance_control()
        desired_omega=float(np.clip(self.cfg.robot.turn_gain*err+avoid_turn,-1.55,1.55))
        consistency_scale=0.45 if self.assessment.consistency<self.cfg.assessment.caution_consistency else 1.0
        front_scale=np.clip((self.assessment.front_clearance-self.cfg.lidar.blocked_forward_distance)/max(1e-6,front_warn-self.cfg.lidar.blocked_forward_distance),0.0,1.0)
        front_scale=max(float(self.cfg.robot.obstacle_min_speed_scale),float(front_scale)**1.05)
        side_scale=np.clip(side_clear/desired_clear,0.18,1.0)
        desired_v=self.cfg.robot.max_speed*consistency_scale*front_scale*side_scale*max(0.16,math.cos(err))*avoid_speed_scale
        v,omega=self._lidar_local_planner(target,float(desired_v),float(desired_omega))
        self.last_command=(float(v),omega); return self.last_command


# ============================================================================
# src / communication.py
# ============================================================================

"""LOS/team packet communication and HOME memory."""
from collections import deque
from dataclasses import dataclass, field


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


# ============================================================================
# src / simulator.py
# ============================================================================

"""Search-CAGE baseline simulator orchestrator."""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MissionStatus:
    phase: str = "SEARCH_TARGET"
    success: bool = False
    message: str = "Searching for hidden target"





@dataclass
class PassageStatus:
    known: bool = False
    safe: bool = False
    score: float = 0.0
    length: float = 0.0
    mean_quality: float = 0.0
    min_clearance: float = 0.0
    unknown_fraction: float = 1.0
    message: str = "No target passage yet"


class Simulator:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig()
        self.cfg.validate()
        self.rng = np.random.default_rng(self.cfg.world.seed)
        self.world = World(self.cfg.world)
        self.time_s = 0.0
        self.step_count = 0
        self.running = True
        self.robots: list[RobotAgent] = []
        self.home_memory = HomeMemory(
            map=OccupancyGrid(self.world.width, self.world.height, self.cfg.mapping),
            graph=RouteGraph(self.cfg.cage.edge_merge_distance),
        )
        self.home_memory.graph.add_node(self.world.home, kind="home", confidence=1.0, allow_merge=False)
        self.communication = CommunicationManager(self.cfg.communication, self.world, self.home_memory, self.cfg.target_reporting)
        self.comm_state = CommunicationState()
        self.mission = MissionStatus()
        self.passage_status = PassageStatus()
        self._last_passage_eval_time = -1.0e9
        self._last_passage_target: tuple[float, float] | None = None
        self.exploration_return_requested = False
        self._exploration_complete_counter = 0
        self._spawn_robots()

    def _spawn_robots(self) -> None:
        self.robots = []
        hx, hy = self.world.home
        n = self.cfg.robot.count
        for i in range(n):
            angle = 2.0 * math.pi * i / max(1, n)
            r = self.cfg.robot.spawn_spacing
            pose = (hx + math.cos(angle) * r, hy + math.sin(angle) * r, angle)
            if not self.world.is_free((pose[0], pose[1]), margin=self.cfg.robot.radius):
                pose = (hx, hy, angle)
            robot_rng = np.random.default_rng(self.cfg.world.seed + 101 * (i + 1))
            self.robots.append(RobotAgent(i, pose, self.cfg, self.world, robot_rng))

    def reset(self, cfg: AppConfig | None = None) -> None:
        self.__init__(cfg or self.cfg)

    def step(self) -> None:
        dt = self.cfg.dt
        self.time_s += dt
        self.step_count += 1

        for robot in self.robots:
            peer_poses=[tuple(other.true_pose) for other in self.robots if other.id!=robot.id]
            robot.step_predict_and_move(self.world, dt, peer_poses=peer_poses)
        for robot in self.robots:
            robot.sense_update_map_and_belief(self.world, self.time_s)

        self.comm_state = self.communication.update(self.robots, self.time_s)
        robot_by_id = {r.id: r for r in self.robots}
        for a_id, b_id in self.comm_state.direct_robot_edges:
            a = robot_by_id[a_id]
            b = robot_by_id[b_id]
            a.update_localization_from_teammate(b, self.world, self.time_s)
            b.update_localization_from_teammate(a, self.world, self.time_s)
        self._update_target_roundtrip_flags()
        self._update_exploration_return_flags()

        reserved_goals: dict[int, tuple[float, float]] = {}
        reserved_frontiers: dict[int, tuple[float, float]] = {}
        for robot in self.robots:
            robot.choose_task_and_plan(self.time_s, reserved_goals=reserved_goals, reserved_frontiers=reserved_frontiers)
            if robot.current_goal is not None and robot.current_task in {"SEARCH_HIER_NBV", "SEARCH_NBV", "SEARCH_FRONTIER", "SEARCH_TARGET_BELIEF", "SEARCH_OPEN_SECTOR", "DEPLOY_FROM_HOME", "EXPLORE_TOWARD_TARGET", "GO_TO_TARGET", "VERIFY_ROUTE_EVIDENCE", "RELAY_EVIDENCE_HOME"}:
                reserved_goals[robot.id] = robot.current_goal
            if robot.current_goal is not None and robot.current_task in {"SEARCH_HIER_NBV", "SEARCH_NBV", "SEARCH_FRONTIER", "SEARCH_TARGET_BELIEF", "DEPLOY_FROM_HOME", "SEARCH_OPEN_SECTOR"}:
                reserved_frontiers[robot.id] = robot.current_goal
        for robot in self.robots:
            robot.compute_control()

        self._update_mission_status()


    def _required_roundtrip_count(self) -> int:
        if self.cfg.cage.require_all_robots_target_roundtrip:
            return len(self.robots)
        return max(1, min(len(self.robots), int(self.cfg.cage.min_robots_completed_roundtrip)))

    def _update_target_roundtrip_flags(self) -> None:
        # Target-roundtrip progression is evaluated by the simulator because it
        # can check physical target visibility and HOME arrival without leaking
        # truth into robot planning. Robots still plan only from their estimated
        # pose and communication-limited maps.
        target_known = self.home_memory.target.detected or any(r.target.detected for r in self.robots)
        if not target_known:
            return
        arrival_radius = max(self.cfg.world.target_radius, float(self.cfg.cage.target_arrival_radius_m))
        for robot in self.robots:
            if not robot.target.detected or robot.completed_target_roundtrip:
                continue
            true_xy = (float(robot.true_pose[0]), float(robot.true_pose[1]))
            if not robot.target_reached:
                est_close = robot.target.xy is not None and math.hypot(robot.est_xy[0]-robot.target.xy[0], robot.est_xy[1]-robot.target.xy[1]) <= arrival_radius
                target_visible = self.world.target_visible(tuple(robot.true_pose), self.cfg.lidar.range)
                if target_visible:
                    robot.mark_target_reached(self.time_s)
                    uploaded = self.communication.upload_robot_to_home(
                        robot,
                        self.time_s,
                        full=True,
                        robots=self.robots,
                        require_connection=self.cfg.target_reporting.require_home_connection_for_target_report,
                    )
                    if uploaded:
                        robot.last_home_full_upload_time = self.time_s
                elif est_close:
                    robot.status.note = "estimated_target_close_without_visual_confirmation"
            if robot.target_reached and self.world.home_base.contains(true_xy):
                robot.mark_target_roundtrip_complete(self.time_s)
                uploaded = self.communication.upload_robot_to_home(
                    robot,
                    self.time_s,
                    full=True,
                    robots=self.robots,
                    require_connection=True,
                )
                if uploaded:
                    robot.last_home_full_upload_time = self.time_s

    def _update_exploration_return_flags(self) -> None:
        # Target workflow has priority over generic exploration completion.
        if self.home_memory.target.detected or any(r.target.detected for r in self.robots):
            return
        if self.exploration_return_requested:
            for robot in self.robots:
                robot.force_return_home = True
            return
        # This check is intentionally throttled because frontier extraction is
        # one of the expensive operations on long runs.
        if self.step_count < 120 or self.step_count % 10 != 0:
            return
        known = self.home_memory.map.known_mask()
        for robot in self.robots:
            known = known | robot.map.known_mask()
        known_ratio = float(np.count_nonzero(known)) / float(max(1, known.size))
        frontier_count = 0
        max_count = max(1, self.cfg.cage.exploration_complete_max_frontiers_per_robot * len(self.robots))
        for robot in self.robots:
            frontiers = robot.map.find_frontiers(
                self.cfg.planning.frontier_min_cluster_size,
                self.cfg.planning.frontier_info_radius_m,
            )
            frontier_count += min(len(frontiers), max_count + 1)
        no_useful_frontiers = frontier_count <= max_count
        enough_known = known_ratio >= self.cfg.cage.exploration_complete_min_known_ratio
        complete_now = enough_known and no_useful_frontiers
        if complete_now:
            self._exploration_complete_counter += 10
        else:
            self._exploration_complete_counter = max(0, self._exploration_complete_counter - 5)
        if self._exploration_complete_counter >= self.cfg.cage.exploration_complete_stable_steps:
            self.exploration_return_requested = True
            for robot in self.robots:
                robot.force_return_home = True

    def _update_mission_status(self) -> None:
        home_target = self.home_memory.target.detected
        local_target = any(r.target.detected for r in self.robots)
        target_xy = self._target_xy_for_passage()
        routes = self.home_memory.graph.top_routes(k=max(1, self.cfg.cage.desired_route_count))
        self.home_memory.best_routes = routes
        returned = [
            r.id
            for r in self.robots
            if self.world.home_base.contains((float(r.true_pose[0]), float(r.true_pose[1])))
        ]
        # A robot that physically returns HOME still performs a full knowledge
        # upload, but target detection alone no longer forces the team home.
        for r in self.robots:
            if r.id in returned and self.time_s - r.last_home_full_upload_time >= self.cfg.communication.packet_period_s:
                uploaded = self.communication.upload_robot_to_home(
                    r,
                    self.time_s,
                    full=True,
                    robots=self.robots,
                    require_connection=True,
                )
                if uploaded:
                    r.last_home_full_upload_time = self.time_s
        all_returned_home = len(returned) == len(self.robots)
        returned_msg = f"{len(returned)}/{len(self.robots)} robots at HOME"

        passage_target_xy = target_xy if (home_target or self.home_memory.route_candidates) else None
        self._maybe_evaluate_passage(passage_target_xy)
        completed = [r.id for r in self.robots if r.completed_target_roundtrip]
        reached = [r.id for r in self.robots if r.target_reached]
        required = self._required_roundtrip_count()
        enough_roundtrips = len(completed) >= required
        roundtrip_msg = f"roundtrip {len(completed)}/{required} complete, reached target {reached or '-'}"
        if target_xy is not None and enough_roundtrips and self.passage_status.safe:
            self.mission = MissionStatus(
                "MISSION_COMPLETE",
                True,
                f"Target roundtrip complete and safe HOME-target passage known: score {self.passage_status.score:.2f}, "
                f"clear {self.passage_status.min_clearance:.2f} m, unknown {self.passage_status.unknown_fraction:.2f}; {roundtrip_msg}",
            )
        elif target_xy is not None and enough_roundtrips:
            self.mission = MissionStatus(
                "ROUNDTRIP_COMPLETE",
                False,
                f"Required robots reached target and returned HOME; passage evaluator says: {self.passage_status.message}; {roundtrip_msg}",
            )
        elif target_xy is not None and self.passage_status.safe:
            self.mission = MissionStatus(
                "SAFE_PASSAGE_KNOWN",
                False,
                f"Safe passage candidate known, but robots still need target roundtrip. {roundtrip_msg}",
            )
        elif home_target:
            self.mission = MissionStatus(
                "TARGET_ROUNDTRIP",
                False,
                f"HOME knows target; every robot goes to target then returns HOME. {self.passage_status.message}; {roundtrip_msg}",
            )
        elif local_target:
            self.mission = MissionStatus(
                "TARGET_REPORTED",
                False,
                "Target found by robot; sharing target and starting target-guided route attempts",
            )
        elif self.exploration_return_requested and all_returned_home:
            self.mission = MissionStatus("COMPLETE", True, "Exploration complete and all robots returned HOME")
        elif self.exploration_return_requested:
            self.mission = MissionStatus("RETURN_HOME_EXPLORATION_COMPLETE", False, f"Exploration complete; returning team to HOME ({returned_msg})")
        else:
            self.mission = MissionStatus("SEARCH_TARGET", False, "Searching for hidden target")

    def _target_xy_for_passage(self):
        if self.home_memory.target.detected and self.home_memory.target.xy is not None:
            return self.home_memory.target.xy
        best = None
        best_conf = -1.0
        for r in self.robots:
            if r.target.detected and r.target.xy is not None and r.target.confidence > best_conf:
                best = r.target.xy
                best_conf = r.target.confidence
        return best

    def _evaluate_passage(self, target_xy) -> None:
        if target_xy is None:
            self.passage_status = PassageStatus(False, False, 0.0, 0.0, 0.0, 0.0, 1.0, "No target passage yet")
            self._last_passage_target = None
            return
        grid = self.home_memory.map
        planner = GridPlanner(self.cfg.planning)
        passage = grid.passage_quality(self.cfg.passage_quality, robot_radius_m=self.cfg.robot.radius)
        result = planner.plan(grid, self.world.home, target_xy, passage_quality=passage)
        self._last_passage_eval_time = float(self.time_s)
        self._last_passage_target = (float(target_xy[0]), float(target_xy[1]))
        if not result.success or len(result.path) < 2:
            self.passage_status = PassageStatus(True, False, 0.0, 0.0, 0.0, 0.0, 1.0, "No connected candidate passage in HOME map")
            return
        cells = []
        seen = set()
        for a, b in zip(result.path[:-1], result.path[1:]):
            ca = grid.world_to_cell(a)
            cb = grid.world_to_cell(b)
            if ca is None or cb is None:
                continue
            for c in grid._bresenham(ca, cb):
                if c not in seen:
                    seen.add(c)
                    cells.append(c)
        if not cells:
            self.passage_status = PassageStatus(True, False, 0.0, 0.0, 0.0, 0.0, 1.0, "Candidate passage had no valid cells")
            return
        known = grid.known_mask()
        clearance = grid.clearance_map(max_radius_m=max(3.0, self.cfg.passage_quality.good_clearance_m * 3.0))
        q_vals = []
        cl_vals = []
        p_vals = []
        unknown = 0
        for i, j in cells:
            if not known[j, i]:
                unknown += 1
            q_vals.append(float(grid.quality[j, i]))
            cl_vals.append(float(clearance[j, i]))
            p_vals.append(float(passage[j, i]))
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(result.path[:-1], result.path[1:]))
        mean_quality = float(np.mean(q_vals)) if q_vals else 0.0
        min_clearance = float(np.min(cl_vals)) if cl_vals else 0.0
        unknown_fraction = float(unknown) / float(max(1, len(cells)))
        passage_score = float(np.mean(p_vals)) if p_vals else 0.0
        safe = (
            passage_score >= self.cfg.cage.safe_passage_score_threshold
            and min_clearance >= self.cfg.cage.safe_passage_min_clearance_m
            and unknown_fraction <= self.cfg.cage.safe_passage_max_unknown_fraction
        )
        msg = f"passage safety {passage_score:.2f}, clear {min_clearance:.2f} m, unknown {unknown_fraction:.2f}"
        self.passage_status = PassageStatus(True, safe, passage_score, length, mean_quality, min_clearance, unknown_fraction, msg)

    def _maybe_evaluate_passage(self, target_xy) -> None:
        if target_xy is None:
            self._evaluate_passage(None)
            return
        target = (float(target_xy[0]), float(target_xy[1]))
        target_changed = (
            self._last_passage_target is None
            or math.hypot(target[0] - self._last_passage_target[0], target[1] - self._last_passage_target[1]) > 1.00
        )
        if target_changed or self.time_s - self._last_passage_eval_time >= float(self.cfg.cage.passage_eval_period_s):
            self._evaluate_passage(target)

    def run_headless(self, steps: int = 400) -> MissionStatus:
        for _ in range(steps):
            self.step()
            if self.mission.success:
                break
        return self.mission


# ============================================================================
# src / ui / snapshots.py
# ============================================================================

"""Small view models for the dashboard.

The dashboard reads these stable fields instead of reaching deep into every
subsystem.  This keeps rendering separate from simulator logic.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotView:
    id: int
    true_pose: Pose
    est_pose: Pose
    cov_trace: float
    task: str
    goal: Point | None
    scan_consistency: float
    front_clearance: float
    blocked_forward: bool


@dataclass(frozen=True)
class MissionView:
    time_s: float
    phase: str
    message: str
    target_reported_home: bool
    success: bool


# ============================================================================
# src / ui / matplotlib_dashboard.py
# ============================================================================

"""Old-style Matplotlib dashboard for the clean Search-CAGE baseline.

The layout mirrors the previous simulator UI style while keeping the new clean
backend:
  * toolbar across the top
  * Global Truth map, simulation/debug only
  * Team Fused/Reported Belief map
  * Mission status panel
  * compact local-belief cards for all robots

Map panels stay mostly free of legends/text.  Route/status information is kept
in the dedicated status and card-title areas to avoid overlap.
"""
import math
from dataclasses import replace
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, TextBox
import numpy as np


class MatplotlibDashboard:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.selected_robot = min(sim.cfg.ui.selected_robot, len(sim.robots) - 1)
        self.show_rays = bool(sim.cfg.ui.show_lidar_rays)
        self.show_passage_quality = bool(sim.cfg.passage_quality.show_by_default)
        self.show_route_graph = bool(sim.cfg.ui.show_route_graph)
        self.show_team_paths = True
        self.controls: dict[str, object] = {}
        self.local_axes = []
        self._render_frame = 0
        self._belief_cache: dict[int, tuple[int, np.ndarray]] = {}
        self._frontier_cache: dict[int, tuple[int, int, list[tuple[float, float]]]] = {}
        self._local_drawn: set[int] = set()
        self.fig = plt.figure(figsize=(sim.cfg.ui.figure_width, sim.cfg.ui.figure_height), constrained_layout=False)
        self.fig.set_facecolor("#f4f6f8")
        self.anim: FuncAnimation | None = None
        self._build_layout()

    def run(self) -> None:
        self.anim = FuncAnimation(
            self.fig,
            self._tick,
            interval=self.sim.cfg.ui.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Layout and controls
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.fig.clf()
        self.fig.set_facecolor("#f4f6f8")
        self.controls = {}
        self.local_axes = []
        self._belief_cache = {}
        self._frontier_cache = {}
        self._local_drawn = set()

        self._build_toolbar()
        self.ax_truth = self.fig.add_axes([0.045, 0.49, 0.44, 0.36])
        self.ax_team = self.fig.add_axes([0.045, 0.06, 0.44, 0.36])
        self.ax_status = self.fig.add_axes([0.51, 0.67, 0.445, 0.18])
        self._build_local_card_axes()
        self.fig.suptitle(
            "Search-CAGE: LiDAR Route Discovery",
            fontsize=14,
            y=0.982,
            fontweight="bold",
        )
        self._redraw_all(force=True)

    def _build_toolbar(self) -> None:
        y = 0.895
        h = 0.038
        x = 0.045
        gap = 0.008

        def style_ax(ax):
            ax.set_facecolor("#ffffff")
            for sp in ax.spines.values():
                sp.set_edgecolor("#d6dde6")
                sp.set_linewidth(1.0)

        def add_button(width: float, label: str, cb):
            nonlocal x
            ax = self.fig.add_axes([x, y, width, h])
            style_ax(ax)
            btn = Button(ax, label, color="#ffffff", hovercolor="#e8eef7")
            btn.label.set_fontsize(9)
            btn.on_clicked(cb)
            x += width + gap
            return btn

        def add_labeled_box(width: float, label: str, initial: str):
            nonlocal x
            label_ax = self.fig.add_axes([x, y + h + 0.003, width, 0.014])
            label_ax.axis("off")
            label_ax.text(0.02, 0.5, label, ha="left", va="center", fontsize=8.2, color="#475569")
            ax = self.fig.add_axes([x, y, width, h])
            style_ax(ax)
            box = TextBox(ax, "", initial=initial, color="white", hovercolor="#f7f7f7")
            box.text_disp.set_fontsize(9)
            x += width + gap
            return box

        self.controls["start"] = add_button(0.058, "Start", self._on_start)
        self.controls["pause"] = add_button(0.058, "Pause", self._on_pause)
        self.controls["rays"] = add_button(0.058, "Rays", self._on_toggle_rays)
        self.controls["quality"] = add_button(0.108, "Passage quality", self._on_toggle_passage_quality)
        self.controls["graph"] = add_button(0.060, "Graph", self._on_toggle_route_graph)
        self.controls["team_paths"] = add_button(0.078, "Team paths", self._on_toggle_team_paths)
        self.controls["reset"] = add_button(0.060, "Reset", self._on_reset)
        self.controls["seed"] = add_labeled_box(0.07, "Seed", str(self.sim.cfg.world.seed))
        self.controls["robots"] = add_labeled_box(0.06, "Robots", str(self.sim.cfg.robot.count))
        self.controls["obstacles"] = add_labeled_box(0.06, "Obst", str(self.sim.cfg.world.obstacle_count))
        self.controls["landmarks"] = add_labeled_box(0.06, "Land", str(self.sim.cfg.world.landmark_count))

    def _build_local_card_axes(self) -> None:
        n = max(1, len(self.sim.robots))
        cols = 2 if n > 1 else 1
        rows = ceil(n / cols)
        x0, y0, w, h = 0.51, 0.06, 0.445, 0.55
        gap_x, gap_y = 0.014, 0.020
        card_w = (w - gap_x * (cols - 1)) / cols
        card_h = (h - gap_y * (rows - 1)) / rows
        for idx in range(n):
            c = idx % cols
            r = idx // cols
            left = x0 + c * (card_w + gap_x)
            bottom = y0 + (rows - 1 - r) * (card_h + gap_y)
            ax = self.fig.add_axes([left, bottom, card_w, card_h])
            self.local_axes.append(ax)

    def _tick(self, _frame: int):
        if self.sim.running and not self.sim.mission.success:
            for _ in range(max(1, int(self.sim.cfg.ui.sim_steps_per_render))):
                if not self.sim.mission.success:
                    self.sim.step()
        self._redraw_all()
        return []

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def _on_start(self, _event) -> None:
        self.sim.running = True

    def _on_pause(self, _event) -> None:
        self.sim.running = False

    def _on_toggle_rays(self, _event) -> None:
        self.show_rays = not self.show_rays
        self._redraw_all(force=True)

    def _on_toggle_passage_quality(self, _event) -> None:
        self.show_passage_quality = not self.show_passage_quality
        self._redraw_all(force=True)

    def _on_toggle_route_graph(self, _event) -> None:
        self.show_route_graph = not self.show_route_graph
        self._redraw_all(force=True)

    def _on_toggle_team_paths(self, _event) -> None:
        self.show_team_paths = not self.show_team_paths
        self._redraw_all(force=True)

    def _textbox_value(self, key: str, default: int) -> int:
        obj = self.controls.get(key)
        raw = getattr(obj, "text", str(default))
        try:
            return int(raw)
        except Exception:
            return default

    def _on_reset(self, _event) -> None:
        old = self.sim.cfg
        seed = self._textbox_value("seed", old.world.seed)
        robots = max(1, min(8, self._textbox_value("robots", old.robot.count)))
        obstacles = max(0, min(40, self._textbox_value("obstacles", old.world.obstacle_count)))
        landmarks = max(0, min(40, self._textbox_value("landmarks", old.world.landmark_count)))
        cfg = replace(
            old,
            world=replace(old.world, seed=seed, obstacle_count=obstacles, landmark_count=landmarks),
            robot=replace(old.robot, count=robots),
        )
        self.sim.reset(cfg)
        self.selected_robot = min(self.selected_robot, len(self.sim.robots) - 1)
        self._build_layout()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _redraw_all(self, force: bool = False) -> None:
        truth_interval = max(1, int(self.sim.cfg.ui.render_truth_every))
        team_interval = max(1, int(self.sim.cfg.ui.render_team_every))
        if force or not self.sim.running or self._render_frame % truth_interval == 0:
            self._draw_truth()
        if force or not self.sim.running or self._render_frame % team_interval == 0:
            self._draw_team_belief()
        self._draw_status()
        self._draw_local_cards(force=force)
        self.fig.canvas.draw_idle()
        self._render_frame += 1

    def _setup_map_axis(self, ax, title: str, ticks: bool = True) -> None:
        ax.clear()
        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d6dde6")
            sp.set_linewidth(1.0)
        ax.set_title(title, fontsize=10.5, pad=7, color="#111827")
        ax.set_xlim(0, self.sim.world.width)
        ax.set_ylim(0, self.sim.world.height)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.08)
        ax.tick_params(labelsize=8, colors="#64748b", width=0.6)
        if ticks:
            ax.set_xlabel("x [m]", fontsize=8, color="#64748b")
            ax.set_ylabel("y [m]", fontsize=8, color="#64748b")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    def _robot_color(self, robot_id: int) -> str:
        palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2", "#be123c", "#65a30d"]
        return palette[robot_id % len(palette)]

    def _short_task(self, task: str) -> str:
        names = {
            "SEARCH_FRONTIER": "FRONTIER",
            "SEARCH_OPEN_SECTOR": "OPEN",
            "GO_TO_TARGET": "TARGET",
            "EXPLORE_TOWARD_TARGET": "T-EXPLORE",
            "RETURN_HOME_AFTER_TARGET": "T-RETURN",
            "WAIT_AT_HOME_DONE": "DONE",
            "ADVANCE_TO_TARGET": "TARGET",
            "CERTIFY_TARGET_EDGE": "CERTIFY",
            "REPORT_TARGET_HOME": "REPORT",
            "RETURN_HOME_CERT_ROUTE": "RETURN",
            "REANCHOR": "ANCHOR",
        }
        return names.get(task, task.replace("_", " "))

    def _draw_home_base(self, ax) -> None:
        hb = self.sim.world.home_base
        ax.add_patch(
            Rectangle(
                (hb.x0, hb.y0),
                hb.x1 - hb.x0,
                hb.y1 - hb.y0,
                facecolor="#dcfce7",
                edgecolor="#16a34a",
                linewidth=1.0,
                alpha=0.55,
                zorder=1,
            )
        )

    def _draw_obstacles_and_landmarks(self, ax) -> None:
        for obs in self.sim.world.obstacles:
            ax.add_patch(
                Rectangle(
                    (obs.x0, obs.y0),
                    obs.x1 - obs.x0,
                    obs.y1 - obs.y0,
                    facecolor="#64748b",
                    edgecolor="#334155",
                    linewidth=0.8,
                    alpha=0.82,
                    zorder=2,
                )
            )
        if self.sim.world.landmarks:
            xs = [lm.xy[0] for lm in self.sim.world.landmarks]
            ys = [lm.xy[1] for lm in self.sim.world.landmarks]
            ax.scatter(xs, ys, marker="*", s=58, c="#facc15", edgecolors="#334155", linewidths=0.45, zorder=4)

    def _draw_truth(self) -> None:
        ax = self.ax_truth
        self._setup_map_axis(ax, "Truth")
        self._draw_home_base(ax)
        self._draw_obstacles_and_landmarks(ax)
        hx, hy = self.sim.world.home
        ax.scatter([hx], [hy], marker="P", s=74, c="#22c55e", edgecolors="#111827", linewidths=0.7, zorder=5)
        if self.sim.cfg.ui.show_truth_target:
            tx, ty = self.sim.world.target
            ax.scatter([tx], [ty], marker="X", s=118, c="#ef4444", edgecolors="#111827", linewidths=1.0, zorder=6)
        for r in self.sim.robots:
            color = self._robot_color(r.id)
            if len(r.true_path) > 1:
                xs, ys = zip(*r.true_path[-self.sim.cfg.ui.max_draw_path_points:])
                ax.plot(xs, ys, color=color, linewidth=1.15, alpha=0.72, zorder=5)
            if r.path and r.path_index < len(r.path):
                px = [r.est_xy[0]] + [p[0] for p in r.path[r.path_index:]]
                py = [r.est_xy[1]] + [p[1] for p in r.path[r.path_index:]]
                ax.plot(px, py, color=color, linewidth=0.9, linestyle="--", alpha=0.42, zorder=5)
            x, y, th_true = r.true_pose
            ex, ey, th_est = r.est_pose
            self._draw_robot_body(ax, float(x), float(y), float(th_true), color, filled=True, zorder=7)
            self._draw_robot_body(ax, float(ex), float(ey), float(th_est), color, filled=False, zorder=8)
            ax.plot([x, ex], [y, ey], color=color, linewidth=0.75, linestyle=":", alpha=0.45, zorder=6)
            if self.show_rays and r.scan is not None:
                stride = max(self.sim.cfg.ui.draw_lidar_stride, len(r.scan.angles) // 24)
                th = float(r.true_pose[2])
                for a, rng in zip(r.scan.angles[::stride], r.scan.ranges[::stride]):
                    x2 = x + math.cos(th + float(a)) * float(rng)
                    y2 = y + math.sin(th + float(a)) * float(rng)
                    ax.plot([x, x2], [y, y2], color=color, alpha=0.08, linewidth=0.45, zorder=3)
        self._draw_comm_links(ax)

    def _draw_robot_body(self, ax, x: float, y: float, theta: float, color: str, filled: bool, zorder: int) -> None:
        length = float(self.sim.cfg.robot.body_length)
        width = float(self.sim.cfg.robot.body_width)
        c, s = math.cos(theta), math.sin(theta)
        forward = np.array([c, s])
        side = np.array([-s, c])
        center = np.array([x, y])
        corners = [
            center + forward * (length * 0.5) + side * (width * 0.5),
            center + forward * (length * 0.5) - side * (width * 0.5),
            center - forward * (length * 0.5) - side * (width * 0.5),
            center - forward * (length * 0.5) + side * (width * 0.5),
        ]
        patch = Polygon(
            corners,
            closed=True,
            facecolor=color if filled else "none",
            edgecolor="#111827" if filled else color,
            linewidth=0.75 if filled else 1.25,
            alpha=0.88 if filled else 0.95,
            zorder=zorder,
        )
        ax.add_patch(patch)
        nose = center + forward * (length * 0.55)
        ax.plot([x, float(nose[0])], [y, float(nose[1])], color="#111827" if filled else color, linewidth=0.8, alpha=0.8, zorder=zorder + 1)

    def _draw_comm_links(self, ax) -> None:
        for a, b in self.sim.comm_state.robot_segments:
            ax.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", color="#0284c7", linewidth=1.35, alpha=0.70, zorder=9)
        for a, b in self.sim.comm_state.home_segments:
            ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=":", color="#16a34a", linewidth=1.55, alpha=0.75, zorder=9)

    def _draw_grid(self, ax, grid: OccupancyGrid, title: str, ticks: bool = True, quality_overlay: bool = False) -> None:
        self._setup_map_axis(ax, title, ticks=ticks)
        ax.imshow(
            self._belief_image(grid, shade_quality=False),
            origin="lower",
            extent=(0, grid.width_m, 0, grid.height_m),
            interpolation="nearest",
            zorder=0,
        )
        if quality_overlay:
            q = grid.passage_quality(self.sim.cfg.passage_quality, robot_radius_m=self.sim.cfg.robot.radius)
            overlay_mask = (
                grid.known_mask()
                & grid.free_mask()
                & np.isfinite(q)
            )
            q = self._masked_neighbor_mean(q, overlay_mask)
            finite = overlay_mask & np.isfinite(q)
            if np.any(finite):
                q_min = float(np.percentile(q[finite], 5.0))
                q_max = float(np.percentile(q[finite], 98.0))
                if q_max > q_min + 1e-9:
                    scaled = np.clip((q - q_min) / (q_max - q_min), 0.0, 1.0)
                else:
                    scaled = np.full_like(q, 0.5, dtype=float)
            else:
                scaled = np.full_like(q, 0.5, dtype=float)
            overlay = np.zeros((grid.ny, grid.nx, 4), dtype=float)
            # Normalize the current passage scores into a red-yellow-green ramp:
            # lowest explored/free score is red, highest is green. Unexplored
            # cells stay as the base unknown color instead of dominating the ramp.
            overlay[..., 0] = np.where(scaled < 0.5, 1.0, 2.0 * (1.0 - scaled))
            overlay[..., 1] = np.where(scaled < 0.5, 2.0 * scaled, 1.0)
            overlay[..., 3] = float(self.sim.cfg.passage_quality.overlay_alpha) * finite
            ax.imshow(
                overlay,
                origin="lower",
                extent=(0, grid.width_m, 0, grid.height_m),
                interpolation="nearest",
                zorder=1,
            )

    def _masked_neighbor_mean(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=float, copy=True)
        total = np.zeros_like(out)
        count = np.zeros_like(out)
        for dy in (-1, 0, 1):
            ys = slice(max(0, -dy), min(out.shape[0], out.shape[0] - dy))
            yd = slice(max(0, dy), min(out.shape[0], out.shape[0] + dy))
            for dx in (-1, 0, 1):
                xs = slice(max(0, -dx), min(out.shape[1], out.shape[1] - dx))
                xd = slice(max(0, dx), min(out.shape[1], out.shape[1] + dx))
                valid = mask[ys, xs]
                total[yd, xd] += np.where(valid, values[ys, xs], 0.0)
                count[yd, xd] += valid.astype(float)
        m = mask & (count > 0.0)
        out[m] = total[m] / count[m]
        return out

    def _belief_image(self, grid: OccupancyGrid, shade_quality: bool = False) -> np.ndarray:
        version = int(getattr(grid, "_version", 0))
        cache_key = (id(grid), bool(shade_quality))
        cached = self._belief_cache.get(cache_key)
        if cached is not None and cached[0] == version:
            return cached[1]
        prob = grid.probability()
        quality = np.clip(grid.quality, 0.0, 1.0)
        free = prob < grid.cfg.prob_free_threshold
        occ = prob > grid.cfg.prob_occ_threshold
        observed = quality > 0.05
        img = np.zeros((grid.ny, grid.nx, 3), dtype=float)
        img[:, :, :] = np.array([0.88, 0.91, 0.95])
        img[observed & ~free & ~occ] = np.array([0.78, 0.83, 0.89])
        free_color = np.array([0.98, 1.00, 0.99])
        occ_color = np.array([0.08, 0.10, 0.12])
        img[free] = free_color
        img[occ] = occ_color
        # Quality is intentionally NOT baked into normal occupancy colors.
        # The red/green confidence layer appears only when the HOME fused map
        # calls _draw_grid(..., quality_overlay=True).
        if shade_quality:
            low_q = observed & (quality < 0.45)
            img[low_q] = 0.65 * img[low_q] + 0.35 * np.array([0.74, 0.79, 0.86])
        self._belief_cache[cache_key] = (version, img)
        return img

    def _draw_frontiers(self, ax, grid: OccupancyGrid, size: float = 12.0, alpha: float = 0.72) -> None:
        pts = self._frontier_points(grid)
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, marker=".", s=size, c="#06b6d4", alpha=alpha, linewidths=0.0, zorder=5)

    def _frontier_points(self, grid: OccupancyGrid) -> list[tuple[float, float]]:
        cache_key = id(grid)
        version = int(getattr(grid, "_version", 0))
        cached = self._frontier_cache.get(cache_key)
        interval = max(1, int(self.sim.cfg.ui.render_frontier_every))
        if cached is not None:
            cached_version, frame, pts = cached
            if cached_version == version or self._render_frame - frame < interval:
                return pts
        frontiers = grid.find_frontiers(self.sim.cfg.planning.frontier_min_cluster_size, self.sim.cfg.planning.frontier_info_radius_m)
        pts = [(float(fr.centroid_world[0]), float(fr.centroid_world[1])) for fr in frontiers[: self.sim.cfg.ui.max_draw_frontiers]]
        self._frontier_cache[cache_key] = (version, self._render_frame, pts)
        return pts

    def _draw_team_belief(self) -> None:
        title = "HOME Fused Belief"
        if self.show_passage_quality:
            title += " + Passage quality"
        self._draw_grid(self.ax_team, self.sim.home_memory.map, title, quality_overlay=self.show_passage_quality)
        self._draw_home_base(self.ax_team)
        self._draw_frontiers(self.ax_team, self.sim.home_memory.map, size=14, alpha=0.58)
        hx, hy = self.sim.world.home
        self.ax_team.scatter([hx], [hy], marker="P", s=60, c="#22c55e", edgecolors="#111827", linewidths=0.6, zorder=7)
        if self.sim.home_memory.target.detected and self.sim.home_memory.target.xy:
            tx, ty = self.sim.home_memory.target.xy
            self.ax_team.scatter([tx], [ty], marker="X", s=78, c="#ef4444", edgecolors="#111827", linewidths=0.9, zorder=8)
        self._draw_home_robot_reports(self.ax_team)
        if self.show_route_graph:
            self._draw_graph(self.ax_team, self.sim.home_memory.graph, self.sim.home_memory.best_routes)
        self._draw_comm_links(self.ax_team)

    def _draw_home_robot_reports(self, ax) -> None:
        memory = self.sim.home_memory
        for rid, pose in memory.known_robot_pose.items():
            stamp = memory.known_robot_time.get(rid, -math.inf)
            age = max(0.0, self.sim.time_s - stamp)
            alpha = max(0.25, min(0.95, 1.0 - age / max(1.0, self.sim.cfg.communication.teammate_intent_timeout_s * 2.0)))
            color = self._robot_color(rid)
            x, y, th = pose
            visits = memory.known_robot_visits.get(rid, [])[-self.sim.cfg.ui.max_draw_teammate_visit_points:]
            if visits:
                ax.scatter([p[0] for p in visits], [p[1] for p in visits], marker=".", s=10, c=[color], alpha=0.18, linewidths=0, zorder=4)
            path = memory.known_robot_paths.get(rid, [])
            if len(path) >= 2:
                ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=0.9, linestyle="--", alpha=0.42, zorder=6)
            goal = memory.known_robot_goal.get(rid)
            if goal is not None:
                ax.scatter([goal[0]], [goal[1]], marker="x", s=38, c=[color], linewidths=1.25, alpha=alpha, zorder=7)
            ax.scatter([x], [y], s=44, c=[color], edgecolors="#111827", linewidths=0.55, alpha=alpha, zorder=8)
            ax.arrow(x, y, math.cos(th) * 0.42, math.sin(th) * 0.42, head_width=0.12, color=color, alpha=alpha, zorder=9)

    def _draw_local_cards(self, force: bool = False) -> None:
        local_interval = max(1, int(self.sim.cfg.ui.render_local_every))
        for ax, robot in zip(self.local_axes, self.sim.robots):
            if (
                not force
                and self.sim.running
                and robot.id != self.selected_robot
                and robot.id in self._local_drawn
                and self._render_frame % local_interval != 0
            ):
                continue
            color = self._robot_color(robot.id)
            known_intents = len(robot.known_teammate_goals)
            title = f"R{robot.id} Knowledge  {self._short_task(robot.current_task)}  S {robot.assessment.consistency:.2f}  F {robot.assessment.front_clearance:.1f}m  I {known_intents}"
            self._draw_grid(ax, robot.map, title, ticks=False)
            self._draw_home_base(ax)
            self._draw_frontiers(ax, robot.map, size=8, alpha=0.48)
            if self.show_team_paths:
                self._draw_known_trajectories(ax, robot)
            self._draw_teammate_context(ax, robot)
            hx, hy = self.sim.world.home
            ax.scatter([hx], [hy], marker="P", s=28, c="#22c55e", edgecolors="#111827", linewidths=0.4, zorder=3)
            if self.show_route_graph and robot.id == self.selected_robot:
                self._draw_graph(ax, robot.graph, robot.best_routes[:2], node_size_scale=0.55, line_scale=0.65)
            ex, ey, eth = robot.est_pose
            ax.scatter([ex], [ey], s=36, c=[color], edgecolors="#111827", linewidths=0.5, zorder=8)
            ax.arrow(ex, ey, math.cos(eth) * 0.42, math.sin(eth) * 0.42, head_width=0.13, color="#111827", zorder=9)
            ell_x, ell_y = covariance_ellipse(robot.estimator.belief.covariance[:2, :2], scale=2.0)
            if len(ell_x):
                ax.plot(ex + ell_x, ey + ell_y, color=color, linewidth=0.9, linestyle="--", alpha=0.62, zorder=7)
            if robot.path and robot.path_index < len(robot.path):
                xs = [ex] + [p[0] for p in robot.path[robot.path_index:]]
                ys = [ey] + [p[1] for p in robot.path[robot.path_index:]]
                ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.82, zorder=7)
            if robot.current_goal is not None:
                gx, gy = robot.current_goal
                ax.scatter([gx], [gy], marker="x", s=34, c=[color], linewidths=1.2, zorder=8)
            if robot.target.detected and robot.target.xy:
                tx, ty = robot.target.xy
                ax.scatter([tx], [ty], marker="X", s=40, c="#ef4444", edgecolors="#111827", linewidths=0.6, zorder=8)
            if self.show_rays and robot.scan is not None:
                stride = max(self.sim.cfg.ui.draw_lidar_stride, len(robot.scan.angles) // 20)
                for a, rng in zip(robot.scan.angles[::stride], robot.scan.ranges[::stride]):
                    x2 = ex + math.cos(eth + float(a)) * float(rng)
                    y2 = ey + math.sin(eth + float(a)) * float(rng)
                    ax.plot([ex, x2], [ey, y2], color=color, alpha=0.08, linewidth=0.38, zorder=5)
            self._local_drawn.add(robot.id)


    def _draw_known_trajectories(self, ax, robot) -> None:
        max_pts = max(8, int(self.sim.cfg.ui.max_draw_teammate_trajectory_points))
        own = list(getattr(robot, "trajectory_from_home", []))[-max_pts:]
        own_color = self._robot_color(robot.id)
        if len(own) >= 2:
            ax.plot(
                [p[0] for p in own],
                [p[1] for p in own],
                color=own_color,
                linewidth=1.15,
                linestyle="-",
                alpha=0.58,
                zorder=5,
            )
        for rid, path in getattr(robot, "known_teammate_trajectories", {}).items():
            if rid == robot.id or len(path) < 2:
                continue
            pts = list(path)[-max_pts:]
            color = self._robot_color(rid)
            stamp = getattr(robot, "known_teammate_trajectory_time", {}).get(rid, -math.inf)
            age = max(0.0, self.sim.time_s - stamp)
            fresh_window = max(1.0, self.sim.cfg.communication.teammate_intent_timeout_s * 2.0)
            alpha = max(0.16, min(0.46, 0.46 * (1.0 - min(0.75, age / fresh_window))))
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                linewidth=0.95,
                linestyle="--",
                alpha=alpha,
                zorder=5,
            )
            ax.scatter([pts[-1][0]], [pts[-1][1]], marker=".", s=18, c=[color], alpha=max(alpha, 0.24), linewidths=0, zorder=6)

    def _draw_teammate_context(self, ax, robot) -> None:
        for rid, pts in robot.known_teammate_visits.items():
            if rid == robot.id or not pts:
                continue
            color = self._robot_color(rid)
            pts = pts[-self.sim.cfg.ui.max_draw_teammate_visit_points:]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=".", s=7, c=[color], alpha=0.18, linewidths=0, zorder=4)
        for rid, path in robot.known_teammate_paths.items():
            if rid == robot.id or len(path) < 2:
                continue
            color = self._robot_color(rid)
            ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=0.7, linestyle="--", alpha=0.24, zorder=5)
        for rid, goal in robot.known_teammate_goals.items():
            if rid == robot.id or goal is None:
                continue
            color = self._robot_color(rid)
            ax.scatter([goal[0]], [goal[1]], marker="x", s=22, c=[color], linewidths=0.9, alpha=0.45, zorder=6)

    def _draw_graph(
        self,
        ax,
        graph: RouteGraph,
        routes: list[RouteCandidate] | tuple[RouteCandidate, ...],
        node_size_scale: float = 1.0,
        line_scale: float = 1.0,
    ) -> None:
        highlighted = set()
        route_node_ids = set()
        for idx, route in enumerate(routes):
            pts = graph.route_points(route)
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    color="#22c55e" if idx == 0 else "#84cc16",
                    linewidth=(3.0 if idx == 0 else 2.0) * line_scale,
                    alpha=0.85 if idx == 0 else 0.55,
                    zorder=6,
                )
                highlighted.update(route.edge_ids)
                route_node_ids.update(route.node_ids)
        ranked_edges = sorted(graph.edges.values(), key=lambda e: e.cert.confidence, reverse=True)[: self.sim.cfg.ui.max_draw_graph_edges]
        node_order: list[int] = []
        seen_nodes: set[int] = set()

        def keep_node(nid: int) -> None:
            if nid in graph.nodes and nid not in seen_nodes and len(node_order) < self.sim.cfg.ui.max_draw_graph_nodes:
                seen_nodes.add(nid)
                node_order.append(nid)

        for nid in route_node_ids:
            keep_node(nid)
        for edge in ranked_edges:
            keep_node(edge.a)
            keep_node(edge.b)
        for nid, node in graph.nodes.items():
            if node.kind in {"home", "target", "anchor"}:
                keep_node(nid)

        for edge in ranked_edges:
            a = graph.nodes.get(edge.a)
            b = graph.nodes.get(edge.b)
            if a is None or b is None:
                continue
            c = edge.cert.confidence
            col = "#22c55e" if c >= 0.7 else "#eab308" if c >= 0.45 else "#ef4444"
            ax.plot(
                [a.xy[0], b.xy[0]],
                [a.xy[1], b.xy[1]],
                color=col,
                linewidth=(2.0 if edge.id in highlighted else 1.0) * line_scale,
                alpha=0.75 if edge.id in highlighted else 0.38,
                zorder=4,
            )
        for nid in node_order:
            node = graph.nodes[nid]
            if node.kind == "home":
                color = "#22c55e"
                marker = "P"
                size = 42
            elif node.kind == "target":
                color = "#ef4444"
                marker = "X"
                size = 46
            elif node.kind == "anchor":
                color = "#f97316"
                marker = "o"
                size = 24
            else:
                color = "white"
                marker = "o"
                size = 16
            ax.scatter(
                [node.xy[0]],
                [node.xy[1]],
                marker=marker,
                s=size * node_size_scale,
                c=[color],
                edgecolors="black",
                linewidths=0.45,
                zorder=7,
            )

    def _draw_status(self) -> None:
        ax = self.ax_status
        ax.clear()
        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d6dde6")
            sp.set_linewidth(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Mission", fontsize=10.5, pad=5, color="#111827")
        status = "RUNNING" if self.sim.running else "PAUSED"
        local = [r.id for r in self.sim.robots if r.target.detected]
        home_target = self.sim.home_memory.target
        returned = [
            r.id
            for r in self.sim.robots
            if self.sim.world.home_base.contains((float(r.true_pose[0]), float(r.true_pose[1])))
        ]
        sel = self.sim.robots[min(self.selected_robot, len(self.sim.robots) - 1)]
        rb = sel.status.reward_breakdown
        home_connected = [rid for rid, ok in sorted(self.sim.comm_state.home_connected.items()) if ok]
        intent_counts = "  ".join(f"R{r.id}:{len(r.known_teammate_goals)}" for r in self.sim.robots)
        reached = [r.id for r in self.sim.robots if getattr(r, "target_reached", False)]
        completed = [r.id for r in self.sim.robots if getattr(r, "completed_target_roundtrip", False)]
        required = self.sim._required_roundtrip_count() if hasattr(self.sim, "_required_roundtrip_count") else len(self.sim.robots)

        left_lines = [
            f"{status}  t={self.sim.time_s:5.1f}s  step={self.sim.step_count}",
            f"Phase   {self.sim.mission.phase}",
            f"Target  HOME {'yes' if home_target.detected else 'no'}   local {local or '-'}",
            f"Passage score {self.sim.passage_status.score:.2f}   clear {self.sim.passage_status.min_clearance:.2f} m   unk {self.sim.passage_status.unknown_fraction:.2f}",
            f"Target  reached {reached or '-'}   done {len(completed)}/{required} {completed or '-'}",
            f"Return  {len(returned)}/{len(self.sim.robots)} at HOME   ids {returned or '-'}",
            f"LOS     robot {len(self.sim.comm_state.direct_robot_edges)}   home {home_connected or '-'}",
            f"View    rays {'on' if self.show_rays else 'off'}   passage quality {'on' if self.show_passage_quality else 'off'}   team paths {'on' if self.show_team_paths else 'off'}   graph {'on' if self.show_route_graph else 'off'}",
        ]
        right_lines = [
            f"Selected R{sel.id}  {self._short_task(sel.current_task)}",
            f"Plan     {'ok' if sel.status.last_plan_success else sel.status.last_plan_reason}",
            f"Clear    {sel.status.last_path_min_clearance:.2f} m",
            f"Intent   {intent_counts or '-'}",
        ]
        if rb:
            right_lines.append(
                f"MDP      {rb.get('mdp_score', rb.get('score', 0.0)):.2f}   belief {rb.get('mdp_target_belief', rb.get('target_belief_gain', 0.0)):.3f}   cert {rb.get('mdp_certificate_gain', 0.0):.2f}"
            )
            right_lines.append(f"Belief   focus {1.0 - sel.target_belief_entropy():.2f}   comm {rb.get('mdp_communication_value', 0.0):.2f}")

        ax.text(0.025, 0.90, "\n".join(left_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.4, family="monospace", color="#111827")
        ax.text(0.53, 0.90, "\n".join(right_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.4, family="monospace", color="#111827")
        ax.plot([0.025, 0.975], [0.37, 0.37], transform=ax.transAxes, color="#e2e8f0", linewidth=0.8)

        routes = self.sim.home_memory.best_routes[: self.sim.cfg.ui.max_status_routes]
        route_lines = ["Routes / target roundtrips"]
        if routes:
            for i, route in enumerate(routes):
                route_lines.append(
                    f"#{i}  len {route.length:5.1f} m   clear {route.min_clearance:.2f}   cert {route.certificate:.2f}   {route.status}"
                )
        else:
            route_lines.append("graph routes: none yet")
        candidates = getattr(self.sim.home_memory, "route_candidates", {})
        if candidates:
            for rid, cand in sorted(candidates.items())[: self.sim.cfg.ui.max_status_routes]:
                done = "done" if cand.get("roundtrip_complete") else "reached"
                route_lines.append(
                    f"R{rid} {done} len {cand.get('route_length', 0.0):4.1f}+{cand.get('return_length', 0.0):4.1f} m "
                    f"q {cand.get('mean_quality', 0.0):.2f} clr {cand.get('min_clearance', 0.0):.2f}"
                )
        ax.text(0.025, 0.30, "\n".join(route_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.2, family="monospace", color="#111827")


# ============================================================================
# src / ui / pygame_dashboard.py
# ============================================================================

"""Realtime pygame dashboard for smooth one-step-per-frame simulation."""
import math
from dataclasses import replace

import numpy as np


class PygameDashboard:
    def __init__(self, sim: Simulator, fps: int = 60, width: int = 1920, height: int = 1080, fullscreen: bool = False):
        import pygame

        self.pg = pygame
        self.sim = sim
        self.fps = max(1, int(fps))
        self.window_size = (max(960, int(width)), max(640, int(height)))
        self.fullscreen = bool(fullscreen)
        self.running = True
        self.show_rays = bool(sim.cfg.ui.show_lidar_rays)
        self.show_passage_quality = bool(sim.cfg.passage_quality.show_by_default)
        self.show_route_graph = bool(sim.cfg.ui.show_route_graph)
        self.show_team_paths = True
        self.selected_robot = min(sim.cfg.ui.selected_robot, len(sim.robots) - 1)
        self._map_cache: dict[tuple[int, bool], tuple[int, object]] = {}
        self._button_rects: list[tuple[object, str]] = []

        pygame.init()
        pygame.display.set_caption("Search-CAGE Realtime")
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
            size = (0, 0)
        else:
            flags |= pygame.RESIZABLE
            size = self.window_size
        try:
            self.screen = pygame.display.set_mode(size, flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode(size, flags)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,dejavusansmono,menlo", 16)
        self.small = pygame.font.SysFont("consolas,dejavusansmono,menlo", 13)
        self.tiny = pygame.font.SysFont("consolas,dejavusansmono,menlo", 11)
        self.title_font = pygame.font.SysFont("arial,dejavusans", 20, bold=True)
        self.header_font = pygame.font.SysFont("arial,dejavusans", 26, bold=True)

    def run(self) -> None:
        while self.running:
            self._handle_events()
            if self.sim.running and not self.sim.mission.success:
                self.sim.step()
            self._draw()
            self.pg.display.flip()
            self.clock.tick(self.fps)
        self.pg.quit()

    def _handle_events(self) -> None:
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    self.running = False
                elif event.key == pg.K_SPACE:
                    self.sim.running = not self.sim.running
                elif event.key == pg.K_l:
                    self.show_rays = not self.show_rays
                elif event.key == pg.K_p:
                    self.show_passage_quality = not self.show_passage_quality
                    self._map_cache.clear()
                elif event.key == pg.K_g:
                    self.show_route_graph = not self.show_route_graph
                elif event.key == pg.K_t:
                    self.show_team_paths = not self.show_team_paths
                elif event.key == pg.K_TAB and self.sim.robots:
                    self.selected_robot = (self.selected_robot + 1) % len(self.sim.robots)
                elif event.key == pg.K_r:
                    self._reset()
                elif event.key == pg.K_F11:
                    self._toggle_fullscreen()
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                for rect, action in self._button_rects:
                    if rect.collidepoint(event.pos):
                        self._button(action)
                        break

    def _button(self, action: str) -> None:
        if action == "start":
            self.sim.running = True
        elif action == "pause":
            self.sim.running = False
        elif action == "rays":
            self.show_rays = not self.show_rays
        elif action == "quality":
            self.show_passage_quality = not self.show_passage_quality
            self._map_cache.clear()
        elif action == "graph":
            self.show_route_graph = not self.show_route_graph
        elif action == "paths":
            self.show_team_paths = not self.show_team_paths
        elif action.startswith("robot:"):
            try:
                rid = int(action.split(":", 1)[1])
            except ValueError:
                return
            if 0 <= rid < len(self.sim.robots):
                self.selected_robot = rid
        elif action == "robot" and self.sim.robots:
            self.selected_robot = (self.selected_robot + 1) % len(self.sim.robots)
        elif action == "reset":
            self._reset()

    def _reset(self) -> None:
        self.sim.reset(self.sim.cfg)
        self.selected_robot = min(self.selected_robot, len(self.sim.robots) - 1)
        self._map_cache.clear()

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        flags = self.pg.DOUBLEBUF | self.pg.HWSURFACE
        if self.fullscreen:
            flags |= self.pg.FULLSCREEN
            size = (0, 0)
        else:
            flags |= self.pg.RESIZABLE
            size = self.window_size
        try:
            self.screen = self.pg.display.set_mode(size, flags, vsync=1)
        except TypeError:
            self.screen = self.pg.display.set_mode(size, flags)

    def _draw(self) -> None:
        pg = self.pg
        screen = self.screen
        w, h = screen.get_size()
        if not self.fullscreen:
            self.window_size = (w, h)
        screen.fill((239, 243, 248))
        self._button_rects = []

        margin = 18
        gap = 16
        toolbar_h = 68
        self._draw_toolbar(pg.Rect(margin, 12, w - margin * 2, toolbar_h - 18))

        content_y = toolbar_h + gap
        content_h = max(420, h - content_y - margin)
        sidebar_w = min(440, max(330, int(w * 0.22)))
        main_w = max(620, w - sidebar_w - margin * 2 - gap)
        sidebar_x = margin + main_w + gap
        main = pg.Rect(margin, content_y, main_w, content_h)
        sidebar = pg.Rect(sidebar_x, content_y, sidebar_w, content_h)

        if w < 1250:
            sidebar_w = max(300, int(w * 0.28))
            main_w = max(560, w - sidebar_w - margin * 2 - gap)
            main = pg.Rect(margin, content_y, main_w, content_h)
            sidebar = pg.Rect(main.right + gap, content_y, sidebar_w, content_h)

        top_h = max(320, int(main.h * 0.51))
        top_w = (main.w - gap) // 2
        truth = pg.Rect(main.x, main.y, top_w, top_h)
        team = pg.Rect(truth.right + gap, main.y, main.w - top_w - gap, top_h)
        selected = pg.Rect(main.x, truth.bottom + gap, main.w, main.bottom - truth.bottom - gap)

        status = pg.Rect(sidebar.x, sidebar.y, sidebar.w, max(245, int(sidebar.h * 0.36)))
        robot_list = pg.Rect(sidebar.x, status.bottom + gap, sidebar.w, sidebar.bottom - status.bottom - gap)

        self._draw_truth(truth)
        self._draw_belief(team, self.sim.home_memory.map, "Team Fused Map", team_view=True)
        sel = self.sim.robots[min(self.selected_robot, len(self.sim.robots) - 1)]
        self._draw_belief(selected, sel.map, f"Selected Robot R{sel.id}", robot=sel)
        self._draw_status(status)
        self._draw_robot_list(robot_list)

    def _draw_toolbar(self, rect) -> None:
        title = self.header_font.render("Search-CAGE Realtime", True, (15, 23, 42))
        self.screen.blit(title, (rect.x, rect.y + 2))
        x = rect.x + title.get_width() + 24
        labels = [
            ("Start", "start", self.sim.running),
            ("Pause", "pause", not self.sim.running),
            ("Rays", "rays", self.show_rays),
            ("Passage", "quality", self.show_passage_quality),
            ("Graph", "graph", self.show_route_graph),
            ("Paths", "paths", self.show_team_paths),
            (f"R{self.selected_robot}", "robot", True),
            ("Reset", "reset", False),
        ]
        if x > rect.right - 620:
            x = rect.x
            rect = self.pg.Rect(rect.x, rect.y + 30, rect.w, max(30, rect.h - 30))
        for label, action, active in labels:
            bw = max(66, 12 + len(label) * 9)
            r = self.pg.Rect(x, rect.y + 3, bw, 34)
            self._button_rects.append((r, action))
            self._draw_button(r, label, active)
            x += bw + 8

        fps = self.clock.get_fps()
        text = f"t={self.sim.time_s:6.1f}s   step={self.sim.step_count}   fps={fps:4.0f}   {self.sim.mission.phase}"
        surf = self.font.render(text, True, (51, 65, 85))
        self.screen.blit(surf, (max(x + 10, rect.right - surf.get_width()), rect.y + 10))

    def _draw_button(self, rect, label: str, active: bool) -> None:
        color = (30, 90, 210) if active else (255, 255, 255)
        edge = (30, 64, 175) if active else (193, 204, 219)
        fg = (255, 255, 255) if active else (15, 23, 42)
        self.pg.draw.rect(self.screen, color, rect, border_radius=5)
        self.pg.draw.rect(self.screen, edge, rect, width=1, border_radius=5)
        surf = self.small.render(label, True, fg)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_panel(self, rect, title: str) -> None:
        shadow = self.pg.Rect(rect.x + 2, rect.y + 3, rect.w, rect.h)
        self.pg.draw.rect(self.screen, (225, 232, 242), shadow, border_radius=7)
        self.pg.draw.rect(self.screen, (255, 255, 255), rect, border_radius=5)
        self.pg.draw.rect(self.screen, (202, 213, 226), rect, width=1, border_radius=5)
        self._text(title, (rect.x + 12, rect.y + 8), (17, 24, 39), self.title_font)

    def _map_rect(self, rect):
        pad_top = 40
        inner = self.pg.Rect(rect.x + 10, rect.y + pad_top, rect.w - 20, rect.h - pad_top - 10)
        aspect = self.sim.world.width / max(1e-6, self.sim.world.height)
        if inner.w / max(1, inner.h) > aspect:
            mh = inner.h
            mw = int(mh * aspect)
        else:
            mw = inner.w
            mh = int(mw / aspect)
        return self.pg.Rect(inner.centerx - mw // 2, inner.centery - mh // 2, mw, mh)

    def _world_to_screen(self, p, mrect):
        x = mrect.x + float(p[0]) / self.sim.world.width * mrect.w
        y = mrect.bottom - float(p[1]) / self.sim.world.height * mrect.h
        return int(round(x)), int(round(y))

    def _world_rect(self, rect, mrect):
        x0, y1 = self._world_to_screen((rect.x0, rect.y0), mrect)
        x1, y0 = self._world_to_screen((rect.x1, rect.y1), mrect)
        return self.pg.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    def _draw_truth(self, rect) -> None:
        self._draw_panel(rect, "Truth")
        mrect = self._map_rect(rect)
        self.pg.draw.rect(self.screen, (248, 250, 252), mrect)
        self._draw_grid_lines(mrect)
        self.pg.draw.rect(self.screen, (148, 163, 184), mrect, width=1)
        self._draw_static_world(mrect)
        if self.sim.cfg.ui.show_truth_target:
            self._draw_marker(self.sim.world.target, mrect, (239, 68, 68), "x", 9)
        for robot in self.sim.robots:
            color = self._robot_color(robot.id)
            if len(robot.true_path) > 1:
                self._polyline(robot.true_path[-self.sim.cfg.ui.max_draw_path_points:], mrect, color, 2)
            if robot.path and robot.path_index < len(robot.path):
                pts = [robot.est_xy] + robot.path[robot.path_index:]
                self._polyline(pts, mrect, color, 1)
            self._draw_robot(tuple(robot.true_pose), mrect, color, filled=True)
            self._draw_robot(robot.est_pose, mrect, color, filled=False)
            self.pg.draw.line(self.screen, color, self._world_to_screen(robot.true_pose[:2], mrect), self._world_to_screen(robot.est_xy, mrect), 1)
            if self.show_rays and robot.scan is not None:
                stride = max(self.sim.cfg.ui.draw_lidar_stride, len(robot.scan.angles) // 24)
                x, y, th = robot.true_pose
                for a, rng in zip(robot.scan.angles[::stride], robot.scan.ranges[::stride]):
                    p2 = (x + math.cos(th + float(a)) * float(rng), y + math.sin(th + float(a)) * float(rng))
                    self.pg.draw.line(self.screen, (*color, 80), self._world_to_screen((x, y), mrect), self._world_to_screen(p2, mrect), 1)
        self._draw_comm_links(mrect)

    def _draw_belief(self, rect, grid: OccupancyGrid, title: str, team_view: bool = False, robot=None) -> None:
        label = title + (" + Passage" if team_view and self.show_passage_quality else "")
        self._draw_panel(rect, label)
        mrect = self._map_rect(rect)
        if robot is not None and rect.w > rect.h * 1.35:
            mrect.x = rect.x + 14
        surf = self._grid_surface(grid, self.show_passage_quality if team_view else False)
        self.screen.blit(self.pg.transform.scale(surf, (mrect.w, mrect.h)), mrect)
        self._draw_grid_lines(mrect, subtle=True)
        self.pg.draw.rect(self.screen, (148, 163, 184), mrect, width=1)
        self._draw_static_world(mrect, landmarks=False, obstacles=False)
        self._draw_marker(self.sim.world.home, mrect, (34, 197, 94), "home", 7)
        if team_view:
            self._draw_team_fused_reports(mrect)
            if self.sim.home_memory.target.detected and self.sim.home_memory.target.xy:
                self._draw_marker(self.sim.home_memory.target.xy, mrect, (239, 68, 68), "x", 8)
            if self.show_route_graph:
                self._draw_graph(self.sim.home_memory.graph, self.sim.home_memory.best_routes[:2], mrect)
        elif robot is not None:
            if self.show_team_paths:
                self._draw_known_paths(robot, mrect)
            if self.show_route_graph and robot.id == self.selected_robot:
                self._draw_graph(robot.graph, robot.best_routes[:2], mrect, scale=0.75)
            if robot.path and robot.path_index < len(robot.path):
                self._polyline([robot.est_xy] + robot.path[robot.path_index:], mrect, self._robot_color(robot.id), 2)
            if robot.current_goal is not None:
                self._draw_marker(robot.current_goal, mrect, self._robot_color(robot.id), "goal", 6)
            if robot.target.detected and robot.target.xy:
                self._draw_marker(robot.target.xy, mrect, (239, 68, 68), "x", 7)
            self._draw_robot(robot.est_pose, mrect, self._robot_color(robot.id), filled=True, scale=0.85)
            ell_x, ell_y = covariance_ellipse(robot.estimator.belief.covariance[:2, :2], scale=2.0)
            if len(ell_x):
                pts = [(robot.est_xy[0] + float(x), robot.est_xy[1] + float(y)) for x, y in zip(ell_x, ell_y)]
                self._polyline(pts, mrect, self._robot_color(robot.id), 1)
            self._draw_robot_overlay_stats(rect, robot)
            self._draw_selected_details(rect, mrect, robot)

    def _draw_robot_list(self, rect) -> None:
        self._draw_panel(rect, "Robots")
        robots = self.sim.robots
        if not robots:
            return
        y = rect.y + 42
        row_h = max(64, min(92, (rect.h - 54) // max(1, len(robots))))
        for robot in robots:
            row = self.pg.Rect(rect.x + 10, y, rect.w - 20, row_h - 8)
            selected = robot.id == self.selected_robot
            fill = (237, 244, 255) if selected else (248, 250, 252)
            edge = (37, 99, 235) if selected else (226, 232, 240)
            self.pg.draw.rect(self.screen, fill, row, border_radius=5)
            self.pg.draw.rect(self.screen, edge, row, width=2 if selected else 1, border_radius=5)
            self._button_rects.append((row, f"robot:{robot.id}"))

            color = self._robot_color(robot.id)
            badge = self.pg.Rect(row.x + 10, row.y + 10, 36, 36)
            self.pg.draw.rect(self.screen, color, badge, border_radius=5)
            label = self.font.render(f"R{robot.id}", True, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=badge.center))

            task = robot.current_task.replace("_", " ")
            self._text(task[:24], (row.x + 56, row.y + 8), (15, 23, 42), self.small)
            self._text(
                f"S {robot.assessment.consistency:.2f}  F {robot.assessment.front_clearance:.1f}m  P {robot.last_pose_quality:.2f}",
                (row.x + 56, row.y + 28),
                (71, 85, 105),
                self.tiny,
            )
            self._metric_bar(self.pg.Rect(row.x + 56, row.bottom - 15, row.w - 72, 6), robot.assessment.consistency, color)
            y += row_h

    def _draw_robot_overlay_stats(self, rect, robot) -> None:
        chips = [
            f"pose {robot.last_pose_quality:.2f}",
            f"front {robot.assessment.front_clearance:.1f}m",
            f"packets {robot.received_packets}",
        ]
        x = rect.x + 14
        y = rect.bottom - 30
        for chip in chips:
            surf = self.tiny.render(chip, True, (51, 65, 85))
            r = self.pg.Rect(x, y, surf.get_width() + 14, 20)
            self.pg.draw.rect(self.screen, (255, 255, 255), r, border_radius=4)
            self.pg.draw.rect(self.screen, (203, 213, 225), r, width=1, border_radius=4)
            self.screen.blit(surf, (r.x + 7, r.y + 4))
            x = r.right + 6

    def _draw_team_fused_reports(self, mrect) -> None:
        memory = self.sim.home_memory
        for rid in sorted(memory.known_robot_pose):
            color = self._robot_color(rid)
            stamp = memory.known_robot_time.get(rid, -math.inf)
            age = max(0.0, self.sim.time_s - stamp)
            freshness = max(0.25, min(1.0, 1.0 - age / max(1.0, self.sim.cfg.communication.teammate_intent_timeout_s * 2.0)))
            faded = self._blend_color((148, 163, 184), color, freshness)

            visits = memory.known_robot_visits.get(rid, [])[-self.sim.cfg.ui.max_draw_teammate_visit_points:]
            for p in visits:
                self.pg.draw.circle(self.screen, self._blend_color((226, 232, 240), color, 0.35), self._world_to_screen(p, mrect), 2)

            traj = memory.known_robot_trajectories.get(rid, [])[-self.sim.cfg.ui.max_draw_teammate_trajectory_points:]
            if self.show_team_paths and len(traj) >= 2:
                self._polyline(traj, mrect, self._blend_color((180, 190, 204), color, 0.55), 1)

            path = memory.known_robot_paths.get(rid, [])
            if self.show_team_paths and len(path) >= 2:
                self._polyline(path, mrect, faded, 2)

            pose = memory.known_robot_pose[rid]
            goal = memory.known_robot_goal.get(rid)
            if goal is not None:
                self._draw_marker(goal, mrect, faded, "goal", 5)

            self._draw_robot(pose, mrect, faded, filled=True, scale=0.82)
            label_pos = self._world_to_screen((float(pose[0]) + 0.35, float(pose[1]) + 0.35), mrect)
            self._text(f"R{rid}", label_pos, (15, 23, 42), self.tiny)

    def _blend_color(self, a, b, t: float):
        t = max(0.0, min(1.0, float(t)))
        return (
            int(round(a[0] * (1.0 - t) + b[0] * t)),
            int(round(a[1] * (1.0 - t) + b[1] * t)),
            int(round(a[2] * (1.0 - t) + b[2] * t)),
        )

    def _draw_selected_details(self, rect, mrect, robot) -> None:
        x = mrect.right + 22
        if x > rect.right - 220:
            return
        y = rect.y + 50
        w = rect.right - x - 16
        color = self._robot_color(robot.id)
        self._text("Telemetry", (x, y), (15, 23, 42), self.font)
        y += 28
        rows = [
            ("task", robot.current_task.replace("_", " ")),
            ("pose", f"{robot.est_xy[0]:.1f}, {robot.est_xy[1]:.1f}, {math.degrees(robot.est_pose[2]):.0f} deg"),
            ("goal", "-" if robot.current_goal is None else f"{robot.current_goal[0]:.1f}, {robot.current_goal[1]:.1f}"),
            ("path", f"{max(0, len(robot.path) - robot.path_index)} waypoints"),
            ("target", "detected" if robot.target.detected else "not detected"),
        ]
        for label, value in rows:
            self._text(label, (x, y), (100, 116, 139), self.tiny)
            self._text(str(value)[:30], (x + 66, y), (51, 65, 85), self.small)
            y += 24

        y += 8
        metrics = [
            ("scan-map", robot.assessment.consistency),
            ("pose quality", robot.last_pose_quality),
            ("front clear", min(1.0, robot.assessment.front_clearance / max(1e-6, self.sim.cfg.lidar.range))),
            ("belief focus", 1.0 - robot.target_belief_entropy()),
        ]
        for label, value in metrics:
            self._text(label, (x, y), (51, 65, 85), self.tiny)
            self._metric_bar(self.pg.Rect(x, y + 16, max(80, w - 8), 8), value, color)
            y += 34

        if robot.status.reward_breakdown:
            y += 4
            self._text("Planner score", (x, y), (15, 23, 42), self.small)
            y += 22
            rb = robot.status.reward_breakdown
            parts = [
                f"mdp {rb.get('mdp_score', rb.get('score', 0.0)):.2f}",
                f"belief {rb.get('mdp_target_belief', rb.get('target_belief_gain', 0.0)):.3f}",
                f"cert {rb.get('mdp_certificate_gain', 0.0):.2f}",
                f"comm {rb.get('mdp_communication_value', 0.0):.2f}",
            ]
            self._text("  ".join(parts), (x, y), (71, 85, 105), self.tiny)
            y += 18
            for item in getattr(robot, "last_mdp_candidates", [])[:3]:
                task = item.task.replace("_", " ").lower()[:20]
                self._text(f"{item.score:5.2f}  {task}", (x, y), (71, 85, 105), self.tiny)
                y += 16

    def _draw_status(self, rect) -> None:
        self._draw_panel(rect, "Mission")
        sel = self.sim.robots[min(self.selected_robot, len(self.sim.robots) - 1)]
        y = rect.y + 42
        status_color = (22, 163, 74) if self.sim.running else (234, 179, 8)
        self._status_row(rect.x + 14, y, "State", "RUNNING" if self.sim.running else "PAUSED", status_color)
        y += 28
        self._status_row(rect.x + 14, y, "Phase", self.sim.mission.phase, (37, 99, 235))
        y += 28
        target = "HOME yes" if self.sim.home_memory.target.detected else "HOME no"
        self._status_row(rect.x + 14, y, "Target", target, (239, 68, 68) if self.sim.home_memory.target.detected else (100, 116, 139))
        y += 30

        self._text(f"t={self.sim.time_s:5.1f}s    step={self.sim.step_count}", (rect.x + 14, y), (51, 65, 85), self.small)
        y += 24
        home_connected = [rid for rid, ok in self.sim.comm_state.home_connected.items() if ok]
        self._text(f"LOS robot {len(self.sim.comm_state.direct_robot_edges)}  home {home_connected or '-'}", (rect.x + 14, y), (51, 65, 85), self.small)
        y += 28

        self._text("Passage", (rect.x + 14, y), (15, 23, 42), self.small)
        self._metric_bar(self.pg.Rect(rect.x + 92, y + 5, rect.w - 112, 8), self.sim.passage_status.score, (34, 197, 94))
        y += 22
        self._text(
            f"clear {self.sim.passage_status.min_clearance:.2f}m  unknown {self.sim.passage_status.unknown_fraction:.2f}",
            (rect.x + 14, y),
            (71, 85, 105),
            self.tiny,
        )
        y += 30

        self._text(f"Selected R{sel.id}", (rect.x + 14, y), (15, 23, 42), self.font)
        y += 22
        self._text(sel.current_task.replace("_", " ")[:34], (rect.x + 14, y), (51, 65, 85), self.small)
        y += 20
        plan = "ok" if sel.status.last_plan_success else sel.status.last_plan_reason
        self._text(f"plan {plan[:34]}", (rect.x + 14, y), (71, 85, 105), self.tiny)

    def _status_row(self, x: int, y: int, label: str, value: str, color) -> None:
        self._text(label, (x, y + 4), (100, 116, 139), self.small)
        surf = self.small.render(value, True, (255, 255, 255))
        chip = self.pg.Rect(x + 76, y, min(230, surf.get_width() + 18), 24)
        self.pg.draw.rect(self.screen, color, chip, border_radius=4)
        self.screen.blit(surf, (chip.x + 9, chip.y + 5))

    def _metric_bar(self, rect, value: float, color) -> None:
        v = max(0.0, min(1.0, float(value)))
        self.pg.draw.rect(self.screen, (226, 232, 240), rect, border_radius=3)
        if rect.w > 0:
            fill = self.pg.Rect(rect.x, rect.y, max(1, int(rect.w * v)), rect.h)
            self.pg.draw.rect(self.screen, color, fill, border_radius=3)

    def _grid_surface(self, grid: OccupancyGrid, quality_overlay: bool):
        key = (id(grid), bool(quality_overlay))
        version = int(getattr(grid, "_version", 0))
        cached = self._map_cache.get(key)
        if cached is not None and cached[0] == version:
            return cached[1]
        prob = grid.probability()
        quality = np.clip(grid.quality, 0.0, 1.0)
        free = prob < grid.cfg.prob_free_threshold
        occ = prob > grid.cfg.prob_occ_threshold
        observed = quality > 0.05
        img = np.empty((grid.ny, grid.nx, 3), dtype=np.uint8)
        img[:, :, :] = (224, 232, 242)
        img[observed & ~free & ~occ] = (198, 211, 226)
        img[free] = (248, 255, 252)
        img[occ] = (20, 24, 30)
        if quality_overlay:
            q = grid.passage_quality(self.sim.cfg.passage_quality, robot_radius_m=self.sim.cfg.robot.radius)
            known_free = grid.known_mask() & free
            if np.any(known_free):
                scaled = np.clip(q, 0.0, 1.0)
                overlay = np.zeros_like(img)
                overlay[..., 0] = np.where(scaled < 0.5, 255, 255 * (2.0 * (1.0 - scaled))).astype(np.uint8)
                overlay[..., 1] = np.where(scaled < 0.5, 255 * (2.0 * scaled), 255).astype(np.uint8)
                alpha = float(self.sim.cfg.passage_quality.overlay_alpha)
                img[known_free] = ((1.0 - alpha) * img[known_free] + alpha * overlay[known_free]).astype(np.uint8)
        screen_img = np.ascontiguousarray(np.flipud(img).swapaxes(0, 1))
        surf = self.pg.surfarray.make_surface(screen_img)
        self._map_cache[key] = (version, surf)
        return surf

    def _draw_static_world(self, mrect, landmarks: bool = True, obstacles: bool = True) -> None:
        self.pg.draw.rect(self.screen, (22, 163, 74), self._world_rect(self.sim.world.home_base, mrect), width=2)
        if obstacles:
            for obs in self.sim.world.obstacles:
                self.pg.draw.rect(self.screen, (100, 116, 139), self._world_rect(obs, mrect))
        if landmarks:
            for lm in self.sim.world.landmarks:
                self._draw_marker(lm.xy, mrect, (250, 204, 21), "star", 5)

    def _draw_grid_lines(self, mrect, subtle: bool = False) -> None:
        color = (226, 232, 240) if not subtle else (236, 241, 247)
        for k in range(0, int(self.sim.world.width) + 1, 5):
            x, _ = self._world_to_screen((k, 0), mrect)
            self.pg.draw.line(self.screen, color, (x, mrect.y), (x, mrect.bottom), 1)
        for k in range(0, int(self.sim.world.height) + 1, 5):
            _, y = self._world_to_screen((0, k), mrect)
            self.pg.draw.line(self.screen, color, (mrect.x, y), (mrect.right, y), 1)

    def _draw_robot(self, pose, mrect, color, filled: bool, scale: float = 1.0) -> None:
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
        length = float(self.sim.cfg.robot.body_length) * scale
        width = float(self.sim.cfg.robot.body_width) * scale
        c, s = math.cos(th), math.sin(th)
        f = np.array([c, s])
        side = np.array([-s, c])
        center = np.array([x, y])
        corners = [
            center + f * length * 0.5 + side * width * 0.5,
            center + f * length * 0.5 - side * width * 0.5,
            center - f * length * 0.5 - side * width * 0.5,
            center - f * length * 0.5 + side * width * 0.5,
        ]
        pts = [self._world_to_screen(p, mrect) for p in corners]
        if filled:
            self.pg.draw.polygon(self.screen, color, pts)
        self.pg.draw.polygon(self.screen, color if not filled else (17, 24, 39), pts, width=2 if not filled else 1)
        nose = center + f * length * 0.65
        self.pg.draw.line(self.screen, (17, 24, 39), self._world_to_screen(center, mrect), self._world_to_screen(nose, mrect), 2)

    def _draw_marker(self, p, mrect, color, kind: str, size: int) -> None:
        x, y = self._world_to_screen(p, mrect)
        if kind == "x":
            self.pg.draw.line(self.screen, color, (x - size, y - size), (x + size, y + size), 3)
            self.pg.draw.line(self.screen, color, (x - size, y + size), (x + size, y - size), 3)
        elif kind == "goal":
            self.pg.draw.circle(self.screen, color, (x, y), size, width=2)
        elif kind == "star":
            self.pg.draw.circle(self.screen, (51, 65, 85), (x, y), size + 1)
            self.pg.draw.circle(self.screen, color, (x, y), size)
        else:
            self.pg.draw.circle(self.screen, color, (x, y), size)

    def _draw_comm_links(self, mrect) -> None:
        for a, b in self.sim.comm_state.robot_segments:
            self.pg.draw.line(self.screen, (2, 132, 199), self._world_to_screen(a, mrect), self._world_to_screen(b, mrect), 2)
        for a, b in self.sim.comm_state.home_segments:
            self.pg.draw.line(self.screen, (22, 163, 74), self._world_to_screen(a, mrect), self._world_to_screen(b, mrect), 2)

    def _draw_known_paths(self, robot, mrect) -> None:
        max_pts = max(8, int(self.sim.cfg.ui.max_draw_teammate_trajectory_points))
        own = list(getattr(robot, "trajectory_from_home", []))[-max_pts:]
        if len(own) >= 2:
            self._polyline(own, mrect, self._robot_color(robot.id), 1)
        for rid, pts in getattr(robot, "known_teammate_trajectories", {}).items():
            if rid == robot.id or len(pts) < 2:
                continue
            self._polyline(list(pts)[-max_pts:], mrect, self._robot_color(rid), 1)

    def _draw_graph(self, graph, routes, mrect, scale: float = 1.0) -> None:
        highlighted = set()
        for route in routes:
            pts = graph.route_points(route)
            if len(pts) >= 2:
                self._polyline(pts, mrect, (34, 197, 94), max(1, int(3 * scale)))
                highlighted.update(route.edge_ids)
        edges = sorted(graph.edges.values(), key=lambda e: e.cert.confidence, reverse=True)[: self.sim.cfg.ui.max_draw_graph_edges]
        for edge in edges:
            a = graph.nodes.get(edge.a)
            b = graph.nodes.get(edge.b)
            if a is None or b is None:
                continue
            c = edge.cert.confidence
            color = (34, 197, 94) if c >= 0.7 else (234, 179, 8) if c >= 0.45 else (239, 68, 68)
            self.pg.draw.line(self.screen, color, self._world_to_screen(a.xy, mrect), self._world_to_screen(b.xy, mrect), 2 if edge.id in highlighted else 1)

    def _polyline(self, pts, mrect, color, width: int) -> None:
        if len(pts) < 2:
            return
        screen_pts = [self._world_to_screen(p, mrect) for p in pts]
        if width <= 1:
            self.pg.draw.aalines(self.screen, color, False, screen_pts)
        else:
            self.pg.draw.lines(self.screen, color, False, screen_pts, max(1, int(width)))

    def _text(self, text: str, pos, color, font) -> None:
        self.screen.blit(font.render(text, True, color), pos)

    def _robot_color(self, robot_id: int):
        palette = [
            (37, 99, 235),
            (220, 38, 38),
            (22, 163, 74),
            (147, 51, 234),
            (249, 115, 22),
            (8, 145, 178),
            (190, 18, 60),
            (101, 163, 13),
        ]
        return palette[robot_id % len(palette)]


def realtime_config(cfg: AppConfig, fps: int = 60) -> AppConfig:
    realtime_dt = min(float(cfg.dt), 5.0 / float(max(1, fps)))
    return replace(
        cfg,
        dt=realtime_dt,
        cage=replace(cfg.cage, passage_eval_period_s=max(float(cfg.cage.passage_eval_period_s), 10.0)),
        passage_quality=replace(cfg.passage_quality, show_by_default=False),
        ui=replace(
            cfg.ui,
            interval_ms=max(1, int(1000 / max(1, fps))),
            sim_steps_per_render=1,
            render_truth_every=1,
            render_team_every=1,
            render_local_every=1,
            render_frontier_every=1,
            max_draw_path_points=220,
            max_draw_graph_edges=70,
            max_draw_graph_nodes=90,
            max_draw_frontiers=10,
            max_draw_teammate_visit_points=8,
            max_draw_teammate_trajectory_points=45,
        ),
    )


# ============================================================================
# main.py
# ============================================================================


def apply_ui_profile(cfg: AppConfig, profile: str) -> AppConfig:
    """Tune Matplotlib drawing cost without changing simulator behavior."""
    if profile == "quality":
        return cfg
    if profile == "balanced":
        return replace(
            cfg,
            ui=replace(
                cfg.ui,
                render_truth_every=4,
                render_team_every=4,
                render_local_every=8,
                render_frontier_every=16,
                max_draw_path_points=300,
                max_draw_frontiers=16,
                max_draw_teammate_visit_points=12,
                max_draw_teammate_trajectory_points=60,
            ),
        )
    if profile == "fast":
        return replace(
            cfg,
            passage_quality=replace(cfg.passage_quality, show_by_default=False),
            ui=replace(
                cfg.ui,
                interval_ms=140,
                sim_steps_per_render=3,
                render_truth_every=8,
                render_team_every=5,
                render_local_every=12,
                render_frontier_every=24,
                max_draw_path_points=180,
                max_draw_graph_edges=45,
                max_draw_graph_nodes=70,
                max_draw_frontiers=10,
                max_draw_teammate_visit_points=8,
                max_draw_teammate_trajectory_points=35,
            ),
        )
    raise ValueError(f"unknown UI profile: {profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-CAGE LiDAR-guided multi-robot baseline")
    parser.add_argument("--headless", action="store_true", help="run without UI")
    parser.add_argument("--steps", type=int, default=500, help="headless simulation steps")
    parser.add_argument("--seed", type=int, default=None, help="override world seed")
    parser.add_argument("--map-resolution", type=float, default=None, help="override occupancy-grid cell size in meters, e.g. 0.20 for sharper maps")
    parser.add_argument(
        "--renderer",
        choices=("pygame", "matplotlib"),
        default="pygame",
        help="interactive renderer",
    )
    parser.add_argument("--fps", type=int, default=60, help="target FPS for the pygame renderer")
    parser.add_argument("--width", type=int, default=1920, help="pygame window width")
    parser.add_argument("--height", type=int, default=1080, help="pygame window height")
    parser.add_argument("--fullscreen", action="store_true", help="start pygame renderer fullscreen")
    parser.add_argument(
        "--ui-profile",
        choices=("fast", "balanced", "quality"),
        default="fast",
        help="matplotlib detail/performance profile, ignored by pygame",
    )
    args = parser.parse_args()

    cfg = AppConfig()
    if args.seed is not None:
        cfg = replace(cfg, world=replace(cfg.world, seed=args.seed))
    if args.map_resolution is not None:
        cfg = replace(cfg, mapping=replace(cfg.mapping, resolution=max(0.05, float(args.map_resolution))))
    if not args.headless:
        if args.renderer == "pygame":
            cfg = realtime_config(cfg, fps=args.fps)
        else:
            cfg = apply_ui_profile(cfg, args.ui_profile)
    sim = Simulator(cfg)

    if args.headless:
        status = sim.run_headless(args.steps)
        print(f"phase={status.phase} success={status.success} message={status.message}")
        print(f"time_s={sim.time_s:.1f} home_target={sim.home_memory.target.detected} routes={len(sim.home_memory.best_routes)}")
        for i, route in enumerate(sim.home_memory.best_routes[:4]):
            print(f"route[{i}] length={route.length:.2f} clearance={route.min_clearance:.2f} cert={route.certificate:.2f} reported={route.reported_home} status={route.status}")
    else:
        if args.renderer == "pygame":
            PygameDashboard(sim, fps=args.fps, width=args.width, height=args.height, fullscreen=args.fullscreen).run()
        else:
            MatplotlibDashboard(sim).run()


if __name__ == "__main__":
    main()
