"""LiDAR-first local safety and scan-map consistency assessment."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import AssessmentConfig, LidarConfig
from .geometry import Pose
from .mapping import OccupancyGrid
from .sensors import LidarScan


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


def _sector_min(scan: LidarScan, center: float, half_width: float, fallback: float) -> float:
    delta = np.angle(np.exp(1j * (scan.angles - center)))
    m = np.abs(delta) <= half_width
    if not np.any(m):
        return fallback
    return float(np.min(scan.ranges[m]))


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
    front = _sector_min(scan, 0.0, front_half, lidar_cfg.range)
    left = _sector_min(scan, math.pi / 2.0, side_half, lidar_cfg.range)
    right = _sector_min(scan, -math.pi / 2.0, side_half, lidar_cfg.range)
    blocked = front <= lidar_cfg.blocked_forward_distance

    open_mask = scan.ranges > max(lidar_cfg.range * 0.58, lidar_cfg.blocked_forward_distance * 2.0)
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
        best = max(sectors, key=lambda ab: (ab[1] - ab[0]) % n)
        mid = (best[0] + ((best[1] - best[0]) % n) / 2.0) % n
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
