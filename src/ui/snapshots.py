"""Small view models for the dashboard.

The dashboard reads these stable fields instead of reaching deep into every
subsystem.  This keeps rendering separate from simulator logic.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..geometry import Point, Pose


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
