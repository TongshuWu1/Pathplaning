"""Fresh simulator state built on top of editable environments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .environment import EditableEnvironment
from .geometry import Pose, wrap_angle


@dataclass
class SimRobot:
    pose: list[float]
    command_v: float = 0.0
    command_omega: float = 0.0

    @property
    def xy(self) -> tuple[float, float]:
        return (float(self.pose[0]), float(self.pose[1]))

    @property
    def theta(self) -> float:
        return float(self.pose[2])

    def as_pose(self) -> Pose:
        return (float(self.pose[0]), float(self.pose[1]), float(self.pose[2]))


class SimulatorState:
    def __init__(self, env: EditableEnvironment, source_path: Path | None = None):
        self.env = env
        self.source_path = source_path
        self.time_s = 0.0
        self.running = False
        self.max_speed_mps = 1.0
        self.max_turn_radps = 2.2
        self.robot = SimRobot([env.home.center[0], env.home.center[1], 0.0])
        self.status = "Ready"

    @classmethod
    def from_environment(cls, env: EditableEnvironment, source_path: Path | None = None) -> "SimulatorState":
        return cls(env, source_path=source_path)

    def set_environment(self, env: EditableEnvironment, source_path: Path | None = None) -> None:
        self.env = env
        self.source_path = source_path
        self.reset()
        self.status = "Environment loaded"

    @property
    def collision_radius(self) -> float:
        half_l = 0.5 * float(self.env.robot_length)
        half_w = 0.5 * float(self.env.robot_width)
        return math.hypot(half_l, half_w) + float(self.env.robot_safety_margin)

    def reset(self) -> None:
        self.time_s = 0.0
        self.running = False
        self.robot.pose[:] = [self.env.home.center[0], self.env.home.center[1], 0.0]
        self.robot.command_v = 0.0
        self.robot.command_omega = 0.0
        self.status = "Reset"

    def set_command(self, v: float, omega: float) -> None:
        self.robot.command_v = max(-self.max_speed_mps, min(self.max_speed_mps, float(v)))
        self.robot.command_omega = max(-self.max_turn_radps, min(self.max_turn_radps, float(omega)))

    def step(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return
        x, y, theta = self.robot.pose
        v = self.robot.command_v
        omega = self.robot.command_omega
        new_theta = wrap_angle(theta + omega * dt)
        mid_theta = theta + 0.5 * omega * dt
        new_x = x + math.cos(mid_theta) * v * dt
        new_y = y + math.sin(mid_theta) * v * dt
        radius = self.collision_radius
        if self.env.segment_free((x, y), (new_x, new_y), margin=radius) and self.env.is_free((new_x, new_y), margin=radius):
            self.robot.pose[:] = [new_x, new_y, new_theta]
            self.status = "Running" if self.running else "Manual"
        else:
            self.robot.command_v = 0.0
            self.status = "Collision blocked"
        self.time_s += dt
