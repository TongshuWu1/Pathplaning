"""Search-CAGE baseline simulator orchestrator."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cage_graph import RouteGraph
from .communication import CommunicationManager, HomeMemory, CommunicationState
from .config import AppConfig
from .mapping import OccupancyGrid
from .robot import RobotAgent
from .world import World


@dataclass
class MissionStatus:
    phase: str = "SEARCH_TARGET"
    success: bool = False
    message: str = "Searching for hidden target"


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
        self.communication = CommunicationManager(self.cfg.communication, self.world, self.home_memory)
        self.comm_state = CommunicationState()
        self.mission = MissionStatus()
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
            robot.step_predict_and_move(self.world, dt)
        for robot in self.robots:
            robot.sense_update_map_and_belief(self.world, self.time_s)

        self.comm_state = self.communication.update(self.robots, self.time_s)

        for robot in self.robots:
            robot.choose_task_and_plan(self.time_s)
        for robot in self.robots:
            robot.compute_control()

        self._update_mission_status()

    def _update_mission_status(self) -> None:
        home_target = self.home_memory.target.detected
        local_target = any(r.target.detected for r in self.robots)
        routes = self.home_memory.graph.top_routes(k=max(1, self.cfg.cage.desired_route_count))
        good_routes = [r for r in routes if r.certificate >= self.cfg.cage.route_cert_threshold]
        self.home_memory.best_routes = routes
        if home_target and good_routes:
            self.mission = MissionStatus("COMPLETE", True, "HOME has target report and certified route")
        elif home_target:
            self.mission = MissionStatus("CERTIFY_ROUTES", False, "HOME knows target; certifying route options")
        elif local_target:
            self.mission = MissionStatus("REPORT_TARGET", False, "Target found by robot; reporting to HOME")
        else:
            self.mission = MissionStatus("SEARCH_TARGET", False, "Searching for hidden target")

    def run_headless(self, steps: int = 400) -> MissionStatus:
        for _ in range(steps):
            self.step()
            if self.mission.success:
                break
        return self.mission
