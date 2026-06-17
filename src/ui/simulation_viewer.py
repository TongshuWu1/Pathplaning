"""Pygame simulator viewer for the fresh simulator."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..environment import EnvironmentRect, load_environment
from ..geometry import Point, Rect
from ..simulation import SimulatorState
from .file_browser import BrowserResult, JsonFileBrowser


@dataclass
class Button:
    rect: Any
    action: str
    label: str


class SimulationViewer:
    def __init__(self, sim: SimulatorState, width: int = 1600, height: int = 950, fps: int = 60):
        import pygame

        self.pg = pygame
        self.sim = sim
        self.fps = max(1, int(fps))
        self.running = True
        self.manual_v = 0.0
        self.manual_omega = 0.0
        self.buttons: list[Button] = []
        self.button_bottom = 80
        self.file_browser: JsonFileBrowser | None = None

        pygame.init()
        pygame.display.set_caption("Pathplaning Simulator")
        self.screen = pygame.display.set_mode((max(1100, int(width)), max(720, int(height))), pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial,dejavusans", 16)
        self.small = pygame.font.SysFont("arial,dejavusans", 13)
        self.title_font = pygame.font.SysFont("arial,dejavusans", 24, bold=True)
        self.mono = pygame.font.SysFont("consolas,dejavusansmono", 13)

        self.sidebar = pygame.Rect(0, 0, 0, 0)
        self.topbar = pygame.Rect(0, 0, 0, 0)
        self.canvas = pygame.Rect(0, 0, 0, 0)
        self.map_rect = pygame.Rect(0, 0, 0, 0)
        self.scale = 1.0
        self._update_layout()

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.fps) / 1000.0
            self._handle_events()
            self._update_manual_command()
            if self.sim.running or abs(self.manual_v) > 1e-6 or abs(self.manual_omega) > 1e-6:
                self.sim.step(dt)
            self._draw()
            self.pg.display.flip()
        self.pg.quit()

    def _update_layout(self) -> None:
        w, h = self.screen.get_size()
        side_w = 310
        top_h = 66
        self.sidebar = self.pg.Rect(0, 0, side_w, h)
        self.topbar = self.pg.Rect(side_w, 0, max(1, w - side_w), top_h)
        self.canvas = self.pg.Rect(side_w + 20, top_h + 18, max(1, w - side_w - 40), max(1, h - top_h - 38))
        self._update_map_transform()
        self._rebuild_buttons()

    def _rebuild_buttons(self) -> None:
        pg = self.pg
        x = 18
        y = 82
        bw = 130
        bh = 30
        gap = 8
        self.buttons = [
            Button(pg.Rect(x, y, bw, bh), "open", "Open Map..."),
            Button(pg.Rect(x + bw + gap, y, bw, bh), "reload", "Reload"),
            Button(pg.Rect(x, y + bh + gap, bw, bh), "reset", "Reset"),
            Button(pg.Rect(x + bw + gap, y + bh + gap, bw, bh), "pause", "Pause/Run"),
        ]
        self.button_bottom = y + 2 * bh + gap

    def _update_map_transform(self) -> None:
        pad = 10
        env = self.sim.env
        available = self.canvas.inflate(-2 * pad, -2 * pad)
        self.scale = min(available.w / max(1e-6, env.width), available.h / max(1e-6, env.height))
        map_w = int(round(env.width * self.scale))
        map_h = int(round(env.height * self.scale))
        self.map_rect = self.pg.Rect(
            available.x + (available.w - map_w) // 2,
            available.y + (available.h - map_h) // 2,
            map_w,
            map_h,
        )

    def _handle_events(self) -> None:
        pg = self.pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif self.file_browser is not None:
                result = self.file_browser.handle_event(event)
                if result is not None:
                    self._handle_file_browser_result(result)
            elif event.type == pg.VIDEORESIZE:
                self.screen = pg.display.set_mode((max(1100, event.w), max(720, event.h)), pg.RESIZABLE | pg.DOUBLEBUF)
                self._update_layout()
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    self.running = False
                elif event.key == pg.K_SPACE:
                    self.sim.running = not self.sim.running
                    self.sim.status = "Running" if self.sim.running else "Paused"
                elif event.key == pg.K_o and (event.mod & pg.KMOD_CTRL):
                    self._open_environment_dialog()
                elif event.key == pg.K_r and (event.mod & pg.KMOD_CTRL):
                    self._reload_environment()
                elif event.key == pg.K_r:
                    self.sim.reset()
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                self._mouse_down(event.pos)

    def _mouse_down(self, pos: tuple[int, int]) -> None:
        for button in self.buttons:
            if button.rect.collidepoint(pos):
                self._run_button(button.action)
                return

    def _run_button(self, action: str) -> None:
        if action == "open":
            self._open_environment_dialog()
        elif action == "reload":
            self._reload_environment()
        elif action == "reset":
            self.sim.reset()
        elif action == "pause":
            self.sim.running = not self.sim.running
            self.sim.status = "Running" if self.sim.running else "Paused"

    def _open_environment_dialog(self) -> None:
        initial_dir = self.sim.source_path.parent if self.sim.source_path else Path("environments")
        initial_file = self.sim.source_path.name if self.sim.source_path else "environment_01.json"
        self.file_browser = JsonFileBrowser(self.pg, "Open Simulation Environment", "open", initial_dir, initial_file)
        self.sim.status = "Choose environment"

    def _handle_file_browser_result(self, result: BrowserResult) -> None:
        self.file_browser = None
        if result.action == "cancel" or result.path is None:
            self.sim.status = "File browser closed"
            return
        self._load_environment_path(result.path)

    def _reload_environment(self) -> None:
        if self.sim.source_path is None:
            self.sim.status = "No environment file to reload"
            return
        self._load_environment_path(self.sim.source_path)

    def _load_environment_path(self, path: Path) -> None:
        try:
            env = load_environment(path)
        except Exception as exc:
            self.sim.status = f"Load failed: {exc}"
            return
        self.sim.set_environment(env, source_path=path)

    def _update_manual_command(self) -> None:
        pg = self.pg
        keys = pg.key.get_pressed()
        v = 0.0
        omega = 0.0
        if keys[pg.K_UP] or keys[pg.K_w]:
            v += self.sim.max_speed_mps
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            v -= 0.55 * self.sim.max_speed_mps
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            omega += self.sim.max_turn_radps
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            omega -= self.sim.max_turn_radps
        self.manual_v = v
        self.manual_omega = omega
        self.sim.set_command(v, omega)

    def _draw(self) -> None:
        self._update_map_transform()
        pg = self.pg
        self.screen.fill((241, 245, 249))
        pg.draw.rect(self.screen, (15, 23, 42), self.sidebar)
        pg.draw.rect(self.screen, (255, 255, 255), self.topbar)
        pg.draw.line(self.screen, (214, 226, 238), (self.topbar.x, self.topbar.bottom), (self.topbar.right, self.topbar.bottom), 1)
        self._draw_sidebar()
        self._draw_topbar()
        self._draw_canvas()
        if self.file_browser is not None:
            self.file_browser.draw(self.screen, self.font, self.small, self.mono)

    def _draw_sidebar(self) -> None:
        self._text("Simulator", (18, 20), self.title_font, (248, 250, 252))
        self._text("Fresh runtime", (18, 48), self.small, (148, 163, 184))
        for button in self.buttons:
            self._button(button.rect, button.label)
        path = self.sim.source_path.name if self.sim.source_path else "unsaved"
        rows = [
            f"Map {self.sim.env.name}",
            f"File {path}",
            f"Time {self.sim.time_s:6.2f} s",
            f"Status {self.sim.status}",
            f"Pose x {self.sim.robot.pose[0]:.2f}",
            f"Pose y {self.sim.robot.pose[1]:.2f}",
            f"Yaw {math.degrees(self.sim.robot.pose[2]):.1f} deg",
            f"Command v {self.sim.robot.command_v:.2f}",
            f"Command w {self.sim.robot.command_omega:.2f}",
            f"Radius {self.sim.collision_radius:.2f} m",
            "",
            "Controls",
            "WASD / arrows drive",
            "Space pause/run",
            "R reset",
            "Esc quit",
        ]
        y = self.button_bottom + 20
        for row in rows:
            if row:
                self._text(row, (18, y), self.small, (226, 232, 240))
            y += 22

    def _draw_topbar(self) -> None:
        env = self.sim.env
        self._text(env.name or "environment", (self.topbar.x + 24, 18), self.title_font, (15, 23, 42))
        self._text(
            f"Target {env.target[0]:.1f}, {env.target[1]:.1f}   Landmarks {len(env.landmarks)}   Robot {env.robot_length:.2f} x {env.robot_width:.2f} m",
            (max(self.topbar.x + 360, self.topbar.right - 540), 24),
            self.small,
            (71, 85, 105),
        )

    def _draw_canvas(self) -> None:
        pg = self.pg
        env = self.sim.env
        pg.draw.rect(self.screen, (226, 232, 240), self.canvas, border_radius=6)
        pg.draw.rect(self.screen, (255, 255, 255), self.map_rect)
        self._draw_grid()
        for item in env.rects:
            self._draw_item(item)
        self._draw_home()
        self._draw_target()
        self._draw_landmarks()
        self._draw_robot()
        pg.draw.rect(self.screen, (100, 116, 139), self.map_rect, width=2)

    def _draw_grid(self) -> None:
        pg = self.pg
        env = self.sim.env
        g = max(0.25, float(env.grid_size))
        n_x = int(env.width / g) + 1
        n_y = int(env.height / g) + 1
        for i in range(n_x + 1):
            xw = min(env.width, i * g)
            x = self._world_to_screen((xw, 0.0))[0]
            major = abs((xw / 5.0) - round(xw / 5.0)) < 1e-6
            color = (203, 213, 225) if major else (232, 238, 245)
            pg.draw.line(self.screen, color, (x, self.map_rect.y), (x, self.map_rect.bottom), 1)
        for j in range(n_y + 1):
            yw = min(env.height, j * g)
            y = self._world_to_screen((0.0, yw))[1]
            major = abs((yw / 5.0) - round(yw / 5.0)) < 1e-6
            color = (203, 213, 225) if major else (232, 238, 245)
            pg.draw.line(self.screen, color, (self.map_rect.x, y), (self.map_rect.right, y), 1)

    def _draw_item(self, item: EnvironmentRect) -> None:
        color = (51, 65, 85) if item.kind == "wall" else (86, 96, 112)
        rect = self._rect_to_screen(item.rect)
        self.pg.draw.rect(self.screen, color, rect)
        self.pg.draw.rect(self.screen, (30, 41, 59), rect, width=1)

    def _draw_home(self) -> None:
        rect = self._rect_to_screen(self.sim.env.home.rect)
        self.pg.draw.rect(self.screen, (187, 247, 208), rect)
        self.pg.draw.rect(self.screen, (22, 163, 74), rect, width=2)
        self._center_text("HOME", rect, self.small, (21, 128, 61))

    def _draw_target(self) -> None:
        pg = self.pg
        cx, cy = self._world_to_screen(self.sim.env.target)
        pg.draw.circle(self.screen, (254, 226, 226), (cx, cy), 10)
        pg.draw.line(self.screen, (220, 38, 38), (cx - 7, cy - 7), (cx + 7, cy + 7), 3)
        pg.draw.line(self.screen, (220, 38, 38), (cx + 7, cy - 7), (cx - 7, cy + 7), 3)

    def _draw_landmarks(self) -> None:
        pg = self.pg
        for landmark in self.sim.env.landmarks:
            x, y = self._world_to_screen(landmark.xy)
            radius = 6
            pts = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
            pg.draw.polygon(self.screen, (250, 204, 21), pts)
            pg.draw.polygon(self.screen, (113, 63, 18), pts, width=1)
            if landmark.name:
                self._text(landmark.name, (x + 9, y - 9), self.small, (92, 64, 14))

    def _draw_robot(self) -> None:
        env = self.sim.env
        pose = self.sim.robot.as_pose()
        half_l = 0.5 * float(env.robot_length)
        half_w = 0.5 * float(env.robot_width)
        margin = float(env.robot_safety_margin)
        safety = self._oriented_box_points((pose[0], pose[1]), half_l + margin, half_w + margin, pose[2])
        body = self._oriented_box_points((pose[0], pose[1]), half_l, half_w, pose[2])
        overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
        self.pg.draw.polygon(overlay, (245, 158, 11, 42), [self._world_to_screen(p) for p in safety])
        self.pg.draw.polygon(overlay, (14, 165, 233, 90), [self._world_to_screen(p) for p in body])
        self.screen.blit(overlay, (0, 0))
        safety_pts = [self._world_to_screen(p) for p in safety]
        body_pts = [self._world_to_screen(p) for p in body]
        self.pg.draw.polygon(self.screen, (217, 119, 6), safety_pts, width=2)
        self.pg.draw.polygon(self.screen, (2, 132, 199), body_pts, width=2)
        center = self._world_to_screen((pose[0], pose[1]))
        nose = self._world_to_screen((pose[0] + math.cos(pose[2]) * half_l, pose[1] + math.sin(pose[2]) * half_l))
        self.pg.draw.line(self.screen, (2, 132, 199), center, nose, 3)

    def _oriented_box_points(self, center: Point, half_l: float, half_w: float, heading: float) -> list[Point]:
        ct = math.cos(heading)
        st = math.sin(heading)
        local = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
        return [
            (center[0] + lx * ct - ly * st, center[1] + lx * st + ly * ct)
            for lx, ly in local
        ]

    def _rect_to_screen(self, rect: Rect) -> Any:
        r = rect.normalized()
        left, top = self._world_to_screen((r.x0, r.y1))
        right, bottom = self._world_to_screen((r.x1, r.y0))
        return self.pg.Rect(left, top, max(1, right - left), max(1, bottom - top))

    def _world_to_screen(self, point: Point) -> tuple[int, int]:
        return (
            int(round(self.map_rect.x + point[0] * self.scale)),
            int(round(self.map_rect.bottom - point[1] * self.scale)),
        )

    def _text(self, text: str, pos: tuple[int, int], font: Any, color: tuple[int, int, int]) -> None:
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, pos)

    def _center_text(self, text: str, rect: Any, font: Any, color: tuple[int, int, int]) -> None:
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2))

    def _button(self, rect: Any, label: str) -> None:
        self.pg.draw.rect(self.screen, (43, 54, 70), rect, border_radius=5)
        self.pg.draw.rect(self.screen, (71, 85, 105), rect, width=1, border_radius=5)
        self._center_text(label, rect, self.small, (226, 232, 240))
