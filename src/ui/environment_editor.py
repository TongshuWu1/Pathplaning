"""Pygame environment editor for custom rooms, corridors, and obstacles."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..environment import EditableEnvironment, EnvironmentPoint, EnvironmentRect, load_environment, make_default_environment, save_environment
from ..geometry import Point, Rect
from .file_browser import BrowserResult, JsonFileBrowser


@dataclass
class Control:
    rect: Any
    action: str
    label: str


class EnvironmentEditor:
    def __init__(
        self,
        env: EditableEnvironment | None = None,
        width: int = 1600,
        height: int = 950,
        env_dir: str | Path = "environments",
        filename: str = "environment_01.json",
    ):
        import pygame

        self.pg = pygame
        self.env = env or make_default_environment()
        self.env_dir = Path(env_dir)
        initial_path = Path(filename)
        if not initial_path.is_absolute():
            initial_path = initial_path if initial_path.parent != Path(".") else self.env_dir / initial_path
        self.current_path = initial_path
        self.env_dir = self.current_path.parent
        self.filename = self.current_path.name
        self.running = True
        self.tool = "select"
        self.snap_enabled = True
        self.show_robot_preview = True
        self.room_door_side = "bottom"
        self.selected_id: str | None = None
        self.status = "Ready"
        self.filename_active = False
        self.drag_start: Point | None = None
        self.drag_current: Point | None = None
        self.drag_original: EnvironmentRect | EnvironmentPoint | None = None
        self.mouse_world: Point | None = None
        self.controls: list[Control] = []
        self.file_browser: JsonFileBrowser | None = None
        self.file_browser_action = ""

        pygame.init()
        pygame.display.set_caption("Pathplaning Environment Editor")
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
        self.filename_rect = pygame.Rect(0, 0, 0, 0)
        self.scale = 1.0
        self._update_layout()

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._draw()
            self.pg.display.flip()
            self.clock.tick(60)
        self.pg.quit()

    def _update_layout(self) -> None:
        w, h = self.screen.get_size()
        side_w = 330
        top_h = 66
        self.sidebar = self.pg.Rect(0, 0, side_w, h)
        self.topbar = self.pg.Rect(side_w, 0, max(1, w - side_w), top_h)
        self.canvas = self.pg.Rect(side_w + 20, top_h + 18, max(1, w - side_w - 40), max(1, h - top_h - 38))
        self._update_map_transform()
        self._rebuild_controls()

    def _update_map_transform(self) -> None:
        pad = 10
        available = self.canvas.inflate(-2 * pad, -2 * pad)
        self.scale = min(available.w / max(1e-6, self.env.width), available.h / max(1e-6, self.env.height))
        map_w = int(round(self.env.width * self.scale))
        map_h = int(round(self.env.height * self.scale))
        self.map_rect = self.pg.Rect(
            available.x + (available.w - map_w) // 2,
            available.y + (available.h - map_h) // 2,
            map_w,
            map_h,
        )

    def _rebuild_controls(self) -> None:
        pg = self.pg
        self.controls = []
        x = 18
        y = 84
        bw = 138
        bh = 30
        gap = 8
        tools = [
            ("select", "Select"),
            ("obstacle", "Obstacle"),
            ("wall", "Wall"),
            ("room", "Room"),
            ("landmark", "Landmark"),
            ("erase", "Erase"),
            ("home", "Home"),
            ("target", "Target"),
        ]
        for idx, (action, label) in enumerate(tools):
            col = idx % 2
            row = idx // 2
            self.controls.append(Control(pg.Rect(x + col * (bw + gap), y + row * (bh + gap), bw, bh), f"tool:{action}", label))

        y += 4 * (bh + gap) + 10
        self.controls.append(Control(pg.Rect(x, y, bw, bh), "new", "New"))
        self.controls.append(Control(pg.Rect(x + bw + gap, y, bw, bh), "save", "Save"))
        y += bh + gap
        self.controls.append(Control(pg.Rect(x, y, bw, bh), "open", "Open..."))
        self.controls.append(Control(pg.Rect(x + bw + gap, y, bw, bh), "save_as", "Save As..."))
        y += bh + gap
        self.controls.append(Control(pg.Rect(x, y, bw, bh), "rename", "Rename..."))
        self.controls.append(Control(pg.Rect(x + bw + gap, y, bw, bh), "delete", "Delete Obj"))

        y += bh + 14
        self.filename_rect = pg.Rect(x, y, 2 * bw + gap, 34)
        y += 48
        self.controls.append(Control(pg.Rect(x, y, bw, bh), "snap", "Grid lock"))
        self.controls.append(Control(pg.Rect(x + bw + gap, y, bw, bh), "door", f"Door {self.room_door_side}"))

        y += bh + 14
        half = 66
        self.controls.append(Control(pg.Rect(x, y, half, bh), "wminus", "W -"))
        self.controls.append(Control(pg.Rect(x + half + 6, y, half, bh), "wplus", "W +"))
        self.controls.append(Control(pg.Rect(x + 2 * (half + 6), y, half, bh), "hminus", "H -"))
        self.controls.append(Control(pg.Rect(x + 3 * (half + 6), y, half, bh), "hplus", "H +"))

        y += bh + 14
        self.controls.append(Control(pg.Rect(x, y, bw, bh), "robot_preview", "Robot preview"))
        self.controls.append(Control(pg.Rect(x + bw + gap, y, bw, bh), "landmark_clear", "Clear LM"))
        y += bh + gap
        self.controls.append(Control(pg.Rect(x, y, half, bh), "robot_minus", "Bot -"))
        self.controls.append(Control(pg.Rect(x + half + 6, y, half, bh), "robot_plus", "Bot +"))
        self.controls.append(Control(pg.Rect(x + 2 * (half + 6), y, half, bh), "margin_minus", "Safe -"))
        self.controls.append(Control(pg.Rect(x + 3 * (half + 6), y, half, bh), "margin_plus", "Safe +"))

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
                self._handle_keydown(event)
            elif event.type == pg.TEXTINPUT and self.filename_active:
                self._insert_filename_text(event.text)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                self._mouse_down(event.pos)
            elif event.type == pg.MOUSEMOTION:
                self._mouse_motion(event.pos)
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                self._mouse_up(event.pos)

    def _handle_keydown(self, event: Any) -> None:
        pg = self.pg
        if self.filename_active:
            if event.key in (pg.K_RETURN, pg.K_ESCAPE):
                self.filename_active = False
            elif event.key == pg.K_BACKSPACE:
                self.filename = self.filename[:-1]
            return
        if event.key in (pg.K_ESCAPE, pg.K_q):
            self.running = False
        elif event.key == pg.K_DELETE:
            self._delete_selected()
        elif event.key == pg.K_s and (event.mod & pg.KMOD_CTRL):
            if event.mod & pg.KMOD_SHIFT:
                self._save_as()
            else:
                self._save()
        elif event.key == pg.K_o and (event.mod & pg.KMOD_CTRL):
            self._open_dialog()
        elif event.key == pg.K_n and (event.mod & pg.KMOD_CTRL):
            self._new_environment()
        elif event.key == pg.K_r and (event.mod & pg.KMOD_CTRL):
            self._rename_dialog()
        elif event.key in (pg.K_1, pg.K_s):
            self.tool = "select"
        elif event.key in (pg.K_2, pg.K_o):
            self.tool = "obstacle"
        elif event.key in (pg.K_3, pg.K_w):
            self.tool = "wall"
        elif event.key in (pg.K_4, pg.K_r):
            self.tool = "room"
        elif event.key in (pg.K_5, pg.K_l):
            self.tool = "landmark"
        elif event.key in (pg.K_6, pg.K_e):
            self.tool = "erase"
        elif event.key == pg.K_b:
            self.show_robot_preview = not self.show_robot_preview

    def _insert_filename_text(self, text: str) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for char in text:
            if char in allowed and len(self.filename) < 80:
                self.filename += char

    def _mouse_down(self, pos: tuple[int, int]) -> None:
        self.filename_active = self.filename_rect.collidepoint(pos)
        if self.filename_active:
            return
        for control in self.controls:
            if control.rect.collidepoint(pos):
                self._run_control(control.action)
                return
        point = self._screen_to_world(pos)
        if point is None:
            self.selected_id = None
            self.mouse_world = None
            return
        point = self._snap(point)
        self.mouse_world = point
        if self.tool == "target":
            self.env.set_target(point)
            self.status = f"Target set at {point[0]:.1f}, {point[1]:.1f}"
            return
        if self.tool == "landmark":
            landmark = self.env.add_landmark(point[0], point[1])
            self.selected_id = landmark.id
            self.status = f"Added landmark {landmark.name}"
            return
        if self.tool == "erase":
            item = self.env.item_at(point)
            if item is not None and item.id != self.env.home.id:
                self.env.remove_item(item.id)
                self.status = f"Removed {item.kind}"
            return
        if self.tool == "select":
            item = self.env.item_at(point)
            self.selected_id = item.id if item is not None else None
            self.drag_start = point if item is not None else None
            self.drag_current = point if item is not None else None
            self.drag_original = item
            return
        self.drag_start = point
        self.drag_current = point
        self.drag_original = None

    def _mouse_motion(self, pos: tuple[int, int]) -> None:
        point = self._screen_to_world(pos)
        if point is None:
            self.mouse_world = None
            return
        point = self._snap(point)
        self.mouse_world = point
        if self.tool == "select" and self.drag_original is not None and self.drag_start is not None:
            dx = point[0] - self.drag_start[0]
            dy = point[1] - self.drag_start[1]
            moved = self.drag_original.moved(dx, dy).clamped(self.env.width, self.env.height)
            self.env.replace_item(moved)
            self.selected_id = moved.id
            self.drag_current = point
        elif self.drag_start is not None:
            self.drag_current = point

    def _mouse_up(self, pos: tuple[int, int]) -> None:
        point = self._screen_to_world(pos)
        if point is not None:
            self.drag_current = self._snap(point)
        if self.tool == "select":
            self.drag_start = None
            self.drag_current = None
            self.drag_original = None
            return
        if self.drag_start is None or self.drag_current is None:
            return
        rect = Rect(self.drag_start[0], self.drag_start[1], self.drag_current[0], self.drag_current[1]).normalized()
        if self.tool == "obstacle":
            item = self.env.add_rect("obstacle", rect.x0, rect.y0, rect.x1, rect.y1, "obstacle")
            if item is not None:
                self.selected_id = None
                self.status = "Obstacle added"
        elif self.tool == "wall":
            item = self._add_wall(rect)
            if item is not None:
                self.selected_id = None
                self.status = "Wall added"
        elif self.tool == "room":
            items = self.env.add_room(rect, door_side=self.room_door_side)
            if items:
                self.selected_id = None
                self.status = "Room added"
        elif self.tool == "home":
            self._set_home(rect)
            self.selected_id = None
            self.status = "Home region updated"
        self.drag_start = None
        self.drag_current = None

    def _add_wall(self, rect: Rect) -> EnvironmentRect | None:
        r = rect.normalized()
        t = max(0.08, float(self.env.wall_thickness))
        if (r.x1 - r.x0) < 0.1 and (r.y1 - r.y0) < 0.1:
            return None
        if (r.x1 - r.x0) >= (r.y1 - r.y0):
            mid = (r.y0 + r.y1) * 0.5
            return self.env.add_rect("wall", r.x0, mid - t * 0.5, r.x1, mid + t * 0.5, "wall")
        mid = (r.x0 + r.x1) * 0.5
        return self.env.add_rect("wall", mid - t * 0.5, r.y0, mid + t * 0.5, r.y1, "wall")

    def _set_home(self, rect: Rect) -> None:
        r = rect.normalized()
        if (r.x1 - r.x0) < 0.6 or (r.y1 - r.y0) < 0.6:
            cx = (r.x0 + r.x1) * 0.5
            cy = (r.y0 + r.y1) * 0.5
            r = Rect(cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5).normalized()
        self.env.set_home(r)

    def _run_control(self, action: str) -> None:
        if action.startswith("tool:"):
            self.tool = action.split(":", 1)[1]
            return
        if action == "new":
            self._new_environment()
        elif action == "save":
            self._save()
        elif action == "open":
            self._open_dialog()
        elif action == "save_as":
            self._save_as()
        elif action == "rename":
            self._rename_dialog()
        elif action == "delete":
            self._delete_selected()
        elif action == "snap":
            self.snap_enabled = True
            self.status = "Grid snapping is locked on"
        elif action == "robot_preview":
            self.show_robot_preview = not self.show_robot_preview
        elif action == "landmark_clear":
            self.env.landmarks.clear()
            if self.selected_id and self.selected_id.startswith("landmark_"):
                self.selected_id = None
            self.status = "Landmarks cleared"
        elif action == "door":
            sides = ["bottom", "right", "top", "left"]
            self.room_door_side = sides[(sides.index(self.room_door_side) + 1) % len(sides)]
            self._rebuild_controls()
        elif action == "wminus":
            self.env.resize(self.env.width - 5.0, self.env.height)
            self._update_map_transform()
        elif action == "wplus":
            self.env.resize(self.env.width + 5.0, self.env.height)
            self._update_map_transform()
        elif action == "hminus":
            self.env.resize(self.env.width, self.env.height - 5.0)
            self._update_map_transform()
        elif action == "hplus":
            self.env.resize(self.env.width, self.env.height + 5.0)
            self._update_map_transform()
        elif action == "robot_minus":
            self._scale_robot(0.90)
        elif action == "robot_plus":
            self._scale_robot(1.10)
        elif action == "margin_minus":
            self.env.robot_safety_margin = max(0.0, self.env.robot_safety_margin - 0.05)
            self.status = f"Safety margin {self.env.robot_safety_margin:.2f} m"
        elif action == "margin_plus":
            self.env.robot_safety_margin = min(1.5, self.env.robot_safety_margin + 0.05)
            self.status = f"Safety margin {self.env.robot_safety_margin:.2f} m"

    def _scale_robot(self, scale: float) -> None:
        self.env.robot_length = min(3.0, max(0.25, self.env.robot_length * scale))
        self.env.robot_width = min(self.env.robot_length, min(2.0, max(0.18, self.env.robot_width * scale)))
        self.status = f"Robot {self.env.robot_length:.2f} x {self.env.robot_width:.2f} m"

    def _new_environment(self) -> None:
        self.env = make_default_environment()
        self.selected_id = None
        self.status = "New environment"
        self._update_map_transform()

    def _delete_selected(self) -> None:
        if self.selected_id is None or self.selected_id == self.env.home.id:
            return
        if self.env.remove_item(self.selected_id):
            self.status = "Selection deleted"
            self.selected_id = None

    def _save(self) -> None:
        path = self._file_path()
        self.env.name = path.stem
        written = save_environment(self.env, path)
        self._set_current_path(written)
        self.status = f"Saved {written}"

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            self.status = f"Missing {path}"
            return
        self.env = load_environment(path)
        self._set_current_path(path)
        self.selected_id = None
        self.status = f"Loaded {path}"
        self._update_map_transform()

    def _file_path(self) -> Path:
        name = Path(self.filename or "environment_01.json").name
        if not name.endswith(".json"):
            name = f"{name}.json"
        typed_path = Path(name)
        if typed_path.name != self.current_path.name:
            self.current_path = self.env_dir / typed_path.name
        return self.current_path

    def _set_current_path(self, path: Path) -> None:
        self.current_path = Path(path)
        self.env_dir = self.current_path.parent
        self.filename = self.current_path.name

    def _open_dialog(self) -> None:
        self.file_browser_action = "open"
        self.file_browser = JsonFileBrowser(self.pg, "Open Environment", "open", self.env_dir, self.current_path.name)

    def _save_as(self) -> None:
        self.file_browser_action = "save_as"
        self.file_browser = JsonFileBrowser(self.pg, "Save Environment As", "save", self.env_dir, self.current_path.name)

    def _rename_dialog(self) -> None:
        self.file_browser_action = "rename"
        self.file_browser = JsonFileBrowser(self.pg, "Rename Environment File", "save", self.env_dir, self.current_path.name)

    def _handle_file_browser_result(self, result: BrowserResult) -> None:
        action = self.file_browser_action
        self.file_browser = None
        self.file_browser_action = ""
        if result.action == "cancel" or result.path is None:
            self.status = "File browser closed"
            return
        if action == "open":
            self._open_path(result.path)
        elif action == "save_as":
            self.env.name = result.path.stem
            written = save_environment(self.env, result.path)
            self._set_current_path(written)
            self.status = f"Saved {written}"
        elif action == "rename":
            old = self._file_path()
            new = result.path
            old_resolved = old.resolve()
            new_resolved = new.resolve()
            if new.exists() and old_resolved != new_resolved:
                self.status = f"Rename blocked: {new.name} already exists"
                return
            self.env.name = new.stem
            written = save_environment(self.env, new)
            self._set_current_path(written)
            if old.exists() and old_resolved != written.resolve():
                old.unlink()
                self.status = f"Renamed to {written}"
            elif old_resolved == written.resolve():
                self.status = f"Saved {written}"
            else:
                self.status = f"Saved {written}; original kept"

    def _snap(self, point: Point) -> Point:
        self.snap_enabled = True
        return self.env.snap_point(point)

    def _world_to_screen(self, point: Point) -> tuple[int, int]:
        return (
            int(round(self.map_rect.x + point[0] * self.scale)),
            int(round(self.map_rect.bottom - point[1] * self.scale)),
        )

    def _screen_to_world(self, pos: tuple[int, int]) -> Point | None:
        if not self.map_rect.collidepoint(pos):
            return None
        x = (pos[0] - self.map_rect.x) / self.scale
        y = (self.map_rect.bottom - pos[1]) / self.scale
        return (float(x), float(y))

    def _selected_item(self) -> EnvironmentRect | EnvironmentPoint | None:
        if self.selected_id == self.env.home.id:
            return self.env.home
        for item in self.env.landmarks:
            if item.id == self.selected_id:
                return item
        for item in self.env.rects:
            if item.id == self.selected_id:
                return item
        return None

    def _draw(self) -> None:
        self._update_map_transform()
        pg = self.pg
        self.screen.fill((242, 245, 249))
        pg.draw.rect(self.screen, (24, 32, 44), self.sidebar)
        pg.draw.rect(self.screen, (255, 255, 255), self.topbar)
        pg.draw.line(self.screen, (216, 226, 236), (self.topbar.x, self.topbar.bottom), (self.topbar.right, self.topbar.bottom), 1)
        self._draw_sidebar()
        self._draw_topbar()
        self._draw_canvas()
        if self.file_browser is not None:
            self.file_browser.draw(self.screen, self.font, self.small, self.mono)

    def _draw_sidebar(self) -> None:
        self._text("Environment", (18, 20), self.title_font, (248, 250, 252))
        self._text("Editor", (18, 47), self.small, (148, 163, 184))
        self._text("Tools", (18, 70), self.small, (203, 213, 225))
        for control in self.controls:
            active = False
            if control.action == f"tool:{self.tool}":
                active = True
            if control.action == "snap" and self.snap_enabled:
                active = True
            if control.action == "robot_preview" and self.show_robot_preview:
                active = True
            self._button(control.rect, control.label, active=active)
        self._text("File", (18, self.filename_rect.y - 20), self.small, (203, 213, 225))
        self._text_box(self.filename_rect, self.filename, active=self.filename_active)

        info_y = max(control.rect.bottom for control in self.controls) + 18 if self.controls else self.filename_rect.y + 150
        selected = self._selected_item()
        rows = [
            f"Map {self.env.width:.0f} x {self.env.height:.0f} m",
            f"Obstacles/walls {len(self.env.rects)}",
            f"Landmarks {len(self.env.landmarks)}",
            f"Grid {self.env.grid_size:.2f} m",
            f"Robot {self.env.robot_length:.2f} x {self.env.robot_width:.2f} m",
            f"Safety +{self.env.robot_safety_margin:.2f} m",
            f"Folder {self.current_path.parent.name}",
            f"Tool {self.tool}",
        ]
        if selected is not None:
            rows.append(f"Selected {selected.kind}")
            if isinstance(selected, EnvironmentPoint):
                rows.append(f"At {selected.x:.1f}, {selected.y:.1f}")
        for idx, row in enumerate(rows):
            y = info_y + idx * 18
            if y > self.sidebar.h - 74:
                break
            self._text(row, (18, y), self.small, (226, 232, 240))
        self._text(self.status, (18, self.sidebar.h - 48), self.small, (186, 230, 253))

    def _draw_topbar(self) -> None:
        title = f"{self.env.name or 'untitled'}"
        self._text(title, (self.topbar.x + 24, 18), self.title_font, (15, 23, 42))
        self._text(
            f"Target {self.env.target[0]:.1f}, {self.env.target[1]:.1f}   Landmarks {len(self.env.landmarks)}   Robot {self.env.robot_length:.2f} x {self.env.robot_width:.2f} m",
            (max(self.topbar.x + 320, self.topbar.right - 520), 24),
            self.small,
            (71, 85, 105),
        )

    def _draw_canvas(self) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (230, 236, 244), self.canvas, border_radius=6)
        pg.draw.rect(self.screen, (255, 255, 255), self.map_rect)
        self._draw_grid()
        for item in self.env.rects:
            self._draw_item(item)
        self._draw_landmarks()
        self._draw_home()
        self._draw_target()
        self._draw_robot_previews()
        self._draw_preview()
        pg.draw.rect(self.screen, (100, 116, 139), self.map_rect, width=2)

    def _draw_grid(self) -> None:
        pg = self.pg
        g = max(0.25, float(self.env.grid_size))
        n_x = int(self.env.width / g) + 1
        n_y = int(self.env.height / g) + 1
        for i in range(n_x + 1):
            xw = min(self.env.width, i * g)
            x = self._world_to_screen((xw, 0.0))[0]
            major = abs((xw / 5.0) - round(xw / 5.0)) < 1e-6
            color = (203, 213, 225) if major else (231, 236, 243)
            pg.draw.line(self.screen, color, (x, self.map_rect.y), (x, self.map_rect.bottom), 1)
        for j in range(n_y + 1):
            yw = min(self.env.height, j * g)
            y = self._world_to_screen((0.0, yw))[1]
            major = abs((yw / 5.0) - round(yw / 5.0)) < 1e-6
            color = (203, 213, 225) if major else (231, 236, 243)
            pg.draw.line(self.screen, color, (self.map_rect.x, y), (self.map_rect.right, y), 1)

    def _draw_item(self, item: EnvironmentRect) -> None:
        color = (51, 65, 85) if item.kind == "wall" else (86, 96, 112)
        selected = self.tool == "select" and item.id == self.selected_id
        outline = (14, 165, 233) if selected else (30, 41, 59)
        rect = self._rect_to_screen(item.rect)
        self.pg.draw.rect(self.screen, color, rect)
        self.pg.draw.rect(self.screen, outline, rect, width=3 if selected else 1)

    def _draw_home(self) -> None:
        rect = self._rect_to_screen(self.env.home.rect)
        selected = self.tool == "select" and self.selected_id == self.env.home.id
        self.pg.draw.rect(self.screen, (187, 247, 208), rect)
        self.pg.draw.rect(self.screen, (22, 163, 74), rect, width=3 if selected else 2)
        self._center_text("HOME", rect, self.small, (21, 128, 61))

    def _draw_landmarks(self) -> None:
        pg = self.pg
        for landmark in self.env.landmarks:
            x, y = self._world_to_screen(landmark.xy)
            selected = self.tool == "select" and landmark.id == self.selected_id
            radius = 8 if selected else 6
            color = (250, 204, 21) if not selected else (251, 146, 60)
            outline = (113, 63, 18) if not selected else (14, 165, 233)
            pts = [(x, y - radius), (x + radius, y), (x, y + radius), (x - radius, y)]
            pg.draw.polygon(self.screen, color, pts)
            pg.draw.polygon(self.screen, outline, pts, width=2 if selected else 1)
            if landmark.name:
                self._text(landmark.name, (x + 9, y - 9), self.small, (92, 64, 14))

    def _draw_target(self) -> None:
        pg = self.pg
        cx, cy = self._world_to_screen(self.env.target)
        pg.draw.circle(self.screen, (254, 226, 226), (cx, cy), 10)
        pg.draw.line(self.screen, (220, 38, 38), (cx - 7, cy - 7), (cx + 7, cy + 7), 3)
        pg.draw.line(self.screen, (220, 38, 38), (cx + 7, cy - 7), (cx - 7, cy + 7), 3)

    def _draw_robot_previews(self) -> None:
        if not self.show_robot_preview:
            return
        home_center = self.env.home.center
        self._draw_robot_footprint(home_center, heading=0.0, selected=False)
        if self.mouse_world is not None and self.map_rect.collidepoint(self.pg.mouse.get_pos()):
            self._draw_robot_footprint(self.mouse_world, heading=0.0, selected=True)

    def _draw_robot_footprint(self, center: Point, heading: float = 0.0, selected: bool = False) -> None:
        half_l = max(0.05, float(self.env.robot_length) * 0.5)
        half_w = max(0.05, float(self.env.robot_width) * 0.5)
        margin = max(0.0, float(self.env.robot_safety_margin))
        safety = self._oriented_box_points(center, half_l + margin, half_w + margin, heading)
        body = self._oriented_box_points(center, half_l, half_w, heading)
        safety_pts = [self._world_to_screen(p) for p in safety]
        body_pts = [self._world_to_screen(p) for p in body]
        overlay = self.pg.Surface(self.screen.get_size(), self.pg.SRCALPHA)
        safe_fill = (245, 158, 11, 42 if selected else 24)
        body_fill = (14, 165, 233, 82 if selected else 56)
        self.pg.draw.polygon(overlay, safe_fill, safety_pts)
        self.pg.draw.polygon(overlay, body_fill, body_pts)
        self.screen.blit(overlay, (0, 0))
        self.pg.draw.polygon(self.screen, (217, 119, 6), safety_pts, width=2 if selected else 1)
        self.pg.draw.polygon(self.screen, (2, 132, 199), body_pts, width=2)
        cx, cy = self._world_to_screen(center)
        nose = self._world_to_screen((center[0] + math.cos(heading) * half_l, center[1] + math.sin(heading) * half_l))
        self.pg.draw.line(self.screen, (2, 132, 199), (cx, cy), nose, 2)

    def _oriented_box_points(self, center: Point, half_l: float, half_w: float, heading: float) -> list[Point]:
        ct = math.cos(heading)
        st = math.sin(heading)
        local = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
        return [
            (center[0] + lx * ct - ly * st, center[1] + lx * st + ly * ct)
            for lx, ly in local
        ]

    def _draw_preview(self) -> None:
        if self.drag_start is None or self.drag_current is None or self.tool == "select":
            return
        rect = Rect(self.drag_start[0], self.drag_start[1], self.drag_current[0], self.drag_current[1]).normalized()
        screen_rect = self._rect_to_screen(rect)
        color = {
            "obstacle": (125, 125, 125),
            "wall": (71, 85, 105),
            "room": (20, 184, 166),
            "home": (34, 197, 94),
            "landmark": (250, 204, 21),
        }.get(self.tool, (59, 130, 246))
        self.pg.draw.rect(self.screen, color, screen_rect, width=3)

    def _rect_to_screen(self, rect: Rect) -> Any:
        r = rect.normalized()
        left, top = self._world_to_screen((r.x0, r.y1))
        right, bottom = self._world_to_screen((r.x1, r.y0))
        return self.pg.Rect(left, top, max(1, right - left), max(1, bottom - top))

    def _button(self, rect: Any, label: str, active: bool = False) -> None:
        fill = (14, 165, 233) if active else (43, 54, 70)
        edge = (125, 211, 252) if active else (71, 85, 105)
        text = (255, 255, 255) if active else (226, 232, 240)
        self.pg.draw.rect(self.screen, fill, rect, border_radius=5)
        self.pg.draw.rect(self.screen, edge, rect, width=1, border_radius=5)
        self._center_text(label, rect, self.small, text)

    def _text_box(self, rect: Any, text: str, active: bool = False) -> None:
        self.pg.draw.rect(self.screen, (248, 250, 252), rect, border_radius=5)
        self.pg.draw.rect(self.screen, (56, 189, 248) if active else (148, 163, 184), rect, width=2 if active else 1, border_radius=5)
        self._text(text or "environment_01.json", (rect.x + 9, rect.y + 9), self.mono, (15, 23, 42))

    def _text(self, text: str, pos: tuple[int, int], font: Any, color: tuple[int, int, int]) -> None:
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, pos)

    def _center_text(self, text: str, rect: Any, font: Any, color: tuple[int, int, int]) -> None:
        surf = font.render(str(text), True, color)
        self.screen.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2))
