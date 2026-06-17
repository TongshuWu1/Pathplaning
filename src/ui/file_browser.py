"""In-app JSON file browser for pygame tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BrowserResult:
    action: str
    path: Path | None = None


@dataclass
class BrowserEntry:
    path: Path
    label: str
    is_dir: bool


class JsonFileBrowser:
    def __init__(
        self,
        pg: Any,
        title: str,
        mode: str,
        initial_dir: Path,
        initial_file: str = "environment.json",
    ):
        self.pg = pg
        self.title = title
        self.mode = mode
        self.current_dir = Path(initial_dir).expanduser()
        if not self.current_dir.exists():
            self.current_dir = Path.cwd()
        if self.current_dir.is_file():
            self.current_dir = self.current_dir.parent
        self.filename = initial_file or "environment.json"
        self.focus_filename = mode != "open"
        self.status = ""
        self.scroll = 0
        self.entries: list[BrowserEntry] = []
        self.entry_rects: list[tuple[Any, BrowserEntry]] = []
        self.up_rect = pg.Rect(0, 0, 0, 0)
        self.filename_rect = pg.Rect(0, 0, 0, 0)
        self.cancel_rect = pg.Rect(0, 0, 0, 0)
        self.confirm_rect = pg.Rect(0, 0, 0, 0)
        self._refresh()

    def _refresh(self) -> None:
        entries: list[BrowserEntry] = []
        try:
            children = list(self.current_dir.iterdir())
        except OSError as exc:
            self.status = f"Cannot read folder: {exc}"
            children = []
        dirs = sorted([p for p in children if p.is_dir() and not p.name.startswith(".")], key=lambda p: p.name.lower())
        files = sorted([p for p in children if p.is_file() and p.suffix.lower() == ".json"], key=lambda p: p.name.lower())
        if self.current_dir.parent != self.current_dir:
            entries.append(BrowserEntry(self.current_dir.parent, "..", True))
        entries.extend(BrowserEntry(p, f"[{p.name}]", True) for p in dirs)
        entries.extend(BrowserEntry(p, p.name, False) for p in files)
        self.entries = entries
        self.scroll = max(0, min(self.scroll, max(0, len(self.entries) - 1)))

    def handle_event(self, event: Any) -> BrowserResult | None:
        pg = self.pg
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                return BrowserResult("cancel")
            if event.key == pg.K_RETURN:
                return self._confirm()
            if event.key == pg.K_BACKSPACE and self.focus_filename:
                self.filename = self.filename[:-1]
            elif event.key == pg.K_UP:
                self.scroll = max(0, self.scroll - 1)
            elif event.key == pg.K_DOWN:
                self.scroll = min(max(0, len(self.entries) - 1), self.scroll + 1)
        elif event.type == pg.TEXTINPUT and self.focus_filename:
            allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- "
            for char in event.text:
                if char in allowed and len(self.filename) < 96:
                    self.filename += char
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 4:
                self.scroll = max(0, self.scroll - 3)
            elif event.button == 5:
                self.scroll = min(max(0, len(self.entries) - 1), self.scroll + 3)
            elif event.button == 1:
                return self._mouse_down(event.pos)
        return None

    def _mouse_down(self, pos: tuple[int, int]) -> BrowserResult | None:
        if self.cancel_rect.collidepoint(pos):
            return BrowserResult("cancel")
        if self.confirm_rect.collidepoint(pos):
            return self._confirm()
        if self.up_rect.collidepoint(pos):
            if self.current_dir.parent != self.current_dir:
                self.current_dir = self.current_dir.parent
                self.scroll = 0
                self._refresh()
            return None
        if self.filename_rect.collidepoint(pos):
            self.focus_filename = True
            return None
        self.focus_filename = False
        for rect, entry in self.entry_rects:
            if rect.collidepoint(pos):
                if entry.is_dir:
                    self.current_dir = entry.path
                    self.scroll = 0
                    self._refresh()
                else:
                    self.filename = entry.path.name
                    if self.mode == "open":
                        return BrowserResult("confirm", entry.path)
                return None
        return None

    def _confirm(self) -> BrowserResult | None:
        name = Path(self.filename).name.strip()
        if not name:
            self.status = "Enter a filename"
            return None
        if not name.endswith(".json"):
            name = f"{name}.json"
        path = self.current_dir / name
        if self.mode == "open" and not path.exists():
            self.status = f"Missing {path.name}"
            return None
        return BrowserResult("confirm", path)

    def draw(self, screen: Any, font: Any, small: Any, mono: Any) -> None:
        pg = self.pg
        w, h = screen.get_size()
        overlay = pg.Surface((w, h), pg.SRCALPHA)
        overlay.fill((15, 23, 42, 135))
        screen.blit(overlay, (0, 0))

        dialog = pg.Rect(0, 0, min(820, w - 80), min(610, h - 80))
        dialog.center = (w // 2, h // 2)
        pg.draw.rect(screen, (248, 250, 252), dialog, border_radius=8)
        pg.draw.rect(screen, (71, 85, 105), dialog, width=2, border_radius=8)

        title = font.render(self.title, True, (15, 23, 42))
        screen.blit(title, (dialog.x + 22, dialog.y + 18))
        folder = self._ellipsize(str(self.current_dir), 86)
        folder_surf = small.render(folder, True, (71, 85, 105))
        screen.blit(folder_surf, (dialog.x + 22, dialog.y + 48))

        self.up_rect = pg.Rect(dialog.right - 122, dialog.y + 22, 94, 30)
        self._button(screen, self.up_rect, "Up", small)

        list_rect = pg.Rect(dialog.x + 22, dialog.y + 82, dialog.w - 44, dialog.h - 190)
        pg.draw.rect(screen, (255, 255, 255), list_rect, border_radius=5)
        pg.draw.rect(screen, (203, 213, 225), list_rect, width=1, border_radius=5)
        self.entry_rects = []
        row_h = 26
        visible = max(1, list_rect.h // row_h)
        start = min(self.scroll, max(0, len(self.entries) - visible))
        for idx, entry in enumerate(self.entries[start:start + visible]):
            row = pg.Rect(list_rect.x + 6, list_rect.y + 6 + idx * row_h, list_rect.w - 12, row_h - 2)
            self.entry_rects.append((row, entry))
            is_selected = entry.path.name == self.filename
            fill = (219, 234, 254) if is_selected else ((241, 245, 249) if idx % 2 else (255, 255, 255))
            pg.draw.rect(screen, fill, row, border_radius=3)
            color = (14, 116, 144) if entry.is_dir else (30, 41, 59)
            label = small.render(entry.label, True, color)
            screen.blit(label, (row.x + 8, row.y + 5))

        self.filename_rect = pg.Rect(dialog.x + 22, dialog.bottom - 86, dialog.w - 244, 34)
        pg.draw.rect(screen, (255, 255, 255), self.filename_rect, border_radius=5)
        pg.draw.rect(screen, (14, 165, 233) if self.focus_filename else (148, 163, 184), self.filename_rect, width=2 if self.focus_filename else 1, border_radius=5)
        name_surf = mono.render(self.filename, True, (15, 23, 42))
        screen.blit(name_surf, (self.filename_rect.x + 9, self.filename_rect.y + 9))

        self.cancel_rect = pg.Rect(dialog.right - 206, dialog.bottom - 84, 82, 34)
        self.confirm_rect = pg.Rect(dialog.right - 112, dialog.bottom - 84, 90, 34)
        self._button(screen, self.cancel_rect, "Cancel", small)
        self._button(screen, self.confirm_rect, "Open" if self.mode == "open" else "Save", small, active=True)

        if self.status:
            status = small.render(self.status, True, (185, 28, 28))
            screen.blit(status, (dialog.x + 22, dialog.bottom - 36))

    def _button(self, screen: Any, rect: Any, label: str, small: Any, active: bool = False) -> None:
        fill = (14, 165, 233) if active else (226, 232, 240)
        edge = (2, 132, 199) if active else (148, 163, 184)
        color = (255, 255, 255) if active else (15, 23, 42)
        self.pg.draw.rect(screen, fill, rect, border_radius=5)
        self.pg.draw.rect(screen, edge, rect, width=1, border_radius=5)
        surf = small.render(label, True, color)
        screen.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2))

    def _ellipsize(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return "..." + text[-max(1, limit - 3):]
