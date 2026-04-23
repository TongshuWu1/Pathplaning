from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


@dataclass
class OccupancyGrid:
    width_m: float
    height_m: float
    res: float

    def __post_init__(self) -> None:
        self.nx = int(np.ceil(self.width_m / self.res))
        self.ny = int(np.ceil(self.height_m / self.res))
        self.data = np.full((self.ny, self.nx), UNKNOWN, dtype=np.int8)

    def reset(self) -> None:
        self.data.fill(UNKNOWN)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(x / self.res)
        gy = int(y / self.res)
        if gx < 0:
            gx = 0
        elif gx >= self.nx:
            gx = self.nx - 1
        if gy < 0:
            gy = 0
        elif gy >= self.ny:
            gy = self.ny - 1
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return (gx + 0.5) * self.res, (gy + 0.5) * self.res

    def mark_free(self, x: float, y: float) -> None:
        gx, gy = self.world_to_grid(x, y)
        if self.data[gy, gx] == UNKNOWN:
            self.data[gy, gx] = FREE

    def mark_occupied(self, x: float, y: float) -> None:
        gx, gy = self.world_to_grid(x, y)
        self.data[gy, gx] = OCCUPIED

    def reveal_disk(self, x: float, y: float, radius: float) -> None:
        gx, gy = self.world_to_grid(x, y)
        cells = int(np.ceil(radius / self.res))
        for yy in range(max(0, gy - cells), min(self.ny, gy + cells + 1)):
            for xx in range(max(0, gx - cells), min(self.nx, gx + cells + 1)):
                wx, wy = self.grid_to_world(xx, yy)
                if (wx - x) ** 2 + (wy - y) ** 2 <= radius ** 2 and self.data[yy, xx] == UNKNOWN:
                    self.data[yy, xx] = FREE


def apply_scan(grid: OccupancyGrid, origin_xy: Tuple[float, float], rays: Sequence[Tuple[float, float, bool]], step: float) -> None:
    ox, oy = origin_xy
    for hx, hy, hit in rays:
        dx = hx - ox
        dy = hy - oy
        dist = float(np.hypot(dx, dy))
        if dist < 1e-9:
            continue
        n = max(1, int(np.ceil(dist / step)))
        for i in range(n):
            t = i / n
            x = ox + t * dx
            y = oy + t * dy
            grid.mark_free(x, y)
        if hit:
            grid.mark_occupied(hx, hy)
        else:
            grid.mark_free(hx, hy)


def frontier_mask(data: np.ndarray) -> np.ndarray:
    free = data == FREE
    unknown = data == UNKNOWN
    out = np.zeros_like(free, dtype=bool)
    out[1:, :] |= free[1:, :] & unknown[:-1, :]
    out[:-1, :] |= free[:-1, :] & unknown[1:, :]
    out[:, 1:] |= free[:, 1:] & unknown[:, :-1]
    out[:, :-1] |= free[:, :-1] & unknown[:, 1:]
    return out


def connected_components(mask: np.ndarray, min_size: int = 4) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(x, y)]
            seen[y, x] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx, dy in nbrs:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            if len(comp) >= min_size:
                comps.append(comp)
    return comps
