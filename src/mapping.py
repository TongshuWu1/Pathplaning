"""LiDAR occupancy map with per-cell quality.

Cells are updated from LiDAR scans transformed by the estimated pose.  Updates
made with poor pose confidence are weaker and cannot blindly overwrite older
higher-quality evidence.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import MappingConfig
from .geometry import Point, Pose, clamp
from .sensors import LidarScan


@dataclass
class FrontierCluster:
    cells: list[tuple[int, int]]
    centroid_world: Point
    information_gain: float


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

    def copy(self) -> "OccupancyGrid":
        other = OccupancyGrid(self.width_m, self.height_m, self.cfg)
        other.logodds = self.logodds.copy()
        other.quality = self.quality.copy()
        other.last_seen = self.last_seen.copy()
        other.source = self.source.copy()
        return other

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

    def _write_cell(self, cell: tuple[int, int], delta: float, obs_quality: float, robot_id: int, time_s: float) -> None:
        i, j = cell
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return
        current_q = self.quality[j, i]
        # Higher-quality observations dominate. Lower-quality observations are
        # allowed to nudge, but not overwrite aggressively.
        if obs_quality + self.cfg.quality_overwrite_margin >= current_q:
            scale = 1.0
        else:
            scale = self.cfg.low_quality_update_scale * max(0.05, obs_quality / max(current_q, 1e-6))
        self.logodds[j, i] = clamp(
            float(self.logodds[j, i] + delta * obs_quality * scale),
            self.cfg.logodds_min,
            self.cfg.logodds_max,
        )
        self.quality[j, i] = max(current_q * 0.995, obs_quality)
        self.last_seen[j, i] = time_s
        self.source[j, i] = robot_id

    def update_from_lidar(self, est_pose: Pose, scan: LidarScan, pose_quality: float, robot_id: int, time_s: float) -> None:
        start = self.world_to_cell((est_pose[0], est_pose[1]))
        if start is None:
            return
        th = est_pose[2]
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
                self._write_cell(c, self.cfg.logodds_free, pose_quality, robot_id, time_s)
            if hit:
                self._write_cell(ray_cells[-1], self.cfg.logodds_occ, pose_quality, robot_id, time_s)

    def predict_scan_ranges(self, est_pose: Pose, angles: np.ndarray, max_range: float) -> np.ndarray:
        out = np.full(len(angles), max_range, dtype=float)
        occ = self.occupied_mask()
        step = max(0.5 * self.res, 0.05)
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

    def clearance_at(self, p: Point, max_radius_m: float = 2.0) -> float:
        cell = self.world_to_cell(p)
        if cell is None:
            return 0.0
        occ = self.occupied_mask()
        ci, cj = cell
        max_r = max(1, int(max_radius_m / self.res))
        best = max_radius_m
        y0 = max(0, cj - max_r)
        y1 = min(self.ny, cj + max_r + 1)
        x0 = max(0, ci - max_r)
        x1 = min(self.nx, ci + max_r + 1)
        ys, xs = np.nonzero(occ[y0:y1, x0:x1])
        if len(xs) == 0:
            return best
        for yy, xx in zip(ys, xs):
            wc = self.cell_to_world((x0 + int(xx), y0 + int(yy)))
            d = math.hypot(wc[0] - p[0], wc[1] - p[1])
            best = min(best, d)
        return float(best)

    def merge_from_digest(self, digest: dict) -> None:
        # Lightweight packet fusion: accept higher-quality cells from teammates.
        idx = digest.get("cells", [])
        vals = digest.get("logodds", [])
        quals = digest.get("quality", [])
        src = int(digest.get("source_robot", -1))
        t = float(digest.get("time_s", 0.0))
        for (i, j), lo, q in zip(idx, vals, quals):
            if not (0 <= i < self.nx and 0 <= j < self.ny):
                continue
            if float(q) > self.quality[j, i] + 0.02:
                self.logodds[j, i] = float(lo)
                self.quality[j, i] = float(q)
                self.source[j, i] = src
                self.last_seen[j, i] = t

    def make_digest(self, robot_id: int, time_s: float, max_cells: int = 650) -> dict:
        known = self.known_mask() & (self.quality > 0.05)
        ys, xs = np.nonzero(known)
        if len(xs) > max_cells:
            # Prefer recent/high-quality cells.
            score = self.quality[ys, xs] + 0.001 * np.maximum(0.0, self.last_seen[ys, xs])
            keep = np.argsort(score)[-max_cells:]
            xs = xs[keep]
            ys = ys[keep]
        cells = [(int(i), int(j)) for i, j in zip(xs, ys)]
        return {
            "source_robot": int(robot_id),
            "time_s": float(time_s),
            "cells": cells,
            "logodds": [float(self.logodds[j, i]) for i, j in cells],
            "quality": [float(self.quality[j, i]) for i, j in cells],
        }
