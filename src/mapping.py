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

from .config import MappingConfig, PassageQualityConfig
from .geometry import Point, Pose, clamp, distance
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
        self.source_mask = np.zeros((self.ny, self.nx), dtype=np.int64)
        self._version = 0
        self._clearance_cache = None
        self._clearance_cache_key = None

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
        self._clearance_cache = None
        self._clearance_cache_key = None

    def clearance_map(self, max_radius_m: float = 3.0) -> np.ndarray:
        """Approximate distance to nearest occupied cell in meters.

        Two-pass chamfer transform: fast, dependency-free, and good enough for
        centerline/clearance-aware planning on the coarser grid.
        """
        max_cells = int(math.ceil(max_radius_m / self.res))
        key = (self._version, max_cells)
        if self._clearance_cache is not None and self._clearance_cache_key == key:
            return self._clearance_cache
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
        self._clearance_cache = clearance
        self._clearance_cache_key = key
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
        radius_cells = max(0, int(math.ceil(radius / self.res)))
        sigma = max(self.res * 0.5, radius * 0.62)
        min_weight = min(1.0, max(0.0, float(self.cfg.lidar_kernel_min_weight)))
        for dj in range(-radius_cells, radius_cells + 1):
            for di in range(-radius_cells, radius_cells + 1):
                dist_m = math.hypot(di * self.res, dj * self.res)
                if dist_m > radius + 1e-9:
                    continue
                weight = math.exp(-0.5 * (dist_m / sigma) ** 2)
                if dist_m > 0.0:
                    weight = max(min_weight, weight)
                self._write_cell((ci + di, cj + dj), delta, obs_quality, robot_id, time_s, weight=weight)

    def update_from_lidar(self, est_pose: Pose, scan: LidarScan, pose_quality: float, robot_id: int, time_s: float) -> None:
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
        return min(self.segment_min_clearance(a, b, max_radius_m) for a, b in zip(path[:-1], path[1:]))

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
