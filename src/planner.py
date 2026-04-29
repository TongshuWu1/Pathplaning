"""Clearance-aware grid A* planner over the robot's local LiDAR map."""
from __future__ import annotations
import heapq, math
from dataclasses import dataclass
import numpy as np
from .config import PlanningConfig
from .geometry import Point, distance
from .mapping import OccupancyGrid

@dataclass
class PlanResult:
    path: list[Point]
    success: bool
    cost: float
    reason: str = ""
    min_clearance: float = 0.0

class GridPlanner:
    def __init__(self, cfg: PlanningConfig): self.cfg = cfg

    def plan(self, grid: OccupancyGrid, start: Point, goal: Point) -> PlanResult:
        start_cell, goal_cell = grid.world_to_cell(start), grid.world_to_cell(goal)
        if start_cell is None or goal_cell is None:
            return PlanResult([], False, math.inf, "start_or_goal_outside_map")
        traversable = grid.traversable_mask(self.cfg.inflation_radius_m)
        clearance = grid.clearance_map(max_radius_m=max(3.0, self.cfg.desired_clearance_m * 3.0))
        goal_cell = self._nearest_good_cell(traversable, clearance, goal_cell) or goal_cell
        if not traversable[start_cell[1], start_cell[0]]:
            start_cell = self._nearest_good_cell(traversable, clearance, start_cell) or start_cell
        if not traversable[start_cell[1], start_cell[0]] or not traversable[goal_cell[1], goal_cell[0]]:
            return PlanResult([], False, math.inf, "no_traversable_start_or_goal")
        prob, known = grid.probability(), grid.known_mask()
        nbrs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),(1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(-1,-1,math.sqrt(2))]
        heap, counter = [(0.0, 0, start_cell)], 0
        came: dict[tuple[int,int], tuple[int,int]] = {}
        g = {start_cell: 0.0}
        expanded = 0
        while heap and expanded < self.cfg.max_a_star_expansions:
            _, _, cur = heapq.heappop(heap); expanded += 1
            if cur == goal_cell:
                cells=[cur]
                while cur in came:
                    cur=came[cur]; cells.append(cur)
                cells.reverse(); path=[grid.cell_to_world(c) for c in cells]
                return PlanResult(path, True, g[goal_cell], "ok", min(float(clearance[j,i]) for i,j in cells))
            ci,cj=cur
            for di,dj,step in nbrs:
                ni,nj=ci+di,cj+dj
                if not (0 <= ni < grid.nx and 0 <= nj < grid.ny) or not traversable[nj,ni]: continue
                if di and dj and (not traversable[cj,ni] or not traversable[nj,ci]): continue
                unknown_cost = self.cfg.unknown_penalty if not known[nj,ni] else 0.0
                occ_soft = max(0.0, float(prob[nj,ni] - 0.5)) * 2.0
                cl = float(clearance[nj,ni])
                deficit = max(0.0, self.cfg.desired_clearance_m - cl) / max(1e-6, self.cfg.desired_clearance_m)
                clearance_cost = self.cfg.clearance_cost_weight * deficit * deficit + (3.0 if cl < self.cfg.critical_clearance_m else 0.0)
                new_g = g[cur] + step * grid.res * (1.0 + unknown_cost + occ_soft + clearance_cost)
                nb=(ni,nj)
                if new_g < g.get(nb, math.inf):
                    came[nb]=cur; g[nb]=new_g; counter += 1
                    heapq.heappush(heap, (new_g + distance(grid.cell_to_world(nb), grid.cell_to_world(goal_cell)), counter, nb))
        return PlanResult([], False, math.inf, "a_star_failed")

    def _nearest_good_cell(self, traversable: np.ndarray, clearance: np.ndarray, cell: tuple[int,int], radius: int = 14) -> tuple[int,int] | None:
        ci,cj=cell; ny,nx=traversable.shape
        min_clearance = min(self.cfg.safe_approach_min_clearance_m, self.cfg.desired_clearance_m)
        if 0 <= cj < ny and 0 <= ci < nx and traversable[cj,ci] and float(clearance[cj,ci]) >= min_clearance:
            return cell
        best=None; best_score=-math.inf
        for r in range(1, radius+1):
            y0,y1=max(0,cj-r),min(ny,cj+r+1); x0,x1=max(0,ci-r),min(nx,ci+r+1)
            ys,xs=np.nonzero(traversable[y0:y1,x0:x1])
            for yy,xx in zip(ys,xs):
                c=(x0+int(xx), y0+int(yy)); d2=(c[0]-ci)**2+(c[1]-cj)**2
                if float(clearance[c[1],c[0]]) < min_clearance:
                    continue
                score=float(clearance[c[1],c[0]]) - 0.04*d2
                if score > best_score: best_score=score; best=c
            if best is not None: return best
        return None
