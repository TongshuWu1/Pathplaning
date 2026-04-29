"""Old-style Matplotlib dashboard for the clean Search-CAGE baseline.

The layout mirrors the previous simulator UI style while keeping the new clean
backend:
  * toolbar across the top
  * Global Truth map, simulation/debug only
  * Team Fused/Reported Belief map
  * Mission status panel
  * compact local-belief cards for all robots

Map panels stay mostly free of legends/text.  Route/status information is kept
in the dedicated status and card-title areas to avoid overlap.
"""
from __future__ import annotations

import math
from dataclasses import replace
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
import numpy as np

from ..cage_graph import RouteGraph, RouteCandidate
from ..config import AppConfig
from ..geometry import covariance_ellipse
from ..mapping import OccupancyGrid
from ..simulator import Simulator


class MatplotlibDashboard:
    def __init__(self, sim: Simulator):
        self.sim = sim
        self.selected_robot = min(sim.cfg.ui.selected_robot, len(sim.robots) - 1)
        self.show_rays = bool(sim.cfg.ui.show_lidar_rays)
        self.show_fused_quality = False
        self.show_route_graph = bool(sim.cfg.ui.show_route_graph)
        self.controls: dict[str, object] = {}
        self.local_axes = []
        self._render_frame = 0
        self._belief_cache: dict[int, tuple[int, np.ndarray]] = {}
        self._frontier_cache: dict[int, tuple[int, int, list[tuple[float, float]]]] = {}
        self._local_drawn: set[int] = set()
        self.fig = plt.figure(figsize=(sim.cfg.ui.figure_width, sim.cfg.ui.figure_height), constrained_layout=False)
        self.fig.set_facecolor("#f4f6f8")
        self.anim: FuncAnimation | None = None
        self._build_layout()

    def run(self) -> None:
        self.anim = FuncAnimation(
            self.fig,
            self._tick,
            interval=self.sim.cfg.ui.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Layout and controls
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.fig.clf()
        self.fig.set_facecolor("#f4f6f8")
        self.controls = {}
        self.local_axes = []
        self._belief_cache = {}
        self._frontier_cache = {}
        self._local_drawn = set()

        self._build_toolbar()
        self.ax_truth = self.fig.add_axes([0.045, 0.49, 0.44, 0.36])
        self.ax_team = self.fig.add_axes([0.045, 0.06, 0.44, 0.36])
        self.ax_status = self.fig.add_axes([0.51, 0.67, 0.445, 0.18])
        self._build_local_card_axes()
        self.fig.suptitle(
            "Search-CAGE: LiDAR Route Discovery",
            fontsize=14,
            y=0.982,
            fontweight="bold",
        )
        self._redraw_all(force=True)

    def _build_toolbar(self) -> None:
        y = 0.895
        h = 0.038
        x = 0.045
        gap = 0.008

        def style_ax(ax):
            ax.set_facecolor("#ffffff")
            for sp in ax.spines.values():
                sp.set_edgecolor("#d6dde6")
                sp.set_linewidth(1.0)

        def add_button(width: float, label: str, cb):
            nonlocal x
            ax = self.fig.add_axes([x, y, width, h])
            style_ax(ax)
            btn = Button(ax, label, color="#ffffff", hovercolor="#e8eef7")
            btn.label.set_fontsize(9)
            btn.on_clicked(cb)
            x += width + gap
            return btn

        def add_labeled_box(width: float, label: str, initial: str):
            nonlocal x
            label_ax = self.fig.add_axes([x, y + h + 0.003, width, 0.014])
            label_ax.axis("off")
            label_ax.text(0.02, 0.5, label, ha="left", va="center", fontsize=8.2, color="#475569")
            ax = self.fig.add_axes([x, y, width, h])
            style_ax(ax)
            box = TextBox(ax, "", initial=initial, color="white", hovercolor="#f7f7f7")
            box.text_disp.set_fontsize(9)
            x += width + gap
            return box

        self.controls["start"] = add_button(0.058, "Start", self._on_start)
        self.controls["pause"] = add_button(0.058, "Pause", self._on_pause)
        self.controls["rays"] = add_button(0.058, "Rays", self._on_toggle_rays)
        self.controls["quality"] = add_button(0.070, "Quality", self._on_toggle_fused_quality)
        self.controls["graph"] = add_button(0.060, "Graph", self._on_toggle_route_graph)
        self.controls["reset"] = add_button(0.060, "Reset", self._on_reset)
        self.controls["seed"] = add_labeled_box(0.07, "Seed", str(self.sim.cfg.world.seed))
        self.controls["robots"] = add_labeled_box(0.06, "Robots", str(self.sim.cfg.robot.count))
        self.controls["obstacles"] = add_labeled_box(0.06, "Obst", str(self.sim.cfg.world.obstacle_count))
        self.controls["landmarks"] = add_labeled_box(0.06, "Land", str(self.sim.cfg.world.landmark_count))

    def _build_local_card_axes(self) -> None:
        n = max(1, len(self.sim.robots))
        cols = 2 if n > 1 else 1
        rows = ceil(n / cols)
        x0, y0, w, h = 0.51, 0.06, 0.445, 0.55
        gap_x, gap_y = 0.014, 0.020
        card_w = (w - gap_x * (cols - 1)) / cols
        card_h = (h - gap_y * (rows - 1)) / rows
        for idx in range(n):
            c = idx % cols
            r = idx // cols
            left = x0 + c * (card_w + gap_x)
            bottom = y0 + (rows - 1 - r) * (card_h + gap_y)
            ax = self.fig.add_axes([left, bottom, card_w, card_h])
            self.local_axes.append(ax)

    def _tick(self, _frame: int):
        if self.sim.running and not self.sim.mission.success:
            for _ in range(max(1, int(self.sim.cfg.ui.sim_steps_per_render))):
                if not self.sim.mission.success:
                    self.sim.step()
        self._redraw_all()
        return []

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def _on_start(self, _event) -> None:
        self.sim.running = True

    def _on_pause(self, _event) -> None:
        self.sim.running = False

    def _on_toggle_rays(self, _event) -> None:
        self.show_rays = not self.show_rays
        self._redraw_all(force=True)

    def _on_toggle_fused_quality(self, _event) -> None:
        self.show_fused_quality = not self.show_fused_quality
        self._redraw_all(force=True)

    def _on_toggle_route_graph(self, _event) -> None:
        self.show_route_graph = not self.show_route_graph
        self._redraw_all(force=True)

    def _textbox_value(self, key: str, default: int) -> int:
        obj = self.controls.get(key)
        raw = getattr(obj, "text", str(default))
        try:
            return int(raw)
        except Exception:
            return default

    def _on_reset(self, _event) -> None:
        old = self.sim.cfg
        seed = self._textbox_value("seed", old.world.seed)
        robots = max(1, min(8, self._textbox_value("robots", old.robot.count)))
        obstacles = max(0, min(40, self._textbox_value("obstacles", old.world.obstacle_count)))
        landmarks = max(0, min(40, self._textbox_value("landmarks", old.world.landmark_count)))
        cfg = replace(
            old,
            world=replace(old.world, seed=seed, obstacle_count=obstacles, landmark_count=landmarks),
            robot=replace(old.robot, count=robots),
        )
        self.sim.reset(cfg)
        self.selected_robot = min(self.selected_robot, len(self.sim.robots) - 1)
        self._build_layout()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _redraw_all(self, force: bool = False) -> None:
        truth_interval = max(1, int(self.sim.cfg.ui.render_truth_every))
        team_interval = max(1, int(self.sim.cfg.ui.render_team_every))
        if force or not self.sim.running or self._render_frame % truth_interval == 0:
            self._draw_truth()
        if force or not self.sim.running or self._render_frame % team_interval == 0:
            self._draw_team_belief()
        self._draw_status()
        self._draw_local_cards(force=force)
        self.fig.canvas.draw_idle()
        self._render_frame += 1

    def _setup_map_axis(self, ax, title: str, ticks: bool = True) -> None:
        ax.clear()
        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d6dde6")
            sp.set_linewidth(1.0)
        ax.set_title(title, fontsize=10.5, pad=7, color="#111827")
        ax.set_xlim(0, self.sim.world.width)
        ax.set_ylim(0, self.sim.world.height)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.35, alpha=0.08)
        ax.tick_params(labelsize=8, colors="#64748b", width=0.6)
        if ticks:
            ax.set_xlabel("x [m]", fontsize=8, color="#64748b")
            ax.set_ylabel("y [m]", fontsize=8, color="#64748b")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    def _robot_color(self, robot_id: int) -> str:
        palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2", "#be123c", "#65a30d"]
        return palette[robot_id % len(palette)]

    def _short_task(self, task: str) -> str:
        names = {
            "SEARCH_FRONTIER": "FRONTIER",
            "SEARCH_OPEN_SECTOR": "OPEN",
            "ADVANCE_TO_TARGET": "TARGET",
            "CERTIFY_TARGET_EDGE": "CERTIFY",
            "REPORT_TARGET_HOME": "REPORT",
            "RETURN_HOME_CERT_ROUTE": "RETURN",
            "REANCHOR": "ANCHOR",
        }
        return names.get(task, task.replace("_", " "))

    def _draw_home_base(self, ax) -> None:
        hb = self.sim.world.home_base
        ax.add_patch(
            Rectangle(
                (hb.x0, hb.y0),
                hb.x1 - hb.x0,
                hb.y1 - hb.y0,
                facecolor="#dcfce7",
                edgecolor="#16a34a",
                linewidth=1.0,
                alpha=0.55,
                zorder=1,
            )
        )

    def _draw_obstacles_and_landmarks(self, ax) -> None:
        for obs in self.sim.world.obstacles:
            ax.add_patch(
                Rectangle(
                    (obs.x0, obs.y0),
                    obs.x1 - obs.x0,
                    obs.y1 - obs.y0,
                    facecolor="#64748b",
                    edgecolor="#334155",
                    linewidth=0.8,
                    alpha=0.82,
                    zorder=2,
                )
            )
        if self.sim.world.landmarks:
            xs = [lm.xy[0] for lm in self.sim.world.landmarks]
            ys = [lm.xy[1] for lm in self.sim.world.landmarks]
            ax.scatter(xs, ys, marker="*", s=58, c="#facc15", edgecolors="#334155", linewidths=0.45, zorder=4)

    def _draw_truth(self) -> None:
        ax = self.ax_truth
        self._setup_map_axis(ax, "Truth")
        self._draw_home_base(ax)
        self._draw_obstacles_and_landmarks(ax)
        hx, hy = self.sim.world.home
        ax.scatter([hx], [hy], marker="P", s=74, c="#22c55e", edgecolors="#111827", linewidths=0.7, zorder=5)
        if self.sim.cfg.ui.show_truth_target:
            tx, ty = self.sim.world.target
            ax.scatter([tx], [ty], marker="X", s=118, c="#ef4444", edgecolors="#111827", linewidths=1.0, zorder=6)
        for r in self.sim.robots:
            color = self._robot_color(r.id)
            if len(r.true_path) > 1:
                xs, ys = zip(*r.true_path[-self.sim.cfg.ui.max_draw_path_points:])
                ax.plot(xs, ys, color=color, linewidth=1.15, alpha=0.72, zorder=5)
            if r.path and r.path_index < len(r.path):
                px = [r.est_xy[0]] + [p[0] for p in r.path[r.path_index:]]
                py = [r.est_xy[1]] + [p[1] for p in r.path[r.path_index:]]
                ax.plot(px, py, color=color, linewidth=0.9, linestyle="--", alpha=0.42, zorder=5)
            x, y, _ = r.true_pose
            ex, ey, _ = r.est_pose
            ax.scatter([x], [y], s=54, c=[color], edgecolors="#111827", linewidths=0.55, zorder=7)
            ax.scatter([ex], [ey], s=48, marker="o", facecolors="none", edgecolors=[color], linewidths=1.25, zorder=8)
            ax.plot([x, ex], [y, ey], color=color, linewidth=0.75, linestyle=":", alpha=0.45, zorder=6)
            if self.show_rays and r.scan is not None:
                stride = max(self.sim.cfg.ui.draw_lidar_stride, len(r.scan.angles) // 24)
                th = float(r.true_pose[2])
                for a, rng in zip(r.scan.angles[::stride], r.scan.ranges[::stride]):
                    x2 = x + math.cos(th + float(a)) * float(rng)
                    y2 = y + math.sin(th + float(a)) * float(rng)
                    ax.plot([x, x2], [y, y2], color=color, alpha=0.08, linewidth=0.45, zorder=3)
        self._draw_comm_links(ax)

    def _draw_comm_links(self, ax) -> None:
        for a, b in self.sim.comm_state.robot_segments:
            ax.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", color="#0284c7", linewidth=1.35, alpha=0.70, zorder=9)
        for a, b in self.sim.comm_state.home_segments:
            ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=":", color="#16a34a", linewidth=1.55, alpha=0.75, zorder=9)

    def _draw_grid(self, ax, grid: OccupancyGrid, title: str, ticks: bool = True) -> None:
        self._setup_map_axis(ax, title, ticks=ticks)
        ax.imshow(
            self._belief_image(grid),
            origin="lower",
            extent=(0, grid.width_m, 0, grid.height_m),
            interpolation="nearest",
            zorder=0,
        )
        if self.show_fused_quality:
            q = np.clip(grid.quality, 0.0, 1.0)
            overlay = np.zeros((grid.ny, grid.nx, 4), dtype=float)
            overlay[..., 1] = q
            overlay[..., 0] = 1.0 - q
            overlay[..., 3] = 0.25 * (q > 0.05)
            ax.imshow(
                overlay,
                origin="lower",
                extent=(0, grid.width_m, 0, grid.height_m),
                interpolation="nearest",
                zorder=1,
            )

    def _belief_image(self, grid: OccupancyGrid) -> np.ndarray:
        version = int(getattr(grid, "_version", 0))
        cache_key = id(grid)
        cached = self._belief_cache.get(cache_key)
        if cached is not None and cached[0] == version:
            return cached[1]
        prob = grid.probability()
        quality = np.clip(grid.quality, 0.0, 1.0)
        free = prob < grid.cfg.prob_free_threshold
        occ = prob > grid.cfg.prob_occ_threshold
        observed = quality > 0.05
        img = np.zeros((grid.ny, grid.nx, 3), dtype=float)
        img[:, :, :] = np.array([0.88, 0.91, 0.95])
        img[observed & ~free & ~occ] = np.array([0.78, 0.83, 0.89])
        free_color = np.array([0.98, 1.00, 0.99])
        occ_color = np.array([0.08, 0.10, 0.12])
        img[free] = free_color
        img[occ] = occ_color
        low_q = observed & (quality < 0.45)
        img[low_q] = 0.65 * img[low_q] + 0.35 * np.array([0.74, 0.79, 0.86])
        self._belief_cache[cache_key] = (version, img)
        return img

    def _draw_frontiers(self, ax, grid: OccupancyGrid, size: float = 12.0, alpha: float = 0.72) -> None:
        pts = self._frontier_points(grid)
        if not pts:
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, marker=".", s=size, c="#06b6d4", alpha=alpha, linewidths=0.0, zorder=5)

    def _frontier_points(self, grid: OccupancyGrid) -> list[tuple[float, float]]:
        cache_key = id(grid)
        version = int(getattr(grid, "_version", 0))
        cached = self._frontier_cache.get(cache_key)
        interval = max(1, int(self.sim.cfg.ui.render_frontier_every))
        if cached is not None:
            cached_version, frame, pts = cached
            if cached_version == version or self._render_frame - frame < interval:
                return pts
        frontiers = grid.find_frontiers(self.sim.cfg.planning.frontier_min_cluster_size, self.sim.cfg.planning.frontier_info_radius_m)
        pts = [(float(fr.centroid_world[0]), float(fr.centroid_world[1])) for fr in frontiers[: self.sim.cfg.ui.max_draw_frontiers]]
        self._frontier_cache[cache_key] = (version, self._render_frame, pts)
        return pts

    def _draw_team_belief(self) -> None:
        title = "HOME Fused Belief"
        if self.show_fused_quality:
            title += " + Quality"
        self._draw_grid(self.ax_team, self.sim.home_memory.map, title)
        self._draw_home_base(self.ax_team)
        self._draw_frontiers(self.ax_team, self.sim.home_memory.map, size=14, alpha=0.58)
        hx, hy = self.sim.world.home
        self.ax_team.scatter([hx], [hy], marker="P", s=60, c="#22c55e", edgecolors="#111827", linewidths=0.6, zorder=7)
        if self.sim.home_memory.target.detected and self.sim.home_memory.target.xy:
            tx, ty = self.sim.home_memory.target.xy
            self.ax_team.scatter([tx], [ty], marker="X", s=78, c="#ef4444", edgecolors="#111827", linewidths=0.9, zorder=8)
        self._draw_home_robot_reports(self.ax_team)
        if self.show_route_graph:
            self._draw_graph(self.ax_team, self.sim.home_memory.graph, self.sim.home_memory.best_routes)
        self._draw_comm_links(self.ax_team)

    def _draw_home_robot_reports(self, ax) -> None:
        memory = self.sim.home_memory
        for rid, pose in memory.known_robot_pose.items():
            stamp = memory.known_robot_time.get(rid, -math.inf)
            age = max(0.0, self.sim.time_s - stamp)
            alpha = max(0.25, min(0.95, 1.0 - age / max(1.0, self.sim.cfg.communication.teammate_intent_timeout_s * 2.0)))
            color = self._robot_color(rid)
            x, y, th = pose
            visits = memory.known_robot_visits.get(rid, [])[-self.sim.cfg.ui.max_draw_teammate_visit_points:]
            if visits:
                ax.scatter([p[0] for p in visits], [p[1] for p in visits], marker=".", s=10, c=[color], alpha=0.18, linewidths=0, zorder=4)
            path = memory.known_robot_paths.get(rid, [])
            if len(path) >= 2:
                ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=0.9, linestyle="--", alpha=0.42, zorder=6)
            goal = memory.known_robot_goal.get(rid)
            if goal is not None:
                ax.scatter([goal[0]], [goal[1]], marker="x", s=38, c=[color], linewidths=1.25, alpha=alpha, zorder=7)
            ax.scatter([x], [y], s=44, c=[color], edgecolors="#111827", linewidths=0.55, alpha=alpha, zorder=8)
            ax.arrow(x, y, math.cos(th) * 0.42, math.sin(th) * 0.42, head_width=0.12, color=color, alpha=alpha, zorder=9)

    def _draw_local_cards(self, force: bool = False) -> None:
        local_interval = max(1, int(self.sim.cfg.ui.render_local_every))
        for ax, robot in zip(self.local_axes, self.sim.robots):
            if (
                not force
                and self.sim.running
                and robot.id != self.selected_robot
                and robot.id in self._local_drawn
                and self._render_frame % local_interval != 0
            ):
                continue
            color = self._robot_color(robot.id)
            known_intents = len(robot.known_teammate_goals)
            title = f"R{robot.id} Local  {self._short_task(robot.current_task)}  S {robot.assessment.consistency:.2f}  F {robot.assessment.front_clearance:.1f}m  I {known_intents}"
            self._draw_grid(ax, robot.map, title, ticks=False)
            self._draw_home_base(ax)
            self._draw_frontiers(ax, robot.map, size=8, alpha=0.48)
            self._draw_teammate_context(ax, robot)
            hx, hy = self.sim.world.home
            ax.scatter([hx], [hy], marker="P", s=28, c="#22c55e", edgecolors="#111827", linewidths=0.4, zorder=3)
            if self.show_route_graph and robot.id == self.selected_robot:
                self._draw_graph(ax, robot.graph, robot.best_routes[:2], node_size_scale=0.55, line_scale=0.65)
            ex, ey, eth = robot.est_pose
            ax.scatter([ex], [ey], s=36, c=[color], edgecolors="#111827", linewidths=0.5, zorder=8)
            ax.arrow(ex, ey, math.cos(eth) * 0.42, math.sin(eth) * 0.42, head_width=0.13, color="#111827", zorder=9)
            ell_x, ell_y = covariance_ellipse(robot.estimator.belief.covariance[:2, :2], scale=2.0)
            if len(ell_x):
                ax.plot(ex + ell_x, ey + ell_y, color=color, linewidth=0.9, linestyle="--", alpha=0.62, zorder=7)
            if robot.path and robot.path_index < len(robot.path):
                xs = [ex] + [p[0] for p in robot.path[robot.path_index:]]
                ys = [ey] + [p[1] for p in robot.path[robot.path_index:]]
                ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.82, zorder=7)
            if robot.current_goal is not None:
                gx, gy = robot.current_goal
                ax.scatter([gx], [gy], marker="x", s=34, c=[color], linewidths=1.2, zorder=8)
            if robot.target.detected and robot.target.xy:
                tx, ty = robot.target.xy
                ax.scatter([tx], [ty], marker="X", s=40, c="#ef4444", edgecolors="#111827", linewidths=0.6, zorder=8)
            if self.show_rays and robot.scan is not None:
                stride = max(self.sim.cfg.ui.draw_lidar_stride, len(robot.scan.angles) // 20)
                for a, rng in zip(robot.scan.angles[::stride], robot.scan.ranges[::stride]):
                    x2 = ex + math.cos(eth + float(a)) * float(rng)
                    y2 = ey + math.sin(eth + float(a)) * float(rng)
                    ax.plot([ex, x2], [ey, y2], color=color, alpha=0.08, linewidth=0.38, zorder=5)
            self._local_drawn.add(robot.id)

    def _draw_teammate_context(self, ax, robot) -> None:
        for rid, pts in robot.known_teammate_visits.items():
            if rid == robot.id or not pts:
                continue
            color = self._robot_color(rid)
            pts = pts[-self.sim.cfg.ui.max_draw_teammate_visit_points:]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=".", s=7, c=[color], alpha=0.18, linewidths=0, zorder=4)
        for rid, path in robot.known_teammate_paths.items():
            if rid == robot.id or len(path) < 2:
                continue
            color = self._robot_color(rid)
            ax.plot([p[0] for p in path], [p[1] for p in path], color=color, linewidth=0.7, linestyle="--", alpha=0.24, zorder=5)
        for rid, goal in robot.known_teammate_goals.items():
            if rid == robot.id or goal is None:
                continue
            color = self._robot_color(rid)
            ax.scatter([goal[0]], [goal[1]], marker="x", s=22, c=[color], linewidths=0.9, alpha=0.45, zorder=6)

    def _draw_graph(
        self,
        ax,
        graph: RouteGraph,
        routes: list[RouteCandidate] | tuple[RouteCandidate, ...],
        node_size_scale: float = 1.0,
        line_scale: float = 1.0,
    ) -> None:
        highlighted = set()
        route_node_ids = set()
        for idx, route in enumerate(routes):
            pts = graph.route_points(route)
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    color="#22c55e" if idx == 0 else "#84cc16",
                    linewidth=(3.0 if idx == 0 else 2.0) * line_scale,
                    alpha=0.85 if idx == 0 else 0.55,
                    zorder=6,
                )
                highlighted.update(route.edge_ids)
                route_node_ids.update(route.node_ids)
        ranked_edges = sorted(graph.edges.values(), key=lambda e: e.cert.confidence, reverse=True)[: self.sim.cfg.ui.max_draw_graph_edges]
        node_order: list[int] = []
        seen_nodes: set[int] = set()

        def keep_node(nid: int) -> None:
            if nid in graph.nodes and nid not in seen_nodes and len(node_order) < self.sim.cfg.ui.max_draw_graph_nodes:
                seen_nodes.add(nid)
                node_order.append(nid)

        for nid in route_node_ids:
            keep_node(nid)
        for edge in ranked_edges:
            keep_node(edge.a)
            keep_node(edge.b)
        for nid, node in graph.nodes.items():
            if node.kind in {"home", "target", "anchor"}:
                keep_node(nid)

        for edge in ranked_edges:
            a = graph.nodes.get(edge.a)
            b = graph.nodes.get(edge.b)
            if a is None or b is None:
                continue
            c = edge.cert.confidence
            col = "#22c55e" if c >= 0.7 else "#eab308" if c >= 0.45 else "#ef4444"
            ax.plot(
                [a.xy[0], b.xy[0]],
                [a.xy[1], b.xy[1]],
                color=col,
                linewidth=(2.0 if edge.id in highlighted else 1.0) * line_scale,
                alpha=0.75 if edge.id in highlighted else 0.38,
                zorder=4,
            )
        for nid in node_order:
            node = graph.nodes[nid]
            if node.kind == "home":
                color = "#22c55e"
                marker = "P"
                size = 42
            elif node.kind == "target":
                color = "#ef4444"
                marker = "X"
                size = 46
            elif node.kind == "anchor":
                color = "#f97316"
                marker = "o"
                size = 24
            else:
                color = "white"
                marker = "o"
                size = 16
            ax.scatter(
                [node.xy[0]],
                [node.xy[1]],
                marker=marker,
                s=size * node_size_scale,
                c=[color],
                edgecolors="black",
                linewidths=0.45,
                zorder=7,
            )

    def _draw_status(self) -> None:
        ax = self.ax_status
        ax.clear()
        ax.set_facecolor("#ffffff")
        for sp in ax.spines.values():
            sp.set_edgecolor("#d6dde6")
            sp.set_linewidth(1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Mission", fontsize=10.5, pad=5, color="#111827")
        status = "RUNNING" if self.sim.running else "PAUSED"
        local = [r.id for r in self.sim.robots if r.target.detected]
        home_target = self.sim.home_memory.target
        returned = [
            r.id
            for r in self.sim.robots
            if self.sim.world.home_base.contains((float(r.true_pose[0]), float(r.true_pose[1])))
        ]
        sel = self.sim.robots[min(self.selected_robot, len(self.sim.robots) - 1)]
        rb = sel.status.reward_breakdown
        home_connected = [rid for rid, ok in sorted(self.sim.comm_state.home_connected.items()) if ok]
        intent_counts = "  ".join(f"R{r.id}:{len(r.known_teammate_goals)}" for r in self.sim.robots)

        left_lines = [
            f"{status}  t={self.sim.time_s:5.1f}s  step={self.sim.step_count}",
            f"Phase   {self.sim.mission.phase}",
            f"Target  HOME {'yes' if home_target.detected else 'no'}   local {local or '-'}",
            f"Return  {len(returned)}/{len(self.sim.robots)} at HOME   ids {returned or '-'}",
            f"LOS     robot {len(self.sim.comm_state.direct_robot_edges)}   home {home_connected or '-'}",
            f"View    rays {'on' if self.show_rays else 'off'}   quality {'on' if self.show_fused_quality else 'off'}   graph {'on' if self.show_route_graph else 'off'}",
        ]
        right_lines = [
            f"Selected R{sel.id}  {self._short_task(sel.current_task)}",
            f"Plan     {'ok' if sel.status.last_plan_success else sel.status.last_plan_reason}",
            f"Clear    {sel.status.last_path_min_clearance:.2f} m",
            f"Intent   {intent_counts or '-'}",
        ]
        if rb:
            right_lines.append(
                f"Score    {rb.get('score', 0.0):.2f}   info {rb.get('info', 0.0):.2f}   clear {rb.get('raw_clearance_m', 0.0):.2f} m"
            )

        ax.text(0.025, 0.90, "\n".join(left_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.4, family="monospace", color="#111827")
        ax.text(0.53, 0.90, "\n".join(right_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.4, family="monospace", color="#111827")
        ax.plot([0.025, 0.975], [0.37, 0.37], transform=ax.transAxes, color="#e2e8f0", linewidth=0.8)

        routes = self.sim.home_memory.best_routes[: self.sim.cfg.ui.max_status_routes]
        route_lines = ["Routes"]
        if routes:
            for i, route in enumerate(routes):
                route_lines.append(
                    f"#{i}  len {route.length:5.1f} m   clear {route.min_clearance:.2f}   cert {route.certificate:.2f}   {route.status}"
                )
        else:
            route_lines.append("none yet")
        ax.text(0.025, 0.30, "\n".join(route_lines), transform=ax.transAxes, va="top", ha="left", fontsize=8.2, family="monospace", color="#111827")
