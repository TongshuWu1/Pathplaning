from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import numpy as np

from .mapping import UNKNOWN, frontier_mask


@dataclass
class UIState:
    fig: plt.Figure
    anim: FuncAnimation


class SimulatorUI:
    """Single-window dashboard focused on one big global view and compact robot cards."""

    def __init__(self, sim):
        self.sim = sim
        self.fig: Optional[plt.Figure] = None
        self.anim: Optional[FuncAnimation] = None
        self.controls: Dict[str, object] = {}

        self.cmap = ListedColormap(['#d9dde3', '#f7fbff', '#55606d'])
        self.norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], self.cmap.N)

        self.global_ax = None
        self.global_robot_scatter = None
        self.global_est_scatter = None
        self.global_target_scatter = None
        self.global_ray_collection = None
        self.global_comm_collection = None
        self.global_home_comm_collection = None
        self.global_path_lines = []
        self.global_est_path_lines = []
        self.global_plan_lines = []
        self.global_drift_lines = []
        self.global_cov_ellipses = []

        self.status_ax = None
        self.status_text = None

        self.axes_local = []
        self.local_info_axes = []
        self.local_images = []
        self.local_frontier_scatter = []
        self.local_packet_scatter = []
        self.local_target_scatter = []
        self.local_shared_keypoint_scatter = []
        self.local_robot_scatter = []
        self.local_est_scatter = []
        self.local_robot_target_scatter = []
        self.local_robot_path_lines = []
        self.local_est_path_lines = []
        self.local_scan_collections = []
        self.local_path_lines = []
        self.local_drift_lines = []
        self.local_cov_ellipses = []
        self.local_texts = []

    def build(self) -> UIState:
        if self.fig is None:
            self.fig = plt.figure(figsize=(16.5, 10.2), constrained_layout=False)
        self._build_layout()
        if self.anim is None:
            self.anim = FuncAnimation(
                self.fig,
                self._tick,
                interval=self.sim.cfg.fps_ms,
                blit=False,
                cache_frame_data=False,
            )
        return UIState(fig=self.fig, anim=self.anim)

    def _build_layout(self) -> None:
        assert self.fig is not None
        self.fig.clf()
        self.fig.set_facecolor('#eef1f5')

        self._reset_handles()
        self._build_toolbar()
        self._build_global_panel()
        self._build_status_panel()
        self._build_local_panels()

        self.fig.suptitle('Multi-robot exploration with cooperative localization + LOS comm graph', fontsize=13, y=0.982)
        self._refresh()
        self.fig.canvas.draw_idle()

    def _reset_handles(self) -> None:
        self.controls = {}
        self.global_path_lines = []
        self.global_est_path_lines = []
        self.global_plan_lines = []
        self.global_drift_lines = []
        self.global_cov_ellipses = []
        self.axes_local = []
        self.local_info_axes = []
        self.local_images = []
        self.local_frontier_scatter = []
        self.local_packet_scatter = []
        self.local_target_scatter = []
        self.local_shared_keypoint_scatter = []
        self.local_robot_scatter = []
        self.local_est_scatter = []
        self.local_robot_target_scatter = []
        self.local_robot_path_lines = []
        self.local_est_path_lines = []
        self.local_scan_collections = []
        self.local_path_lines = []
        self.local_drift_lines = []
        self.local_cov_ellipses = []
        self.local_texts = []

    def _build_toolbar(self) -> None:
        assert self.fig is not None
        y = 0.92
        h = 0.036
        x = 0.045
        gap = 0.008

        def style_ax(ax):
            ax.set_facecolor('#fbfcfd')
            for sp in ax.spines.values():
                sp.set_edgecolor('#cfd8e3')
                sp.set_linewidth(1.0)

        def add_button(width: float, label: str, cb):
            nonlocal x
            ax = self.fig.add_axes([x, y, width, h])
            style_ax(ax)
            btn = Button(ax, label, color='#fbfcfd', hovercolor='#e9eef6')
            btn.label.set_fontsize(9)
            btn.on_clicked(cb)
            x += width + gap
            return btn

        def add_labeled_box(width: float, label: str, initial: str):
            nonlocal x
            label_ax = self.fig.add_axes([x, y + h + 0.003, width, 0.014])
            label_ax.axis('off')
            label_ax.text(0.02, 0.5, label, ha='left', va='center', fontsize=8.5, color='#334155')

            ax = self.fig.add_axes([x, y, width, h])
            style_ax(ax)
            box = TextBox(ax, '', initial=initial, color='white', hovercolor='#f7f7f7')
            box.text_disp.set_fontsize(9)
            x += width + gap
            return box

        self.controls['start_btn'] = add_button(0.06, 'Start', self._on_start)
        self.controls['stop_btn'] = add_button(0.06, 'Pause', self._on_stop)
        self.controls['rays_btn'] = add_button(0.085, 'Toggle rays', self._on_toggle_rays)
        self.controls['reset_btn'] = add_button(0.065, 'Reset', self._on_reset)
        self.controls['seed_box'] = add_labeled_box(0.07, 'Seed', str(self.sim.cfg.seed))
        self.controls['robot_box'] = add_labeled_box(0.06, 'Robots', str(self.sim.cfg.robot_count))
        self.controls['obstacle_box'] = add_labeled_box(0.06, 'Obst', str(self.sim.cfg.obstacle_count))
        self.controls['landmark_box'] = add_labeled_box(0.06, 'Land', str(self.sim.cfg.landmark_count))

    def _build_global_panel(self) -> None:
        assert self.fig is not None
        self.global_ax = self.fig.add_axes([0.045, 0.08, 0.58, 0.80])
        ax = self.global_ax
        ax.set_facecolor('#fbfcfd')
        ax.set_title('Global truth / belief view', fontsize=11, pad=8)
        ax.set_xlim(0, self.sim.cfg.world_w)
        ax.set_ylim(0, self.sim.cfg.world_h)
        ax.set_aspect('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.grid(True, alpha=0.16)

        self._draw_home_base(ax)
        for obs in self.sim.world.obstacles:
            ax.add_patch(
                patches.Rectangle(
                    (obs.xmin, obs.ymin), obs.w, obs.h,
                    facecolor='#606a78', edgecolor='#2f3640', lw=1.0, zorder=2
                )
            )
        ax.scatter(
            [lm.x for lm in self.sim.world.landmarks],
            [lm.y for lm in self.sim.world.landmarks],
            marker='*', s=75, c='#ffcc00', edgecolors='black', linewidths=0.55, zorder=4,
        )
        ax.scatter(
            [self.sim.world.home_marker.x], [self.sim.world.home_marker.y],
            marker='P', s=92, c='#22c55e', edgecolors='black', linewidths=0.8, zorder=5,
        )
        initial_robot_xy = np.array([[r.x, r.y] for r in self.sim.robots], dtype=float)
        initial_est_xy = np.array([[r.x_est, r.y_est] for r in self.sim.robots], dtype=float)
        initial_target_xy = np.array([
            r.current_target if r.current_target is not None else (np.nan, np.nan)
            for r in self.sim.robots
        ], dtype=float)
        self.global_robot_scatter = ax.scatter(
            initial_robot_xy[:, 0], initial_robot_xy[:, 1],
            s=68, c=[r.color for r in self.sim.robots], edgecolors='black', linewidths=0.65, zorder=7,
        )
        self.global_est_scatter = ax.scatter(
            initial_est_xy[:, 0], initial_est_xy[:, 1],
            s=56, marker='o', facecolors='none', edgecolors=[r.color for r in self.sim.robots], linewidths=1.3, zorder=8,
        )
        self.global_target_scatter = ax.scatter(
            initial_target_xy[:, 0], initial_target_xy[:, 1],
            s=38, marker='x', c=[r.color for r in self.sim.robots], linewidths=1.4, zorder=6,
        )
        self.global_ray_collection = LineCollection([], colors='tab:blue', linewidths=0.65, alpha=0.18, zorder=3)
        self.global_comm_collection = LineCollection([], colors='#0ea5e9', linewidths=1.8, alpha=0.8, zorder=4)
        self.global_home_comm_collection = LineCollection([], colors='#16a34a', linewidths=2.0, alpha=0.9, zorder=4)
        ax.add_collection(self.global_ray_collection)
        ax.add_collection(self.global_comm_collection)
        ax.add_collection(self.global_home_comm_collection)
        for r in self.sim.robots:
            hist_line, = ax.plot([], [], color=r.color, linewidth=1.45, alpha=0.95, zorder=5)
            est_hist_line, = ax.plot([], [], color=r.color, linewidth=1.0, alpha=0.75, linestyle=':', zorder=5)
            plan_line, = ax.plot([], [], color=r.color, linewidth=1.0, alpha=0.65, linestyle='--', zorder=5)
            drift_line, = ax.plot([], [], color=r.color, linewidth=1.0, alpha=0.85, linestyle='-.', zorder=8)
            ell = patches.Ellipse((r.x_est, r.y_est), width=0.01, height=0.01, angle=0.0,
                                  facecolor='none', edgecolor=r.color, linewidth=1.8, linestyle='--', alpha=0.98, zorder=8)
            ax.add_patch(ell)
            self.global_path_lines.append(hist_line)
            self.global_est_path_lines.append(est_hist_line)
            self.global_plan_lines.append(plan_line)
            self.global_drift_lines.append(drift_line)
            self.global_cov_ellipses.append(ell)

    def _build_status_panel(self) -> None:
        assert self.fig is not None
        self.status_ax = self.fig.add_axes([0.645, 0.77, 0.31, 0.11])
        self.status_ax.set_facecolor('#fbfcfd')
        for sp in self.status_ax.spines.values():
            sp.set_edgecolor('#cfd8e3')
            sp.set_linewidth(1.0)
        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])
        self.status_ax.set_title('Mission status / legend', fontsize=10, pad=4)
        self.status_text = self.status_ax.text(
            0.02, 0.95, '', transform=self.status_ax.transAxes,
            va='top', ha='left', fontsize=8.0, family='monospace', color='#111827'
        )
        lx = 0.72
        ys = [0.84, 0.68, 0.52, 0.36, 0.20]
        labels = ['true pose', 'estimated pose', '2.5σ ellipse', 'robot LOS', 'home link']
        self.status_ax.scatter([lx], [ys[0]], s=36, c=['tab:blue'], edgecolors='black', linewidths=0.5, transform=self.status_ax.transAxes)
        self.status_ax.scatter([lx], [ys[1]], s=36, marker='o', facecolors='none', edgecolors=['tab:blue'], linewidths=1.2, transform=self.status_ax.transAxes)
        self.status_ax.add_patch(patches.Ellipse((lx, ys[2]), 0.07, 0.036, facecolor='none', edgecolor='tab:blue', linewidth=1.5, transform=self.status_ax.transAxes))
        self.status_ax.plot([lx - 0.025, lx + 0.025], [ys[3], ys[3]], color='#0ea5e9', linewidth=2.0, transform=self.status_ax.transAxes)
        self.status_ax.plot([lx - 0.025, lx + 0.025], [ys[4], ys[4]], color='#16a34a', linewidth=2.0, transform=self.status_ax.transAxes)
        for y, label in zip(ys, labels):
            self.status_ax.text(lx + 0.055, y, label, transform=self.status_ax.transAxes, va='center', fontsize=8.3)

    def _build_local_panels(self) -> None:
        assert self.fig is not None
        n = max(1, len(self.sim.robots))
        cols = 2 if n > 1 else 1
        rows = ceil(n / cols)
        x0, y0, w, h = 0.645, 0.08, 0.31, 0.66
        gap_x, gap_y = 0.012, 0.018
        card_w = (w - gap_x * (cols - 1)) / cols
        card_h = (h - gap_y * (rows - 1)) / rows
        info_h = min(0.072, card_h * 0.30)
        inner_gap = 0.004

        for idx, robot in enumerate(self.sim.robots):
            c = idx % cols
            r = idx // cols
            ax_left = x0 + c * (card_w + gap_x)
            ax_bottom = y0 + (rows - 1 - r) * (card_h + gap_y)

            info_ax = self.fig.add_axes([ax_left, ax_bottom + card_h - info_h, card_w, info_h])
            info_ax.set_facecolor('#fbfcfd')
            for sp in info_ax.spines.values():
                sp.set_edgecolor('#cfd8e3')
                sp.set_linewidth(1.0)
            info_ax.set_xticks([])
            info_ax.set_yticks([])

            ax = self.fig.add_axes([ax_left, ax_bottom, card_w, card_h - info_h - inner_gap])
            ax.set_facecolor('#fbfcfd')
            for sp in ax.spines.values():
                sp.set_edgecolor('#cfd8e3')
                sp.set_linewidth(1.0)
            ax.set_xlim(0, self.sim.cfg.world_w)
            ax.set_ylim(0, self.sim.cfg.world_h)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            img = ax.imshow(
                robot.local_map.data,
                origin='lower', interpolation='nearest',
                cmap=self.cmap, norm=self.norm,
                extent=[0, self.sim.cfg.world_w, 0, self.sim.cfg.world_h],
                alpha=self.sim.cfg.local_view_alpha,
                zorder=1,
            )
            self._draw_home_base(ax)
            ax.scatter(
                [lm.x for lm in self.sim.world.landmarks], [lm.y for lm in self.sim.world.landmarks],
                marker='*', s=20, c='#f2c641', edgecolors='black', linewidths=0.35, alpha=0.30, zorder=2,
            )
            ax.scatter(
                [self.sim.world.home_marker.x], [self.sim.world.home_marker.y],
                marker='P', s=34, c='#22c55e', edgecolors='black', linewidths=0.45, alpha=0.9, zorder=2,
            )

            frontier_scatter = ax.scatter([], [], s=5, c='#22c1c3', alpha=0.45, linewidths=0, zorder=3)
            own_scan = LineCollection([], colors=robot.color, linewidths=0.55, alpha=0.15, zorder=4)
            ax.add_collection(own_scan)
            own_path_line, = ax.plot([], [], color=robot.color, linewidth=1.15, alpha=0.9, zorder=5)
            own_est_path_line, = ax.plot([], [], color=robot.color, linewidth=0.9, alpha=0.75, linestyle=':', zorder=5)
            own_robot_scatter = ax.scatter([robot.x], [robot.y], s=34, c=[robot.color], edgecolors='black', linewidths=0.45, zorder=7)
            own_est_scatter = ax.scatter([robot.x_est], [robot.y_est], s=28, marker='o', facecolors='none', edgecolors=[robot.color], linewidths=1.1, zorder=8)
            own_drift_line, = ax.plot([], [], color=robot.color, linewidth=0.95, alpha=0.85, linestyle='-.', zorder=8)
            own_cov = patches.Ellipse((robot.x_est, robot.y_est), width=0.01, height=0.01, angle=0.0,
                                      facecolor='none', edgecolor=robot.color, linewidth=1.0, alpha=0.85, zorder=8)
            ax.add_patch(own_cov)
            own_target_scatter = ax.scatter([], [], s=28, marker='x', c=[robot.color], linewidths=1.1, zorder=6)
            packet_scatter = ax.scatter([], [], s=30, marker='s', c='none', edgecolors='tab:red', linewidths=1.2, zorder=6)
            target_scatter = ax.scatter([], [], s=34, marker='x', c='tab:red', linewidths=1.2, zorder=6)
            keypoint_scatter = ax.scatter([], [], s=11, marker='o', c='#ef4444', alpha=0.50, linewidths=0, zorder=5)

            teammate_lines = []
            for other in self.sim.robots:
                line, = ax.plot([], [], linestyle='--', linewidth=0.95, alpha=0.65, color=other.color, zorder=5)
                teammate_lines.append(line)

            txt = info_ax.text(
                0.02, 0.96, '', transform=info_ax.transAxes,
                va='top', ha='left', fontsize=6.8, family='monospace', color='#111827'
            )

            self.local_info_axes.append(info_ax)
            self.axes_local.append(ax)
            self.local_images.append(img)
            self.local_frontier_scatter.append(frontier_scatter)
            self.local_packet_scatter.append(packet_scatter)
            self.local_target_scatter.append(target_scatter)
            self.local_shared_keypoint_scatter.append(keypoint_scatter)
            self.local_robot_scatter.append(own_robot_scatter)
            self.local_est_scatter.append(own_est_scatter)
            self.local_drift_lines.append(own_drift_line)
            self.local_cov_ellipses.append(own_cov)
            self.local_robot_target_scatter.append(own_target_scatter)
            self.local_robot_path_lines.append(own_path_line)
            self.local_est_path_lines.append(own_est_path_line)
            self.local_scan_collections.append(own_scan)
            self.local_path_lines.append(teammate_lines)
            self.local_texts.append(txt)

    def _tick(self, _frame_idx: int):
        self.sim.step()
        self._refresh()
        artists = [
            self.global_robot_scatter,
            self.global_est_scatter,
            self.global_target_scatter,
            self.global_ray_collection,
            self.global_comm_collection,
            self.global_home_comm_collection,
            self.status_text,
        ]
        artists.extend(self.global_path_lines)
        artists.extend(self.global_est_path_lines)
        artists.extend(self.global_plan_lines)
        artists.extend(self.global_drift_lines)
        artists.extend(self.global_cov_ellipses)
        artists.extend(self.local_images)
        artists.extend(self.local_frontier_scatter)
        artists.extend(self.local_packet_scatter)
        artists.extend(self.local_target_scatter)
        artists.extend(self.local_shared_keypoint_scatter)
        artists.extend(self.local_robot_scatter)
        artists.extend(self.local_est_scatter)
        artists.extend(self.local_drift_lines)
        artists.extend(self.local_cov_ellipses)
        artists.extend(self.local_robot_target_scatter)
        artists.extend(self.local_robot_path_lines)
        artists.extend(self.local_est_path_lines)
        artists.extend(self.local_scan_collections)
        artists.extend(self.local_texts)
        for lines in self.local_path_lines:
            artists.extend(lines)
        return artists

    def _refresh(self) -> None:
        robot_xy = np.array([[r.x, r.y] for r in self.sim.robots], dtype=float)
        est_xy = np.array([[r.x_est, r.y_est] for r in self.sim.robots], dtype=float)
        self.global_robot_scatter.set_offsets(robot_xy)
        self.global_est_scatter.set_offsets(est_xy)
        tgt = np.array([
            r.current_target if r.current_target is not None else (np.nan, np.nan)
            for r in self.sim.robots
        ], dtype=float)
        self.global_target_scatter.set_offsets(tgt)
        segs = []
        if self.sim.cfg.show_rays:
            for r in self.sim.robots:
                for hx, hy, _ in r.last_scan:
                    segs.append([(r.x, r.y), (hx, hy)])
        self.global_ray_collection.set_segments(segs)

        robot_lookup = {r.robot_id: r for r in self.sim.robots}
        comm_segs = [[robot_lookup[a].pose_xy(), robot_lookup[b].pose_xy()] for a, b in self.sim.robot_comm_edges]
        home_segs = [[robot_lookup[rid].pose_xy(), anchor] for rid, anchor in self.sim.home_comm_links]
        self.global_comm_collection.set_segments(comm_segs)
        self.global_home_comm_collection.set_segments(home_segs)

        for hist_line, est_hist_line, plan_line, drift_line, ell, robot in zip(
            self.global_path_lines, self.global_est_path_lines, self.global_plan_lines,
            self.global_drift_lines, self.global_cov_ellipses, self.sim.robots
        ):
            if len(robot.path_history) >= 2:
                xs = [p[0] for p in robot.path_history[-160:]]
                ys = [p[1] for p in robot.path_history[-160:]]
                hist_line.set_data(xs, ys)
            else:
                hist_line.set_data([], [])
            if len(robot.est_path_history) >= 2:
                xs = [p[0] for p in robot.est_path_history[-160:]]
                ys = [p[1] for p in robot.est_path_history[-160:]]
                est_hist_line.set_data(xs, ys)
            else:
                est_hist_line.set_data([], [])
            if robot.current_path:
                xs = [robot.x_est] + [p[0] for p in robot.current_path]
                ys = [robot.y_est] + [p[1] for p in robot.current_path]
                plan_line.set_data(xs, ys)
            else:
                plan_line.set_data([], [])
            drift_line.set_data([robot.x, robot.x_est], [robot.y, robot.y_est])
            self._set_covariance_ellipse(ell, robot.x_est, robot.y_est, robot.P[:2, :2])

        status = 'RUNNING' if self.sim.running else 'PAUSED'
        known_counts = [100.0 * float(np.mean(r.local_map.data != UNKNOWN)) for r in self.sim.robots]
        self.status_text.set_text(
            f'{status}\n'
            f't={self.sim.time_s:5.1f}s   steps={self.sim.step_count}   replans={self.sim.replan_count}\n'
            f'coverage≈{self.sim.estimated_coverage():5.1f}%   mean local known={np.mean(known_counts):4.1f}%\n'
            f'home-connected={self.sim.connected_robot_count()}/{len(self.sim.robots)}   robot-links={len(self.sim.robot_comm_edges)}   range={self.sim.cfg.comm_radius:3.1f}m\n'
            f'mean loc err={self.sim.mean_localization_error():4.2f}m   max err={self.sim.max_localization_error():4.2f}m   mean tr(Pxy)={self.sim.mean_covariance_trace():4.2f}'
        )

        robot_color_by_id = {r.robot_id: r.color for r in self.sim.robots}

        for idx, robot in enumerate(self.sim.robots):
            self.local_images[idx].set_data(robot.local_map.data)

            frontier = frontier_mask(robot.local_map.data)
            rr, cc = np.where(frontier)
            if len(rr) > 350:
                step = max(1, len(rr) // 350)
                rr = rr[::step]
                cc = cc[::step]
            if len(rr):
                xs = (cc + 0.5) * robot.local_map.res
                ys = (rr + 0.5) * robot.local_map.res
                self.local_frontier_scatter[idx].set_offsets(np.column_stack([xs, ys]))
            else:
                self.local_frontier_scatter[idx].set_offsets(np.empty((0, 2), dtype=float))

            packets = robot.received_packets
            packet_xy_list = [xy for xy, _ts in robot.shared_pose_memory.values()]
            packet_tgt_list = [xy for xy, _ts in robot.shared_target_memory.values()]
            packet_xy = np.asarray(packet_xy_list, dtype=float).reshape((-1, 2)) if packet_xy_list else np.empty((0, 2), dtype=float)
            packet_tgt = np.asarray(packet_tgt_list, dtype=float).reshape((-1, 2)) if packet_tgt_list else np.empty((0, 2), dtype=float)
            self.local_packet_scatter[idx].set_offsets(packet_xy)
            self.local_target_scatter[idx].set_offsets(packet_tgt)
            if len(packet_xy):
                pose_colors = [robot_color_by_id.get(rid, 'tab:red') for rid in robot.shared_pose_memory.keys()]
                self.local_packet_scatter[idx].set_edgecolors(pose_colors)
            if len(packet_tgt):
                tgt_colors = [robot_color_by_id.get(rid, 'tab:red') for rid in robot.shared_target_memory.keys()]
                self.local_target_scatter[idx].set_color(tgt_colors)

            keypoints = []
            key_colors = []
            for rid, pts in robot.shared_path_memory.items():
                for px, py, _ts in pts:
                    keypoints.append((px, py))
                    key_colors.append(robot_color_by_id.get(rid, '#ef4444'))
            if keypoints:
                self.local_shared_keypoint_scatter[idx].set_offsets(np.asarray(keypoints, dtype=float).reshape((-1, 2)))
                self.local_shared_keypoint_scatter[idx].set_color(key_colors)
            else:
                self.local_shared_keypoint_scatter[idx].set_offsets(np.empty((0, 2), dtype=float))

            self.local_robot_scatter[idx].set_offsets(np.array([[robot.x, robot.y]], dtype=float))
            self.local_est_scatter[idx].set_offsets(np.array([[robot.x_est, robot.y_est]], dtype=float))
            self.local_drift_lines[idx].set_data([robot.x, robot.x_est], [robot.y, robot.y_est])
            self._set_covariance_ellipse(self.local_cov_ellipses[idx], robot.x_est, robot.y_est, robot.P[:2, :2])
            if robot.current_target is not None:
                self.local_robot_target_scatter[idx].set_offsets(np.array([robot.current_target], dtype=float))
            else:
                self.local_robot_target_scatter[idx].set_offsets(np.empty((0, 2), dtype=float))
            if len(robot.path_history) >= 2:
                xs = [p[0] for p in robot.path_history[-120:]]
                ys = [p[1] for p in robot.path_history[-120:]]
                self.local_robot_path_lines[idx].set_data(xs, ys)
            else:
                self.local_robot_path_lines[idx].set_data([], [])
            if len(robot.est_path_history) >= 2:
                xs = [p[0] for p in robot.est_path_history[-120:]]
                ys = [p[1] for p in robot.est_path_history[-120:]]
                self.local_est_path_lines[idx].set_data(xs, ys)
            else:
                self.local_est_path_lines[idx].set_data([], [])

            own_segs = []
            if self.sim.cfg.show_rays:
                for hx, hy, _ in robot.last_scan:
                    own_segs.append([(robot.x, robot.y), (hx, hy)])
            self.local_scan_collections[idx].set_segments(own_segs)

            for j, line in enumerate(self.local_path_lines[idx]):
                if j >= len(self.sim.robots):
                    line.set_data([], [])
                    continue
                other = self.sim.robots[j]
                if other.robot_id == robot.robot_id:
                    line.set_data([], [])
                    continue
                pkt = next((p for p in packets if p.robot_id == other.robot_id), None)
                if pkt is None or len(pkt.path_xy) < 2:
                    line.set_data([], [])
                    continue
                xs = [p[0] for p in pkt.path_xy]
                ys = [p[1] for p in pkt.path_xy]
                line.set_data(xs, ys)

            known_pct = 100.0 * float(np.mean(robot.local_map.data != UNKNOWN))
            frontier_cells = int(frontier.sum())
            cur_tgt = '-' if robot.current_target is None else f'({robot.current_target[0]:.1f},{robot.current_target[1]:.1f})'
            score_text = robot.last_choice_debug.replace('\n', '   ')
            home_state = 'yes' if robot.home_connected else 'no'
            hops = '-' if robot.home_hops is None else str(robot.home_hops)
            err = float(np.hypot(robot.x - robot.x_est, robot.y - robot.y_est))
            shared_pose_n = len(robot.shared_pose_memory)
            shared_key_n = sum(len(v) for v in robot.shared_path_memory.values())
            self.local_texts[idx].set_text(
                f'{robot.name}\n'
                f'known {known_pct:4.1f}%   frontier {frontier_cells}   pkts {len(robot.received_packets)}\n'
                f'shared pose {shared_pose_n}   shared keypts {shared_key_n}   motion {robot.motion_state}\n'
                f'home {home_state}   hops {hops}   nbrs {len(robot.direct_neighbors)}\n'
                f'true ({robot.x:4.1f},{robot.y:4.1f})   est ({robot.x_est:4.1f},{robot.y_est:4.1f})\n'
                f'err {err:4.2f}m   tr(Pxy) {robot.covariance_trace():4.2f}   lm {robot.last_landmark_updates}   team {robot.last_teammate_updates}\n'
                f'goal {cur_tgt}   {score_text}'
            )

    def _set_covariance_ellipse(self, ellipse, x: float, y: float, cov_xy: np.ndarray) -> None:
        cov = 0.5 * (cov_xy + cov_xy.T)
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            vals = np.array([1e-6, 1e-6], dtype=float)
            vecs = np.eye(2)
        vals = np.clip(vals, 1e-6, None)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        n_sigma = 2.5
        ellipse.center = (x, y)
        ellipse.width = max(0.30, 2.0 * n_sigma * float(np.sqrt(vals[0])))
        ellipse.height = max(0.22, 2.0 * n_sigma * float(np.sqrt(vals[1])))
        ellipse.angle = angle

    def _draw_home_base(self, ax) -> None:
        base = self.sim.world.home_base
        ax.add_patch(
            patches.Rectangle(
                (base.xmin, base.ymin), base.w, base.h,
                facecolor='#cfeec5', edgecolor='#2d7f38', lw=1.3, alpha=0.52, zorder=1,
            )
        )

    def _parse_box_int(self, key: str, fallback: int, lo: int, hi: int) -> int:
        box = self.controls[key]
        try:
            value = int(float(box.text))
        except Exception:
            value = fallback
        value = max(lo, min(hi, value))
        if str(value) != box.text.strip():
            box.set_val(str(value))
        return value

    def _on_start(self, _event) -> None:
        self.sim.start()
        self._refresh()
        self.fig.canvas.draw_idle()

    def _on_stop(self, _event) -> None:
        self.sim.stop()
        self._refresh()
        self.fig.canvas.draw_idle()

    def _on_toggle_rays(self, _event) -> None:
        self.sim.cfg = replace(self.sim.cfg, show_rays=not self.sim.cfg.show_rays)
        self._refresh()
        self.fig.canvas.draw_idle()

    def _on_reset(self, _event) -> None:
        new_cfg = replace(
            self.sim.cfg,
            seed=self._parse_box_int('seed_box', self.sim.cfg.seed, 0, 10_000_000),
            robot_count=self._parse_box_int('robot_box', self.sim.cfg.robot_count, 1, 12),
            obstacle_count=self._parse_box_int('obstacle_box', self.sim.cfg.obstacle_count, 0, 80),
            landmark_count=self._parse_box_int('landmark_box', self.sim.cfg.landmark_count, 0, 80),
        )
        self.sim.reset(new_cfg)
        self._build_layout()
