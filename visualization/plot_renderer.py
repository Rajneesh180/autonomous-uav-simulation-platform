"""
Plot Renderer — Professional 3D-First Visualization Suite
===========================================================
Publication-quality figures with bold 3D spatial rendering (dark theme)
and clean 2D analytical charts (light theme).  Designed for IEEE /
BTP thesis reports.

Colour palette: neon/cyberpunk-inspired for 3D; clean material for 2D.
All outputs: dual PNG (300 DPI) + vector PDF.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
import numpy as np
import os
from config.feature_toggles import FeatureToggles


class PlotRenderer:
    """Professional UAV simulation visualiser — IEEE publication-quality."""

    # ── IEEE / Research-Grade Colour Palette ─────────────────
    C = {
        "visited":    "#2E7D32",   # forest green
        "waiting":    "#1565C0",   # strong blue
        "idle":       "#9E9E9E",   # neutral gray
        "critical":   "#C62828",   # deep red
        "uav":        "#0277BD",   # ocean blue
        "uav_edge":   "#01579B",
        "base":       "#E65100",   # burnt orange
        "base_edge":  "#BF360C",
        "obstacle":   "#B71C1C",   # dark red
        "obs_edge":   "#7F0000",
        "risk":       "#F9A825",   # golden amber
        "risk_edge":  "#F57F17",
        "trail":      "#4527A0",   # indigo
        "trail_hot":  "#AD1457",   # deep rose
        "rp":         "#FF8F00",   # dark amber
        "cluster":    "#6A1B9A",   # purple
    }

    # Theme constants
    _DARK_BG   = "#1A1A2E"
    _DARK_GRID = "#3A3A5C"
    _DARK_TXT  = "#E0E0E0"
    _LIGHT_BG  = "#FFFFFF"
    _LIGHT_AX  = "#FAFAFA"
    _LIGHT_TXT = "#212121"
    _LIGHT_GRD = "#BDBDBD"

    # ═══════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _use_3d():
        return FeatureToggles.RENDER_MODE in ("3D", "both")

    @staticmethod
    def _set_style_3d():
        """Dark theme for 3D spatial plots."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.10,
            "grid.color": PlotRenderer._DARK_GRID,
            "text.color": PlotRenderer._DARK_TXT,
            "axes.labelcolor": PlotRenderer._DARK_TXT,
            "xtick.color": PlotRenderer._DARK_TXT,
            "ytick.color": PlotRenderer._DARK_TXT,
        })

    @staticmethod
    def _set_style_2d():
        """Clean light theme for 2D analytical charts."""
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": PlotRenderer._LIGHT_GRD,
            "text.color": PlotRenderer._LIGHT_TXT,
            "axes.labelcolor": PlotRenderer._LIGHT_TXT,
            "xtick.color": PlotRenderer._LIGHT_TXT,
            "ytick.color": PlotRenderer._LIGHT_TXT,
            "axes.facecolor": PlotRenderer._LIGHT_AX,
            "figure.facecolor": PlotRenderer._LIGHT_BG,
        })

    @staticmethod
    def _set_style():
        PlotRenderer._set_style_2d()

    @staticmethod
    def _dark_fig(fig, ax=None):
        """Apply dark background to figure and 3D axes panes."""
        fig.patch.set_facecolor(PlotRenderer._DARK_BG)

    @staticmethod
    def _save_dual(fig, save_dir, basename):
        fig.savefig(os.path.join(save_dir, f"{basename}.png"),
                    dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(os.path.join(save_dir, f"{basename}.pdf"),
                    format="pdf", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    @staticmethod
    def _alt_color(z, z_max=80.0):
        """Altitude color: blue(low) -> purple(mid) -> amber(high)."""
        f = min(1.0, max(0.0, z / z_max))
        if f < 0.5:
            t = f * 2
            return (0.27 * (1 - t) + 0.49 * t,
                    0.54 * (1 - t) + 0.30 * t,
                    1.00 * (1 - t) + 1.00 * t)
        t = (f - 0.5) * 2
        return (0.49 * (1 - t) + 1.00 * t,
                0.30 * (1 - t) + 0.84 * t,
                1.00 * (1 - t) + 0.25 * t)

    # ── 3D Building Blocks ───────────────────────────────────
    @staticmethod
    def _draw_obstacles_3d(ax, obstacles, z_ceil=30, alpha=0.30):
        C = PlotRenderer.C
        for obs in obstacles:
            v = [
                [obs.x1, obs.y1, 0],      [obs.x2, obs.y1, 0],
                [obs.x2, obs.y2, 0],      [obs.x1, obs.y2, 0],
                [obs.x1, obs.y1, z_ceil], [obs.x2, obs.y1, z_ceil],
                [obs.x2, obs.y2, z_ceil], [obs.x1, obs.y2, z_ceil],
            ]
            faces = [
                [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]],
            ]
            ax.add_collection3d(Poly3DCollection(
                faces, alpha=alpha, facecolors=C["obstacle"],
                edgecolors=C["obs_edge"], linewidths=0.7))

    @staticmethod
    def _draw_obstacles_2d(ax, obstacles, alpha=0.28):
        C = PlotRenderer.C
        for obs in obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                facecolor=C["obstacle"], alpha=alpha,
                edgecolor=C["obs_edge"], linewidth=1.0)
            ax.add_patch(rect)

    @staticmethod
    def _draw_risk_3d(ax, risk_zones, z_ceil=20):
        C = PlotRenderer.C
        for rz in risk_zones:
            v = [
                [rz.x1, rz.y1, 0],      [rz.x2, rz.y1, 0],
                [rz.x2, rz.y2, 0],      [rz.x1, rz.y2, 0],
                [rz.x1, rz.y1, z_ceil], [rz.x2, rz.y1, z_ceil],
                [rz.x2, rz.y2, z_ceil], [rz.x1, rz.y2, z_ceil],
            ]
            faces = [
                [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]],
            ]
            ax.add_collection3d(Poly3DCollection(
                faces, alpha=0.22, facecolors=C["risk"],
                edgecolors=C["risk_edge"], linewidths=0.5))

    @staticmethod
    def _draw_risk_2d(ax, risk_zones):
        C = PlotRenderer.C
        for rz in risk_zones:
            rect = plt.Rectangle(
                (rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1,
                facecolor=C["risk"], alpha=0.22,
                edgecolor=C["risk_edge"], linewidth=1.0)
            ax.add_patch(rect)

    @staticmethod
    def _setup_3d_ax(ax, w, h, z_max=80, elev=25, azim=-45):
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_zlim(0, z_max)
        ax.set_xlabel("X (m)", labelpad=10)
        ax.set_ylabel("Y (m)", labelpad=10)
        ax.set_zlabel("Altitude (m)", labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(PlotRenderer._DARK_GRID)
            pane.set_alpha(0.08)
        ax.grid(True, alpha=0.08, color=PlotRenderer._DARK_GRID)
        ax.tick_params(colors=PlotRenderer._DARK_TXT, labelsize=8)

    @staticmethod
    def _setup_2d_ax(ax, w, h):
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")

    @staticmethod
    def _draw_trail_3d(ax, trail, lw=2.5, alpha=0.92):
        """Draw altitude-gradient coloured 3D trail."""
        if len(trail) < 2:
            return
        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]
        zs = [p[2] if len(p) > 2 else 50.0 for p in trail]
        for i in range(len(xs) - 1):
            c = PlotRenderer._alt_color(zs[i])
            ax.plot(xs[i:i + 2], ys[i:i + 2], zs[i:i + 2],
                    color=c, linewidth=lw, alpha=alpha, solid_capstyle="round")

    # ═══════════════════════════════════════════════════════════
    #  1. ENVIRONMENT OVERVIEW (3D dark)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_environment(env, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        if is_3d:
            PlotRenderer._set_style_3d()
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            PlotRenderer._dark_fig(fig)

            for node in env.sensors:
                z = getattr(node, "z", 0)
                ax.scatter(node.x, node.y, z, c=C["waiting"],
                           s=120, marker="o", edgecolors="white",
                           linewidths=0.6, depthshade=True, zorder=3)
                ax.text(node.x, node.y, z + 4, str(node.id),
                        fontsize=7, color="#E6EDF3", ha="center",
                        fontweight="bold")

            if hasattr(env, "uav") and env.uav:
                ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=280,
                           marker="s", edgecolors=C["base_edge"],
                           linewidths=1.2, zorder=10, label="Base Station")

            PlotRenderer._draw_obstacles_3d(ax, env.obstacles)
            PlotRenderer._draw_risk_3d(ax, env.risk_zones)
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            PlotRenderer._set_style_2d()
            fig, ax = plt.subplots(figsize=(12, 9))
            for node in env.sensors:
                ax.scatter(node.x, node.y, c=C["waiting"], s=100,
                           marker="o", edgecolors="black", linewidths=0.5,
                           zorder=3)
                ax.annotate(str(node.id), (node.x + 5, node.y + 5),
                            fontsize=7, color="#546E7A")

            if hasattr(env, "uav") and env.uav:
                ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=240,
                           marker="s", edgecolors=C["base_edge"],
                           linewidths=1.0, zorder=10, label="Base Station")

            PlotRenderer._draw_obstacles_2d(ax, env.obstacles)
            PlotRenderer._draw_risk_2d(ax, env.risk_zones)
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        ax.set_title("UAV Mission Environment", fontsize=16, fontweight="bold",
                      color=PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT)
        ax.legend(loc="upper right", framealpha=0.7, facecolor="#161B22" if is_3d else "white",
                  labelcolor=PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT)
        PlotRenderer._save_dual(fig, save_dir, "environment")

    # ═══════════════════════════════════════════════════════════
    #  2. ENERGY & VISIT BAR CHARTS (2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_energy_plots(visited, energy_consumed, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(["Visited\nNodes"], [visited], color="#00FF87",
                     edgecolor="#00C853", linewidth=1.2, width=0.45)
        axes[0].set_title("Nodes Visited", fontweight="bold", fontsize=13)
        axes[0].set_ylabel("Count", fontsize=11)
        axes[0].text(0, visited + 0.3, str(visited), ha="center",
                     fontsize=14, fontweight="bold", color="#00C853")

        axes[1].bar(["Energy\nConsumed"], [energy_consumed], color="#FF6D00",
                     edgecolor="#E65100", linewidth=1.2, width=0.45)
        axes[1].set_title("Energy Consumed", fontweight="bold", fontsize=13)
        axes[1].set_ylabel("Joules", fontsize=11)
        axes[1].text(0, energy_consumed + 0.3, f"{energy_consumed:.1f}",
                     ha="center", fontsize=12, fontweight="bold", color="#E65100")

        fig.suptitle("Mission Resource Summary", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        PlotRenderer._save_dual(fig, save_dir, "energy_visited")

    # ═══════════════════════════════════════════════════════════
    #  3. METRICS SNAPSHOT (2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_metrics_snapshot(completion_pct, efficiency, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(["Completion %", "Efficiency"],
                       [completion_pct, efficiency],
                       color=["#58A6FF", "#00FF87"],
                       edgecolor=["#1F6FEB", "#00C853"],
                       linewidth=1.2, width=0.5)
        ax.set_title("Mission Metrics Snapshot", fontweight="bold", fontsize=14)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=12,
                    fontweight="bold")
        plt.tight_layout()
        PlotRenderer._save_dual(fig, save_dir, "metrics_snapshot")

    # ═══════════════════════════════════════════════════════════
    #  4. TIME-SERIES PLOTS (2D light, wide)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_time_series(visited_hist, battery_hist, replan_hist, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        fig.suptitle("Mission Time-Series", fontsize=16, fontweight="bold", y=1.03)

        # Visited
        ax = axes[0]
        ax.plot(visited_hist, color="#00FF87", linewidth=2.0)
        ax.fill_between(range(len(visited_hist)), visited_hist,
                        alpha=0.15, color="#00FF87")
        ax.set_title("Visited Nodes", fontsize=13)
        ax.set_xlabel("Step"); ax.set_ylabel("Count")

        # Battery
        ax = axes[1]
        ax.plot(battery_hist, color="#58A6FF", linewidth=2.0)
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.12, color="#58A6FF")
        ax.set_title("Battery Level", fontsize=13)
        ax.set_xlabel("Step"); ax.set_ylabel("Joules")

        # Replans
        ax = axes[2]
        ax.plot(replan_hist, color="#FF6D00", linewidth=2.0)
        ax.fill_between(range(len(replan_hist)), replan_hist,
                        alpha=0.12, color="#FF6D00")
        ax.set_title("Cumulative Replans", fontsize=13)
        ax.set_xlabel("Step"); ax.set_ylabel("Count")

        plt.tight_layout()
        PlotRenderer._save_dual(fig, save_dir, "time_series")

    # ═══════════════════════════════════════════════════════════
    #  5. ENVIRONMENT FRAME (per-step, 3D dark)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_environment_frame(env, save_dir, step, mission=None):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        if is_3d:
            PlotRenderer._set_style_3d()
        else:
            PlotRenderer._set_style_2d()

        fig = plt.figure(figsize=(12, 9) if is_3d else (10, 7))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
        if is_3d:
            PlotRenderer._dark_fig(fig)

        visited_ids = mission.visited if mission else set()

        # ── Nodes ──
        for node in env.sensors:
            if node.id in visited_ids:
                colour, sz = C["visited"], 110
            else:
                buf_ratio = node.current_buffer / max(node.buffer_capacity, 1e-6)
                colour = C["waiting"] if buf_ratio > 0.5 else C["idle"]
                sz = 70
            z = getattr(node, "z", 0)
            if is_3d:
                ax.scatter(node.x, node.y, z, c=colour, s=sz, marker="o",
                           edgecolors="white", linewidths=0.5, depthshade=True)
            else:
                ax.scatter(node.x, node.y, c=colour, s=sz, marker="o",
                           edgecolors="black", linewidths=0.5, zorder=4)

        # ── Obstacles & risk zones ──
        if is_3d:
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles)
            PlotRenderer._draw_risk_3d(ax, env.risk_zones)
        else:
            PlotRenderer._draw_obstacles_2d(ax, env.obstacles)
            PlotRenderer._draw_risk_2d(ax, env.risk_zones)

        # ── UAV trail + marker ──
        if hasattr(env, "uav"):
            if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
                if is_3d:
                    PlotRenderer._draw_trail_3d(ax, env.uav_trail, lw=2.0)
                else:
                    tx = [p[0] for p in env.uav_trail]
                    ty = [p[1] for p in env.uav_trail]
                    ax.plot(tx, ty, linewidth=1.8, alpha=0.8,
                            color=C["trail"], zorder=5)

            if is_3d:
                ax.scatter(env.uav.x, env.uav.y, env.uav.z,
                           s=220, marker="^", c=C["uav"],
                           edgecolors=C["uav_edge"], linewidths=1.2, zorder=10)
            else:
                ax.scatter(env.uav.x, env.uav.y,
                           s=200, marker="^", c=C["uav"],
                           edgecolors=C["uav_edge"], linewidths=1.0, zorder=10)

            if not is_3d and mission and mission.current_target:
                tgt = mission.current_target
                ax.plot([env.uav.x, tgt.x], [env.uav.y, tgt.y],
                        linestyle="--", color=C["trail_hot"],
                        linewidth=1.0, alpha=0.6, zorder=6)

        # ── Cluster boundaries ──
        if mission and hasattr(mission, 'active_labels') and len(mission.active_labels) > 0:
            from scipy.spatial import ConvexHull
            cluster_colors = ['#FF6B6B', '#00FF87', '#00E5FF', '#FFD93D', '#FF87AB',
                              '#C77DFF', '#72EFDD', '#FFB347']
            label_arr = np.array(mission.active_labels)
            unique_labels = sorted(set(label_arr))
            sensor_list = [n for n in env.sensors if not getattr(n, '_is_dynamic', False)]
            for lbl in unique_labels:
                if lbl < 0:
                    continue
                idxs = np.where(label_arr == lbl)[0]
                if len(idxs) < 3:
                    continue
                pts = np.array([[sensor_list[i].x, sensor_list[i].y]
                                for i in idxs if i < len(sensor_list)])
                if len(pts) < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    clr = cluster_colors[lbl % len(cluster_colors)]
                    if is_3d:
                        ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                                [0]*len(hull_pts), color=clr,
                                linewidth=1.5, alpha=0.6, linestyle='--')
                    else:
                        ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                                color=clr, linewidth=1.5, alpha=0.5,
                                linestyle='--', zorder=3)
                        ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                                color=clr, alpha=0.08, zorder=2)
                except Exception:
                    pass

        # ── Axes ──
        if is_3d:
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        # ── HUD ──
        total_nodes = len(env.sensors)
        n_visited = len(visited_ids)
        battery_pct = 0.0
        if hasattr(env, "uav"):
            from config.config import Config
            battery_pct = (env.uav.current_battery / Config.BATTERY_CAPACITY) * 100
        title_col = PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT
        ax.set_title(
            f"Step {step}  |  Battery: {battery_pct:.0f}%  |  "
            f"Visited: {n_visited}/{total_nodes}",
            fontsize=13, fontweight="bold", color=title_col)

        # ── Replan flash ──
        if hasattr(env, "temporal_engine"):
            flash = env.temporal_engine.consume_replan_flash()
            if flash:
                txt_fn = ax.text2D if is_3d else ax.text
                txt_fn(0.5, 0.93, "⚡ REPLAN", transform=ax.transAxes,
                       ha="center", fontsize=13, color="#FF6B6B",
                       fontweight="bold")

        fig.savefig(os.path.join(save_dir, f"{step:04d}.png"), dpi=150,
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  6. RADAR CHART (dark polar)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_radar_chart(results: dict, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_3d()

        labels = [
            "Data Collection\nRate (%)", "Coverage\nRatio (%)",
            "Network\nLifetime", "Path\nStability",
            "Priority\nSatisfaction (%)", "Data\nFreshness",
        ]
        max_aoi = 800.0
        values = [
            min(100.0, results.get("data_collection_rate_percent", 0)),
            min(100.0, results.get("coverage_ratio_percent", 0)),
            results.get("network_lifetime_residual", 0) * 100,
            results.get("path_stability_index", 0) * 100,
            results.get("priority_satisfaction_percent", 0),
            max(0, 100 * (1 - results.get("average_aoi_s", 0) / max_aoi)),
        ]

        num = len(labels)
        angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        PlotRenderer._dark_fig(fig)
        ax.set_facecolor("#0D111700")

        ax.fill(angles, values, color="#00E5FF", alpha=0.20)
        ax.plot(angles, values, color="#00E5FF", linewidth=2.5,
                marker="o", markersize=8, markerfacecolor="#00FF87",
                markeredgecolor="white", markeredgewidth=1.0)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, color=PlotRenderer._DARK_TXT)
        ax.set_ylim(0, 100)
        ax.set_title("Mission Performance Radar",
                      fontsize=16, fontweight="bold", pad=25,
                      color=PlotRenderer._DARK_TXT)
        ax.tick_params(axis="y", colors=PlotRenderer._DARK_TXT, labelsize=8)
        ax.spines["polar"].set_color(PlotRenderer._DARK_GRID)
        ax.yaxis.grid(color=PlotRenderer._DARK_GRID, alpha=0.3)
        ax.xaxis.grid(color=PlotRenderer._DARK_GRID, alpha=0.3)

        PlotRenderer._save_dual(fig, save_dir, "radar_chart")

    # ═══════════════════════════════════════════════════════════
    #  7. NODE ENERGY HEATMAP (3D dark)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_node_energy_heatmap(nodes: list, env_width: float,
                                    env_height: float, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()

        from config.config import Config
        initial = Config.NODE_BATTERY_CAPACITY_J
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        zs = [getattr(n, "z", 0) for n in nodes]
        residuals = [
            getattr(n, "node_battery_J", initial) / max(initial, 1e-6)
            for n in nodes
        ]

        if is_3d:
            PlotRenderer._set_style_3d()
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            PlotRenderer._dark_fig(fig)
            sc = ax.scatter(xs, ys, zs, c=residuals, cmap="plasma",
                            s=180, edgecolors="white", linewidths=0.6,
                            vmin=0, vmax=1, depthshade=True)
            PlotRenderer._setup_3d_ax(ax, env_width, env_height)
            cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
        else:
            PlotRenderer._set_style_2d()
            fig, ax = plt.subplots(figsize=(12, 8))
            sc = ax.scatter(xs, ys, c=residuals, cmap="plasma", s=150,
                            edgecolors="black", linewidths=0.6, vmin=0, vmax=1)
            PlotRenderer._setup_2d_ax(ax, env_width, env_height)
            cbar = plt.colorbar(sc, ax=ax)

        cbar.set_label("Residual Battery Fraction", fontsize=11)
        title_col = PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT
        ax.set_title("IoT Node Residual Energy Map",
                      fontsize=15, fontweight="bold", color=title_col)
        PlotRenderer._save_dual(fig, save_dir, "node_energy_heatmap")

    # ═══════════════════════════════════════════════════════════
    #  8. TRAJECTORY SUMMARY (3D dark, flagship)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_trajectory_summary(env, visited_ids: set, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        if is_3d:
            PlotRenderer._set_style_3d()
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            PlotRenderer._dark_fig(fig)
        else:
            PlotRenderer._set_style_2d()
            fig, ax = plt.subplots(figsize=(12, 9))

        # Obstacles & risk zones
        if is_3d:
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles)
            PlotRenderer._draw_risk_3d(ax, env.risk_zones)
        else:
            PlotRenderer._draw_obstacles_2d(ax, env.obstacles)
            PlotRenderer._draw_risk_2d(ax, env.risk_zones)

        # Nodes
        for node in env.sensors:
            is_visited = node.id in visited_ids
            colour = C["visited"] if is_visited else C["critical"]
            marker = "o" if is_visited else "X"
            sz = 120 if is_visited else 90
            z = getattr(node, "z", 0)
            edge = "white" if is_3d else "black"
            if is_3d:
                ax.scatter(node.x, node.y, z, c=colour, s=sz, marker=marker,
                           edgecolors=edge, linewidths=0.5,
                           depthshade=True, zorder=5)
                ax.text(node.x, node.y, z + 4, str(node.id),
                        fontsize=7, color="#E6EDF3", ha="center",
                        fontweight="bold")
            else:
                ax.scatter(node.x, node.y, c=colour, s=sz, marker=marker,
                           edgecolors=edge, linewidths=0.5, zorder=5)
                ax.annotate(str(node.id), (node.x + 5, node.y + 5),
                            fontsize=7, color="gray")

        # Base station
        base = env.uav
        if is_3d:
            ax.scatter(base.x, base.y, 0, c=C["base"], s=300, marker="s",
                       edgecolors=C["base_edge"], linewidths=1.2,
                       zorder=10, label="Base Station")
        else:
            ax.scatter(base.x, base.y, c=C["base"], s=260, marker="s",
                       edgecolors=C["base_edge"], linewidths=1.0,
                       zorder=10, label="Base Station")

        # UAV trail
        if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
            if is_3d:
                PlotRenderer._draw_trail_3d(ax, env.uav_trail, lw=2.5)
                ax.plot([], [], [], color=C["trail"], linewidth=2.0,
                        label="UAV Trajectory")
            else:
                tx = [p[0] for p in env.uav_trail]
                ty = [p[1] for p in env.uav_trail]
                ax.plot(tx, ty, linewidth=1.8, alpha=0.7, color=C["trail"],
                        label="UAV Trajectory", zorder=3)

        if is_3d:
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        title_col = PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT
        ax.set_title("Mission Trajectory Summary",
                      fontsize=16, fontweight="bold", color=title_col)
        ax.legend(loc="upper right", framealpha=0.7,
                  facecolor="#161B22" if is_3d else "white",
                  labelcolor=PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT,
                  fontsize=10)
        PlotRenderer._save_dual(fig, save_dir, "trajectory_summary")

    # ═══════════════════════════════════════════════════════════
    #  9. DASHBOARD PANEL (2×3, 2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_dashboard_panel(results: dict, battery_hist: list,
                                visited_hist: list, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Performance Dashboard",
                      fontsize=17, fontweight="bold", y=0.98)

        # [0,0] Battery
        ax = axes[0, 0]
        ax.plot(battery_hist, color="#58A6FF", linewidth=2.0)
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.10, color="#58A6FF")
        ax.set_title("Battery Discharge", fontsize=12)
        ax.set_xlabel("Step"); ax.set_ylabel("J")

        # [0,1] Visited
        ax = axes[0, 1]
        ax.plot(visited_hist, color="#00FF87", linewidth=2.0)
        ax.fill_between(range(len(visited_hist)), visited_hist,
                        alpha=0.10, color="#00FF87")
        ax.set_title("Nodes Visited", fontsize=12)
        ax.set_xlabel("Step"); ax.set_ylabel("Count")

        # [0,2] Coverage pie
        ax = axes[0, 2]
        vis = results.get("nodes_visited", 0)
        total = results.get("total_nodes", 1)
        colors = [PlotRenderer.C["visited"], PlotRenderer.C["critical"]]
        wedges, texts, autotexts = ax.pie(
            [vis, total - vis], labels=["Visited", "Unvisited"],
            colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": 11})
        for t in autotexts:
            t.set_fontweight("bold")
        ax.set_title("Node Coverage", fontsize=12)

        # [1,0] DR%
        ax = axes[1, 0]
        dr = results.get("data_collection_rate_percent", 0)
        ax.bar(["DR%"], [dr], color="#FF6D00", width=0.45,
               edgecolor="#E65100", linewidth=1.2)
        ax.text(0, dr + 1, f"{dr:.1f}%", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 110); ax.set_title("Data Collection Rate", fontsize=12)
        ax.set_ylabel("%")

        # [1,1] AoI
        ax = axes[1, 1]
        aoi = results.get("average_aoi_s", 0)
        ax.bar(["Avg AoI"], [aoi], color="#E040FB", width=0.45,
               edgecolor="#AA00FF", linewidth=1.2)
        ax.text(0, aoi + 1, f"{aoi:.1f}s", ha="center", fontsize=12, fontweight="bold")
        ax.set_title("Mean Peak AoI", fontsize=12); ax.set_ylabel("Seconds")

        # [1,2] Energy/Node
        ax = axes[1, 2]
        epn = results.get("energy_per_node_J", 0)
        ax.bar(["J/node"], [epn], color="#00E5FF", width=0.45,
               edgecolor="#0097A7", linewidth=1.2)
        ax.text(0, epn + 0.3, f"{epn:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_title("Energy per Node", fontsize=12); ax.set_ylabel("Joules")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        PlotRenderer._save_dual(fig, save_dir, "dashboard_panel")

    # ═══════════════════════════════════════════════════════════
    #  10. 3D TRAJECTORY (multi-angle, dark)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_3d_trajectory(env, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_3d()
        C = PlotRenderer.C

        views = [
            ("isometric",  25, -45),
            ("top_down",   90, -90),
            ("side_view",   5, -90),
        ]
        for view_name, elev, azim in views:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            PlotRenderer._dark_fig(fig)

            # Nodes coloured by AoI
            for node in env.sensors:
                aoi = getattr(node, "aoi_timer", 0.0)
                colour = (C["visited"] if aoi < 50
                          else (C["rp"] if aoi < 200 else C["critical"]))
                ax.scatter(node.x, node.y, getattr(node, "z", 0),
                           c=colour, s=100, edgecolors="white",
                           linewidths=0.5, depthshade=True)

            # Base
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=250,
                       marker="s", edgecolors=C["base_edge"],
                       linewidths=1.0, zorder=10)

            PlotRenderer._draw_obstacles_3d(ax, env.obstacles, z_ceil=35, alpha=0.20)

            # Flight trail — altitude gradient
            if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
                PlotRenderer._draw_trail_3d(ax, env.uav_trail, lw=2.5)

            PlotRenderer._setup_3d_ax(ax, env.width, env.height,
                                       z_max=120, elev=elev, azim=azim)
            ax.set_title(
                f"3D Trajectory — {view_name.replace('_', ' ').title()}",
                fontsize=15, fontweight="bold", color=PlotRenderer._DARK_TXT)

            PlotRenderer._save_dual(fig, save_dir,
                                     f"trajectory_3d_{view_name}")

    # ═══════════════════════════════════════════════════════════
    #  11. SPEED HEATMAP (2D light — LineCollection)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_trajectory_heatmap(env, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        if not hasattr(env, "uav_trail") or len(env.uav_trail) < 3:
            return

        trail = env.uav_trail
        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]

        speeds = []
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]
            speeds.append((dx ** 2 + dy ** 2) ** 0.5)
        speeds = np.array(speeds)
        norm_speeds = speeds / max(speeds.max(), 1e-6)

        fig, ax = plt.subplots(figsize=(12, 8))
        PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.15)

        for node in env.sensors:
            ax.scatter(node.x, node.y, c=PlotRenderer.C["idle"],
                       s=40, edgecolors="black", linewidths=0.4, zorder=3)

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="magma", linewidths=2.5, zorder=5)
        lc.set_array(norm_speeds)
        ax.add_collection(lc)

        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label("Relative Speed (0 = hover, 1 = max transit)", fontsize=11)

        PlotRenderer._setup_2d_ax(ax, env.width, env.height)
        ax.set_title("Trajectory Speed Heatmap",
                      fontsize=15, fontweight="bold")
        PlotRenderer._save_dual(fig, save_dir, "trajectory_heatmap")

    # ═══════════════════════════════════════════════════════════
    #  12. AoI TIMELINE (2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_aoi_timeline(aoi_history: dict, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        if not aoi_history:
            return

        peaks = {nid: max(vals) for nid, vals in aoi_history.items()}
        top_nodes = sorted(peaks, key=peaks.get, reverse=True)[:10]

        neon_colors = ["#00E5FF", "#00FF87", "#FF6B6B", "#FFD740",
                       "#E040FB", "#7C4DFF", "#FF6D00", "#00BFA5",
                       "#64DD17", "#F50057"]

        fig, ax = plt.subplots(figsize=(14, 6))
        for i, nid in enumerate(top_nodes):
            ax.plot(aoi_history[nid], label=f"Node {nid}",
                    color=neon_colors[i % len(neon_colors)],
                    linewidth=1.5, alpha=0.90)

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Age of Information (s)", fontsize=12)
        ax.set_title("Per-Node AoI Timeline (Top 10 Peak)",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", ncol=2, fontsize=9, framealpha=0.8)
        PlotRenderer._save_dual(fig, save_dir, "aoi_timeline")

    # ═══════════════════════════════════════════════════════════
    #  13. BATTERY + REPLAN OVERLAY (2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_battery_with_replans(battery_hist: list, replan_steps: list,
                                     save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(battery_hist, color="#58A6FF", linewidth=2.0, label="Battery (J)")
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.10, color="#58A6FF")

        for rs in replan_steps:
            if 0 <= rs < len(battery_hist):
                ax.axvline(x=rs, color="#FF6B6B", linestyle="--",
                           linewidth=1.0, alpha=0.7)
        if replan_steps:
            ax.axvline(x=replan_steps[0], color="#FF6B6B", linestyle="--",
                       linewidth=1.0, alpha=0.7, label="Replan Event")

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Battery (J)", fontsize=12)
        ax.set_title("Battery Discharge with Replan Events",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        PlotRenderer._save_dual(fig, save_dir, "battery_replans")

    # ═══════════════════════════════════════════════════════════
    #  14. RUN COMPARISON (2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_run_comparison(run_a: dict, run_b: dict, save_dir: str,
                               label_a: str = "Run A",
                               label_b: str = "Run B"):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        metrics = [
            ("Coverage (%)", "coverage_ratio_percent"),
            ("DR (%)", "data_collection_rate_percent"),
            ("Nodes Visited", "nodes_visited"),
            ("Avg AoI (s)", "average_aoi_s"),
            ("Energy/Node (J)", "energy_per_node_J"),
            ("Steps", "steps"),
        ]
        labels = [m[0] for m in metrics]
        vals_a = [run_a.get(m[1], 0) for m in metrics]
        vals_b = [run_b.get(m[1], 0) for m in metrics]
        x = np.arange(len(labels))
        w = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        ba = ax.bar(x - w / 2, vals_a, w, label=label_a,
                    color="#58A6FF", alpha=0.90, edgecolor="#1F6FEB",
                    linewidth=1.0)
        bb = ax.bar(x + w / 2, vals_b, w, label=label_b,
                    color="#FF6D00", alpha=0.90, edgecolor="#E65100",
                    linewidth=1.0)

        for bar in list(ba) + list(bb):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title("Run Comparison", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        PlotRenderer._save_dual(fig, save_dir, "run_comparison")

    # ═══════════════════════════════════════════════════════════
    #  15. SEMANTIC CLUSTERING (3D dark)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_semantic_clustering(env, active_labels, active_centroids,
                                    save_dir: str):
        from scipy.spatial import ConvexHull
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        cluster_colors = ["#00E5FF", "#00FF87", "#FF6B6B", "#FFD740",
                          "#E040FB", "#7C4DFF", "#FF6D00", "#00BFA5",
                          "#64DD17", "#F50057"]
        unique_labels = sorted(set(active_labels))
        nodes = env.sensors

        if is_3d:
            PlotRenderer._set_style_3d()
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            PlotRenderer._dark_fig(fig)
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles, alpha=0.12)
        else:
            PlotRenderer._set_style_2d()
            fig, ax = plt.subplots(figsize=(12, 9))
            PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.15)

        for label in unique_labels:
            mask = [i for i, l in enumerate(active_labels)
                    if l == label and i < len(nodes)]
            if not mask:
                continue
            colour = "#8B949E" if label == -1 else cluster_colors[label % len(cluster_colors)]
            lbl = f"Cluster {label}" if label != -1 else "Noise"
            xs = [nodes[i].x for i in mask]
            ys = [nodes[i].y for i in mask]
            zs = [getattr(nodes[i], "z", 0) for i in mask]

            if is_3d:
                ax.scatter(xs, ys, zs, c=[colour] * len(xs), s=110,
                           edgecolors="white", linewidths=0.5,
                           depthshade=True, zorder=3, label=lbl)
            else:
                ax.scatter(xs, ys, c=[colour] * len(xs), s=80,
                           edgecolors="black", linewidths=0.5,
                           zorder=3, label=lbl)
                if label != -1 and len(mask) >= 3:
                    pts = np.array(list(zip(xs, ys)))
                    try:
                        hull = ConvexHull(pts)
                        hp = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(pts[hp, 0], pts[hp, 1], color=colour,
                                linewidth=1.5, linestyle="--", alpha=0.6)
                    except Exception:
                        pass

        # Centroids
        for idx, centroid in enumerate(active_centroids):
            if np.all(np.array(centroid) == 0):
                continue
            cz = centroid[2] if len(centroid) > 2 else 0
            colour = cluster_colors[idx % len(cluster_colors)]
            if is_3d:
                ax.scatter(centroid[0], centroid[1], cz, marker="*", s=350,
                           facecolors=colour, edgecolors="white",
                           linewidths=1.0, zorder=5)
            else:
                ax.scatter(centroid[0], centroid[1], marker="*", s=300,
                           facecolors=colour, edgecolors="black",
                           linewidths=1.0, zorder=5)
                ax.annotate(f"C{idx}", (centroid[0], centroid[1]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=8, fontweight="bold")

        # Base station
        base = env.uav
        if is_3d:
            ax.scatter(base.x, base.y, 0, c=C["base"], s=250, marker="s",
                       edgecolors=C["base_edge"], linewidths=1.0,
                       zorder=6, label="Base Station")
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            ax.scatter(base.x, base.y, c=C["base"], s=220, marker="s",
                       edgecolors=C["base_edge"], linewidths=1.0,
                       zorder=6, label="Base Station")
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        title_col = PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT
        ax.set_title("Semantic Clustering — Geographic Distribution",
                      fontsize=16, fontweight="bold", color=title_col)
        ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.7,
                  facecolor="#161B22" if is_3d else "white",
                  labelcolor=PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT)
        PlotRenderer._save_dual(fig, save_dir, "semantic_clustering_geo")

    # ═══════════════════════════════════════════════════════════
    #  16. CLUSTERING PCA SPACE (2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_clustering_pca_space(reduced_features, active_labels,
                                     save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        arr = np.array(reduced_features)
        if arr.ndim < 2 or arr.shape[1] < 2:
            return

        cluster_colors = ["#00E5FF", "#00FF87", "#FF6B6B", "#FFD740",
                          "#E040FB", "#7C4DFF", "#FF6D00", "#00BFA5",
                          "#64DD17", "#F50057"]
        unique_labels = sorted(set(active_labels))

        fig, ax = plt.subplots(figsize=(10, 7))
        for label in unique_labels:
            mask = [i for i, l in enumerate(active_labels) if l == label]
            colour = "#8B949E" if label == -1 else cluster_colors[label % len(cluster_colors)]
            lbl = f"Cluster {label}" if label != -1 else "Noise / Outliers"
            ax.scatter(arr[mask, 0], arr[mask, 1],
                       c=[colour] * len(mask), s=80,
                       edgecolors="black", linewidths=0.4, alpha=0.90,
                       label=lbl)

        ax.set_xlabel("PCA Component 1", fontsize=12)
        ax.set_ylabel("PCA Component 2", fontsize=12)
        ax.set_title("Semantic Clustering — PCA Latent Space",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.8)
        PlotRenderer._save_dual(fig, save_dir, "semantic_clustering_pca")

    # ═══════════════════════════════════════════════════════════
    #  17. ROUTING PIPELINE (3-panel, 3D dark)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_routing_pipeline(env, rp_nodes, rp_member_map,
                                 route_sequence, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        if is_3d:
            PlotRenderer._set_style_3d()
        else:
            PlotRenderer._set_style_2d()

        fig = plt.figure(figsize=(22, 7) if is_3d else (18, 6))
        if is_3d:
            PlotRenderer._dark_fig(fig)
        fig.suptitle("Routing Pipeline Compression",
                      fontsize=16, fontweight="bold",
                      color=PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT)

        all_nodes = env.sensors
        projections = {"projection": "3d"} if is_3d else {}

        def _draw_obs(ax):
            if is_3d:
                PlotRenderer._draw_obstacles_3d(ax, env.obstacles, alpha=0.12)
            else:
                PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.15)

        def _setup(ax):
            if is_3d:
                PlotRenderer._setup_3d_ax(ax, env.width, env.height)
            else:
                PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        xs = [n.x for n in all_nodes]
        ys = [n.y for n in all_nodes]
        zs = [getattr(n, "z", 0) for n in all_nodes]

        # Panel 1: raw nodes
        ax = fig.add_subplot(131, **projections)
        _draw_obs(ax)
        if is_3d:
            ax.scatter(xs, ys, zs, c="#58A6FF", s=70, edgecolors="white",
                       linewidths=0.4, depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=200,
                       marker="s", edgecolors=C["base_edge"], linewidths=1.0)
        else:
            ax.scatter(xs, ys, c="#58A6FF", s=50, edgecolors="black",
                       linewidths=0.4)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=180,
                       marker="s", edgecolors=C["base_edge"], linewidths=0.8)
        _setup(ax)
        tc = PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT
        ax.set_title(f"(a) Raw Deployment [{len(all_nodes)} nodes]",
                      fontsize=11, color=tc)

        # Panel 2: RP compression
        ax = fig.add_subplot(132, **projections)
        _draw_obs(ax)
        rp_list = rp_nodes or []
        if is_3d:
            ax.scatter(xs, ys, zs, c="#8B949E", s=18, alpha=0.4)
            for rp in rp_list:
                theta = np.linspace(0, 2 * np.pi, 40)
                rx = rp.x + 120 * np.cos(theta)
                ry = rp.y + 120 * np.sin(theta)
                ax.plot(rx, ry, np.zeros_like(rx), color=C["rp"],
                        alpha=0.25, linewidth=1.0)
            rp_xs = [n.x for n in rp_list]
            rp_ys = [n.y for n in rp_list]
            rp_zs = [getattr(n, "z", 0) for n in rp_list]
            ax.scatter(rp_xs, rp_ys, rp_zs, c=C["rp"], s=140, marker="D",
                       edgecolors="white", linewidths=0.7, zorder=4,
                       depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=200,
                       marker="s", edgecolors=C["base_edge"], linewidths=1.0)
        else:
            ax.scatter(xs, ys, c="#8B949E", s=18, edgecolors="none", alpha=0.5)
            for rp in rp_list:
                circle = plt.Circle((rp.x, rp.y), 120, color=C["rp"],
                                     alpha=0.15)
                ax.add_patch(circle)
            ax.scatter([n.x for n in rp_list], [n.y for n in rp_list],
                       c=C["rp"], s=110, marker="D",
                       edgecolors="black", linewidths=0.6, zorder=4)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=180,
                       marker="s", edgecolors=C["base_edge"], linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(b) RP Compression [{len(rp_list)} RPs]",
                      fontsize=11, color=tc)

        # Panel 3: optimised route
        ax = fig.add_subplot(133, **projections)
        _draw_obs(ax)
        seq = route_sequence or []
        if is_3d:
            ax.scatter(xs, ys, zs, c="#8B949E", s=18, alpha=0.35)
            if len(seq) >= 2:
                rxs = [env.uav.x] + [n.x for n in seq]
                rys = [env.uav.y] + [n.y for n in seq]
                rzs = [0] + [getattr(n, "z", 0) for n in seq]
                ax.plot(rxs, rys, rzs, color=C["visited"],
                        linewidth=2.0, zorder=3)
            sq_xs = [n.x for n in seq]
            sq_ys = [n.y for n in seq]
            sq_zs = [getattr(n, "z", 0) for n in seq]
            ax.scatter(sq_xs, sq_ys, sq_zs, c=C["visited"], s=100,
                       edgecolors="white", linewidths=0.5, zorder=4,
                       depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=200,
                       marker="s", edgecolors=C["base_edge"], linewidths=1.0)
        else:
            ax.scatter(xs, ys, c="#8B949E", s=18, edgecolors="none", alpha=0.35)
            if len(seq) >= 2:
                rxs = [env.uav.x] + [n.x for n in seq]
                rys = [env.uav.y] + [n.y for n in seq]
                ax.plot(rxs, rys, color=C["visited"], linewidth=1.8, zorder=3)
                for i, (x, y) in enumerate(zip(rxs[1:], rys[1:]), 1):
                    ax.annotate(str(i), (x, y), textcoords="offset points",
                                xytext=(5, 5), fontsize=7, color="#00C853",
                                fontweight="bold")
            ax.scatter([n.x for n in seq], [n.y for n in seq],
                       c=C["visited"], s=80, edgecolors="black",
                       linewidths=0.5, zorder=4)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=180,
                       marker="s", edgecolors=C["base_edge"], linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(c) Optimised Route [{len(seq)} waypoints]",
                      fontsize=11, color=tc)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        PlotRenderer._save_dual(fig, save_dir, "routing_pipeline")

    # ═══════════════════════════════════════════════════════════
    #  18. COMMUNICATION QUALITY (2D light, 2-panel)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_communication_quality(nodes, uav_trail, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        sensor_nodes = [n for n in nodes if n.id != 0]
        if not sensor_nodes:
            return

        if uav_trail and len(uav_trail) > 0:
            mean_x = float(np.mean([p[0] for p in uav_trail]))
            mean_y = float(np.mean([p[1] for p in uav_trail]))
        else:
            mean_x, mean_y = 400.0, 300.0

        distances = [((n.x - mean_x) ** 2 + (n.y - mean_y) ** 2) ** 0.5
                     for n in sensor_nodes]
        data_rates = [getattr(n, "current_rate_mbps", 0.0) for n in sensor_nodes]
        buffer_fill = [
            min(100.0, getattr(n, "buffer_fill_mbits", 0.0) /
                max(getattr(n, "buffer_cap_mbits", 1.0), 1e-6) * 100)
            for n in sensor_nodes
        ]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Communication Quality Analysis",
                      fontsize=15, fontweight="bold")

        # Panel 1: link rate vs distance
        ax = axes[0]
        ax.scatter(distances, data_rates, c="#58A6FF", s=70,
                   edgecolors="#1F6FEB", linewidths=0.5, alpha=0.80)
        ax.set_xlabel("UAV–Node Distance (m)", fontsize=11)
        ax.set_ylabel("Data Rate (Mbps)", fontsize=11)
        ax.set_title("(a) Link Rate vs Distance", fontsize=12)
        d_sorted = sorted(distances)
        if d_sorted and d_sorted[-1] > 0:
            trend_d = np.linspace(max(1, d_sorted[0]), d_sorted[-1], 200)
            max_rate = max(data_rates) if data_rates else 1.0
            scale = max_rate * (min(d_sorted) ** 2) if d_sorted[0] > 0 else max_rate
            trend_r = scale / (trend_d ** 2 + 1e-6)
            ax.plot(trend_d, trend_r, color="#FF6B6B", linewidth=1.5,
                    linestyle="--", label=r"$1/d^2$ reference")
            ax.legend(fontsize=9)

        # Panel 2: buffer histogram
        ax = axes[1]
        ax.hist(buffer_fill, bins=15, facecolor="#FF6D00",
                edgecolor="#E65100", linewidth=0.8, alpha=0.85)
        ax.set_xlabel("Buffer Occupancy (%)", fontsize=11)
        ax.set_ylabel("Node Count", fontsize=11)
        ax.set_title("(b) Buffer Fill Distribution", fontsize=12)
        ax.axvline(x=80, color="#FF6B6B", linestyle="--",
                   linewidth=1.5, label="80% threshold")
        ax.legend(fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        PlotRenderer._save_dual(fig, save_dir, "communication_quality")

    # ═══════════════════════════════════════════════════════════
    #  19. MISSION PROGRESS COMBINED (2×2, 2D light)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_mission_progress_combined(visited_hist: list,
                                          battery_hist: list,
                                          data_hist: list,
                                          aoi_mean_hist: list,
                                          replan_steps: list,
                                          save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style_2d()

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        fig.suptitle("Mission Progress Overview",
                      fontsize=16, fontweight="bold")

        series = [
            (axes[0, 0], visited_hist,  "#00FF87", "Nodes Visited",  "Count"),
            (axes[0, 1], battery_hist,  "#58A6FF", "Battery (J)",    "Joules"),
            (axes[1, 0], data_hist,     "#FF6D00", "Data (Mbits)",   "Mbits"),
            (axes[1, 1], aoi_mean_hist, "#E040FB", "Mean AoI",       "Steps"),
        ]
        for ax, hist, colour, title, ylabel in series:
            if hist:
                ax.plot(hist, color=colour, linewidth=2.0)
                ax.fill_between(range(len(hist)), hist,
                                alpha=0.10, color=colour)
            ax.set_title(title, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            for rs in replan_steps:
                if hist and 0 <= rs < len(hist):
                    ax.axvline(x=rs, color="#FF6B6B", linestyle="--",
                               linewidth=0.8, alpha=0.6)

        axes[1, 0].set_xlabel("Step", fontsize=11)
        axes[1, 1].set_xlabel("Step", fontsize=11)
        if replan_steps:
            axes[0, 0].axvline(x=-1, color="#FF6B6B", linestyle="--",
                               linewidth=1.0, label="Replan")
            axes[0, 0].legend(fontsize=9, loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        PlotRenderer._save_dual(fig, save_dir, "mission_progress_combined")

    # ═══════════════════════════════════════════════════════════
    #  20. RENDEZVOUS COMPRESSION (3D dark, before/after)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_rendezvous_compression(env, all_nodes, rp_nodes,
                                       rp_member_map, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        if is_3d:
            PlotRenderer._set_style_3d()
        else:
            PlotRenderer._set_style_2d()

        member_colors = ["#00E5FF", "#00FF87", "#FF6B6B", "#FFD740",
                         "#E040FB", "#7C4DFF", "#FF6D00", "#00BFA5",
                         "#64DD17", "#F50057", "#00B0FF", "#76FF03",
                         "#FF4081", "#FFAB00", "#536DFE", "#69F0AE",
                         "#FF6E40", "#40C4FF", "#B2FF59", "#EA80FC"]
        rp_list = rp_nodes or []
        rp_map = rp_member_map or {}

        node_colour = {}
        for rp_idx, (rp, members) in enumerate(rp_map.items()):
            for m in members:
                node_colour[m] = member_colors[rp_idx % len(member_colors)]

        projections = {"projection": "3d"} if is_3d else {}
        fig = plt.figure(figsize=(18, 8) if is_3d else (16, 7))
        if is_3d:
            PlotRenderer._dark_fig(fig)
        fig.suptitle(
            f"Rendezvous Point Compression  "
            f"({len(all_nodes)} nodes → {len(rp_list)} RPs, "
            f"ratio = {len(all_nodes) / max(len(rp_list), 1):.1f}×)",
            fontsize=15, fontweight="bold",
            color=PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT)

        def _draw_obs(ax):
            if is_3d:
                PlotRenderer._draw_obstacles_3d(ax, env.obstacles, alpha=0.12)
            else:
                PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.15)

        def _setup(ax):
            if is_3d:
                PlotRenderer._setup_3d_ax(ax, env.width, env.height)
            else:
                PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        tc = PlotRenderer._DARK_TXT if is_3d else PlotRenderer._LIGHT_TXT

        # Left: all nodes coloured by RP membership
        ax = fig.add_subplot(121, **projections)
        _draw_obs(ax)
        for n in all_nodes:
            col = node_colour.get(n.id, "#8B949E")
            z = getattr(n, "z", 0)
            if is_3d:
                ax.scatter(n.x, n.y, z, c=[col], s=70,
                           edgecolors="white", linewidths=0.4, depthshade=True)
            else:
                ax.scatter(n.x, n.y, c=[col], s=55,
                           edgecolors="black", linewidths=0.4)
        base_kw = {"c": C["base"], "s": 250, "marker": "s",
                   "edgecolors": C["base_edge"], "linewidths": 1.0}
        if is_3d:
            ax.scatter(env.uav.x, env.uav.y, 0, **base_kw)
        else:
            ax.scatter(env.uav.x, env.uav.y, **base_kw)
        _setup(ax)
        ax.set_title(f"(a) All Nodes [{len(all_nodes)}]",
                      fontsize=12, color=tc)

        # Right: RP waypoints with coverage circles
        ax = fig.add_subplot(122, **projections)
        _draw_obs(ax)
        all_xs = [n.x for n in all_nodes]
        all_ys = [n.y for n in all_nodes]
        all_zs = [getattr(n, "z", 0) for n in all_nodes]
        if is_3d:
            ax.scatter(all_xs, all_ys, all_zs, c="#8B949E", s=14, alpha=0.40)
            for rp_idx, rp in enumerate(rp_list):
                colour = member_colors[rp_idx % len(member_colors)]
                theta = np.linspace(0, 2 * np.pi, 40)
                rx = rp.x + 120 * np.cos(theta)
                ry = rp.y + 120 * np.sin(theta)
                ax.plot(rx, ry, np.zeros_like(rx), color=colour,
                        alpha=0.25, linewidth=1.0)
                ax.scatter(rp.x, rp.y, getattr(rp, "z", 0),
                           c=[colour], s=160, marker="D",
                           edgecolors="white", linewidths=0.7, zorder=4,
                           depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, **base_kw)
        else:
            ax.scatter(all_xs, all_ys, c="#8B949E", s=14,
                       edgecolors="none", alpha=0.40)
            for rp_idx, rp in enumerate(rp_list):
                colour = member_colors[rp_idx % len(member_colors)]
                circle = plt.Circle((rp.x, rp.y), 120, color=colour,
                                     alpha=0.15)
                ax.add_patch(circle)
                ax.scatter(rp.x, rp.y, c=[colour], s=130, marker="D",
                           edgecolors="black", linewidths=0.6, zorder=4)
            ax.scatter(env.uav.x, env.uav.y, **base_kw)
        _setup(ax)
        ax.set_title(f"(b) Rendezvous Points [{len(rp_list)}]",
                      fontsize=12, color=tc)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        PlotRenderer._save_dual(fig, save_dir, "rendezvous_compression")
