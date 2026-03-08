"""
Plot Renderer — 3D-First Professional Visualization Suite
==========================================================
Redesigned for publication-quality IEEE / thesis-grade figures with
modern 3D-default rendering.  All spatial / environment plots render
in 3D by default.  Toggle to 2D via  FeatureToggles.RENDER_MODE = "2D".
Analytical metric charts (radar, dashboard, timeline) stay 2D.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
import numpy as np
import os
from config.feature_toggles import FeatureToggles


class PlotRenderer:
    """3D-first plot renderer for UAV data-collection simulation."""

    # ── Modern Material-Design Colour Palette ────────────────
    C = {
        "visited":    "#00E676",  # bright green
        "waiting":    "#42A5F5",  # blue
        "idle":       "#90A4AE",  # blue-grey
        "critical":   "#FF5252",  # red accent
        "uav":        "#00BCD4",  # cyan
        "uav_edge":   "#FFFFFF",
        "base":       "#37474F",  # dark blue-grey
        "obstacle":   "#EF5350",  # red
        "obs_edge":   "#B71C1C",  # dark red
        "risk":       "#FFA726",  # orange
        "risk_edge":  "#E65100",
        "trail":      "#1565C0",  # indigo blue
        "trail_hot":  "#E53935",  # hot red (high altitude)
        "rp":         "#FF9800",  # amber
        "cluster":    "#AB47BC",  # purple
    }

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
    def _set_style():
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "#FFFFFF",
        })

    @staticmethod
    def _save_dual(fig, save_dir, basename):
        fig.savefig(os.path.join(save_dir, f"{basename}.png"),
                    dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(os.path.join(save_dir, f"{basename}.pdf"),
                    format="pdf", bbox_inches="tight")
        plt.close(fig)

    # ── 3D building blocks ───────────────────────────────────
    @staticmethod
    def _draw_obstacles_3d(ax, obstacles, z_ceil=25, alpha=0.18):
        C = PlotRenderer.C
        for obs in obstacles:
            v = [
                [obs.x1, obs.y1, 0],      [obs.x2, obs.y1, 0],
                [obs.x2, obs.y2, 0],      [obs.x1, obs.y2, 0],
                [obs.x1, obs.y1, z_ceil], [obs.x2, obs.y1, z_ceil],
                [obs.x2, obs.y2, z_ceil], [obs.x1, obs.y2, z_ceil],
            ]
            faces = [
                [v[0], v[1], v[2], v[3]],  # bottom
                [v[4], v[5], v[6], v[7]],  # top
                [v[0], v[1], v[5], v[4]],  # front
                [v[2], v[3], v[7], v[6]],  # back
                [v[1], v[2], v[6], v[5]],  # right
                [v[0], v[3], v[7], v[4]],  # left
            ]
            ax.add_collection3d(Poly3DCollection(
                faces, alpha=alpha, facecolors=C["obstacle"],
                edgecolors=C["obs_edge"], linewidths=0.4))

    @staticmethod
    def _draw_obstacles_2d(ax, obstacles, alpha=0.22):
        C = PlotRenderer.C
        for obs in obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                facecolor=C["obstacle"], alpha=alpha,
                edgecolor=C["obs_edge"], linewidth=0.8)
            ax.add_patch(rect)

    @staticmethod
    def _draw_risk_3d(ax, risk_zones, z_ceil=18):
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
                faces, alpha=0.12, facecolors=C["risk"],
                edgecolors=C["risk_edge"], linewidths=0.3))

    @staticmethod
    def _draw_risk_2d(ax, risk_zones):
        C = PlotRenderer.C
        for rz in risk_zones:
            rect = plt.Rectangle(
                (rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1,
                facecolor=C["risk"], alpha=0.15,
                edgecolor=C["risk_edge"], linewidth=0.8)
            ax.add_patch(rect)

    @staticmethod
    def _setup_3d_ax(ax, w, h, z_max=60, elev=28, azim=-50):
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_zlim(0, z_max)
        ax.set_xlabel("X (m)", labelpad=8)
        ax.set_ylabel("Y (m)", labelpad=8)
        ax.set_zlabel("Altitude (m)", labelpad=6)
        ax.view_init(elev=elev, azim=azim)
        # Transparent panes
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#E0E0E0")
        ax.grid(True, alpha=0.15)

    @staticmethod
    def _setup_2d_ax(ax, w, h):
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")

    # ═══════════════════════════════════════════════════════════
    #  1. ENVIRONMENT OVERVIEW (initial snapshot)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_environment(env, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
        is_3d = PlotRenderer._use_3d()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

        # Nodes
        for node in env.sensors:
            z = getattr(node, "z", 0)
            if is_3d:
                ax.scatter(node.x, node.y, z, c=PlotRenderer.C["waiting"],
                           s=70, marker="o", edgecolors="black", linewidths=0.4,
                           depthshade=True, zorder=3)
                ax.text(node.x, node.y, z + 3, str(node.id),
                        fontsize=6, color="#546E7A", ha="center")
            else:
                ax.scatter(node.x, node.y, c=PlotRenderer.C["waiting"],
                           s=70, marker="o", edgecolors="black", linewidths=0.4, zorder=3)
                ax.annotate(str(node.id), (node.x + 4, node.y + 4),
                            fontsize=6, color="#546E7A")

        # UAV / base
        if hasattr(env, "uav") and env.uav:
            bx, by = env.uav.x, env.uav.y
            if is_3d:
                ax.scatter(bx, by, 0, c=PlotRenderer.C["base"], s=160,
                           marker="s", edgecolors="white", linewidths=0.8,
                           zorder=10, label="Base Station")
            else:
                ax.scatter(bx, by, c=PlotRenderer.C["base"], s=160,
                           marker="s", edgecolors="white", linewidths=0.8,
                           zorder=10, label="Base Station")

        # Obstacles & risk zones
        if is_3d:
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles)
            PlotRenderer._draw_risk_3d(ax, env.risk_zones)
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            PlotRenderer._draw_obstacles_2d(ax, env.obstacles)
            PlotRenderer._draw_risk_2d(ax, env.risk_zones)
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        ax.set_title("UAV Mission Environment", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.8)

        PlotRenderer._save_dual(fig, save_dir, "environment")

    # ═══════════════════════════════════════════════════════════
    #  2. ENERGY & VISIT BAR CHARTS
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_energy_plots(visited, energy_consumed, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(["Visited Nodes"], [visited], color="#00E676",
                     edgecolor="black", linewidth=0.5, width=0.4)
        axes[0].set_title("Nodes Visited", fontweight="bold")
        axes[0].set_ylabel("Count")

        axes[1].bar(["Energy (J)"], [energy_consumed], color="#FF7043",
                     edgecolor="black", linewidth=0.5, width=0.4)
        axes[1].set_title("Energy Consumed", fontweight="bold")
        axes[1].set_ylabel("Joules")

        plt.tight_layout()
        PlotRenderer._save_dual(fig, save_dir, "energy_visited")

    # ═══════════════════════════════════════════════════════════
    #  3. METRICS SNAPSHOT
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_metrics_snapshot(completion_pct, efficiency, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Completion %", "Efficiency"], [completion_pct, efficiency],
                       color=["#42A5F5", "#66BB6A"], edgecolor="black",
                       linewidth=0.5, width=0.45)
        ax.set_title("Mission Metrics Snapshot", fontweight="bold")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        PlotRenderer._save_dual(fig, save_dir, "metrics_snapshot")

    # ═══════════════════════════════════════════════════════════
    #  4. TIME-SERIES PLOTS
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_time_series(visited_hist, battery_hist, replan_hist, save_dir):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        fig.suptitle("Mission Time-Series", fontsize=14, fontweight="bold", y=1.02)

        ax = axes[0]
        ax.plot(visited_hist, color="#00E676", linewidth=1.3)
        ax.fill_between(range(len(visited_hist)), visited_hist,
                        alpha=0.1, color="#00E676")
        ax.set_title("Visited Nodes"); ax.set_xlabel("Step"); ax.set_ylabel("Count")

        ax = axes[1]
        ax.plot(battery_hist, color="#42A5F5", linewidth=1.3)
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.1, color="#42A5F5")
        ax.set_title("Battery Level"); ax.set_xlabel("Step"); ax.set_ylabel("J")

        ax = axes[2]
        ax.plot(replan_hist, color="#FF7043", linewidth=1.3)
        ax.set_title("Cumulative Replans"); ax.set_xlabel("Step"); ax.set_ylabel("Count")

        plt.tight_layout()
        PlotRenderer._save_dual(fig, save_dir, "time_series")

    # ═══════════════════════════════════════════════════════════
    #  5. ENVIRONMENT FRAME (per-step animation frame)
    # ═══════════════════════════════════════════════════════════
    @staticmethod
    def render_environment_frame(env, save_dir, step, mission=None):
        PlotRenderer._ensure_dir(save_dir)
        is_3d = PlotRenderer._use_3d()

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

        visited_ids = mission.visited if mission else set()
        C = PlotRenderer.C

        # ── Nodes (colour by visit status) ──
        for node in env.sensors:
            if node.id in visited_ids:
                colour, sz = C["visited"], 80
            else:
                buf_ratio = node.current_buffer / max(node.buffer_capacity, 1e-6)
                colour = C["waiting"] if buf_ratio > 0.5 else C["idle"]
                sz = 55

            if is_3d:
                ax.scatter(node.x, node.y, getattr(node, "z", 0),
                           c=colour, s=sz, marker="o",
                           edgecolors="black", linewidths=0.4, depthshade=True)
            else:
                ax.scatter(node.x, node.y, c=colour, s=sz, marker="o",
                           edgecolors="black", linewidths=0.4, zorder=4)

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
                tx = [p[0] for p in env.uav_trail]
                ty = [p[1] for p in env.uav_trail]
                if is_3d:
                    tz = [p[2] if len(p) > 2 else 50.0 for p in env.uav_trail]
                    # Altitude-gradient colouring
                    for i in range(len(tx) - 1):
                        frac = min(1.0, tz[i] / 120.0)
                        seg_c = (frac, 0.35, 1.0 - frac)
                        ax.plot(tx[i:i+2], ty[i:i+2], tz[i:i+2],
                                color=seg_c, linewidth=1.2, alpha=0.85)
                else:
                    ax.plot(tx, ty, linewidth=1.2, alpha=0.8,
                            color=C["trail"], zorder=5)

            # UAV marker (triangle)
            if is_3d:
                ax.scatter(env.uav.x, env.uav.y, env.uav.z,
                           s=160, marker="^", c=C["uav"],
                           edgecolors=C["uav_edge"], linewidths=1.0, zorder=10)
            else:
                ax.scatter(env.uav.x, env.uav.y,
                           s=160, marker="^", c=C["uav"],
                           edgecolors=C["uav_edge"], linewidths=1.0, zorder=10)

            # Target connection line (2D only)
            if not is_3d and mission and mission.current_target:
                tgt = mission.current_target
                ax.plot([env.uav.x, tgt.x], [env.uav.y, tgt.y],
                        linestyle="--", color=C["trail_hot"],
                        linewidth=0.8, alpha=0.6, zorder=6)

        # ── Axes setup ──
        if is_3d:
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        # ── HUD title ──
        total_nodes = len(env.sensors)
        n_visited = len(visited_ids)
        battery_pct = 0.0
        if hasattr(env, "uav"):
            from config.config import Config
            battery_pct = (env.uav.current_battery / Config.BATTERY_CAPACITY) * 100
        ax.set_title(
            f"Step {step}  |  Battery: {battery_pct:.0f}%  |  "
            f"Visited: {n_visited}/{total_nodes}",
            fontsize=11, fontweight="bold")

        # ── Replan flash ──
        if hasattr(env, "temporal_engine"):
            flash = env.temporal_engine.consume_replan_flash()
            if flash:
                ax.set_facecolor((1.0, 0.88, 0.88))
                txt_fn = ax.text2D if is_3d else ax.text
                txt_fn(0.5, 0.92, "REPLAN", transform=ax.transAxes,
                       ha="center", fontsize=11, color="red", fontweight="bold")

        fig.savefig(os.path.join(save_dir, f"{step:04d}.png"), dpi=150)
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════
    #  IEEE-Grade Post-Run Visualisations
    # ═══════════════════════════════════════════════════════════

    # ── 6. Radar Chart (6 KPIs) ──────────────────────────────
    @staticmethod
    def render_radar_chart(results: dict, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

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

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="#42A5F5", alpha=0.25)
        ax.plot(angles, values, color="#1565C0", linewidth=2,
                marker="o", markersize=6)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Mission Performance Radar",
                      fontsize=14, fontweight="bold", pad=20)
        PlotRenderer._save_dual(fig, save_dir, "radar_chart")

    # ── 7. Node Energy Heatmap (3D scatter) ──────────────────
    @staticmethod
    def render_node_energy_heatmap(nodes: list, env_width: float,
                                    env_height: float, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
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

        fig = plt.figure(figsize=(10, 7))
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(xs, ys, zs, c=residuals, cmap="RdYlGn",
                            s=100, edgecolors="black", linewidths=0.5,
                            vmin=0, vmax=1, depthshade=True)
            PlotRenderer._setup_3d_ax(ax, env_width, env_height)
        else:
            ax = fig.add_subplot(111)
            sc = ax.scatter(xs, ys, c=residuals, cmap="RdYlGn", s=90,
                            edgecolors="black", linewidths=0.5, vmin=0, vmax=1)
            PlotRenderer._setup_2d_ax(ax, env_width, env_height)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.75 if is_3d else 1.0)
        cbar.set_label("Residual Battery Fraction")
        ax.set_title("IoT Node Residual Energy Map",
                      fontsize=13, fontweight="bold")
        PlotRenderer._save_dual(fig, save_dir, "node_energy_heatmap")

    # ── 8. Trajectory Summary (3D-first) ─────────────────────
    @staticmethod
    def render_trajectory_summary(env, visited_ids: set, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

        # Obstacles & risk zones
        if is_3d:
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles)
            PlotRenderer._draw_risk_3d(ax, env.risk_zones)
        else:
            PlotRenderer._draw_obstacles_2d(ax, env.obstacles)
            PlotRenderer._draw_risk_2d(ax, env.risk_zones)

        # Nodes — visited (green circle) vs unvisited (red X)
        for node in env.sensors:
            colour = C["visited"] if node.id in visited_ids else C["critical"]
            marker = "o" if node.id in visited_ids else "X"
            sz = 80 if node.id in visited_ids else 60
            z = getattr(node, "z", 0)
            if is_3d:
                ax.scatter(node.x, node.y, z, c=colour, s=sz, marker=marker,
                           edgecolors="black", linewidths=0.4,
                           depthshade=True, zorder=5)
                ax.text(node.x, node.y, z + 3, str(node.id),
                        fontsize=6, color="#546E7A", ha="center")
            else:
                ax.scatter(node.x, node.y, c=colour, s=sz, marker=marker,
                           edgecolors="black", linewidths=0.4, zorder=5)
                ax.annotate(str(node.id), (node.x + 4, node.y + 4),
                            fontsize=6, color="gray")

        # Base station
        base = env.uav
        if is_3d:
            ax.scatter(base.x, base.y, 0, c=C["base"], s=180, marker="s",
                       edgecolors="white", linewidths=0.8,
                       zorder=10, label="Base Station")
        else:
            ax.scatter(base.x, base.y, c=C["base"], s=180, marker="s",
                       edgecolors="white", linewidths=0.8,
                       zorder=10, label="Base Station")

        # UAV trail
        if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
            trail = env.uav_trail
            tx = [p[0] for p in trail]
            ty = [p[1] for p in trail]
            if is_3d:
                tz = [p[2] if len(p) > 2 else 50.0 for p in trail]
                for i in range(len(tx) - 1):
                    frac = min(1.0, tz[i] / 120.0)
                    seg_c = (frac, 0.35, 1.0 - frac)
                    ax.plot(tx[i:i+2], ty[i:i+2], tz[i:i+2],
                            color=seg_c, linewidth=1.0, alpha=0.8)
                ax.plot([], [], [], color=C["trail"], linewidth=1.0,
                        label="UAV Trajectory")  # legend entry
            else:
                ax.plot(tx, ty, linewidth=1.0, alpha=0.7, color=C["trail"],
                        label="UAV Trajectory", zorder=3)

        if is_3d:
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        ax.set_title("Mission Trajectory Summary",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.8)
        PlotRenderer._save_dual(fig, save_dir, "trajectory_summary")

    # ── 9. Dashboard Panel (2×3) ─────────────────────────────
    @staticmethod
    def render_dashboard_panel(results: dict, battery_hist: list,
                                visited_hist: list, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Performance Dashboard",
                      fontsize=15, fontweight="bold", y=0.98)

        # [0,0] Battery
        ax = axes[0, 0]
        ax.plot(battery_hist, color="#1565C0", linewidth=1.2)
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.08, color="#1565C0")
        ax.set_title("Battery Discharge"); ax.set_xlabel("Step"); ax.set_ylabel("J")

        # [0,1] Visited
        ax = axes[0, 1]
        ax.plot(visited_hist, color="#00E676", linewidth=1.2)
        ax.fill_between(range(len(visited_hist)), visited_hist,
                        alpha=0.08, color="#00E676")
        ax.set_title("Nodes Visited"); ax.set_xlabel("Step"); ax.set_ylabel("Count")

        # [0,2] Coverage pie
        ax = axes[0, 2]
        vis = results.get("nodes_visited", 0)
        total = results.get("total_nodes", 1)
        ax.pie([vis, total - vis], labels=["Visited", "Unvisited"],
               colors=[PlotRenderer.C["visited"], PlotRenderer.C["critical"]],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "black", "linewidth": 0.4})
        ax.set_title("Node Coverage")

        # [1,0] DR%
        ax = axes[1, 0]
        dr = results.get("data_collection_rate_percent", 0)
        ax.bar(["DR%"], [dr], color="#FF9800", width=0.4,
               edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 100); ax.set_title("Data Collection Rate"); ax.set_ylabel("%")

        # [1,1] AoI
        ax = axes[1, 1]
        aoi = results.get("average_aoi_s", 0)
        ax.bar(["Avg AoI"], [aoi], color="#AB47BC", width=0.4,
               edgecolor="black", linewidth=0.5)
        ax.set_title("Mean Peak AoI"); ax.set_ylabel("Seconds")

        # [1,2] Energy/Node
        ax = axes[1, 2]
        epn = results.get("energy_per_node_J", 0)
        ax.bar(["J/node"], [epn], color="#00BCD4", width=0.4,
               edgecolor="black", linewidth=0.5)
        ax.set_title("Energy per Node"); ax.set_ylabel("Joules")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        PlotRenderer._save_dual(fig, save_dir, "dashboard_panel")

    # ── 10. 3D Trajectory (multi-angle) ──────────────────────
    @staticmethod
    def render_3d_trajectory(env, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
        C = PlotRenderer.C

        views = [
            ("isometric",  28, -50),
            ("top_down",   90, -90),
            ("side_view",   0, -90),
        ]
        for view_name, elev, azim in views:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Nodes (coloured by AoI)
            for node in env.sensors:
                aoi = getattr(node, "aoi_timer", 0.0)
                colour = (C["visited"] if aoi < 50
                          else (C["rp"] if aoi < 200 else C["critical"]))
                ax.scatter(node.x, node.y, getattr(node, "z", 0),
                           c=colour, s=60, edgecolors="black",
                           linewidths=0.4, depthshade=True)

            # Base
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=140,
                       marker="s", edgecolors="white", linewidths=0.8, zorder=10)

            # Obstacles
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles, z_ceil=30,
                                             alpha=0.14)

            # Flight trail — altitude gradient
            if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
                trail = env.uav_trail
                xs = [p[0] for p in trail]
                ys = [p[1] for p in trail]
                zs = [p[2] if len(p) > 2 else 50.0 for p in trail]
                for i in range(len(xs) - 1):
                    frac = min(1.0, zs[i] / 120.0)
                    seg_c = (frac, 0.35, 1.0 - frac)
                    ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                            color=seg_c, linewidth=1.2, alpha=0.85)

            PlotRenderer._setup_3d_ax(ax, env.width, env.height,
                                       z_max=150, elev=elev, azim=azim)
            ax.set_title(
                f"3D Trajectory — {view_name.replace('_', ' ').title()}",
                fontsize=13, fontweight="bold")

            PlotRenderer._save_dual(fig, save_dir,
                                     f"trajectory_3d_{view_name}")

    # ── 11. Speed Heatmap (2D — LineCollection) ──────────────
    @staticmethod
    def render_trajectory_heatmap(env, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

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

        fig, ax = plt.subplots(figsize=(10, 7))
        PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.12)

        for node in env.sensors:
            ax.scatter(node.x, node.y, c=PlotRenderer.C["idle"],
                       s=25, edgecolors="black", linewidths=0.3, zorder=3)

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="coolwarm", linewidths=1.5, zorder=5)
        lc.set_array(norm_speeds)
        ax.add_collection(lc)

        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label("Relative Speed (0 = hover, 1 = max transit)")

        PlotRenderer._setup_2d_ax(ax, env.width, env.height)
        ax.set_title("Trajectory Speed Heatmap",
                      fontsize=14, fontweight="bold")
        PlotRenderer._save_dual(fig, save_dir, "trajectory_heatmap")

    # ── 12. AoI Timeline ─────────────────────────────────────
    @staticmethod
    def render_aoi_timeline(aoi_history: dict, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        if not aoi_history:
            return

        peaks = {nid: max(vals) for nid, vals in aoi_history.items()}
        top_nodes = sorted(peaks, key=peaks.get, reverse=True)[:10]

        fig, ax = plt.subplots(figsize=(12, 5))
        cmap = plt.cm.get_cmap("tab10")
        for i, nid in enumerate(top_nodes):
            ax.plot(aoi_history[nid], label=f"Node {nid}",
                    color=cmap(i), linewidth=0.9, alpha=0.85)

        ax.set_xlabel("Step"); ax.set_ylabel("Age of Information (s)")
        ax.set_title("Per-Node AoI Timeline (Top 10 Peak)",
                      fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        PlotRenderer._save_dual(fig, save_dir, "aoi_timeline")

    # ── 13. Battery + Replan Overlay ─────────────────────────
    @staticmethod
    def render_battery_with_replans(battery_hist: list, replan_steps: list,
                                     save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(battery_hist, color="#1565C0", linewidth=1.2, label="Battery (J)")
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.08, color="#1565C0")

        for rs in replan_steps:
            if 0 <= rs < len(battery_hist):
                ax.axvline(x=rs, color="#E53935", linestyle="--",
                           linewidth=0.6, alpha=0.6)
        if replan_steps:
            ax.axvline(x=replan_steps[0], color="#E53935", linestyle="--",
                       linewidth=0.6, alpha=0.6, label="Replan Event")

        ax.set_xlabel("Step"); ax.set_ylabel("Battery (J)")
        ax.set_title("Battery Discharge with Replan Events",
                      fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        PlotRenderer._save_dual(fig, save_dir, "battery_replans")

    # ── 14. Run Comparison ───────────────────────────────────
    @staticmethod
    def render_run_comparison(run_a: dict, run_b: dict, save_dir: str,
                               label_a: str = "Run A",
                               label_b: str = "Run B"):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

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

        fig, ax = plt.subplots(figsize=(12, 5))
        ba = ax.bar(x - w / 2, vals_a, w, label=label_a,
                    color="#1565C0", alpha=0.85, edgecolor="black", linewidth=0.4)
        bb = ax.bar(x + w / 2, vals_b, w, label=label_b,
                    color="#FF9800", alpha=0.85, edgecolor="black", linewidth=0.4)

        for bar in list(ba) + list(bb):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Run Comparison", fontsize=14, fontweight="bold")
        ax.legend()
        PlotRenderer._save_dual(fig, save_dir, "run_comparison")

    # ── 15. Semantic Clustering — Geographic (3D-first) ──────
    @staticmethod
    def render_semantic_clustering(env, active_labels, active_centroids,
                                    save_dir: str):
        from scipy.spatial import ConvexHull
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        cmap = plt.cm.get_cmap("tab10")
        unique_labels = sorted(set(active_labels))
        nodes = env.sensors

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

        # Obstacles
        if is_3d:
            PlotRenderer._draw_obstacles_3d(ax, env.obstacles, alpha=0.1)
        else:
            PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.12)

        for label in unique_labels:
            mask = [i for i, l in enumerate(active_labels)
                    if l == label and i < len(nodes)]
            if not mask:
                continue
            colour = "#9E9E9E" if label == -1 else cmap(label % 10)
            lbl = f"Cluster {label}" if label != -1 else "Noise"
            xs = [nodes[i].x for i in mask]
            ys = [nodes[i].y for i in mask]
            zs = [getattr(nodes[i], "z", 0) for i in mask]

            if is_3d:
                ax.scatter(xs, ys, zs, c=[colour] * len(xs), s=70,
                           edgecolors="black", linewidths=0.4,
                           depthshade=True, zorder=3, label=lbl)
            else:
                ax.scatter(xs, ys, c=[colour] * len(xs), s=50,
                           edgecolors="black", linewidths=0.4,
                           zorder=3, label=lbl)
                # Convex hull
                if label != -1 and len(mask) >= 3:
                    pts = np.array(list(zip(xs, ys)))
                    try:
                        hull = ConvexHull(pts)
                        hp = np.append(hull.vertices, hull.vertices[0])
                        ax.plot(pts[hp, 0], pts[hp, 1], color=colour,
                                linewidth=1.2, linestyle="--", alpha=0.6)
                    except Exception:
                        pass

        # Centroids
        for idx, centroid in enumerate(active_centroids):
            if np.all(np.array(centroid) == 0):
                continue
            cz = centroid[2] if len(centroid) > 2 else 0
            if is_3d:
                ax.scatter(centroid[0], centroid[1], cz, marker="*", s=250,
                           facecolors=cmap(idx % 10), edgecolors="black",
                           linewidths=0.8, zorder=5)
            else:
                ax.scatter(centroid[0], centroid[1], marker="*", s=250,
                           facecolors=cmap(idx % 10), edgecolors="black",
                           linewidths=0.8, zorder=5)
                ax.annotate(f"C{idx}", (centroid[0], centroid[1]),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=7)

        # Base station
        base = env.uav
        if is_3d:
            ax.scatter(base.x, base.y, 0, c=C["base"], s=140, marker="s",
                       edgecolors="white", linewidths=0.8,
                       zorder=6, label="Base Station")
            PlotRenderer._setup_3d_ax(ax, env.width, env.height)
        else:
            ax.scatter(base.x, base.y, c=C["base"], s=140, marker="s",
                       edgecolors="white", linewidths=0.8,
                       zorder=6, label="Base Station")
            PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        ax.set_title("Semantic Clustering — Geographic Distribution",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.8)
        PlotRenderer._save_dual(fig, save_dir, "semantic_clustering_geo")

    # ── 16. Clustering PCA Space (2D) ────────────────────────
    @staticmethod
    def render_clustering_pca_space(reduced_features, active_labels,
                                     save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        arr = np.array(reduced_features)
        if arr.ndim < 2 or arr.shape[1] < 2:
            return

        cmap = plt.cm.get_cmap("tab10")
        unique_labels = sorted(set(active_labels))

        fig, ax = plt.subplots(figsize=(8, 6))
        for label in unique_labels:
            mask = [i for i, l in enumerate(active_labels) if l == label]
            colour = "#9E9E9E" if label == -1 else cmap(label % 10)
            lbl = f"Cluster {label}" if label != -1 else "Noise / Outliers"
            ax.scatter(arr[mask, 0], arr[mask, 1],
                       c=[colour] * len(mask), s=55,
                       edgecolors="black", linewidths=0.3, alpha=0.85,
                       label=lbl)

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("Semantic Clustering — PCA Latent Space",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=8, framealpha=0.8)
        PlotRenderer._save_dual(fig, save_dir, "semantic_clustering_pca")

    # ── 17. Routing Pipeline (3-panel, 3D-first) ────────────
    @staticmethod
    def render_routing_pipeline(env, rp_nodes, rp_member_map,
                                 route_sequence, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        fig = plt.figure(figsize=(18, 6))
        fig.suptitle("Routing Pipeline Compression",
                      fontsize=14, fontweight="bold")

        all_nodes = env.sensors
        projections = {"projection": "3d"} if is_3d else {}

        def _draw_obs(ax):
            if is_3d:
                PlotRenderer._draw_obstacles_3d(ax, env.obstacles, alpha=0.1)
            else:
                PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.12)

        def _setup(ax):
            if is_3d:
                PlotRenderer._setup_3d_ax(ax, env.width, env.height)
            else:
                PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        # Panel 1: raw nodes
        ax = fig.add_subplot(131, **projections)
        _draw_obs(ax)
        xs = [n.x for n in all_nodes]
        ys = [n.y for n in all_nodes]
        zs = [getattr(n, "z", 0) for n in all_nodes]
        if is_3d:
            ax.scatter(xs, ys, zs, c="#42A5F5", s=45, edgecolors="black",
                       linewidths=0.3, depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=120,
                       marker="s", edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(xs, ys, c="#42A5F5", s=30, edgecolors="black",
                       linewidths=0.3)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=120,
                       marker="s", edgecolors="white", linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(a) Raw Deployment [{len(all_nodes)} nodes]", fontsize=10)

        # Panel 2: RP compression
        ax = fig.add_subplot(132, **projections)
        _draw_obs(ax)
        rp_list = rp_nodes or []
        if is_3d:
            ax.scatter(xs, ys, zs, c="#BDBDBD", s=15, alpha=0.35)
            for rp in rp_list:
                theta = np.linspace(0, 2 * np.pi, 40)
                rx = rp.x + 120 * np.cos(theta)
                ry = rp.y + 120 * np.sin(theta)
                ax.plot(rx, ry, np.zeros_like(rx), color=C["rp"],
                        alpha=0.2, linewidth=0.8)
            rp_xs = [n.x for n in rp_list]
            rp_ys = [n.y for n in rp_list]
            rp_zs = [getattr(n, "z", 0) for n in rp_list]
            ax.scatter(rp_xs, rp_ys, rp_zs, c=C["rp"], s=90, marker="D",
                       edgecolors="black", linewidths=0.5, zorder=4,
                       depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=120,
                       marker="s", edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(xs, ys, c="#BDBDBD", s=15, edgecolors="none", alpha=0.4)
            for rp in rp_list:
                circle = plt.Circle((rp.x, rp.y), 120, color=C["rp"],
                                     alpha=0.12)
                ax.add_patch(circle)
            ax.scatter([n.x for n in rp_list], [n.y for n in rp_list],
                       c=C["rp"], s=80, marker="D",
                       edgecolors="black", linewidths=0.5, zorder=4)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=120,
                       marker="s", edgecolors="white", linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(b) RP Compression [{len(rp_list)} RPs]", fontsize=10)

        # Panel 3: optimised route
        ax = fig.add_subplot(133, **projections)
        _draw_obs(ax)
        seq = route_sequence or []
        if is_3d:
            ax.scatter(xs, ys, zs, c="#BDBDBD", s=15, alpha=0.3)
            if len(seq) >= 2:
                rxs = [env.uav.x] + [n.x for n in seq]
                rys = [env.uav.y] + [n.y for n in seq]
                rzs = [0] + [getattr(n, "z", 0) for n in seq]
                ax.plot(rxs, rys, rzs, color=C["visited"],
                        linewidth=1.3, zorder=3)
            sq_xs = [n.x for n in seq]
            sq_ys = [n.y for n in seq]
            sq_zs = [getattr(n, "z", 0) for n in seq]
            ax.scatter(sq_xs, sq_ys, sq_zs, c=C["visited"], s=65,
                       edgecolors="black", linewidths=0.4, zorder=4,
                       depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=120,
                       marker="s", edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(xs, ys, c="#BDBDBD", s=15, edgecolors="none", alpha=0.3)
            if len(seq) >= 2:
                rxs = [env.uav.x] + [n.x for n in seq]
                rys = [env.uav.y] + [n.y for n in seq]
                ax.plot(rxs, rys, color=C["visited"],
                        linewidth=1.3, zorder=3)
                for i, (x, y) in enumerate(zip(rxs[1:], rys[1:]), 1):
                    ax.annotate(str(i), (x, y), textcoords="offset points",
                                xytext=(4, 4), fontsize=6, color="#1B5E20")
            ax.scatter([n.x for n in seq], [n.y for n in seq],
                       c=C["visited"], s=55, edgecolors="black",
                       linewidths=0.4, zorder=4)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=120,
                       marker="s", edgecolors="white", linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(c) Optimised Route [{len(seq)} waypoints]", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        PlotRenderer._save_dual(fig, save_dir, "routing_pipeline")

    # ── 18. Communication Quality (2-panel) ──────────────────
    @staticmethod
    def render_communication_quality(nodes, uav_trail, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

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

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Communication Quality Analysis",
                      fontsize=13, fontweight="bold")

        # Panel 1: link rate vs distance
        ax = axes[0]
        ax.scatter(distances, data_rates, c="#1565C0", s=45,
                   edgecolors="black", linewidths=0.3, alpha=0.75)
        ax.set_xlabel("UAV–Node Distance (m)")
        ax.set_ylabel("Data Rate (Mbps)")
        ax.set_title("(a) Link Rate vs Distance")
        d_sorted = sorted(distances)
        if d_sorted and d_sorted[-1] > 0:
            trend_d = np.linspace(max(1, d_sorted[0]), d_sorted[-1], 200)
            max_rate = max(data_rates) if data_rates else 1.0
            scale = max_rate * (min(d_sorted) ** 2) if d_sorted[0] > 0 else max_rate
            trend_r = scale / (trend_d ** 2 + 1e-6)
            ax.plot(trend_d, trend_r, color="#E53935", linewidth=1,
                    linestyle="--", label="1/d² reference")
            ax.legend(fontsize=8)

        # Panel 2: buffer histogram
        ax = axes[1]
        ax.hist(buffer_fill, bins=15, facecolor="#FF9800",
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Buffer Occupancy (%)")
        ax.set_ylabel("Node Count")
        ax.set_title("(b) Buffer Fill Distribution")
        ax.axvline(x=80, color="#E53935", linestyle="--",
                   linewidth=1, label="80% threshold")
        ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        PlotRenderer._save_dual(fig, save_dir, "communication_quality")

    # ── 19. Mission Progress Combined (4-panel) ──────────────
    @staticmethod
    def render_mission_progress_combined(visited_hist: list,
                                          battery_hist: list,
                                          data_hist: list,
                                          aoi_mean_hist: list,
                                          replan_steps: list,
                                          save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        fig.suptitle("Mission Progress Overview",
                      fontsize=14, fontweight="bold")

        series = [
            (axes[0, 0], visited_hist,  "#00E676", "Nodes Visited",   "Count"),
            (axes[0, 1], battery_hist,  "#1565C0", "Battery (J)",     "Joules"),
            (axes[1, 0], data_hist,     "#FF9800", "Data (Mbits)",    "Mbits"),
            (axes[1, 1], aoi_mean_hist, "#AB47BC", "Mean AoI",        "Steps"),
        ]
        for ax, hist, colour, title, ylabel in series:
            if hist:
                ax.plot(hist, color=colour, linewidth=1.2)
                ax.fill_between(range(len(hist)), hist,
                                alpha=0.08, color=colour)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            for rs in replan_steps:
                if hist and 0 <= rs < len(hist):
                    ax.axvline(x=rs, color="#E53935", linestyle="--",
                               linewidth=0.5, alpha=0.5)

        axes[1, 0].set_xlabel("Step")
        axes[1, 1].set_xlabel("Step")
        if replan_steps:
            axes[0, 0].axvline(x=-1, color="#E53935", linestyle="--",
                               linewidth=0.8, label="Replan")
            axes[0, 0].legend(fontsize=8, loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        PlotRenderer._save_dual(fig, save_dir, "mission_progress_combined")

    # ── 20. Rendezvous Compression (3D-first before/after) ───
    @staticmethod
    def render_rendezvous_compression(env, all_nodes, rp_nodes,
                                       rp_member_map, save_dir: str):
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_style()
        is_3d = PlotRenderer._use_3d()
        C = PlotRenderer.C

        cmap = plt.cm.get_cmap("tab20")
        rp_list = rp_nodes or []
        rp_map = rp_member_map or {}

        node_colour = {}
        for rp_idx, (rp, members) in enumerate(rp_map.items()):
            for m in members:
                node_colour[m] = cmap(rp_idx % 20)

        projections = {"projection": "3d"} if is_3d else {}
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(
            f"Rendezvous Point Compression  "
            f"({len(all_nodes)} nodes → {len(rp_list)} RPs, "
            f"ratio = {len(all_nodes) / max(len(rp_list), 1):.1f}×)",
            fontsize=13, fontweight="bold")

        def _draw_obs(ax):
            if is_3d:
                PlotRenderer._draw_obstacles_3d(ax, env.obstacles, alpha=0.1)
            else:
                PlotRenderer._draw_obstacles_2d(ax, env.obstacles, alpha=0.12)

        def _setup(ax):
            if is_3d:
                PlotRenderer._setup_3d_ax(ax, env.width, env.height)
            else:
                PlotRenderer._setup_2d_ax(ax, env.width, env.height)

        # Left: all nodes coloured by RP membership
        ax = fig.add_subplot(121, **projections)
        _draw_obs(ax)
        for n in all_nodes:
            col = node_colour.get(n.id, "#9E9E9E")
            z = getattr(n, "z", 0)
            if is_3d:
                ax.scatter(n.x, n.y, z, c=[col], s=45,
                           edgecolors="black", linewidths=0.3, depthshade=True)
            else:
                ax.scatter(n.x, n.y, c=[col], s=35,
                           edgecolors="black", linewidths=0.3)
        if is_3d:
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=140,
                       marker="s", edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=140,
                       marker="s", edgecolors="white", linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(a) All Nodes [{len(all_nodes)}]", fontsize=10)

        # Right: RP waypoints with coverage circles
        ax = fig.add_subplot(122, **projections)
        _draw_obs(ax)
        all_xs = [n.x for n in all_nodes]
        all_ys = [n.y for n in all_nodes]
        all_zs = [getattr(n, "z", 0) for n in all_nodes]
        if is_3d:
            ax.scatter(all_xs, all_ys, all_zs, c="#BDBDBD", s=12,
                       alpha=0.35)
            for rp_idx, rp in enumerate(rp_list):
                colour = cmap(rp_idx % 20)
                theta = np.linspace(0, 2 * np.pi, 40)
                rx = rp.x + 120 * np.cos(theta)
                ry = rp.y + 120 * np.sin(theta)
                ax.plot(rx, ry, np.zeros_like(rx), color=colour,
                        alpha=0.2, linewidth=0.8)
                ax.scatter(rp.x, rp.y, getattr(rp, "z", 0),
                           c=[colour], s=100, marker="D",
                           edgecolors="black", linewidths=0.5, zorder=4,
                           depthshade=True)
            ax.scatter(env.uav.x, env.uav.y, 0, c=C["base"], s=140,
                       marker="s", edgecolors="white", linewidths=0.8)
        else:
            ax.scatter(all_xs, all_ys, c="#BDBDBD", s=12,
                       edgecolors="none", alpha=0.35)
            for rp_idx, rp in enumerate(rp_list):
                colour = cmap(rp_idx % 20)
                circle = plt.Circle((rp.x, rp.y), 120, color=colour,
                                     alpha=0.12)
                ax.add_patch(circle)
                ax.scatter(rp.x, rp.y, c=[colour], s=90, marker="D",
                           edgecolors="black", linewidths=0.5, zorder=4)
            ax.scatter(env.uav.x, env.uav.y, c=C["base"], s=140,
                       marker="s", edgecolors="white", linewidths=0.8)
        _setup(ax)
        ax.set_title(f"(b) Rendezvous Points [{len(rp_list)}]", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        PlotRenderer._save_dual(fig, save_dir, "rendezvous_compression")
