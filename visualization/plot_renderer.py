import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import os
from config.feature_toggles import FeatureToggles


class PlotRenderer:

    # ----------------------------------------------------
    # INTERNAL
    # ----------------------------------------------------
    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    # ----------------------------------------------------
    # ENVIRONMENT VISUAL
    # ----------------------------------------------------
    @staticmethod
    def render_environment(env, save_dir):
        PlotRenderer._ensure_dir(save_dir)

        fig = plt.figure(figsize=(8, 6))
        is_3d = FeatureToggles.DIMENSIONS == "3D"
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # -------- Nodes --------
        xs = [node.x for node in env.nodes]
        ys = [node.y for node in env.nodes]
        
        if is_3d:
            zs = [node.z for node in env.nodes]
            ax.scatter(xs, ys, zs, c="blue", label="Nodes")
        else:
            ax.scatter(xs, ys, c="blue", label="Nodes")

        # -------- Obstacles --------
        for obs in env.obstacles:
            if is_3d:
                # 3D Extrusion (Height = 20)
                z_floor, z_ceil = 0, 20
                dx = obs.x2 - obs.x1
                dy = obs.y2 - obs.y1
                
                # Define 8 vertices of the prism
                vertices = [
                    [obs.x1, obs.y1, z_floor], [obs.x2, obs.y1, z_floor], [obs.x2, obs.y2, z_floor], [obs.x1, obs.y2, z_floor],
                    [obs.x1, obs.y1, z_ceil], [obs.x2, obs.y1, z_ceil], [obs.x2, obs.y2, z_ceil], [obs.x1, obs.y2, z_ceil]
                ]
                # Define the 6 faces connecting vertices
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]], # Bottom
                    [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
                    [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
                    [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
                    [vertices[1], vertices[2], vertices[6], vertices[5]], # Right
                    [vertices[0], vertices[3], vertices[7], vertices[4]]  # Left
                ]
                collection = art3d.Poly3DCollection(faces, alpha=0.3, facecolors="red", edgecolors="darkred")
                ax.add_collection3d(collection)
            else:
                rect = plt.Rectangle((obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1, color="red", alpha=0.4)
                ax.add_patch(rect)

        # -------- Risk Zones --------
        for rz in env.risk_zones:
            if is_3d:
                z_floor, z_ceil = 0, 15
                vertices = [
                    [rz.x1, rz.y1, z_floor], [rz.x2, rz.y1, z_floor], [rz.x2, rz.y2, z_floor], [rz.x1, rz.y2, z_floor],
                    [rz.x1, rz.y1, z_ceil], [rz.x2, rz.y1, z_ceil], [rz.x2, rz.y2, z_ceil], [rz.x1, rz.y2, z_ceil]
                ]
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]], 
                    [vertices[4], vertices[5], vertices[6], vertices[7]], 
                    [vertices[0], vertices[1], vertices[5], vertices[4]], 
                    [vertices[2], vertices[3], vertices[7], vertices[6]], 
                    [vertices[1], vertices[2], vertices[6], vertices[5]], 
                    [vertices[0], vertices[3], vertices[7], vertices[4]]  
                ]
                collection = art3d.Poly3DCollection(faces, alpha=0.2, facecolors="orange", edgecolors="darkorange")
                ax.add_collection3d(collection)
            else:
                rect = plt.Rectangle((rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1, color="orange", alpha=0.3)
                ax.add_patch(rect)

        ax.set_title("UAV Environment Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        if is_3d:
            ax.set_zlabel("Z (Altitude)")
            ax.set_zlim(0, 50)
            
        ax.grid(True)
        ax.legend()

        plt.savefig(
            os.path.join(save_dir, "environment.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # ----------------------------------------------------
    # ENERGY & VISIT PLOTS
    # ----------------------------------------------------
    @staticmethod
    def render_energy_plots(visited, energy_consumed, save_dir):
        PlotRenderer._ensure_dir(save_dir)

        # Visited Nodes
        plt.figure()
        plt.bar(["Visited Nodes"], [visited])
        plt.title("Visited Nodes")
        plt.savefig(
            os.path.join(save_dir, "visited_nodes.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Energy Consumed
        plt.figure()
        plt.bar(["Energy Consumed"], [energy_consumed])
        plt.title("Energy Consumption")
        plt.savefig(
            os.path.join(save_dir, "energy_consumed.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # ----------------------------------------------------
    # METRICS SNAPSHOT
    # ----------------------------------------------------
    @staticmethod
    def render_metrics_snapshot(completion_pct, efficiency, save_dir):
        PlotRenderer._ensure_dir(save_dir)

        plt.figure()
        plt.bar(["Completion %", "Efficiency"], [completion_pct, efficiency])
        plt.title("Mission Metrics Snapshot")
        plt.savefig(
            os.path.join(save_dir, "metrics_snapshot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # ----------------------------------------------------
    # TIME SERIES PLOTS
    # ----------------------------------------------------
    @staticmethod
    def render_time_series(visited_hist, battery_hist, replan_hist, save_dir):
        PlotRenderer._ensure_dir(save_dir)

        # Visited Over Time
        plt.figure()
        plt.plot(visited_hist)
        plt.title("Visited Nodes Over Time")
        plt.xlabel("Step")
        plt.ylabel("Visited")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "visited_over_time.png"))
        plt.close()

        # Battery Over Time
        plt.figure()
        plt.plot(battery_hist)
        plt.title("Battery Over Time")
        plt.xlabel("Step")
        plt.ylabel("Battery Level")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "battery_over_time.png"))
        plt.close()

        # Replans Over Time
        plt.figure()
        plt.plot(replan_hist)
        plt.title("Replans Over Time")
        plt.xlabel("Step")
        plt.ylabel("Replan Count")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "replans_over_time.png"))
        plt.close()

    # ----------------------------------------------------
    # ENVIRONMENT FRAME (TEMPORAL SNAPSHOT)
    # ----------------------------------------------------
    @staticmethod
    def render_environment_frame(env, save_dir, step):
        PlotRenderer._ensure_dir(save_dir)

        fig = plt.figure(figsize=(8, 6))
        is_3d = FeatureToggles.DIMENSIONS == "3D"
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # -------- Nodes --------
        xs = [node.x for node in env.nodes]
        ys = [node.y for node in env.nodes]
        
        if is_3d:
            zs = [node.z for node in env.nodes]
            ax.scatter(xs, ys, zs, c="blue")
        else:
            ax.scatter(xs, ys, c="blue")

        # -------- Obstacles --------
        for obs in env.obstacles:
            if is_3d:
                z_floor, z_ceil = 0, 20
                vertices = [
                    [obs.x1, obs.y1, z_floor], [obs.x2, obs.y1, z_floor], [obs.x2, obs.y2, z_floor], [obs.x1, obs.y2, z_floor],
                    [obs.x1, obs.y1, z_ceil], [obs.x2, obs.y1, z_ceil], [obs.x2, obs.y2, z_ceil], [obs.x1, obs.y2, z_ceil]
                ]
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]], 
                    [vertices[4], vertices[5], vertices[6], vertices[7]], 
                    [vertices[0], vertices[1], vertices[5], vertices[4]], 
                    [vertices[2], vertices[3], vertices[7], vertices[6]], 
                    [vertices[1], vertices[2], vertices[6], vertices[5]], 
                    [vertices[0], vertices[3], vertices[7], vertices[4]]  
                ]
                collection = art3d.Poly3DCollection(faces, alpha=0.3, facecolors="red", edgecolors="darkred")
                ax.add_collection3d(collection)
            else:
                rect = plt.Rectangle((obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1, color="red", alpha=0.4)
                ax.add_patch(rect)

        # -------- Risk Zones --------
        for rz in env.risk_zones:
            if is_3d:
                z_floor, z_ceil = 0, 15
                vertices = [
                    [rz.x1, rz.y1, z_floor], [rz.x2, rz.y1, z_floor], [rz.x2, rz.y2, z_floor], [rz.x1, rz.y2, z_floor],
                    [rz.x1, rz.y1, z_ceil], [rz.x2, rz.y1, z_ceil], [rz.x2, rz.y2, z_ceil], [rz.x1, rz.y2, z_ceil]
                ]
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]], 
                    [vertices[4], vertices[5], vertices[6], vertices[7]], 
                    [vertices[0], vertices[1], vertices[5], vertices[4]], 
                    [vertices[2], vertices[3], vertices[7], vertices[6]], 
                    [vertices[1], vertices[2], vertices[6], vertices[5]], 
                    [vertices[0], vertices[3], vertices[7], vertices[4]]  
                ]
                collection = art3d.Poly3DCollection(faces, alpha=0.2, facecolors="orange", edgecolors="darkorange")
                ax.add_collection3d(collection)
            else:
                rect = plt.Rectangle((rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1, color="orange", alpha=0.3)
                ax.add_patch(rect)

        # -------- UAV Rendering --------
        if hasattr(env, "uav"):

            # --- UAV Trail ---
            if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
                trail_x = [p[0] for p in env.uav_trail]
                trail_y = [p[1] for p in env.uav_trail]

                if is_3d:
                    trail_z = [p[2] for p in env.uav_trail if len(p) > 2]
                    # Ensure z array matches length just in case old 2D data is inside
                    if len(trail_z) == len(trail_x):
                        ax.plot(trail_x, trail_y, zs=trail_z, linewidth=1.2, alpha=0.8, color="black", zorder=5)
                else:
                    ax.plot(trail_x, trail_y, linewidth=1.2, alpha=0.8, color="black", zorder=5)

            # --- UAV Marker ---
            if is_3d:
                ax.scatter(env.uav.x, env.uav.y, env.uav.z, s=120, marker="^", c="black", edgecolors="white", linewidths=0.8, zorder=10)
            else:
                ax.scatter(env.uav.x, env.uav.y, s=120, marker="^", c="black", edgecolors="white", linewidths=0.8, zorder=10)

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        if is_3d:
            ax.set_zlim(0, 50)
            
        ax.set_title(f"Step {step}")

        # -------- Replan Flash --------
        if hasattr(env, "temporal_engine"):
            flash = env.temporal_engine.consume_replan_flash()
            if flash:
                ax.set_facecolor((1.0, 0.85, 0.85))
                if is_3d:
                    ax.text2D(
                        0.5,
                        0.95,
                        "REPLAN TRIGGERED",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=12,
                        color="red",
                        fontweight="bold",
                    )
                else:
                    ax.text(
                        0.5,
                        0.95,
                        "REPLAN TRIGGERED",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=12,
                        color="red",
                        fontweight="bold",
                    )

        filename = f"{step:04d}.png"
        fig.savefig(f"{save_dir}/{filename}", dpi=200)
        plt.close(fig)

    # ============================================================
    #  IEEE-Grade Post-Run Visualisations  (v0.5)
    # ============================================================

    @staticmethod
    def _set_ieee_style():
        """Configure matplotlib for IEEE paper standards."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'axes.grid': True,
            'grid.alpha': 0.25,
        })

    @staticmethod
    def _save_dual(fig, save_dir, basename):
        """Save figure as both PNG and PDF."""
        fig.savefig(os.path.join(save_dir, f"{basename}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{basename}.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------
    # 1. Radar Chart — 6 Normalised KPIs
    # ----------------------------------------------------------
    @staticmethod
    def render_radar_chart(results: dict, save_dir: str):
        """
        Spider/radar chart of 6 normalised mission KPIs:
        DR%, Coverage%, Network Lifetime, Path Stability,
        Priority Satisfaction, Data Freshness (1 − AoI_norm).
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        labels = [
            "Data Collection\nRate (%)",
            "Coverage\nRatio (%)",
            "Network\nLifetime",
            "Path\nStability",
            "Priority\nSatisfaction (%)",
            "Data\nFreshness",
        ]

        max_aoi = 800.0  # normalisation ceiling
        values = [
            min(100.0, results.get("data_collection_rate_percent", 0)),
            min(100.0, results.get("coverage_ratio_percent", 0)),
            results.get("network_lifetime_residual", 0) * 100,
            results.get("path_stability_index", 0) * 100,
            results.get("priority_satisfaction_percent", 0),
            max(0, 100 * (1 - results.get("average_aoi_s", 0) / max_aoi)),
        ]

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="#2196F3", alpha=0.25)
        ax.plot(angles, values, color="#1565C0", linewidth=2, marker="o", markersize=5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_title("Mission Performance Radar", fontsize=14, fontweight="bold", pad=20)

        PlotRenderer._save_dual(fig, save_dir, "radar_chart")

    # ----------------------------------------------------------
    # 2. Node Energy Heatmap
    # ----------------------------------------------------------
    @staticmethod
    def render_node_energy_heatmap(nodes: list, env_width: float,
                                    env_height: float, save_dir: str):
        """
        2D scatter heatmap showing IoT node residual battery energy
        overlaid on the mission environment.
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]

        from config.config import Config
        initial = Config.NODE_BATTERY_J
        residuals = [
            getattr(n, "node_battery_J", initial) / max(initial, 1e-6)
            for n in nodes
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(xs, ys, c=residuals, cmap="RdYlGn", s=80,
                        edgecolors="black", linewidths=0.5, vmin=0, vmax=1)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Residual Battery Fraction")
        ax.set_xlim(0, env_width)
        ax.set_ylim(0, env_height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("IoT Node Residual Energy Map", fontsize=13, fontweight="bold")
        ax.set_aspect("equal")

        PlotRenderer._save_dual(fig, save_dir, "node_energy_heatmap")

    # ----------------------------------------------------------
    # 3. Trajectory Summary — Final Snapshot
    # ----------------------------------------------------------
    @staticmethod
    def render_trajectory_summary(env, visited_ids: set, save_dir: str):
        """
        Final environment snapshot: full UAV trajectory trail,
        visited nodes (green), unvisited (red), obstacles, base station.
        """
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        fig, ax = plt.subplots(figsize=(10, 7))

        # Obstacles
        for obs in env.obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                color="#B71C1C", alpha=0.2, linewidth=1, edgecolor="#B71C1C",
            )
            ax.add_patch(rect)

        # Risk zones
        for rz in env.risk_zones:
            rect = plt.Rectangle(
                (rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1,
                color="orange", alpha=0.15, linewidth=1, edgecolor="darkorange",
            )
            ax.add_patch(rect)

        # Nodes — visited vs unvisited
        for node in env.nodes[1:]:
            colour = "#4CAF50" if node.id in visited_ids else "#E53935"
            marker = "o" if node.id in visited_ids else "x"
            ax.scatter(node.x, node.y, c=colour, s=50, marker=marker,
                       edgecolors="black", linewidths=0.3, zorder=5)
            ax.annotate(str(node.id), (node.x + 4, node.y + 4), fontsize=6, color="gray")

        # Base station
        base = env.nodes[0]
        ax.scatter(base.x, base.y, c="black", s=150, marker="s", zorder=10, label="Base Station")

        # UAV trail
        if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
            trail_x = [p[0] for p in env.uav_trail]
            trail_y = [p[1] for p in env.uav_trail]
            ax.plot(trail_x, trail_y, linewidth=1.0, alpha=0.7, color="#1565C0",
                    label="UAV Trajectory", zorder=3)

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Mission Trajectory Summary", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", framealpha=0.8)
        ax.set_aspect("equal")

        PlotRenderer._save_dual(fig, save_dir, "trajectory_summary")

    # ----------------------------------------------------------
    # 4. Dashboard Panel — 2×3 Multi-Figure
    # ----------------------------------------------------------
    @staticmethod
    def render_dashboard_panel(results: dict, battery_hist: list,
                                visited_hist: list, save_dir: str):
        """
        Publication-quality 2×3 panel combining:
        [0,0] Battery curve  [0,1] Visited curve   [0,2] Coverage pie
        [1,0] DR bar         [1,1] AoI bar          [1,2] Energy efficiency bar
        """
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("IEEE Performance Dashboard", fontsize=15, fontweight="bold", y=0.98)

        # [0,0] Battery over time
        ax = axes[0, 0]
        ax.plot(battery_hist, color="#1565C0", linewidth=1)
        ax.set_title("Battery Discharge")
        ax.set_xlabel("Step")
        ax.set_ylabel("Battery (J)")

        # [0,1] Visited nodes over time
        ax = axes[0, 1]
        ax.plot(visited_hist, color="#4CAF50", linewidth=1)
        ax.set_title("Nodes Visited")
        ax.set_xlabel("Step")
        ax.set_ylabel("Count")

        # [0,2] Coverage pie
        ax = axes[0, 2]
        visited = results.get("nodes_visited", 0)
        total = results.get("total_nodes", 1)
        ax.pie(
            [visited, total - visited],
            labels=["Visited", "Unvisited"],
            colors=["#4CAF50", "#E53935"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Node Coverage")

        # [1,0] Data Collection Rate
        ax = axes[1, 0]
        dr = results.get("data_collection_rate_percent", 0)
        ax.bar(["DR%"], [dr], color="#FF9800", width=0.4)
        ax.set_ylim(0, 100)
        ax.set_title("Data Collection Rate")
        ax.set_ylabel("%")

        # [1,1] Average AoI
        ax = axes[1, 1]
        aoi = results.get("average_aoi_s", 0)
        ax.bar(["Avg AoI (s)"], [aoi], color="#9C27B0", width=0.4)
        ax.set_title("Mean Peak Age of Information")
        ax.set_ylabel("Seconds")

        # [1,2] Energy Efficiency
        ax = axes[1, 2]
        epn = results.get("energy_per_node_J", 0)
        ax.bar(["J/node"], [epn], color="#00BCD4", width=0.4)
        ax.set_title("Energy per Node Visited")
        ax.set_ylabel("Joules")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        PlotRenderer._save_dual(fig, save_dir, "dashboard_panel")

    # ----------------------------------------------------------
    # 5. 3D Trajectory Render with Gaussian Obstacle Surfaces
    # ----------------------------------------------------------
    @staticmethod
    def render_3d_trajectory(env, save_dir: str):
        """
        3D Matplotlib figure with:
        - Full 3D flight path coloured by altitude gradient
        - Obstacle prisms
        - Node markers coloured by AoI
        - Multiple viewing angles saved
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        views = [
            ("isometric", 30, -60),
            ("top_down", 90, -90),
            ("side_view", 0, -90),
        ]

        for view_name, elev, azim in views:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")

            # Nodes
            for node in env.nodes[1:]:
                aoi = getattr(node, "aoi_timer", 0.0)
                colour = "#4CAF50" if aoi < 50 else ("#FF9800" if aoi < 200 else "#E53935")
                ax.scatter(node.x, node.y, getattr(node, "z", 0),
                           c=colour, s=30, edgecolors="black", linewidths=0.3)

            # Base station
            base = env.nodes[0]
            ax.scatter(base.x, base.y, 0, c="black", s=100, marker="s", zorder=10)

            # Obstacles as prisms
            for obs in env.obstacles:
                z_ceil = getattr(obs, "height", 30)
                dx = obs.x2 - obs.x1
                dy = obs.y2 - obs.y1
                ax.bar3d(obs.x1, obs.y1, 0, dx, dy, z_ceil,
                         color="#B71C1C", alpha=0.15, edgecolor="#B71C1C")

            # UAV trail coloured by altitude
            if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
                trail = env.uav_trail
                xs = [p[0] for p in trail]
                ys = [p[1] for p in trail]
                zs = [p[2] if len(p) > 2 else 50.0 for p in trail]

                # Segment-by-segment colour by z
                for i in range(len(xs) - 1):
                    z_frac = min(1.0, zs[i] / 150.0)
                    colour = (z_frac, 0.3, 1.0 - z_frac)
                    ax.plot(xs[i:i+2], ys[i:i+2], zs[i:i+2],
                            color=colour, linewidth=1.0, alpha=0.8)

            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_zlim(0, 150)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Altitude (m)")
            ax.set_title(f"3D Trajectory — {view_name.replace('_', ' ').title()}",
                         fontsize=13, fontweight="bold")
            ax.view_init(elev=elev, azim=azim)

            PlotRenderer._save_dual(fig, save_dir, f"trajectory_3d_{view_name}")
