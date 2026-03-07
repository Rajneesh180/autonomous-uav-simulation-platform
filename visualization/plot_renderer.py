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
        is_3d = FeatureToggles.RENDER_MODE in ("3D", "both")
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # -------- Nodes --------
        all_nodes = [env.uav] + env.sensors if hasattr(env, 'uav') and env.uav else env.sensors
        xs = [node.x for node in all_nodes]
        ys = [node.y for node in all_nodes]
        
        if is_3d:
            zs = [node.z for node in all_nodes]
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
    def render_environment_frame(env, save_dir, step, mission=None):
        PlotRenderer._ensure_dir(save_dir)

        fig = plt.figure(figsize=(10, 7))
        is_3d = FeatureToggles.RENDER_MODE in ("3D", "both")
        
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # -------- Determine visited set --------
        visited_ids = set()
        if mission is not None:
            visited_ids = mission.visited

        # -------- Nodes (colour by visit status) --------
        for node in env.sensors:  # skip UAV anchor
            if node.id in visited_ids:
                colour, marker = "#4CAF50", "o"   # green = visited
            else:
                buf_ratio = node.current_buffer / max(node.buffer_capacity, 1e-6)
                if buf_ratio > 0.5:
                    colour = "#2196F3"             # blue = data waiting
                else:
                    colour = "#9E9E9E"             # grey = low buffer
                marker = "o"

            if is_3d:
                ax.scatter(node.x, node.y, getattr(node, 'z', 0),
                           c=colour, s=40, marker=marker, edgecolors="black", linewidths=0.3)
            else:
                ax.scatter(node.x, node.y, c=colour, s=40, marker=marker,
                           edgecolors="black", linewidths=0.3, zorder=4)

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
                    if len(trail_z) == len(trail_x):
                        ax.plot(trail_x, trail_y, zs=trail_z, linewidth=1.2, alpha=0.8, color="#1565C0", zorder=5)
                else:
                    ax.plot(trail_x, trail_y, linewidth=1.2, alpha=0.8, color="#1565C0", zorder=5)

            # --- UAV Marker ---
            if is_3d:
                ax.scatter(env.uav.x, env.uav.y, env.uav.z, s=120, marker="^", c="black", edgecolors="white", linewidths=0.8, zorder=10)
            else:
                ax.scatter(env.uav.x, env.uav.y, s=120, marker="^", c="black", edgecolors="white", linewidths=0.8, zorder=10)

            # --- Target connection line ---
            if mission is not None and mission.current_target and not is_3d:
                tgt = mission.current_target
                ax.plot([env.uav.x, tgt.x], [env.uav.y, tgt.y],
                        linestyle="--", color="red", linewidth=0.8, alpha=0.6, zorder=6)

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        if is_3d:
            ax.set_zlim(0, 50)

        # -------- HUD Overlay (step, battery, coverage) --------
        total_nodes = len(env.sensors)
        n_visited = len(visited_ids) if visited_ids else 0
        battery_pct = 0.0
        if hasattr(env, "uav"):
            from config.config import Config
            battery_pct = (env.uav.current_battery / Config.BATTERY_CAPACITY) * 100

        title = f"Step {step}  |  Battery: {battery_pct:.0f}%  |  Visited: {n_visited}/{total_nodes}"
        ax.set_title(title, fontsize=11, fontweight="bold")

        # -------- Replan Flash --------
        if hasattr(env, "temporal_engine"):
            flash = env.temporal_engine.consume_replan_flash()
            if flash:
                ax.set_facecolor((1.0, 0.85, 0.85))
                if is_3d:
                    ax.text2D(0.5, 0.92, "REPLAN", transform=ax.transAxes,
                              ha="center", fontsize=11, color="red", fontweight="bold")
                else:
                    ax.text(0.5, 0.92, "REPLAN", transform=ax.transAxes,
                            ha="center", fontsize=11, color="red", fontweight="bold")

        filename = f"{step:04d}.png"
        fig.savefig(f"{save_dir}/{filename}", dpi=150)
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
        initial = Config.NODE_BATTERY_CAPACITY_J
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
                facecolor="#B71C1C", alpha=0.2, linewidth=1, edgecolor="#B71C1C",
            )
            ax.add_patch(rect)

        # Risk zones
        for rz in env.risk_zones:
            rect = plt.Rectangle(
                (rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1,
                facecolor="orange", alpha=0.15, linewidth=1, edgecolor="darkorange",
            )
            ax.add_patch(rect)

        # Nodes — visited vs unvisited
        for node in env.sensors:
            colour = "#4CAF50" if node.id in visited_ids else "#E53935"
            marker = "o" if node.id in visited_ids else "x"
            ax.scatter(node.x, node.y, c=colour, s=50, marker=marker,
                       edgecolors="black", linewidths=0.3, zorder=5)
            ax.annotate(str(node.id), (node.x + 4, node.y + 4), fontsize=6, color="gray")

        # Base station
        base = env.uav
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
            for node in env.sensors:
                aoi = getattr(node, "aoi_timer", 0.0)
                colour = "#4CAF50" if aoi < 50 else ("#FF9800" if aoi < 200 else "#E53935")
                ax.scatter(node.x, node.y, getattr(node, "z", 0),
                           c=colour, s=30, edgecolors="black", linewidths=0.3)

            # Base station
            base = env.uav
            ax.scatter(base.x, base.y, 0, c="black", s=100, marker="s", zorder=10)

            # Obstacles as prisms
            for obs in env.obstacles:
                z_ceil = getattr(obs, "height", 30)
                dx = obs.x2 - obs.x1
                dy = obs.y2 - obs.y1
                ax.bar3d(obs.x1, obs.y1, 0, dx, dy, z_ceil,
                         color="#B71C1C", alpha=0.15)

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

    # ----------------------------------------------------------
    # 6. Speed-Coloured Trajectory Heatmap
    # ----------------------------------------------------------
    @staticmethod
    def render_trajectory_heatmap(env, save_dir: str):
        """
        2D trajectory with path segments coloured by instantaneous speed.
        Cool blue = slow (hovering), hot red = fast (transit).
        """
        import numpy as np
        from matplotlib.collections import LineCollection
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        if not hasattr(env, "uav_trail") or len(env.uav_trail) < 3:
            return

        trail = env.uav_trail
        xs = [p[0] for p in trail]
        ys = [p[1] for p in trail]

        # Compute per-segment speed
        speeds = []
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i-1]
            dy = ys[i] - ys[i-1]
            speeds.append((dx**2 + dy**2) ** 0.5)
        speeds = np.array(speeds)

        # Normalise
        if speeds.max() > 0:
            norm_speeds = speeds / speeds.max()
        else:
            norm_speeds = np.zeros_like(speeds)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Obstacles
        for obs in env.obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                facecolor="#B71C1C", alpha=0.15, edgecolor="#B71C1C",
            )
            ax.add_patch(rect)

        # Nodes
        for node in env.sensors:
            ax.scatter(node.x, node.y, c="#9E9E9E", s=20, edgecolors="black",
                       linewidths=0.3, zorder=3)

        # Coloured trajectory segments
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="coolwarm", linewidths=1.5, zorder=5)
        lc.set_array(norm_speeds)
        ax.add_collection(lc)

        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label("Relative Speed (0=hover, 1=max transit)")

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Trajectory Speed Heatmap", fontsize=14, fontweight="bold")
        ax.set_aspect("equal")

        PlotRenderer._save_dual(fig, save_dir, "trajectory_heatmap")

    # ----------------------------------------------------------
    # 7. Per-Node AoI Timeline
    # ----------------------------------------------------------
    @staticmethod
    def render_aoi_timeline(aoi_history: dict, save_dir: str):
        """
        Time-series plot showing AoI progression for the top-10
        highest-peak-AoI nodes.

        aoi_history : dict mapping node_id → list of AoI values per step
        """
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        if not aoi_history:
            return

        # Sort by peak AoI, take top 10
        peaks = {nid: max(vals) for nid, vals in aoi_history.items()}
        top_nodes = sorted(peaks, key=peaks.get, reverse=True)[:10]

        fig, ax = plt.subplots(figsize=(12, 5))

        cmap = plt.cm.get_cmap("tab10")
        for i, nid in enumerate(top_nodes):
            vals = aoi_history[nid]
            ax.plot(vals, label=f"Node {nid}", color=cmap(i), linewidth=0.8, alpha=0.8)

        ax.set_xlabel("Step")
        ax.set_ylabel("Age of Information (s)")
        ax.set_title("Per-Node AoI Timeline (Top 10 Peak)", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", ncol=2, fontsize=8)

        PlotRenderer._save_dual(fig, save_dir, "aoi_timeline")

    # ----------------------------------------------------------
    # 8. Battery Discharge with Replan Event Overlay
    # ----------------------------------------------------------
    @staticmethod
    def render_battery_with_replans(battery_hist: list, replan_steps: list,
                                     save_dir: str):
        """
        Battery discharge curve with vertical lines marking replan events.
        """
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(battery_hist, color="#1565C0", linewidth=1.2, label="Battery (J)")
        ax.fill_between(range(len(battery_hist)), battery_hist,
                        alpha=0.1, color="#1565C0")

        # Replan vertical markers
        for rs in replan_steps:
            if 0 <= rs < len(battery_hist):
                ax.axvline(x=rs, color="#E53935", linestyle="--", linewidth=0.6, alpha=0.6)

        # Add one labelled line for the legend
        if replan_steps:
            ax.axvline(x=replan_steps[0], color="#E53935", linestyle="--",
                       linewidth=0.6, alpha=0.6, label="Replan Event")

        ax.set_xlabel("Step")
        ax.set_ylabel("Battery (J)")
        ax.set_title("Battery Discharge with Replan Events", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")

        PlotRenderer._save_dual(fig, save_dir, "battery_replans")

    # ----------------------------------------------------------
    # 9. Run Comparison Viewer
    # ----------------------------------------------------------
    @staticmethod
    def render_run_comparison(run_a: dict, run_b: dict, save_dir: str,
                               label_a: str = "Run A", label_b: str = "Run B"):
        """
        Side-by-side bar chart comparing key metrics from two simulation runs.
        Renders as a table + grouped bar chart.
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

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
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        bars_a = ax.bar(x - width/2, vals_a, width, label=label_a,
                        color="#1565C0", alpha=0.8)
        bars_b = ax.bar(x + width/2, vals_b, width, label=label_b,
                        color="#FF9800", alpha=0.8)

        # Value labels on bars
        for bar in bars_a:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)
        for bar in bars_b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Run Comparison", fontsize=14, fontweight="bold")
        ax.legend()

        PlotRenderer._save_dual(fig, save_dir, "run_comparison")

    # ----------------------------------------------------------
    # 10. Semantic Clustering — Geographic Space Overlay
    # ----------------------------------------------------------
    @staticmethod
    def render_semantic_clustering(env, active_labels, active_centroids, save_dir: str):
        """
        2D geographic map with colour-coded node clusters, centroid markers, and
        convex-hull cluster boundaries.  Noise points (label == -1) are shown grey.
        """
        import numpy as np
        from scipy.spatial import ConvexHull
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        cmap = plt.cm.get_cmap("tab10")
        unique_labels = sorted(set(active_labels))
        nodes = env.sensors  # exclude UAV anchor

        fig, ax = plt.subplots(figsize=(10, 7))

        # Obstacles
        for obs in env.obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                facecolor="#B71C1C", alpha=0.15, edgecolor="#B71C1C", linewidth=1,
            )
            ax.add_patch(rect)

        for label in unique_labels:
            mask = [i for i, l in enumerate(active_labels) if l == label and i < len(nodes)]
            if not mask:
                continue
            colour = "#9E9E9E" if label == -1 else cmap(label % 10)
            xs = [nodes[i].x for i in mask]
            ys = [nodes[i].y for i in mask]
            ax.scatter(xs, ys, c=[colour]*len(xs), s=40,
                       edgecolors="black", linewidths=0.4, zorder=3,
                       label=f"Cluster {label}" if label != -1 else "Noise")

            # Convex hull boundary for valid clusters with ≥3 points
            if label != -1 and len(mask) >= 3:
                pts = np.array(list(zip(xs, ys)))
                try:
                    hull = ConvexHull(pts)
                    hull_pts = np.append(hull.vertices, hull.vertices[0])
                    ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                            color=colour, linewidth=1.2, linestyle="--", alpha=0.6)
                except Exception:
                    pass

        # Centroid markers
        for idx, centroid in enumerate(active_centroids):
            if np.all(np.array(centroid) == 0):
                continue
            ax.scatter(centroid[0], centroid[1], marker="*", s=220,
                       facecolors=cmap(idx % 10), edgecolors="black", linewidths=0.8,
                       zorder=5)
            ax.annotate(f"C{idx}", (centroid[0], centroid[1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7)

        # Base station
        base = env.uav
        ax.scatter(base.x, base.y, c="black", s=120, marker="s", zorder=6, label="Base Station")

        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Semantic Clustering — Geographic Distribution", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.8)
        ax.set_aspect("equal")

        PlotRenderer._save_dual(fig, save_dir, "semantic_clustering_geo")

    # ----------------------------------------------------------
    # 11. Semantic Clustering — PCA Latent Space
    # ----------------------------------------------------------
    @staticmethod
    def render_clustering_pca_space(reduced_features, active_labels, save_dir: str):
        """
        2D PCA-space scatter plot showing cluster assignments in the latent feature space.

        reduced_features : (N, ≥2) array-like of PCA-projected node features
        active_labels    : list/array of integer cluster labels (len == N)
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        arr = np.array(reduced_features)
        if arr.ndim < 2 or arr.shape[1] < 2:
            return

        cmap = plt.cm.get_cmap("tab10")
        unique_labels = sorted(set(active_labels))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label in unique_labels:
            mask = [i for i, l in enumerate(active_labels) if l == label]
            colour = "#9E9E9E" if label == -1 else cmap(label % 10)
            ax.scatter(arr[mask, 0], arr[mask, 1],
                       c=[colour]*len(mask), s=45,
                       edgecolors="black", linewidths=0.3, alpha=0.85,
                       label=f"Cluster {label}" if label != -1 else "Noise / Outliers")

        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_title("Semantic Clustering — PCA Latent Space", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=8, framealpha=0.8)

        PlotRenderer._save_dual(fig, save_dir, "semantic_clustering_pca")

    # ----------------------------------------------------------
    # 12. Routing Pipeline Compression — 3-Panel Figure
    # ----------------------------------------------------------
    @staticmethod
    def render_routing_pipeline(env, rp_nodes, rp_member_map, route_sequence, save_dir: str):
        """
        3-panel figure showing the full routing compression pipeline:
        Panel 1 — Raw IoT node deployment
        Panel 2 — RP compression (greedy neighbourhood)
        Panel 3 — Final optimised UAV route
        """
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Routing Pipeline Compression", fontsize=14, fontweight="bold")

        all_nodes = env.sensors

        def _draw_obstacles(ax):
            for obs in env.obstacles:
                rect = plt.Rectangle(
                    (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                    facecolor="#B71C1C", alpha=0.15, edgecolor="#B71C1C",
                )
                ax.add_patch(rect)

        # Panel 1: raw nodes
        ax = axes[0]
        _draw_obstacles(ax)
        ax.scatter([n.x for n in all_nodes], [n.y for n in all_nodes],
                   c="#42A5F5", s=25, edgecolors="black", linewidths=0.3)
        ax.scatter(env.uav.x, env.uav.y, c="black", s=100, marker="s")
        ax.set_title(f"(a) Raw Deployment  [{len(all_nodes)} nodes]", fontsize=10)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

        # Panel 2: RP compression
        ax = axes[1]
        _draw_obstacles(ax)
        # dim out non-RP nodes
        ax.scatter([n.x for n in all_nodes], [n.y for n in all_nodes],
                   c="#BDBDBD", s=15, edgecolors="none", alpha=0.4)
        # coverage circles for each RP
        for rp in (rp_nodes or []):
            circle = plt.Circle((rp.x, rp.y), 120, color="#FF9800", alpha=0.12)
            ax.add_patch(circle)
        ax.scatter([n.x for n in (rp_nodes or [])], [n.y for n in (rp_nodes or [])],
                   c="#FF9800", s=70, edgecolors="black", linewidths=0.5, zorder=4, marker="D")
        ax.scatter(env.uav.x, env.uav.y, c="black", s=100, marker="s")
        ax.set_title(f"(b) RP Compression  [{len(rp_nodes or [])} RPs]", fontsize=10)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)")

        # Panel 3: optimised route
        ax = axes[2]
        _draw_obstacles(ax)
        ax.scatter([n.x for n in all_nodes], [n.y for n in all_nodes],
                   c="#BDBDBD", s=15, edgecolors="none", alpha=0.3)
        seq = route_sequence or []
        if len(seq) >= 2:
            xs = [env.uav.x] + [n.x for n in seq]
            ys = [env.uav.y] + [n.y for n in seq]
            ax.plot(xs, ys, color="#4CAF50", linewidth=1.2, zorder=3)
            for i, (x, y) in enumerate(zip(xs[1:], ys[1:]), start=1):
                ax.annotate(str(i), (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=6, color="#1B5E20")
        ax.scatter([n.x for n in seq], [n.y for n in seq],
                   c="#4CAF50", s=50, edgecolors="black", linewidths=0.4, zorder=4)
        ax.scatter(env.uav.x, env.uav.y, c="black", s=100, marker="s")
        ax.set_title(f"(c) Optimised Route  [{len(seq)} waypoints]", fontsize=10)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        PlotRenderer._save_dual(fig, save_dir, "routing_pipeline")

    # ----------------------------------------------------------
    # 13. Communication Quality — Data Rate & Buffer Levels
    # ----------------------------------------------------------
    @staticmethod
    def render_communication_quality(nodes, uav_trail, save_dir: str):
        """
        Two-panel communication quality figure:
        Panel 1 — Scatter of achievable data rate vs UAV–node 2D distance
        Panel 2 — Histogram of buffer occupancy across active nodes at mission end
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        sensor_nodes = [n for n in nodes if n.id != 0]
        if not sensor_nodes:
            return

        # Compute mean UAV position from trail for distance proxy
        if uav_trail and len(uav_trail) > 0:
            mean_x = float(np.mean([p[0] for p in uav_trail]))
            mean_y = float(np.mean([p[1] for p in uav_trail]))
        else:
            mean_x, mean_y = 400.0, 300.0

        distances = [((n.x - mean_x)**2 + (n.y - mean_y)**2) ** 0.5 for n in sensor_nodes]
        data_rates = [getattr(n, "current_rate_mbps", 0.0) for n in sensor_nodes]
        buffer_fill = [
            min(100.0, getattr(n, "buffer_fill_mbits", 0.0) /
                max(getattr(n, "buffer_cap_mbits", 1.0), 1e-6) * 100)
            for n in sensor_nodes
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Communication Quality Analysis", fontsize=13, fontweight="bold")

        # Panel 1: data rate vs distance
        ax = axes[0]
        ax.scatter(distances, data_rates, c="#1565C0", s=35,
                   edgecolors="black", linewidths=0.3, alpha=0.75)
        ax.set_xlabel("UAV–Node Distance (m)")
        ax.set_ylabel("Achievable Data Rate (Mbps)")
        ax.set_title("(a) Link Rate vs Distance")

        # Fit a simple 1/d^2 trend line for visual reference
        d_sorted = sorted(distances)
        if d_sorted and d_sorted[-1] > 0:
            trend_d = np.linspace(max(1, d_sorted[0]), d_sorted[-1], 200)
            max_rate = max(data_rates) if data_rates else 1.0
            # Normalised inverse-square reference
            scale = max_rate * (min(d_sorted) ** 2) if d_sorted[0] > 0 else max_rate
            trend_r = scale / (trend_d ** 2 + 1e-6)
            ax.plot(trend_d, trend_r, color="#E53935", linewidth=1,
                    linestyle="--", label="1/d² reference")
            ax.legend(fontsize=8)

        # Panel 2: buffer occupancy histogram
        ax = axes[1]
        ax.hist(buffer_fill, bins=15, facecolor="#FF9800", edgecolor="black",
                linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Buffer Occupancy (%)")
        ax.set_ylabel("Node Count")
        ax.set_title("(b) Buffer Fill Distribution")
        ax.axvline(x=80, color="#E53935", linestyle="--", linewidth=1,
                   label="80% threshold")
        ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        PlotRenderer._save_dual(fig, save_dir, "communication_quality")

    # ----------------------------------------------------------
    # 14. Mission Progress Combined — 4-Panel
    # ----------------------------------------------------------
    @staticmethod
    def render_mission_progress_combined(
        visited_hist: list,
        battery_hist: list,
        data_hist: list,
        aoi_mean_hist: list,
        replan_steps: list,
        save_dir: str,
    ):
        """
        4-panel mission progress figure:
        [0] Visited nodes  [1] Battery discharge  [2] Data collected  [3] Mean AoI
        Replan events are overlaid as vertical markers on all panels.
        """
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        fig.suptitle("Mission Progress Overview", fontsize=14, fontweight="bold")

        series = [
            (axes[0, 0], visited_hist,   "#4CAF50", "Nodes Visited",         "Count"),
            (axes[0, 1], battery_hist,   "#1565C0", "Battery Remaining (J)", "Joules"),
            (axes[1, 0], data_hist,      "#FF9800", "Data Collected (Mbits)","Mbits"),
            (axes[1, 1], aoi_mean_hist,  "#9C27B0", "Mean AoI (steps)",      "Steps"),
        ]

        for ax, hist, colour, title, ylabel in series:
            if hist:
                ax.plot(hist, color=colour, linewidth=1.2)
                ax.fill_between(range(len(hist)), hist, alpha=0.08, color=colour)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            # Replan event markers
            for rs in replan_steps:
                if hist and 0 <= rs < len(hist):
                    ax.axvline(x=rs, color="#E53935", linestyle="--",
                               linewidth=0.5, alpha=0.5)

        axes[1, 0].set_xlabel("Simulation Step")
        axes[1, 1].set_xlabel("Simulation Step")

        # Single legend entry for replan events
        if replan_steps:
            axes[0, 0].axvline(x=-1, color="#E53935", linestyle="--",
                               linewidth=0.8, label="Replan")
            axes[0, 0].legend(fontsize=8, loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        PlotRenderer._save_dual(fig, save_dir, "mission_progress_combined")

    # ----------------------------------------------------------
    # 15. Rendezvous Compression — Before / After
    # ----------------------------------------------------------
    @staticmethod
    def render_rendezvous_compression(env, all_nodes, rp_nodes, rp_member_map, save_dir: str):
        """
        Side-by-side before/after figure illustrating RP compression:
        - Left: all nodes coloured by RP membership
        - Right: only the RP waypoints with coverage radii
        Includes a compression ratio annotation.
        """
        import numpy as np
        PlotRenderer._ensure_dir(save_dir)
        PlotRenderer._set_ieee_style()

        cmap = plt.cm.get_cmap("tab20")
        rp_list = rp_nodes or []
        rp_map  = rp_member_map or {}

        # Build node → RP colour map
        node_colour = {}
        for rp_idx, (rp, members) in enumerate(rp_map.items()):
            for m in members:
                node_colour[m] = cmap(rp_idx % 20)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Rendezvous Point Compression  "
            f"({len(all_nodes)} nodes → {len(rp_list)} RPs, "
            f"ratio = {len(all_nodes)/max(len(rp_list), 1):.1f}×)",
            fontsize=13, fontweight="bold",
        )

        def _draw_obs(ax):
            for obs in env.obstacles:
                rect = plt.Rectangle(
                    (obs.x1, obs.y1), obs.x2 - obs.x1, obs.y2 - obs.y1,
                    facecolor="#B71C1C", alpha=0.12, edgecolor="#B71C1C",
                )
                ax.add_patch(rect)

        # Left panel — all nodes coloured by cluster
        ax = axes[0]
        _draw_obs(ax)
        for n in all_nodes:
            col = node_colour.get(n.id, "#9E9E9E")
            ax.scatter(n.x, n.y, c=[col], s=30, edgecolors="black", linewidths=0.3)
        ax.scatter(env.uav.x, env.uav.y, c="black", s=120, marker="s")
        ax.set_title(f"(a) All Nodes  [{len(all_nodes)}]", fontsize=10)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")

        # Right panel — RP waypoints with coverage radii
        ax = axes[1]
        _draw_obs(ax)
        # dim underlying nodes
        ax.scatter([n.x for n in all_nodes], [n.y for n in all_nodes],
                   c="#BDBDBD", s=10, edgecolors="none", alpha=0.35)
        for rp_idx, rp in enumerate(rp_list):
            colour = cmap(rp_idx % 20)
            circle = plt.Circle((rp.x, rp.y), 120, color=colour, alpha=0.12)
            ax.add_patch(circle)
            ax.scatter(rp.x, rp.y, c=[colour], s=80, marker="D",
                       edgecolors="black", linewidths=0.5, zorder=4)
        ax.scatter(env.uav.x, env.uav.y, c="black", s=120, marker="s")
        ax.set_title(f"(b) Rendezvous Points  [{len(rp_list)}]", fontsize=10)
        ax.set_xlim(0, env.width); ax.set_ylim(0, env.height)
        ax.set_xlabel("X (m)"); ax.set_aspect("equal")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        PlotRenderer._save_dual(fig, save_dir, "rendezvous_compression")
