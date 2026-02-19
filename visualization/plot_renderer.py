import matplotlib.pyplot as plt
import os


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

        plt.figure(figsize=(8, 6))

        # -------- Nodes --------
        xs = [node.x for node in env.nodes]
        ys = [node.y for node in env.nodes]
        plt.scatter(xs, ys, c="blue", label="Nodes")

        # -------- Obstacles --------
        for obs in env.obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1),
                obs.x2 - obs.x1,
                obs.y2 - obs.y1,
                color="red",
                alpha=0.4,
            )
            plt.gca().add_patch(rect)

        # -------- Risk Zones --------
        for rz in env.risk_zones:
            rect = plt.Rectangle(
                (rz.x1, rz.y1),
                rz.x2 - rz.x1,
                rz.y2 - rz.y1,
                color="orange",
                alpha=0.3,
            )
            plt.gca().add_patch(rect)

        plt.title("UAV Environment Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        plt.grid(True)
        plt.legend()

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

        plt.figure(figsize=(8, 6))

        # -------- Nodes --------
        xs = [node.x for node in env.nodes]
        ys = [node.y for node in env.nodes]
        plt.scatter(xs, ys, c="blue")

        # -------- Obstacles --------
        for obs in env.obstacles:
            rect = plt.Rectangle(
                (obs.x1, obs.y1),
                obs.x2 - obs.x1,
                obs.y2 - obs.y1,
                color="red",
                alpha=0.4,
            )
            plt.gca().add_patch(rect)

        # -------- Risk Zones --------
        for rz in env.risk_zones:
            rect = plt.Rectangle(
                (rz.x1, rz.y1),
                rz.x2 - rz.x1,
                rz.y2 - rz.y1,
                color="orange",
                alpha=0.3,
            )
            plt.gca().add_patch(rect)

        # -------- UAV Rendering --------
        if hasattr(env, "uav"):

            # --- UAV Trail ---
            if hasattr(env, "uav_trail") and len(env.uav_trail) > 1:
                trail_x = [p[0] for p in env.uav_trail]
                trail_y = [p[1] for p in env.uav_trail]

                plt.plot(
                    trail_x,
                    trail_y,
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=5,
                )

            # --- UAV Marker ---
            plt.scatter(
                env.uav.x,
                env.uav.y,
                s=120,
                marker="^",
                c="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=10,
            )

        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        plt.title(f"Step {step}")

        # -------- Replan Flash --------
        if hasattr(env, "temporal_engine"):
            flash = env.temporal_engine.consume_replan_flash()
            if flash:
                ax = plt.gca()
                ax.set_facecolor((1.0, 0.85, 0.85))
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
        plt.savefig(f"{save_dir}/{filename}", dpi=200)
        plt.close()
