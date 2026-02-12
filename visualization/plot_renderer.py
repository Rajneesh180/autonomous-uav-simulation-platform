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
