import matplotlib.pyplot as plt
import os


class PlotRenderer:

    @staticmethod
    def _ensure_dirs():
        os.makedirs("artifacts/figures", exist_ok=True)
        os.makedirs("artifacts/plots", exist_ok=True)

    # ----------------------------------------------------
    # ENVIRONMENT VISUAL
    # ----------------------------------------------------
    @staticmethod
    def render_environment(env):
        PlotRenderer._ensure_dirs()

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
                (rz.x1, rz.y1), rz.x2 - rz.x1, rz.y2 - rz.y1, color="orange", alpha=0.3
            )
            plt.gca().add_patch(rect)

        plt.title("UAV Environment Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        plt.grid(True)

        plt.savefig(
            "artifacts/figures/env_phase3_dynamic.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # ----------------------------------------------------
    # ENERGY & VISIT PLOTS
    # ----------------------------------------------------
    @staticmethod
    def render_energy_plots(visited, energy_consumed):
        PlotRenderer._ensure_dirs()

        # Visited Nodes
        plt.figure()
        plt.bar(["Visited Nodes"], [visited])
        plt.title("Visited Nodes")
        plt.savefig(
            "artifacts/plots/visited_nodes_phase3.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Energy Consumed
        plt.figure()
        plt.bar(["Energy Consumed"], [energy_consumed])
        plt.title("Energy Consumption")
        plt.savefig(
            "artifacts/plots/energy_consumed_phase3.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # ----------------------------------------------------
    # METRICS SNAPSHOT
    # ----------------------------------------------------
    @staticmethod
    def render_metrics_snapshot(completion_pct, efficiency):
        PlotRenderer._ensure_dirs()

        plt.figure()
        plt.bar(["Completion %", "Efficiency"], [completion_pct, efficiency])
        plt.title("Mission Metrics Snapshot")
        plt.savefig(
            "artifacts/plots/metrics_snapshot_phase3.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
