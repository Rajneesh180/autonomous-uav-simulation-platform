import matplotlib.pyplot as plt


class PlotRenderer:

    @staticmethod
    def render_environment(env):
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
                label="Obstacle"
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
                label="Risk Zone"
            )
            plt.gca().add_patch(rect)

        plt.title("UAV Environment Visualization")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper right")
        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
        plt.grid(True)

        plt.savefig("artifacts/figures/env_phase2.png", dpi=300, bbox_inches="tight")
        plt.show()
