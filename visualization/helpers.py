

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes                   # noqa: E402
from matplotlib.patches import Arc, Circle, Polygon  # noqa: E402

from config.settings import MAP_W, MAP_H


def draw_uav_icon(
    ax: Axes, x: float, y: float, size: float = 18, color: str = "#2196F3",
) -> None:
    # small drone icon with 4 rotors
    body = Circle((x, y), size * 0.35, color=color, zorder=10)
    ax.add_patch(body)
    offsets: list[tuple[int, int]] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in offsets:
        arm_x: float = x + dx * size * 0.6
        arm_y: float = y + dy * size * 0.6
        ax.plot([x, arm_x], [y, arm_y], color=color, linewidth=2, zorder=9)
        rotor = Circle(
            (arm_x, arm_y), size * 0.22, color=color, alpha=0.5, zorder=10,
        )
        ax.add_patch(rotor)


def draw_base_station(ax: Axes, x: float, y: float, size: float = 12) -> None:
    # triangle + signal arcs
    triangle = Polygon(
        [(x, y + size), (x - size * 0.6, y - size * 0.4),
         (x + size * 0.6, y - size * 0.4)],
        color="#FF5722", zorder=10,
    )
    ax.add_patch(triangle)
    for r in [size * 0.8, size * 1.2]:
        arc = Arc(
            (x, y + size), r, r, angle=0, theta1=30, theta2=150,
            color="#FF5722", linewidth=1.5, zorder=9,
        )
        ax.add_patch(arc)


def setup_2d_axes(ax: Axes, title: str = "") -> None:
    # common axis setup for 2d plots
    ax.set_xlim(-20, MAP_W + 20)
    ax.set_ylim(-20, MAP_H + 20)
    ax.set_xlabel("X Position (m)", fontsize=11)
    ax.set_ylabel("Y Position (m)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#FAFAFA")
