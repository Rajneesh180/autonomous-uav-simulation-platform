from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.cm import ScalarMappable           # noqa: E402
from matplotlib.colors import Normalize            # noqa: E402
from matplotlib.lines import Line2D                # noqa: E402
from matplotlib.patches import (                   # noqa: E402
    Circle, FancyBboxPatch, Patch, Rectangle,
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]  # noqa: E402

from config.settings import (
    MAP_W, MAP_H, NODE_COUNT, RP_RADIUS,
    BASE_STATION, UAV_ALTITUDE, BATTERY_CAPACITY, OUT_DIR,
    P_HOVER, E_FLY_PER_M, HOVER_TIME,
    Node, Obstacle,
)
from core.rp_selection import select_rendezvous_points
from core.path_planning import compute_expected_path
from visualization.helpers import draw_uav_icon, draw_base_station, setup_2d_axes


# --- 1. iot deployment ---

def diagram_iot_deployment(nodes: list[Node], obstacles: list[Obstacle]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    setup_2d_axes(ax, "IoT Sensor Network Deployment")

    boundary = Rectangle(
        (0, 0), MAP_W, MAP_H, linewidth=2, edgecolor="black",
        facecolor="none", linestyle="-", zorder=1,
    )
    ax.add_patch(boundary)

    for o in obstacles:
        rect = FancyBboxPatch(
            (o["x1"], o["y1"]), o["x2"] - o["x1"], o["y2"] - o["y1"],
            boxstyle="round,pad=3", facecolor="#FFCDD2", edgecolor="#D32F2F",
            linewidth=1.5, alpha=0.4, linestyle="--", zorder=2,
        )
        ax.add_patch(rect)

    cmap = plt.get_cmap("RdYlGn_r")
    norm = Normalize(1, 10)

    for n in nodes:
        color = cmap(norm(n["priority"]))
        circle = Circle(
            (n["x"], n["y"]), 12, color=color, ec="black", linewidth=0.8, zorder=5,
        )
        ax.add_patch(circle)
        ax.annotate(
            f'N{n["id"]}', (n["x"], n["y"]), fontsize=7,
            ha="center", va="center", fontweight="bold", zorder=6,
        )

    for n in nodes:
        comm_range = Circle(
            (n["x"], n["y"]), RP_RADIUS, fill=False,
            edgecolor="#90CAF9", alpha=0.25, linewidth=0.5, linestyle=":", zorder=3,
        )
        ax.add_patch(comm_range)

    draw_base_station(ax, BASE_STATION[0], BASE_STATION[1])
    ax.annotate(
        "Base Station", (BASE_STATION[0], BASE_STATION[1] - 20),
        fontsize=9, ha="center", color="#FF5722", fontweight="bold",
    )

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, label="Node Priority (1=Low, 10=High)")

    legend_elements = [
        Patch(facecolor="#FFCDD2", edgecolor="#D32F2F", alpha=0.4,
              label="Obstacle Zone"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=10, label="IoT Sensor Node"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.text(
        MAP_W / 2, MAP_H + 10,
        f"Deployment Area: {MAP_W}m \u00d7 {MAP_H}m  |  N = {NODE_COUNT} nodes",
        ha="center", fontsize=10, style="italic",
    )

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase1_iot_deployment.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  1/10  iot_deployment")


# --- 2. obstacle deployment ---

def diagram_obstacle_deployment(nodes: list[Node], obstacles: list[Obstacle]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    setup_2d_axes(ax, "Obstacle Deployment in Operational Area")

    boundary = Rectangle(
        (0, 0), MAP_W, MAP_H, linewidth=2, edgecolor="black", facecolor="none", zorder=1,
    )
    ax.add_patch(boundary)

    for o in obstacles:
        rect = Rectangle(
            (o["x1"], o["y1"]), o["x2"] - o["x1"], o["y2"] - o["y1"],
            facecolor="#EF5350", edgecolor="#B71C1C", linewidth=2, alpha=0.8, zorder=4,
        )
        ax.add_patch(rect)
        ax.annotate(
            f'Obs-{o["id"]}\nh={o["height"]}m',
            (o["cx"], o["cy"]), fontsize=8, ha="center", va="center",
            color="white", fontweight="bold", zorder=5,
        )

    for n in nodes:
        is_in = any(
            o["x1"] <= n["x"] <= o["x2"] and o["y1"] <= n["y"] <= o["y2"]
            for o in obstacles
        )
        color = "#FF9800" if is_in else "#4CAF50"
        marker = "s" if is_in else "o"
        ax.plot(n["x"], n["y"], marker, color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=0.8, zorder=6)
        ax.annotate(f'N{n["id"]}', (n["x"] + 10, n["y"] + 8), fontsize=6.5, zorder=7)

    draw_base_station(ax, BASE_STATION[0], BASE_STATION[1])

    legend_elements = [
        Patch(facecolor="#EF5350", edgecolor="#B71C1C",
              label="Obstacle (Rectangular)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50",
               markersize=8, label="Safe Node"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF9800",
               markersize=8, label="Node Inside Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase2_obstacle_deployment.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  2/10  obstacle_deployment")


# --- 3. obstacle 3d ---

def diagram_obstacle_3d(obstacles: list[Obstacle]) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Obstacle Model: Rectangular/Cuboid Representation",
                 fontsize=13, fontweight="bold", pad=15)

    colors = ["#EF5350", "#42A5F5", "#66BB6A"]

    for idx, o in enumerate(obstacles):
        x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
        h = o["height"]
        c = colors[idx % len(colors)]

        verts_bottom = [(x1, y1, 0), (x2, y1, 0), (x2, y2, 0), (x1, y2, 0)]
        verts_top = [(x1, y1, h), (x2, y1, h), (x2, y2, h), (x1, y2, h)]
        sides = [
            [verts_bottom[0], verts_bottom[1], verts_top[1], verts_top[0]],
            [verts_bottom[1], verts_bottom[2], verts_top[2], verts_top[1]],
            [verts_bottom[2], verts_bottom[3], verts_top[3], verts_top[2]],
            [verts_bottom[3], verts_bottom[0], verts_top[0], verts_top[3]],
        ]

        for face in [verts_bottom, verts_top] + sides:
            poly = Poly3DCollection(
                [face], alpha=0.35, facecolor=c, edgecolor="black", linewidth=0.5,
            )
            ax.add_collection3d(poly)

        ax.text(o["cx"], o["cy"], h + 5,
                f'Obs-{o["id"]}\n({int(x2 - x1)}\u00d7{int(y2 - y1)}\u00d7{h}m)',
                fontsize=8, ha="center", zorder=10)

    xx, yy = np.meshgrid(np.linspace(0, MAP_W, 5), np.linspace(0, MAP_H, 5))
    ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.08, color="#8BC34A")

    ax.plot([MAP_W / 2, MAP_W / 2], [MAP_H / 2, MAP_H / 2], [0, UAV_ALTITUDE],
            "b--", linewidth=1.5, alpha=0.6)
    ax.text(MAP_W / 2 + 20, MAP_H / 2, UAV_ALTITUDE,
            f"UAV Alt = {UAV_ALTITUDE}m", fontsize=9, color="blue")

    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    ax.set_xlim(0, MAP_W)
    ax.set_ylim(0, MAP_H)
    ax.set_zlim(0, UAV_ALTITUDE + 20)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase2_obstacle_3d_model.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  3/10  obstacle_3d")


# --- 4. rp selection ---

def diagram_rp_selection(
    nodes: list[Node], obstacles: list[Obstacle],
    rps: list[int], member_map: dict[int, list[int]],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    setup_2d_axes(ax, "Rendezvous Point (RP) Selection")

    boundary = Rectangle(
        (0, 0), MAP_W, MAP_H, linewidth=2, edgecolor="black", facecolor="none", zorder=1,
    )
    ax.add_patch(boundary)

    for o in obstacles:
        rect = Rectangle(
            (o["x1"], o["y1"]), o["x2"] - o["x1"], o["y2"] - o["y1"],
            facecolor="#EF5350", edgecolor="#B71C1C", linewidth=1.5, alpha=0.5, zorder=2,
        )
        ax.add_patch(rect)

    rp_colors = plt.get_cmap("Set2")(np.linspace(0, 1, max(len(rps), 1)))

    for idx, rp in enumerate(rps):
        color = rp_colors[idx]
        circle = Circle(
            (nodes[rp]["x"], nodes[rp]["y"]), RP_RADIUS,
            fill=True, facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.5, linestyle="--", zorder=3,
        )
        ax.add_patch(circle)

    for idx, rp in enumerate(rps):
        color = rp_colors[idx]
        for member in member_map.get(rp, []):
            ax.plot(
                [nodes[rp]["x"], nodes[member]["x"]],
                [nodes[rp]["y"], nodes[member]["y"]],
                color=color, linewidth=1, alpha=0.5, linestyle=":", zorder=3,
            )

    rp_set = set(rps)
    for n in nodes:
        if n["id"] not in rp_set:
            ax.plot(n["x"], n["y"], "o", color="#9E9E9E", markersize=7,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)
            ax.annotate(f'N{n["id"]}', (n["x"] + 8, n["y"] + 6),
                        fontsize=6, color="#666", zorder=6)

    for idx, rp in enumerate(rps):
        color = rp_colors[idx]
        ax.plot(nodes[rp]["x"], nodes[rp]["y"], "^", color=color, markersize=14,
                markeredgecolor="black", markeredgewidth=1.5, zorder=7)
        ax.annotate(f"RP{idx}", (nodes[rp]["x"] + 12, nodes[rp]["y"] + 10),
                    fontsize=9, fontweight="bold", color=color, zorder=8)

    draw_base_station(ax, BASE_STATION[0], BASE_STATION[1])
    ax.annotate("BS", (BASE_STATION[0], BASE_STATION[1] - 18),
                fontsize=9, ha="center", color="#FF5722", fontweight="bold")

    ax.text(
        MAP_W / 2, MAP_H + 10,
        f"N = {NODE_COUNT} nodes \u2192 {len(rps)} RPs  |  "
        f"Compression: {100 * (1 - len(rps) / NODE_COUNT):.0f}%  |  "
        f"Coverage Radius: {RP_RADIUS}m",
        ha="center", fontsize=10, style="italic",
    )

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#4CAF50",
               markersize=12, label="Rendezvous Point (RP)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E",
               markersize=8, label="Member Node"),
        Patch(facecolor="#EF5350", alpha=0.5, label="Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase3_rp_selection.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  4/10  rp_selection")


# --- 5. obstacle effect on RP ---

def diagram_obstacle_rp_effect(
    nodes: list[Node], obstacles: list[Obstacle],
    rps: list[int], member_map: dict[int, list[int]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for panel, (ax, title, show_obs) in enumerate(zip(
        axes,
        ["Without Obstacles", "With Obstacles (Obstacle-Aware Selection)"],
        [False, True],
    )):
        setup_2d_axes(ax, title)
        boundary = Rectangle(
            (0, 0), MAP_W, MAP_H, linewidth=1.5, edgecolor="black",
            facecolor="none", zorder=1,
        )
        ax.add_patch(boundary)

        if show_obs:
            for o in obstacles:
                rect = Rectangle(
                    (o["x1"], o["y1"]), o["x2"] - o["x1"], o["y2"] - o["y1"],
                    facecolor="#EF5350", edgecolor="#B71C1C", linewidth=1.5,
                    alpha=0.6, zorder=2,
                )
                ax.add_patch(rect)

        cur_rps = rps if show_obs else select_rendezvous_points(nodes, [], RP_RADIUS)[0]
        rp_set = set(cur_rps)

        for n in nodes:
            if n["id"] in rp_set:
                ax.plot(n["x"], n["y"], "^", color="#2196F3", markersize=12,
                        markeredgecolor="black", markeredgewidth=1, zorder=6)
            else:
                ax.plot(n["x"], n["y"], "o", color="#9E9E9E", markersize=6,
                        markeredgecolor="black", markeredgewidth=0.5, zorder=5)

        ax.text(10, MAP_H - 20, f"RPs: {len(cur_rps)}", fontsize=11,
                fontweight="bold", color="#2196F3")

    fig.suptitle("Impact of Obstacles on RP Selection",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase3_obstacle_rp_effect.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  5/10  obstacle_rp_effect")


# --- 6. expected path ---

def diagram_expected_path(
    nodes: list[Node], obstacles: list[Obstacle],
    rps: list[int], member_map: dict[int, list[int]],
) -> float:
    path, _ = compute_expected_path(nodes, rps, BASE_STATION)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
    setup_2d_axes(ax, "Expected UAV Trajectory (Shortest Path Estimate)")

    boundary = Rectangle(
        (0, 0), MAP_W, MAP_H, linewidth=2, edgecolor="black", facecolor="none", zorder=1,
    )
    ax.add_patch(boundary)

    for o in obstacles:
        rect = Rectangle(
            (o["x1"], o["y1"]), o["x2"] - o["x1"], o["y2"] - o["y1"],
            facecolor="#EF5350", edgecolor="#B71C1C", linewidth=1.5, alpha=0.5, zorder=2,
        )
        ax.add_patch(rect)

    for n in nodes:
        ax.plot(n["x"], n["y"], "o", color="#BDBDBD", markersize=5,
                markeredgecolor="#757575", markeredgewidth=0.5, zorder=3)

    rp_colors = plt.get_cmap("Set1")(np.linspace(0, 0.8, len(rps)))
    for idx, rp in enumerate(rps):
        ax.plot(nodes[rp]["x"], nodes[rp]["y"], "^", color=rp_colors[idx],
                markersize=14, markeredgecolor="black", markeredgewidth=1.5, zorder=7)
        circle = Circle(
            (nodes[rp]["x"], nodes[rp]["y"]), RP_RADIUS,
            fill=False, edgecolor=rp_colors[idx], linewidth=1,
            linestyle=":", alpha=0.4, zorder=4,
        )
        ax.add_patch(circle)

    if len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, "-", color="#1565C0", linewidth=2.5, alpha=0.8, zorder=6)

        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            dist = math.hypot(dx, dy)
            if dist > 0:
                mid_x = path[i][0] + dx * 0.5
                mid_y = path[i][1] + dy * 0.5
                ax.annotate(
                    "", xy=(mid_x + dx * 0.02, mid_y + dy * 0.02),
                    xytext=(mid_x - dx * 0.02, mid_y - dy * 0.02),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=2),
                )

    total_dist: float = sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )

    draw_uav_icon(ax, BASE_STATION[0] + 30, BASE_STATION[1] + 30, size=15)
    draw_base_station(ax, BASE_STATION[0], BASE_STATION[1])

    for i, (px, py) in enumerate(path[1:-1], 1):
        ax.annotate(
            str(i), (px - 5, py - 15), fontsize=10, fontweight="bold",
            color="#1565C0",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#1565C0", alpha=0.9),
            zorder=8,
        )

    ax.text(
        MAP_W / 2, MAP_H + 10,
        f"Expected Path: BS \u2192 {len(rps)} RPs \u2192 BS  |  "
        f"Estimated Distance: {total_dist:.0f}m",
        ha="center", fontsize=10, style="italic",
    )

    legend_elements = [
        Line2D([0], [0], color="#1565C0", linewidth=2.5, label="Expected UAV Path"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#E91E63",
               markersize=12, label="Rendezvous Point"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#BDBDBD",
               markersize=6, label="IoT Sensor Node"),
        Patch(facecolor="#EF5350", alpha=0.5, label="Obstacle"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "phase4_expected_path.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  6/10  expected_path")
    return total_dist


# --- 7. energy profile ---

def diagram_energy_profile(nodes: list[Node], rps: list[int], total_dist: float) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    n_rps: int = len(rps)
    segments: int = n_rps + 1
    segment_dist: float = total_dist / segments if segments > 0 else 0

    cumulative_energy: list[float] = [0]
    labels: list[str] = ["BS"]

    for i in range(n_rps):
        e_flight = segment_dist * E_FLY_PER_M
        cumulative_energy.append(cumulative_energy[-1] + e_flight)
        labels.append(f"Fly\u2192RP{i + 1}")

        e_hover = P_HOVER * HOVER_TIME
        cumulative_energy.append(cumulative_energy[-1] + e_hover)
        labels.append(f"Hover@RP{i + 1}")

    e_return = segment_dist * E_FLY_PER_M
    cumulative_energy.append(cumulative_energy[-1] + e_return)
    labels.append("Return\u2192BS")

    battery_pct = [(BATTERY_CAPACITY - e) / BATTERY_CAPACITY * 100
                   for e in cumulative_energy]

    x_pos = range(len(cumulative_energy))

    ax.bar(x_pos, [e / 1000 for e in cumulative_energy],
           color="#42A5F5", alpha=0.7, edgecolor="#1565C0")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cumulative Energy Consumed (kJ)", fontsize=11, color="#1565C0")
    ax.set_title("Expected UAV Energy Consumption Profile",
                 fontsize=13, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(list(x_pos), battery_pct, "r-o", linewidth=2, markersize=6, label="Battery %")
    ax2.set_ylabel("Remaining Battery (%)", fontsize=11, color="red")
    ax2.set_ylim(0, 105)
    ax2.axhline(y=15, color="red", linestyle="--", alpha=0.5,
                label="Return Threshold (15%)")
    ax2.legend(loc="upper right")

    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "expected_energy_profile.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  7/10  energy_profile")


# --- 8. rl framework ---

def diagram_rl_framework() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Proposed RL-Based UAV Path Planning Framework",
                 fontsize=14, fontweight="bold", pad=15)

    # Agent box
    agent_box = FancyBboxPatch(
        (0.5, 4), 3, 2, boxstyle="round,pad=0.3",
        facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2,
    )
    ax.add_patch(agent_box)
    ax.text(2, 5.3, "RL Agent", fontsize=13, fontweight="bold",
            ha="center", color="#1565C0")
    ax.text(2, 4.7, "(UAV Controller)", fontsize=10, ha="center", color="#1565C0")
    ax.text(2, 4.2, "Policy \u03c0(s) \u2192 a", fontsize=9,
            ha="center", color="#666", style="italic")

    # Environment box
    env_box = FancyBboxPatch(
        (7.5, 4), 4, 2, boxstyle="round,pad=0.3",
        facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2,
    )
    ax.add_patch(env_box)
    ax.text(9.5, 5.3, "Environment", fontsize=13, fontweight="bold",
            ha="center", color="#2E7D32")
    ax.text(9.5, 4.7, "(IoT Network + Obstacles)", fontsize=10,
            ha="center", color="#2E7D32")
    ax.text(9.5, 4.2, "State s, Reward r", fontsize=9,
            ha="center", color="#666", style="italic")

    # Action arrow
    ax.annotate("", xy=(7.3, 5.5), xytext=(3.7, 5.5),
                arrowprops=dict(arrowstyle="-|>", color="#FF5722", lw=2.5))
    ax.text(5.5, 5.8, "Action a", fontsize=11, ha="center",
            fontweight="bold", color="#FF5722")
    ax.text(5.5, 5.5, "(Move direction, hover)", fontsize=8,
            ha="center", color="#FF5722")

    # State+Reward arrow
    ax.annotate("", xy=(3.7, 4.5), xytext=(7.3, 4.5),
                arrowprops=dict(arrowstyle="-|>", color="#7B1FA2", lw=2.5))
    ax.text(5.5, 4.0, "State s\u2032, Reward r", fontsize=11,
            ha="center", fontweight="bold", color="#7B1FA2")

    # State components
    state_items: list[str] = [
        "State Space:",
        "\u2022 UAV position (x, y)",
        "\u2022 Remaining battery",
        "\u2022 Nearest obstacle distance",
        "\u2022 Unvisited node locations",
        "\u2022 Node data levels",
    ]
    for i, item in enumerate(state_items):
        weight = "bold" if i == 0 else "normal"
        ax.text(1.2, 3.2 - i * 0.4, item, fontsize=9, fontweight=weight, color="#333")

    # Action components
    action_items = [
        "Action Space:",
        "\u2022 Move in 8 directions",
        "\u2022 Hover & collect data",
        "\u2022 Return to base",
    ]
    for i, item in enumerate(action_items):
        weight = "bold" if i == 0 else "normal"
        ax.text(5, 3.2 - i * 0.4, item, fontsize=9, fontweight=weight, color="#333")

    # Reward components
    reward_items = [
        "Reward Function:",
        "\u2022 +\u03b1\u2081 per node visited",
        "\u2022 \u2212\u03b1\u2082 \u00d7 energy consumed",
        "\u2022 \u2212\u03b1\u2083 \u00d7 collision penalty",
        "\u2022 +\u03b1\u2084 \u00d7 data collected",
        "\u2022 \u2212\u03b1\u2085 \u00d7 path length",
    ]
    for i, item in enumerate(reward_items):
        weight = "bold" if i == 0 else "normal"
        ax.text(8.5, 3.2 - i * 0.4, item, fontsize=9, fontweight=weight, color="#333")

    # Background boxes
    for bx, by, bw, bh, bc in [
        (0.5, 0.5, 3.8, 3, "#E3F2FD"),
        (4.3, 0.5, 3.5, 2.5, "#FFF3E0"),
        (8, 0.5, 3.5, 3, "#F3E5F5"),
    ]:
        box = FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.2",
            facecolor=bc, edgecolor="#999", linewidth=1, alpha=0.5,
        )
        ax.add_patch(box)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rl_framework_concept.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  8/10  rl_framework")


# --- 9. methodology flowchart ---

def diagram_methodology_flowchart() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("Proposed Methodology: Step-wise Workflow",
                 fontsize=14, fontweight="bold", pad=15)

    boxes = [
        (5, 13, "IoT Network Deployment", "#E3F2FD", "#1565C0",
         "Deploy N sensor nodes with\nrandom positions & priorities"),
        (5, 11, "Obstacle Deployment", "#FFEBEE", "#C62828",
         "Place rectangular obstacles\nwith 3D height model (cuboid)"),
        (5, 9, "Rendezvous Point Selection", "#E8F5E9", "#2E7D32",
         "Greedy dominating set to\nselect rendezvous points"),
        (5, 7, "Expected Path Planning", "#FFF3E0", "#E65100",
         "Estimate shortest UAV path\nthrough selected RPs"),
        (5, 5, "RL-Based Optimization", "#F3E5F5", "#6A1B9A",
         "Train RL agent for adaptive\nobstacle-aware path planning"),
        (5, 3, "Evaluation & Analysis", "#E0F7FA", "#00695C",
         "Compare expected vs. RL path\nEnergy, coverage, mission time"),
    ]

    for cx, cy, title, bg, ec, desc in boxes:
        box = FancyBboxPatch(
            (cx - 2.8, cy - 0.8), 5.6, 1.6, boxstyle="round,pad=0.3",
            facecolor=bg, edgecolor=ec, linewidth=2,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.3, title, fontsize=11, fontweight="bold",
                ha="center", color=ec)
        ax.text(cx, cy - 0.3, desc, fontsize=9, ha="center", color="#333")

    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(5, boxes[i + 1][1] + 0.8),
                    xytext=(5, boxes[i][1] - 0.8),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=2))

    # Side annotations
    side_labels = [
        (13, "Work\nDone \u2713", "#2E7D32", "#C8E6C9"),
        (11, "Work\nDone \u2713", "#2E7D32", "#C8E6C9"),
        (9,  "Work\nDone \u2713", "#2E7D32", "#C8E6C9"),
        (7,  "Work\nDone \u2713", "#2E7D32", "#C8E6C9"),
        (5,  "In\nProgress", "#E65100", "#FFE0B2"),
        (3,  "Future\nWork", "#666", "#EEEEEE"),
    ]
    for y, text, color, bg in side_labels:
        ec = "#2E7D32" if "Done" in text else ("#E65100" if "Progress" in text else "#999")
        ax.text(8.5, y, text, fontsize=10, ha="center", color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor=bg, edgecolor=ec))

    # Mid-Term bracket
    ax.annotate("", xy=(1, 6.2), xytext=(1, 13.8),
                arrowprops=dict(arrowstyle="-", color="#1565C0", lw=1.5,
                                connectionstyle="arc3,rad=0.1"))
    ax.text(0.3, 10, "Mid-Term\nScope", fontsize=10, ha="center",
            color="#1565C0", fontweight="bold", rotation=90)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "methodology_flowchart.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  9/10  methodology_flowchart")


# --- 10. timeline ---

def diagram_timeline() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Project Timeline: Work Done & Work In Progress",
                 fontsize=14, fontweight="bold", pad=15)

    phases = [
        {"name": "Literature\nReview",       "start": 0.5, "width": 2,   "color": "#4CAF50", "status": "Done"},
        {"name": "Environment\nSetup",       "start": 2,   "width": 2.5, "color": "#4CAF50", "status": "Done"},
        {"name": "RP Selection",             "start": 4,   "width": 2,   "color": "#4CAF50", "status": "Done"},
        {"name": "Expected Path\nPlanning",  "start": 5.5, "width": 2,   "color": "#4CAF50", "status": "Done"},
        {"name": "RL Study &\nImplementation","start": 7,  "width": 2.5, "color": "#FF9800", "status": "In Progress"},
        {"name": "Evaluation &\nFinal Report","start": 9,  "width": 2.5, "color": "#BDBDBD", "status": "Planned"},
    ]

    for p in phases:
        y = 3.5
        bar = FancyBboxPatch(
            (p["start"], y - 0.4), p["width"], 0.8, boxstyle="round,pad=0.1",
            facecolor=p["color"], edgecolor="black", linewidth=1, alpha=0.8,
        )
        ax.add_patch(bar)
        text_c = "white" if p["color"] != "#BDBDBD" else "black"
        ax.text(p["start"] + p["width"] / 2, y, p["name"], fontsize=8,
                ha="center", va="center", fontweight="bold", color=text_c)
        ax.text(p["start"] + p["width"] / 2, y - 0.7, p["status"], fontsize=8,
                ha="center", color=p["color"], fontweight="bold")

    months = ["Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "May 2026"]
    for i, m in enumerate(months):
        x = 0.5 + i * 2.5
        ax.text(x, 2.2, m, fontsize=9, ha="center", color="#666")
        ax.plot([x, x], [2.4, 2.6], "k-", linewidth=1)

    ax.plot([0.5, 11], [2.5, 2.5], "k-", linewidth=1.5)

    ax.annotate("Mid-Term\n(Mar 2026)", xy=(5.5, 2.5), xytext=(5.5, 1.2),
                fontsize=10, ha="center", fontweight="bold", color="#D32F2F",
                arrowprops=dict(arrowstyle="-|>", color="#D32F2F", lw=2))

    for x, c, label in [(1, "#4CAF50", "Completed"),
                         (4, "#FF9800", "In Progress"),
                         (7, "#BDBDBD", "Planned")]:
        box = FancyBboxPatch(
            (x, 0.5), 0.4, 0.3, boxstyle="round,pad=0.05",
            facecolor=c, edgecolor="black", linewidth=0.5,
        )
        ax.add_patch(box)
        ax.text(x + 0.7, 0.65, label, fontsize=9, va="center")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "project_timeline.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(" 10/10  timeline")


# --- generate all ---

def generate_all_diagrams(
    nodes: list[Node],
    obstacles: list[Obstacle],
    rps: list[int],
    member_map: dict[int, list[int]],
) -> None:
    # generates all 10 figures and saves them
    os.makedirs(OUT_DIR, exist_ok=True)

    print("generating diagrams ...")
    print(f"  {NODE_COUNT} nodes, {len(obstacles)} obstacles, {len(rps)} RPs\n")

    diagram_iot_deployment(nodes, obstacles)
    diagram_obstacle_deployment(nodes, obstacles)
    diagram_obstacle_3d(obstacles)
    diagram_rp_selection(nodes, obstacles, rps, member_map)
    diagram_obstacle_rp_effect(nodes, obstacles, rps, member_map)
    total_dist = diagram_expected_path(nodes, obstacles, rps, member_map)
    diagram_energy_profile(nodes, rps, total_dist)
    diagram_rl_framework()
    diagram_methodology_flowchart()
    diagram_timeline()

    print(f"\nall 10 diagrams saved to {OUT_DIR}")
