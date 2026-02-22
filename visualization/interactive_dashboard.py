import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import math

class InteractiveDashboard:
    """
    High-Fidelity IEEE-Style Matplotlib Live Dashboard for Phase 3.6.
    Replaces Pygame with an interactive 2D/3D toggleable view.
    """
    def __init__(self, env):
        self.env = env
        self.is_3d = False
        
        # Turn on interactive mode
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Autonomous UAV Simulation - Phase 3.6")
        
        # Setup Axes
        self.ax = self.fig.add_subplot(111)
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.set_visible(False)
        
        self.current_ax = self.ax
        
        # Setup UI Button
        self.ax_toggle = plt.axes([0.8, 0.05, 0.1, 0.05])
        self.btn_toggle = widgets.Button(self.ax_toggle, 'Toggle 2D/3D')
        self.btn_toggle.on_clicked(self._toggle_view)
        
        self.cluster_centers = []
            
    def _toggle_view(self, event):
        self.is_3d = not self.is_3d
        self.ax.set_visible(not self.is_3d)
        self.ax_3d.set_visible(self.is_3d)
        self.current_ax = self.ax_3d if self.is_3d else self.ax
        plt.draw()
        
    def render(self, uav, current_target, step, base_pos, active_centroids=None):
        self.current_ax.clear()
        
        ax = self.current_ax
        is_3d = self.is_3d
        
        # Load Semantic Clusters
        self.cluster_centers = []
        if active_centroids is not None:
            for i, centroid in enumerate(active_centroids):
                # Ignore zero-vector noise from DBSCAN
                if not np.all(centroid == 0):
                    self.cluster_centers.append((centroid[0], centroid[1], f"Cluster {i+1}"))
        
        # Setup Grid
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        if is_3d:
            ax.set_zlim(0, 50)
            ax.set_zlabel("Altitude (m)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"IEEE-Style UAV Trajectory - Step {step} | Battery: {uav.current_battery/1000:.1f}kJ")
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # 1. Base Station
        if is_3d:
            ax.scatter(base_pos[0], base_pos[1], 0, c='black', marker='^', s=200, label="Base Station")
        else:
            ax.scatter(base_pos[0], base_pos[1], c='black', marker='^', s=200, label="Base Station")
            
        # 2. Cluster Communication Zones
        comm_radius = 120
        for cx, cy, label in self.cluster_centers:
            if is_3d:
                # 3D Circle is tricky in matplotlib without complex patches, so plot ring points
                theta = np.linspace(0, 2*np.pi, 50)
                x_ring = cx + comm_radius * np.cos(theta)
                y_ring = cy + comm_radius * np.sin(theta)
                ax.plot(x_ring, y_ring, 0, color='purple', alpha=0.3, linestyle='-.')
                ax.text(cx, cy, 2, label, color='purple', fontsize=12, fontweight='bold')
            else:
                circle = Circle((cx, cy), comm_radius, fill=True, color='purple', alpha=0.1, linestyle='-.', lw=2)
                ax.add_patch(circle)
                ax.text(cx, cy + comm_radius + 5, f"{label} Zone", color='purple', fontsize=10, ha='center')
                
        # 3. Render Nodes
        xs, ys, zs, colors, sizes = [], [], [], [], []
        for node in self.env.nodes[1:]:
            xs.append(node.x)
            ys.append(node.y)
            zs.append(getattr(node, 'z', 0.0))
            
            # Buffer draining visualization
            cap_ratio = node.current_buffer / (node.buffer_capacity + 1e-6)
            sizes.append(20 + 100 * cap_ratio) # Size shrinks as buffer drains
            
            if cap_ratio > 0.95: colors.append('red')
            elif cap_ratio > 0.5: colors.append('orange')
            elif cap_ratio < 0.01: colors.append('gray')
            else: colors.append('green')
            
        if is_3d:
            ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolors='black')
        else:
            ax.scatter(xs, ys, c=colors, s=sizes, edgecolors='black')
            # Data draining text
            for node in self.env.nodes[1:]:
                if node.current_buffer > 0.01:
                    ax.text(node.x + 5, node.y + 5, f"{node.current_buffer:.1f}", fontsize=8)

        # 4. Render UAV & Trail
        trail_x = [p[0] for p in self.env.uav_trail]
        trail_y = [p[1] for p in self.env.uav_trail]
        
        if is_3d:
            trail_z = [p[2] for p in self.env.uav_trail] if len(self.env.uav_trail) > 0 and len(self.env.uav_trail[0]) > 2 else [getattr(uav, 'z', 0.0)] * len(trail_x)
            ax.plot(trail_x, trail_y, trail_z, c='blue', alpha=0.6, linewidth=2, label="Trajectory")
            ax.scatter(uav.x, uav.y, getattr(uav, 'z', 0.0), c='cyan', marker='o', s=150, edgecolors='blue')
            if current_target:
                ax.plot([uav.x, current_target.x], [uav.y, current_target.y], [getattr(uav, 'z', 0.0), getattr(current_target, 'z', 0.0)], c='red', linestyle=':')
        else:
            ax.plot(trail_x, trail_y, c='blue', alpha=0.6, linewidth=2, label="Trajectory")
            ax.scatter(uav.x, uav.y, c='cyan', marker='o', s=150, edgecolors='blue')
            if current_target:
                ax.plot([uav.x, current_target.x], [uav.y, current_target.y], c='red', linestyle=':')

        # Only standard axes need a legend in top right (avoid button collision)
        if step == 1 or step % 50 == 0:
            ax.legend(loc="upper right")
            
        plt.pause(0.001)
