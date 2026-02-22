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
        self.fig.canvas.manager.set_window_title("Latent C-Space Trajectory & Data Acquisition - Phase 3.7")
        
        # Setup Axes
        self.ax = self.fig.add_subplot(111)
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.set_visible(False)
        
        self.current_ax = self.ax
        
        # Setup UI Keyboard Toggle (Prevents UI threading freezes)
        self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)
        
        self.cluster_centers = []
        
    def _on_keypress(self, event):
        if event.key == 't' or event.key == 'T':
            self._toggle_view()
            
    def _toggle_view(self):
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
        
        # Setup C-Space Grid
        ax.set_xlim(0, self.env.width)
        ax.set_ylim(0, self.env.height)
        if is_3d:
            ax.set_zlim(0, 50)
            ax.set_zlabel("Z-Axis Elevation (m)")
        ax.set_xlabel("X-Axis Spatial Coordinate (m)")
        ax.set_ylabel("Y-Axis Spatial Coordinate (m)")
        ax.set_title(f"Dynamic Service Time (DST-BA) Trajectory - Iteration {step} | Energy Reserve: {uav.current_battery/1000:,.1f} kJ\n[Press 'T' to Toggle 2D/3D View]")
        ax.grid(True, linestyle=":", alpha=0.5)
        
        # 1. Primary Sink Node (Base Station)
        if is_3d:
            ax.scatter(base_pos[0], base_pos[1], 0, c='black', marker='s', s=150, label="Primary Sink Node")
        else:
            ax.scatter(base_pos[0], base_pos[1], c='black', marker='s', s=150, label="Primary Sink Node")
            
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
                
        # 3. Render Nodes (Age of Information Visualization)
        xs, ys, zs, colors, sizes = [], [], [], [], []
        
        from config.config import Config
        max_aoi = getattr(Config, 'MAX_AOI_LIMIT', 200.0)
        
        for node in self.env.nodes[1:]:
            xs.append(node.x)
            ys.append(node.y)
            zs.append(getattr(node, 'z', 0.0))
            
            # Size mapped to Buffer density
            cap_ratio = node.current_buffer / (node.buffer_capacity + 1e-6)
            sizes.append(20 + 100 * cap_ratio) # Size shrinks as buffer drains
            
            # Color mapped to Age of Information (AoI) Data Freshness
            aoi = getattr(node, 'aoi_timer', 0.0)
            
            if node.current_buffer < 0.01:
                colors.append('darkgray') # Idle / Complete
            else:
                aoi_ratio = min(1.0, aoi / max_aoi)
                if aoi_ratio > 0.8:
                    colors.append('darkred')      # Critical Staleness
                elif aoi_ratio > 0.5:
                    colors.append('orangered')
                elif aoi_ratio > 0.2:
                    colors.append('orange')
                else:
                    colors.append('mediumseagreen') # Fresh Data
            
        if is_3d:
            ax.scatter(xs, ys, zs, c=colors, s=sizes, edgecolors='black')
        else:
            ax.scatter(xs, ys, c=colors, s=sizes, edgecolors='black')
            # Text Tag rendering (Buffer / AoI)
            for node in self.env.nodes[1:]:
                if node.current_buffer > 0.01:
                    ax.text(node.x + 5, node.y + 5, f"AoI: {getattr(node, 'aoi_timer', 0.0):.1f}s", fontsize=7)

        # 4. Environment Obstacles (Collision Avoidance Prisms)
        import matplotlib.patches as patches
        for obs in self.env.obstacles:
            dx = obs.x2 - obs.x1
            dy = obs.y2 - obs.y1
            if is_3d:
                # Plot physical 3D block
                ax.bar3d(obs.x1, obs.y1, 0, dx, dy, 50, color='darkred', alpha=0.15, edgecolor='maroon')
            else:
                rect = patches.Rectangle((obs.x1, obs.y1), dx, dy, linewidth=1, edgecolor='maroon', facecolor='darkred', alpha=0.15)
                ax.add_patch(rect)

        # 5. Render UAV & Spatial Trajectory
        trail_x = [p[0] for p in self.env.uav_trail]
        trail_y = [p[1] for p in self.env.uav_trail]
        
        # Flight State Inference
        uav_state = "Transit Maneuver"
        if current_target and current_target.current_buffer > 0.01:
            uav_state = f"Active Acquisition [Target ID: {current_target.id}]"
            
        if is_3d:
            trail_z = [p[2] for p in self.env.uav_trail] if len(self.env.uav_trail) > 0 and len(self.env.uav_trail[0]) > 2 else [getattr(uav, 'z', 0.0)] * len(trail_x)
            ax.plot(trail_x, trail_y, trail_z, c='#1f77b4', alpha=0.8, linewidth=1.5, label="UAV Trajectory")
            ax.scatter(uav.x, uav.y, getattr(uav, 'z', 0.0), c='cyan', marker='^', s=120, edgecolors='blue')
            ax.text(uav.x, uav.y, getattr(uav, 'z', 0.0) + 5, f"State: {uav_state}", color='teal', fontsize=8, fontweight='bold')
            
            # Active LOS Data Link
            if current_target and current_target.current_buffer > 0.01:
                ax.plot([uav.x, current_target.x], [uav.y, current_target.y], [getattr(uav, 'z', 0.0), getattr(current_target, 'z', 0.0)], c='red', linestyle='--', linewidth=2, alpha=0.7)
                
        else:
            ax.plot(trail_x, trail_y, c='#1f77b4', alpha=0.8, linewidth=1.5, label="UAV Trajectory")
            ax.scatter(uav.x, uav.y, c='cyan', marker='^', s=120, edgecolors='blue')
            ax.text(uav.x + 5, uav.y + 10, f"State: {uav_state}", color='teal', fontsize=8, fontweight='bold')
            
            # Active LOS Data Link
            if current_target and current_target.current_buffer > 0.01:
                ax.plot([uav.x, current_target.x], [uav.y, current_target.y], c='red', linestyle='--', linewidth=2, alpha=0.7)

        # Only standard axes need a legend in top right (avoid button collision)
        if step == 1 or step % 50 == 0:
            ax.legend(loc="upper right")
            
        plt.pause(0.001)
