import sys
import math

try:
    import pygame
except ImportError:
    pygame = None

class PygameRenderer:
    """
    Real-Time 3D/2D Projection Visualizer for Autonomous UAV Simulation.
    Translates mathematical buffer boundaries and PCA-GLS targets into visual signals.
    """
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        if pygame:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Autonomous UAV Simulation - DST-BA | PCA-GLS")
            self.font = pygame.font.SysFont(None, 24)
            self.clock = pygame.time.Clock()
            print("[Pygame] Canvas initialized successfully.")
        else:
            print("[Warning] Pygame is not installed. Live rendering is disabled.")
        self.enabled = pygame is not None

    def render(self, env, uav, current_target, step):
        if not self.enabled:
            return

        # Prevent frozen UI locks during calculations
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Canvas background (dark theme for premium contrast)
        self.screen.fill((25, 28, 36))  

        # 1. Render Environment Obstacles
        for obs in env.obstacles:
            rect = (obs.x1, obs.y1, obs.x2 - obs.x1, obs.y2 - obs.y1)
            pygame.draw.rect(self.screen, (200, 80, 80), rect, 2)
            # Semi-transparent emulation via dot grid or stippling (simplified here)
            center = ((obs.x1 + obs.x2)//2, (obs.y1 + obs.y2)//2)
            # Just an indicator
            pygame.draw.circle(self.screen, (200, 80, 80, 100), center, 2)

        # 2. Render Risk Zones
        for rz in env.risk_zones:
            rect = (rz.x1, rz.y1, rz.x2 - rz.x1, rz.y2 - rz.y1)
            pygame.draw.rect(self.screen, (220, 180, 50), rect, 1)

        # 3. Render Nodes (Buffer Sensitive Coloring)
        for node in env.nodes[1:]: # Skip Base Station if node 0 is base
            # DST-BA Core Visualization: Red = Full/Urgent, Green = Empty/Safe
            capacity_ratio = node.current_buffer / (node.buffer_capacity + 1e-6)
            
            if capacity_ratio >= 0.95:
                color = (255, 60, 60)   # Urgent
            elif capacity_ratio > 0.5:
                color = (255, 170, 0)   # Warning
            else:
                color = (60, 255, 120)  # Nominal

            pos = (int(node.x), int(node.y))
            pygame.draw.circle(self.screen, color, pos, 6)
            
            # Active target lock-on reticle
            if current_target and current_target.id == node.id:
                pygame.draw.circle(self.screen, (255, 255, 255), pos, 12, 1)
                pygame.draw.line(self.screen, (255, 255, 255), (pos[0]-15, pos[1]), (pos[0]+15, pos[1]), 1)
                pygame.draw.line(self.screen, (255, 255, 255), (pos[0], pos[1]-15), (pos[0], pos[1]+15), 1)

        # 4. Render UAV Drone
        uav_pos = (int(uav.x), int(uav.y))
        pygame.draw.polygon(self.screen, (100, 220, 255), [
            (uav_pos[0], uav_pos[1] - 12),
            (uav_pos[0] - 10, uav_pos[1] + 10),
            (uav_pos[0] + 10, uav_pos[1] + 10)
        ])

        # 5. Draw Target Telemetry Line
        if current_target:
            tgt_pos = (int(current_target.x), int(current_target.y))
            pygame.draw.aaline(self.screen, (100, 220, 255, 150), uav_pos, tgt_pos)

        # 6. Render Heads-Up Display (HUD)
        info = [
            f"TEMP/STEP: {step}",
            f"PWR/BATTR: {uav.current_battery:.0f} J",
            f"TGT/NODE: {current_target.id if current_target else 'IDLE'}",
            f"TGT/BUFF: {current_target.current_buffer:.2f} Mb" if current_target else ""
        ]
        
        for i, text in enumerate(info):
            if text:
                surface = self.font.render(text, True, (220, 220, 220))
                self.screen.blit(surface, (15, 15 + i * 25))

        pygame.display.flip()
        self.clock.tick(60) # Smooth 60 FPS
