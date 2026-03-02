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
            pygame.display.set_caption("Autonomous UAV Simulation - DST-BA | ISAC Core")
            self.font = pygame.font.SysFont("Helvetica", 24)
            self.large_font = pygame.font.SysFont("Helvetica", 32, bold=True)
            self.clock = pygame.time.Clock()
            print("[Pygame] Premium UI Canvas initialized successfully.")
        else:
            print("[Warning] Pygame is not installed. Live rendering is disabled.")
        self.enabled = pygame is not None

    def render(self, env, uav, current_target, step, base_pos):
        if not self.enabled:
            return

        # Prevent frozen UI locks during calculations
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Canvas background (Premium Light Theme)
        self.screen.fill((245, 247, 250))  

        # 1. Render Environment Obstacles (Subtle outlines)
        for obs in env.obstacles:
            rect = (obs.x1, obs.y1, obs.x2 - obs.x1, obs.y2 - obs.y1)
            pygame.draw.rect(self.screen, (220, 100, 100), rect, 2, border_radius=4)
            
            # Stippling/fill effect for obstacles
            s = pygame.Surface((obs.x2 - obs.x1, obs.y2 - obs.y1), pygame.SRCALPHA)
            s.fill((220, 100, 100, 30))
            self.screen.blit(s, (obs.x1, obs.y1))

        # 1.5 Render Base Station
        bx, by = int(base_pos[0]), int(base_pos[1])
        pygame.draw.circle(self.screen, (30, 120, 255), (bx, by), 12)
        pygame.draw.circle(self.screen, (245, 247, 250), (bx, by), 6) # Inner white dot
        base_text = self.font.render("BASE STATION", True, (80, 80, 80))
        self.screen.blit(base_text, (bx - 50, by + 15))
        
        # Base Station Communication Range (Faint)
        pygame.draw.circle(self.screen, (30, 120, 255, 40), (bx, by), 150, 1)

        # 2. Render Risk Zones
        for rz in env.risk_zones:
            rect = (rz.x1, rz.y1, rz.x2 - rz.x1, rz.y2 - rz.y1)
            pygame.draw.rect(self.screen, (220, 180, 50), rect, 2, border_radius=4)

        # 3. Render Nodes (Buffer Visuals)
        for node in env.nodes[1:]: # Skip Base Station index 0 if any
            capacity_ratio = node.current_buffer / (node.buffer_capacity + 1e-6)
            
            if capacity_ratio >= 0.95:
                color = (240, 60, 60)   # Urgent
            elif capacity_ratio > 0.5:
                color = (255, 160, 40)  # Warning
            elif node.current_buffer <= 1e-3:
                color = (150, 150, 150) # Empty/Inactive
            else:
                color = (50, 200, 100)  # Nominal
                
            pos = (int(node.x), int(node.y))
            
            # Active target lock-on reticle
            if current_target and current_target.id == node.id:
                # Target lock styling
                pygame.draw.circle(self.screen, (30, 140, 255), pos, 18, 2)
                
                # Active Communication Arc Simulation
                sensing_surface = pygame.Surface((300, 300), pygame.SRCALPHA)
                pygame.draw.circle(sensing_surface, (50, 200, 100, 25), (150, 150), 100)
                self.screen.blit(sensing_surface, (pos[0]-150, pos[1]-150))
            
            # Draw Node Base
            pygame.draw.circle(self.screen, color, pos, 6)
            
            # Data Draining Visual Ring (Arc proportional to buffer size)
            if node.current_buffer > 1e-3:
                rect = pygame.Rect(pos[0]-10, pos[1]-10, 20, 20)
                end_angle = 2 * math.pi * capacity_ratio
                pygame.draw.arc(self.screen, color, rect, 0, end_angle, 2)
                
                # Draw text above it
                buf_text = self.font.render(f"{node.current_buffer:.1f}Mb", True, (100, 100, 100))
                self.screen.blit(buf_text, (pos[0] - 15, pos[1] - 25))

        # 4. Render UAV Drone
        uav_pos = (int(uav.x), int(uav.y))
        
        # UAV Drop Shadow
        pygame.draw.polygon(self.screen, (180, 180, 180), [
            (uav_pos[0]+2, uav_pos[1] - 10),
            (uav_pos[0] - 8, uav_pos[1] + 12),
            (uav_pos[0] + 12, uav_pos[1] + 12)
        ])
        
        # UAV Triangle
        pygame.draw.polygon(self.screen, (40, 50, 60), [
            (uav_pos[0], uav_pos[1] - 12),
            (uav_pos[0] - 10, uav_pos[1] + 10),
            (uav_pos[0] + 10, uav_pos[1] + 10)
        ])

        # 5. Draw Target Telemetry Line
        if current_target:
            tgt_pos = (int(current_target.x), int(current_target.y))
            pygame.draw.aaline(self.screen, (80, 140, 220), uav_pos, tgt_pos)
            
            # Distance tracker text
            dist = math.hypot(tgt_pos[0] - uav_pos[0], tgt_pos[1] - uav_pos[1])
            mid_pos = ((uav_pos[0]+tgt_pos[0])//2, (uav_pos[1]+tgt_pos[1])//2)
            dist_text = self.font.render(f"{dist:.1f}m", True, (60, 100, 150))
            self.screen.blit(dist_text, mid_pos)

        # 6. Render Heads-Up Display (HUD)
        # Background box for HUD
        hud_bg = pygame.Surface((280, 180), pygame.SRCALPHA)
        hud_bg.fill((255, 255, 255, 230))
        self.screen.blit(hud_bg, (10, 10))
        pygame.draw.rect(self.screen, (200, 205, 210), (10, 10, 280, 180), 2, border_radius=5)
        
        # HUD Title
        title = self.large_font.render("UAV TELEMETRY", True, (40, 40, 40))
        self.screen.blit(title, (20, 20))

        info = [
            f"Simulation Step:  {step}",
            f"Power Reserve:  {uav.current_battery/1000:.1f} kJ",
            f"Target Node:  {current_target.id if current_target else 'IDLE'}",
            f"Active Buffer:  {current_target.current_buffer:.2f} Mb" if current_target else "",
            f"Flight Altitude:  {uav.z:.1f} m"
        ]
        
        for i, text in enumerate(info):
            if text:
                surface = self.font.render(text, True, (80, 90, 100))
                self.screen.blit(surface, (20, 60 + i * 25))

        pygame.display.flip()
        self.clock.tick(60) # Smooth 60 FPS
