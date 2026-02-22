import math
from typing import List, Tuple
from core.obstacle_model import Obstacle
from config.config import Config
from metrics.metric_engine import MetricEngine
import copy

class DigitalTwinMap:
    """
    Phase 3.5: Digital Twin / ISAC localized obstacle memory.
    The UAV builds a local map of obstacles using an ISAC sensing radius instead 
    of relying on omniscient global ground truth maps.
    """
    def __init__(self):
        # Localized memory of obstacles
        self.known_obstacles: List[Obstacle] = []
        
    def scan_environment(self, uav_pos: tuple, real_obstacles: List[Obstacle]):
        """
        Simulates the ISAC localized sensing signal. Any real obstacle that intersects 
        with the UAV's sensing radius is added/updated in the Digital Twin.
        """
        if not hasattr(Config, "ISAC_SENSING_RADIUS"):
            return # Configuration missing
            
        sensing_radius = Config.ISAC_SENSING_RADIUS
        
        for obs in real_obstacles:
            # Bounding box shortest Euclidean distance
            dx = max(obs.x1 - uav_pos[0], 0, uav_pos[0] - obs.x2)
            dy = max(obs.y1 - uav_pos[1], 0, uav_pos[1] - obs.y2)
            dist = math.hypot(dx, dy)
            
            if dist <= sensing_radius:
                # Sensed!
                self._update_twin(obs)
                
    def _update_twin(self, sensed_obs: Obstacle):
        """
        Updates the internal digital twin representation of the obstacle.
        """
        for existing in self.known_obstacles:
            if existing.id == sensed_obs.id:
                # Update moving obstacle telemetry
                existing.x1 = sensed_obs.x1
                existing.y1 = sensed_obs.y1
                existing.x2 = sensed_obs.x2
                existing.y2 = sensed_obs.y2
                existing.vx = sensed_obs.vx
                existing.vy = sensed_obs.vy
                return
        
        # Unseen obstacle discovered
        self.known_obstacles.append(copy.deepcopy(sensed_obs))
        print(f"[ISAC] Discovered previously unknown obstacle #{sensed_obs.id} at ({sensed_obs.x1:.1f}, {sensed_obs.y1:.1f})")
        
    def calculate_collision_risk(self, uav_pos: tuple) -> float:
        """
        Computes the continuous collision risk metric (r_risk) based on proximity
        to the known obstacles in the digital twin.
        """
        risk = 0.0
        for obs in self.known_obstacles:
            dx = max(obs.x1 - uav_pos[0], 0, uav_pos[0] - obs.x2)
            dy = max(obs.y1 - uav_pos[1], 0, uav_pos[1] - obs.y2)
            dist = math.hypot(dx, dy)
            
            if dist < 1e-3:
                risk += 1000.0  # Hard collision
            else:
                risk += min(100.0, 1.0 / (dist ** 2))
        return risk
