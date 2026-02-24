import math
from config.config import Config


class Obstacle:
    _id_counter = 0
    
    def __init__(self, x1, y1, x2, y2, obstacle_type="hard", height=None):
        self.id = Obstacle._id_counter
        Obstacle._id_counter += 1
        
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.type = obstacle_type
        self.vx = 0.3
        self.vy = 0.2

        # Peak height for 3D Gaussian altitude model (Gap 5 — Zheng & Liu IEEE TVT 2025)
        # If not specified, derive from bounding box dimensions as proxy
        if height is not None:
            self.height = float(height)
        else:
            width  = self.x2 - self.x1
            depth  = self.y2 - self.y1
            self.height = max(20.0, min(width, depth) * 0.3)

        self.cx = (self.x1 + self.x2) / 2.0   # centre x
        self.cy = (self.y1 + self.y2) / 2.0   # centre y

    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersects_line(self, x3, y3, x4, y4):
        if (
            max(x3, x4) < self.x1
            or min(x3, x4) > self.x2
            or max(y3, y4) < self.y1
            or min(y3, y4) > self.y2
        ):
            return False
        return True

    def gaussian_height(self, x, y,
                        ax: float = None, ay: float = None) -> float:
        """
        Gaussian altitude profile for this obstacle at point (x, y):

            z_obs_i(x, y) = hᵢ · exp[ -((x − cxᵢ)/axᵢ)² − ((y − cyᵢ)/ayᵢ)² ]

        Aligned with: Zheng & Liu (IEEE TVT 2025) — Eq. 36.
        """
        if ax is None:
            ax = Config.GAUSSIAN_SPREAD_X
        if ay is None:
            ay = Config.GAUSSIAN_SPREAD_Y
        return self.height * math.exp(
            -((x - self.cx) / ax) ** 2 - ((y - self.cy) / ay) ** 2
        )

    def move(self, width, height):
        self.x1 += self.vx
        self.x2 += self.vx
        self.y1 += self.vy
        self.y2 += self.vy

        if self.x1 < 0 or self.x2 > width:
            self.vx *= -1
            self.x1 = max(0, self.x1)
            self.x2 = min(width, self.x2)

        if self.y1 < 0 or self.y2 > height:
            self.vy *= -1
            self.y1 = max(0, self.y1)
            self.y2 = min(height, self.y2)

        # Update centre after movement
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0


class ObstacleHeightModel:
    """
    3D Gaussian Obstacle Height Model — Environment-level Aggregator.

    Computes the combined altitude profile z_obs(x,y) = Σᵢ z_obs_i(x,y) for
    all obstacles in the environment. The UAV must stay above
        z_obs(x,y) + Config.VERTICAL_CLEARANCE
    at every hovering position and waypoint.

    Aligned with: Zheng & Liu (IEEE TVT 2025) — Section III-F, Eq. 36.
    """

    @staticmethod
    def required_altitude(x: float, y: float, obstacles: list) -> float:
        """
        Returns the minimum safe UAV altitude at position (x, y):
            z_req = z_obs(x, y) + VERTICAL_CLEARANCE

        Parameters
        ----------
        x, y       : horizontal position (metres)
        obstacles  : list of Obstacle instances

        Returns
        -------
        float : minimum safe altitude (metres), always ≥ 0.
        """
        if not Config.ENABLE_GAUSSIAN_HEIGHT or not obstacles:
            return 0.0
        z_surface = sum(obs.gaussian_height(x, y) for obs in obstacles)
        return z_surface + Config.VERTICAL_CLEARANCE

    @staticmethod
    def enforce_altitude(x: float, y: float, z: float,
                         obstacles: list) -> float:
        """
        Clamp the candidate z so the UAV never flies below the obstacle surface.
        Returns the safe altitude (z unchanged if already above surface).
        """
        z_req = ObstacleHeightModel.required_altitude(x, y, obstacles)
        return max(z, z_req)
