import math


class RiskZone:
    def __init__(self, x1, y1, x2, y2, multiplier=1.5):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

        self.base_multiplier = multiplier
        self.current_multiplier = multiplier

    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def fluctuate(self, step):
        delta = 0.2 * math.sin(step / 5)
        self.current_multiplier = self.base_multiplier + delta
