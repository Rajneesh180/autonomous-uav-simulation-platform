class Obstacle:
    def __init__(self, x1, y1, x2, y2, obstacle_type="hard"):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.type = obstacle_type  # "hard" or "nofly"

    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def intersects_line(self, x3, y3, x4, y4):
        # Simple bounding box intersection approximation
        if (
            max(x3, x4) < self.x1 or
            min(x3, x4) > self.x2 or
            max(y3, y4) < self.y1 or
            min(y3, y4) > self.y2
        ):
            return False
        return True
