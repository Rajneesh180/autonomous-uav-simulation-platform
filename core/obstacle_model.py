class Obstacle:
    def __init__(self, x1, y1, x2, y2, obstacle_type="hard"):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.type = obstacle_type
        self.vx = 0.3
        self.vy = 0.2

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
