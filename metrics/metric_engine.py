import math
import time

class MetricEngine:
    @staticmethod
    def euclidean_distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    @staticmethod
    def path_length(points):
        if len(points) < 2:
            return 0.0
        dist = 0.0
        for i in range(len(points) - 1):
            dist += MetricEngine.euclidean_distance(points[i], points[i+1])
        return dist

    @staticmethod
    def start_timer():
        return time.time()

    @staticmethod
    def end_timer(start_time):
        return round(time.time() - start_time, 5)
