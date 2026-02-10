import math
import time


class MetricEngine:

    # -------- Distance --------
    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def path_length(points):
        total = 0
        for i in range(len(points) - 1):
            total += MetricEngine.euclidean_distance(points[i], points[i + 1])
        return total

    # -------- Timer --------
    @staticmethod
    def start_timer():
        return time.time()

    @staticmethod
    def end_timer(start):
        return round(time.time() - start, 5)

    # -------- Mission Metrics --------
    @staticmethod
    def mission_completion(visited, attempted):
        if attempted == 0:
            return 0.0
        return round((visited / attempted) * 100, 2)

    @staticmethod
    def energy_efficiency(energy_consumed, visited_nodes):
        if visited_nodes == 0:
            return 0
        return round(energy_consumed / visited_nodes, 2)

    @staticmethod
    def abort_flag(reason):
        return 1 if reason is not None else 0

    @staticmethod
    def return_flag(triggered):
        return 1 if triggered else 0

    @staticmethod
    def coverage_ratio(visited, total):
        if total == 0:
            return 0.0
        return round((visited / total) * 100, 2)

    @staticmethod
    def constraint_violation_flag(collisions, unsafe_returns):
        return 1 if (collisions > 0 or unsafe_returns > 0) else 0

