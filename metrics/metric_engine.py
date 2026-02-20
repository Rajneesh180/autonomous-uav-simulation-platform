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

    # -------- Phase-3 Stability Metrics --------
    @staticmethod
    def compute_stability_metrics(results):

        total_steps = results["steps"] if results["steps"] > 0 else 1

        # --- Primary rates ---
        replan_frequency = results["replans"] / total_steps
        collision_rate = results["collisions"] / total_steps

        # --- Correct causal latency ---
        adaptation_latency = MetricEngine.compute_adaptation_latency(
            results["event_timestamps"], results["replan_timestamps"]
        )

        # --- Properly bounded stability index ---
        # PSI = 1 / (1 + RF)
        path_stability_index = 1 / (1 + replan_frequency)

        # --- Properly normalized churn impact ---
        node_churn_impact = results["event_count"] / total_steps
        energy_prediction_error = results.get("energy_prediction_error", 0.0)

        return {
            "replan_frequency": replan_frequency,
            "collision_rate": collision_rate,
            "adaptation_latency": adaptation_latency,
            "path_stability_index": path_stability_index,
            "node_churn_impact": node_churn_impact,
            "energy_prediction_error": energy_prediction_error,
        }

    @staticmethod
    def compute_adaptation_latency(event_times, replan_times):
        if not event_times or not replan_times:
            return 0.0

        latencies = []
        r_index = 0

        for e in event_times:
            while r_index < len(replan_times) and replan_times[r_index] <= e:
                r_index += 1

            if r_index < len(replan_times):
                latency = replan_times[r_index] - e
                latencies.append(latency)

        if not latencies:
            return 0.0

        return round(sum(latencies) / len(latencies), 4)

    # -------- Phase-4 Semantic Metrics --------
    @staticmethod
    def compute_semantic_metrics(visited_nodes, all_nodes, active_labels):
        """
        Calculates Phase-4 Semantic Execution performance.
        Priority Satisfaction: Percentage of high-priority nodes successfully serviced.
        Semantic Purity: Intra-cluster feature variance proxy (lower is better).
        """
        import numpy as np
        
        # Priority Satisfaction
        high_priority_total = sum(1 for n in all_nodes if n.priority >= 5)
        high_priority_visited = sum(1 for n in visited_nodes if n.priority >= 5)
        
        priority_satisfaction = 100.0
        if high_priority_total > 0:
            priority_satisfaction = round((high_priority_visited / high_priority_total) * 100, 2)
            
        # Semantic Purity (Intra-cluster variance of priorities and risks)
        purity_score = 1.0 # default
        if len(active_labels) > 0 and len(active_labels) <= len(all_nodes):
            variances = []
            unique_labels = set(active_labels)
            for k in unique_labels:
                if k == -1: continue # Skip noise
                # Extract features for this cluster
                cluster_nodes = [all_nodes[i] for i, label in enumerate(active_labels) if label == k]
                if not cluster_nodes: continue
                
                priorities = [n.priority for n in cluster_nodes]
                risks = [n.risk for n in cluster_nodes]
                # Purity proxy: variance of core features
                var_p = np.var(priorities) if len(priorities) > 1 else 0
                var_r = np.var(risks) if len(risks) > 1 else 0
                variances.append(var_p + var_r)
                
            if variances:
                purity_score = round(float(np.mean(variances)), 4)
                
        return {
            "priority_satisfaction_percent": priority_satisfaction,
            "semantic_purity_index": purity_score
        }
