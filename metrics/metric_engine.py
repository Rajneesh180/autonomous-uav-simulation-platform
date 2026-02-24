import math
import time


class MetricEngine:

    # -------- Distance --------
    @staticmethod
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1[:2], p2[:2])))

    @staticmethod
    def euclidean_distance_3d(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

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
        replan_frequency = results.get("replans", 0) / total_steps
        # Support both legacy 'collisions' (int) and new 'collision_rate' (float) keys
        collision_count = results.get("collisions", results.get("collision_count", 0))
        collision_rate = collision_count / total_steps

        # --- Correct causal latency ---
        adaptation_latency = MetricEngine.compute_adaptation_latency(
            results["event_timestamps"], results["replan_timestamps"]
        )

        # PSI = 1 / (1 + RF) — perfectly bounded stability index
        path_stability_index = 1 / (1 + replan_frequency)

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

    # -------- Semantic Metrics --------
    @staticmethod
    def compute_semantic_metrics(visited_nodes, all_nodes, active_labels):
        """
        Computes semantic execution quality:
        - Priority Satisfaction: fraction of high-priority nodes serviced.
        - Semantic Purity Index: intra-cluster feature variance proxy (lower = tighter clusters).
        """
        import numpy as np

        high_priority_total = sum(1 for n in all_nodes if n.priority >= 5)
        high_priority_visited = sum(1 for n in visited_nodes if n.priority >= 5)

        priority_satisfaction = 100.0
        if high_priority_total > 0:
            priority_satisfaction = round((high_priority_visited / high_priority_total) * 100, 2)

        purity_score = 1.0
        if len(active_labels) > 0 and len(active_labels) <= len(all_nodes):
            variances = []
            unique_labels = set(active_labels)
            for k in unique_labels:
                if k == -1:
                    continue
                cluster_nodes = [all_nodes[i] for i, label in enumerate(active_labels) if label == k]
                if not cluster_nodes:
                    continue
                priorities = [n.priority for n in cluster_nodes]
                risks = [n.risk for n in cluster_nodes]
                var_p = np.var(priorities) if len(priorities) > 1 else 0
                var_r = np.var(risks) if len(risks) > 1 else 0
                variances.append(var_p + var_r)

            if variances:
                purity_score = round(float(np.mean(variances)), 4)

        return {
            "priority_satisfaction_percent": priority_satisfaction,
            "semantic_purity_index": purity_score
        }

    # ========================================================
    #  MetricsDashboard  — IEEE-aligned performance evaluation
    # ========================================================

    @staticmethod
    def compute_mission_success(visited: int, total_nodes: int,
                                collision_count: int, deadline_violations: int) -> bool:
        """
        Mission Success Rate (SR) criterion:
        A mission is deemed successful if ALL nodes are processed,
        the UAV experiences zero hard collisions with static obstacles,
        and no time-window deadlines were violated.

        Aligned with: Wang et al. (D3QN paper, IEEE IoT 2022) — SR definition.
        """
        return (visited >= total_nodes) and (collision_count == 0) and (deadline_violations == 0)

    @staticmethod
    def compute_data_collection_rate(nodes, collected_data_total_mbits: float) -> dict:
        """
        Data Collection Rate (DR):
        Ratio of data collected to total data available across all IoT nodes
        at mission start (sum of buffer capacities).

        Aligned with: Wang et al. (D3QN, IEEE IoT 2022) — DR metric.

        Returns:
            dict with 'data_collection_rate_percent' and 'total_available_mbits'.
        """
        total_available = sum(n.buffer_capacity for n in nodes)
        if total_available <= 0:
            return {"data_collection_rate_percent": 0.0, "total_available_mbits": 0.0}

        dr = min(100.0, round((collected_data_total_mbits / total_available) * 100, 4))
        return {
            "data_collection_rate_percent": dr,
            "total_available_mbits": round(total_available, 4),
        }

    @staticmethod
    def compute_collision_rate(collision_count: int, total_steps: int) -> float:
        """
        Collision Rate (CR): fraction of time steps in which a hard collision escape
        was triggered. Normalised per unit time.

        Aligned with: Wang et al. (D3QN, IEEE IoT 2022) — CR metric.
        """
        if total_steps <= 0:
            return 0.0
        return round(collision_count / total_steps, 6)

    @staticmethod
    def compute_mission_completion_time(total_steps: int, time_step_seconds: float) -> float:
        """
        Total Mission Completion Time T_total (seconds):
        Product of discrete time steps executed and the simulation Δt.

        Aligned with: Zheng & Liu (3D Trajectory paper, IEEE TVT 2025) — T_total objective.
        """
        return round(total_steps * time_step_seconds, 4)

    @staticmethod
    def compute_average_aoi(nodes) -> float:
        """
        Mean Age of Information (AoI) across all IoT nodes at mission end.
        AoI measures data freshness: lower is better.

        Aligned with:
        - Zheng & Liu (IEEE TVT 2025): T_data = t_s + t_Fs ≤ T_limit
        - Donipati et al. (DST-BA, IEEE TNSM 2025): AoI-aware buffer management
        """
        if not nodes:
            return 0.0
        aoi_values = [n.aoi_timer for n in nodes]
        return round(sum(aoi_values) / len(aoi_values), 4)

    @staticmethod
    def compute_average_achievable_rate(rate_log: list) -> float:
        """
        Time-averaged achievable Shannon communication rate E[R_c] (Mbps):
        Averaged over all time steps at which a data collection event occurred.

        Aligned with:
        - Chen et al. (TD3+ISAC+DT, IEEE IoT 2025): objective = maximise E[R_c]
        - Donipati et al. (DST-BA): R_{u,n}(t) = B log2(1 + αP_t / (L_γ * σ²))
        """
        if not rate_log:
            return 0.0
        return round(sum(rate_log) / len(rate_log), 6)

    @staticmethod
    def compute_deadline_violations(nodes, current_step: int) -> int:
        """
        Count of IoT nodes whose time-window constraint [e_i, l_i] was violated
        (i.e., service did not begin before l_i).

        Aligned with: Donipati et al. (DST-BA): time window constraints e_j ≤ t_j ≤ l_j - s_j.
        """
        violations = 0
        for n in nodes:
            if n.time_window_end < float("inf") and current_step > n.time_window_end:
                violations += 1
        return violations

    @staticmethod
    def compute_network_lifetime(nodes) -> float:
        """
        Network Lifetime Residual Energy Fraction:
        Mean residual battery across all ground IoT nodes normalised by their
        initial capacity. Value in [0, 1] — 1.0 means all nodes fully alive.

        Uses node.node_battery_J (first-order radio TX energy model, Gap 2),
        aligned with: Donipati et al. (DST-BA, IEEE TNSM 2025) network lifetime KPI.
        """
        from config.config import Config
        ground_nodes = [n for n in nodes if n.id != 0]
        if not ground_nodes:
            return 1.0
        init_cap = Config.NODE_BATTERY_CAPACITY_J
        if init_cap <= 0:
            return 1.0
        fractions = [n.node_battery_J / init_cap for n in ground_nodes]
        return round(sum(fractions) / len(fractions), 6)

    @staticmethod
    def compute_full_dashboard(mission, env, temporal, time_step: float,
                               collected_data_mbits: float,
                               rate_log: list) -> dict:
        """
        Master dashboard aggregator:
        Computes and returns the full IEEE-aligned performance evaluation dict
        to be serialised into the run summary JSON.

        Parameters
        ----------
        mission        : MissionController instance post-simulation
        env            : Environment instance
        temporal       : TemporalEngine instance
        time_step      : Config.TIME_STEP float (seconds per step)
        collected_data_mbits : total Mbits collected during mission
        rate_log       : list of per-step achievable rates (Mbps)
        """
        total_nodes = len(env.nodes) - 1  # exclude UAV anchor node
        steps = temporal.current_step if temporal.current_step > 0 else 1

        deadline_violations = MetricEngine.compute_deadline_violations(
            env.nodes[1:], steps
        )

        mission_success = MetricEngine.compute_mission_success(
            visited=len(mission.visited),
            total_nodes=total_nodes,
            collision_count=mission.collision_count,
            deadline_violations=deadline_violations,
        )

        dr_result = MetricEngine.compute_data_collection_rate(env.nodes[1:], collected_data_mbits)

        return {
            # ---- IEEE-aligned core metrics ----
            "mission_success": bool(mission_success),
            "success_rate_flag": 1 if mission_success else 0,
            "data_collection_rate_percent": dr_result["data_collection_rate_percent"],
            "total_available_mbits": dr_result["total_available_mbits"],
            "total_collected_mbits": round(collected_data_mbits, 4),
            "collision_rate": MetricEngine.compute_collision_rate(mission.collision_count, steps),
            "mission_completion_time_s": MetricEngine.compute_mission_completion_time(steps, time_step),
            "average_aoi_s": MetricEngine.compute_average_aoi(env.nodes[1:]),
            "average_achievable_rate_mbps": MetricEngine.compute_average_achievable_rate(rate_log),
            "network_lifetime_residual": MetricEngine.compute_network_lifetime(env.nodes),
            "deadline_violations": deadline_violations,
            # ---- Coverage & efficiency ----
            "nodes_visited": len(mission.visited),
            "total_nodes": total_nodes,
            "coverage_ratio_percent": MetricEngine.coverage_ratio(len(mission.visited), total_nodes),
            "energy_consumed_total_J": round(mission.energy_consumed_total, 4),
            "energy_per_node_J": MetricEngine.energy_efficiency(
                mission.energy_consumed_total, len(mission.visited)
            ),
            "final_battery_J": round(mission.uav.current_battery, 4),
        }
