import math
from core.comms.communication import CommunicationEngine
from config.config import Config


class BufferAwareManager:
    """
    Implements Dynamic Service Time and Buffer-Aware (DST-BA) logic.
    Determines whether the UAV should 'Center-Hover' (buffer full)
    or 'Chord-Fly' (buffer partially full) to optimize data collection times.
    """

    @staticmethod
    def calculate_service_time(uav_pos, node, is_buffer_full: bool, env=None) -> float:
        """
        Calculates the required service time (ST_i) to offload the buffer 
        based on the Shannon capacity limits and Rician fading.
        """
        # Data rate in Mbps (from Shannon Capacity with Rician Fading)
        rate_mbps = CommunicationEngine.achievable_data_rate(
            node.position(), uav_pos, env
        )
        
        if rate_mbps <= 0.0:
            return float('inf')  # Cannot transmit
            
        # Time required to transmit the entire current buffer
        required_time = node.current_buffer / rate_mbps
        return required_time

    @staticmethod
    def get_optimal_hover_strategy(uav_pos, node, env=None) -> dict:
        """
        Returns the optimal strategy (Center-Hover vs Chord-Fly) and the
        minimum mandatory hover time derived from the multi-trial sensing model.

        Gap 3 (Zheng & Liu, IEEE TVT 2025): the UAV must hover for at least
        n̂_s = ceil(log(1-ω)/log(1-e^{-τd})) sensing slots before the cumulative
        success probability exceeds the target threshold ω.
        """
        from metrics.metric_engine import MetricEngine
        is_full = node.current_buffer >= (node.buffer_capacity * 0.95)
        required_time = BufferAwareManager.calculate_service_time(uav_pos, node, is_full, env)

        # Multi-trial sensing: minimum hover time budget
        dist = MetricEngine.euclidean_distance(node.position(), uav_pos)
        min_hover_s = CommunicationEngine.minimum_hover_time(dist)

        strategy = "Center-Hover" if is_full else "Chord-Fly"
        return {
            "strategy": strategy,
            "required_service_time": required_time,
            "min_hover_time_s": min_hover_s,
            "buffer_drained": node.current_buffer
        }

    @staticmethod
    def process_data_collection(uav_pos, node, dt: float, env=None,
                                active_node_id: int = None) -> float:
        """
        Processes data collection over a time step dt.
        Returns the amount of data collected in Mbits.

        Gap 7 (Wang et al., IEEE IoT 2022): When ENABLE_TDMA_SCHEDULING is True,
        only the designated active node (active_node_id) may transmit per slot.
        All other nodes are silenced — enforcing TDMA discipline.
        """
        # TDMA enforcement: only the scheduled node transmits this step
        if Config.ENABLE_TDMA_SCHEDULING and active_node_id is not None:
            if node.id != active_node_id:
                return 0.0  # TDMA silence — node waits for its slot

        from metrics.metric_engine import MetricEngine
        dist = MetricEngine.euclidean_distance(node.position(), uav_pos)

        if not CommunicationEngine.probabilistic_sensing_success(dist):
            return 0.0

        rate_mbps = CommunicationEngine.achievable_data_rate(
            node.position(), uav_pos, env
        )

        collectable_data = rate_mbps * dt
        data_collected = min(node.current_buffer, collectable_data)

        node.current_buffer -= data_collected
        if data_collected > 0:
            node.max_aoi_timer = max(node.max_aoi_timer, node.aoi_timer)  # record peak before reset
            node.aoi_timer = 0.0  # Reset AoI timer on successful collection

        if node.current_buffer < 0:
            node.current_buffer = 0.0

        return data_collected
