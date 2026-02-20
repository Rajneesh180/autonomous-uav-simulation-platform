import math
from core.communication import CommunicationEngine
from config.config import Config


class BufferAwareManager:
    """
    Implements Dynamic Service Time and Buffer-Aware (DST-BA) logic.
    Determines whether the UAV should 'Center-Hover' (buffer full)
    or 'Chord-Fly' (buffer partially full) to optimize data collection times.
    """

    @staticmethod
    def calculate_service_time(uav_pos, node, is_buffer_full: bool) -> float:
        """
        Calculates the required service time (ST_i) to offload the buffer 
        based on the Shannon capacity limits and Rician fading.
        """
        # Data rate in Mbps (from Shannon Capacity with Rician Fading)
        rate_mbps = CommunicationEngine.achievable_data_rate(
            node.position(), uav_pos
        )
        
        if rate_mbps <= 0.0:
            return float('inf')  # Cannot transmit
            
        # Time required to transmit the entire current buffer
        required_time = node.current_buffer / rate_mbps
        return required_time

    @staticmethod
    def get_optimal_hover_strategy(uav_pos, node) -> dict:
        """
        Returns the optimal strategy (Center-Hover vs Chord-Fly).
        If buffer is full (current >= capacity * 0.95), UAV must center-hover.
        Otherwise, it can chord-fly.
        """
        # Using 95% threshold to define 'Full' to prevent floating point edge cases
        is_full = node.current_buffer >= (node.buffer_capacity * 0.95)
        
        required_time = BufferAwareManager.calculate_service_time(uav_pos, node, is_full)
        
        strategy = "Center-Hover" if is_full else "Chord-Fly"
        
        return {
            "strategy": strategy,
            "required_service_time": required_time,
            "buffer_drained": node.current_buffer
        }

    @staticmethod
    def process_data_collection(uav_pos, node, dt: float) -> float:
        """
        Processes data collection over a time step dt.
        Returns the amount of data collected in Mbits.
        """
        rate_mbps = CommunicationEngine.achievable_data_rate(
            node.position(), uav_pos
        )
        
        collectable_data = rate_mbps * dt
        data_collected = min(node.current_buffer, collectable_data)
        
        node.current_buffer -= data_collected
        
        # Structure safeguard
        if node.current_buffer < 0:
            node.current_buffer = 0.0
            
        return data_collected
