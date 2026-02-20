import math
from config.config import Config


class CommunicationEngine:
    """
    Rician Fading Channel & Line-of-Sight (LoS) Estimator.
    Handles buffer dynamics, Shannon capacity, and probabilistic throughput.
    """

    @staticmethod
    def elevation_angle(node_pos: tuple, uav_pos: tuple) -> float:
        """
        Calculates elevation angle in radians between ground node and UAV.
        """
        dx = uav_pos[0] - node_pos[0]
        dy = uav_pos[1] - node_pos[1]
        horizontal_dist = math.hypot(dx, dy)
        
        # UAV z is assumed to be uav_pos[2], Node z is node_pos[2]. 
        # Fallback to config altitude if z missing
        uav_z = uav_pos[2] if len(uav_pos) > 2 else Config.UAV_FLIGHT_ALTITUDE
        node_z = node_pos[2] if len(node_pos) > 2 else 0.0
        
        dz = max(0.0, uav_z - node_z)
        if horizontal_dist < 1e-6:
            return math.pi / 2.0
            
        return math.atan2(dz, horizontal_dist)

    @staticmethod
    def prob_los(elevation_angle_rad: float) -> float:
        """
        Calculates the probability of Line-of-Sight (LoS) based on the environmental
        parameters and the elevation angle (theta).
        """
        theta_degrees = math.degrees(elevation_angle_rad)
        a = Config.LOS_PARAM_A
        b = Config.LOS_PARAM_B
        
        # Sigmoid function model for LoS
        exponent = -b * (theta_degrees - a)
        # Prevent math domain errors from extremely large exponent
        exponent = max(-100.0, min(100.0, exponent))
        
        p_los = 1.0 / (1.0 + a * math.exp(exponent))
        return p_los

    @staticmethod
    def path_loss(distance_3d: float, p_los: float) -> float:
        """
        Calculates the expected path loss (linear scale) blending LoS and NLoS components.
        Simplified FSPL equivalent modeling.
        """
        # Free Space Path Loss constant term: (4 * pi * f_c / c)^2
        c = 3e8 # speed of light
        fc = Config.CARRIER_FREQUENCY
        
        if distance_3d < 1.0:
            distance_3d = 1.0
            
        # Linear FSPL multiplier at 1m
        fspl_1m = (4.0 * math.pi * fc / c) ** 2
        
        loss_los = fspl_1m * (distance_3d ** Config.PATH_LOSS_LOS)
        loss_nlos = fspl_1m * (distance_3d ** Config.PATH_LOSS_NLOS)
        
        expected_loss = p_los * loss_los + (1.0 - p_los) * loss_nlos
        return expected_loss

    @staticmethod
    def achievable_data_rate(node_pos: tuple, uav_pos: tuple) -> float:
        """
        Calculates theoretical capacity based on SNR from the expected path loss
        Returns data rate in Mbps.
        """
        distance_3d = math.hypot(
            uav_pos[0] - node_pos[0],
            uav_pos[1] - node_pos[1]
        )
        uav_z = uav_pos[2] if len(uav_pos) > 2 else Config.UAV_FLIGHT_ALTITUDE
        node_z = node_pos[2] if len(node_pos) > 2 else 0.0
        
        full_dist = math.sqrt(distance_3d**2 + (uav_z - node_z)**2)
        
        angle = CommunicationEngine.elevation_angle(node_pos, uav_pos)
        p_los = CommunicationEngine.prob_los(angle)
        
        loss_linear = CommunicationEngine.path_loss(full_dist, p_los)
        tx_power = Config.NODE_TX_POWER
        rx_power = tx_power / loss_linear
        
        # Noise Power = N0 * B (linear scale from dBm/Hz)
        # N0_dBm = -174 -> N0_W = 10**(-174/10) * 1e-3
        n0_w = (10 ** (Config.NOISE_POWER_DENSITY / 10.0)) * 1e-3
        noise_power = n0_w * Config.BANDWIDTH
        
        snr_linear = rx_power / noise_power
        
        # Shannon Capacity: C = B * log2(1 + SNR)
        # Rate in bps
        rate_bps = Config.BANDWIDTH * math.log2(1.0 + snr_linear)
        
        # Return in Mbps for scaling
        return rate_bps / 1e6

    @staticmethod
    def fill_buffer(node, dt: float):
        """
        Increases the node's buffer filling based on generation rate over time dt.
        """
        generated = node.data_generation_rate * dt
        node.current_buffer = min(node.buffer_capacity, node.current_buffer + generated)

