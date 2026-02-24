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
    def is_line_of_sight_clear(node_pos: tuple, uav_pos: tuple, env) -> bool:
        """
        Phase 3.8: Implements precise Ray-Casting Object Intersection.
        Checks if the 3D ray between the UAV and the IoT node is obstructed by any physical Environment Obstacle.
        Returns True if LoS is clear, False if NLoS (obstructed).
        """
        # Segment from node to UAV
        x1, y1 = node_pos[0], node_pos[1]
        x2, y2 = uav_pos[0], uav_pos[1]
        
        # We assume obstacles are infinitely tall pillars for now or check against their height.
        # Check intersection with every obstacle's 2D bounding box
        for obs in env.obstacles:
            # Check if either point is inside the obstacle
            if obs.contains_point(x1, y1) or obs.contains_point(x2, y2):
                return False
                
            # Check if line segment intersects the obstacle's bounding lines
            # A simple bounding box ray intersection
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            # If the bounding boxes of the segment and the obstacle don't overlap, no intersection
            if max_x < obs.x1 or min_x > obs.x2 or max_y < obs.y1 or min_y > obs.y2:
                continue
                
            # Use cross product line intersection or simple line-rectangle intersection
            if obs.intersects_line(x1, y1, x2, y2):
                return False

        return True

    @staticmethod
    def prob_los(elevation_angle_rad: float, is_los_clear: bool = True) -> float:
        """
        Calculates the probability of Line-of-Sight (LoS) based on the environmental
        parameters and the elevation angle (theta). Overridden perfectly to 0 if a static 
        obstacle interrupts the raycast.
        """
        if not is_los_clear:
            return 0.0
            
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
    def achievable_data_rate(node_pos: tuple, uav_pos: tuple, env=None) -> float:
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
        
        is_clear = True
        if env is not None:
            is_clear = CommunicationEngine.is_line_of_sight_clear(node_pos, uav_pos, env)
            
        p_los = CommunicationEngine.prob_los(angle, is_clear)
        
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
    def probabilistic_sensing_success(distance: float) -> bool:
        if not Config.ENABLE_PROBABILISTIC_SENSING:
            return True
            
        import random
        prob = math.exp(-Config.SENSING_TAU * distance)
        
        if prob < Config.MIN_SENSING_PROB_THRESH:
            prob = 0.0
            
        return random.random() <= prob

    @staticmethod
    def compute_node_tx_energy(bits: float, distance_m: float) -> float:
        """
        First-order radio energy model for IoT node transmission.
        Aligned with Heinzelman et al. (canonical WSN energy model) and
        Donipati et al. (DST-BA, IEEE TNSM 2025) network-lifetime formulation.

            E_tx(b, d) = E_elec * b + E_amp * b * d^2  [Joules]

        Parameters
        ----------
        bits       : number of bits transmitted in this time step
        distance_m : Euclidean distance from node to UAV (metres)
        """
        return (Config.NODE_E_ELEC_J_PER_BIT * bits
                + Config.NODE_E_AMP_J_PER_BIT_M2 * bits * distance_m ** 2)

    @staticmethod
    def fill_buffer(node, dt: float, uav_pos: tuple = None):
        """
        Increases the node's buffer filling based on data generation rate over dt.
        Also tracks Age of Information (AoI) and expires data if too old.

        Gap 2 (Donipati et al. DST-BA): when the UAV is within communication range
        (uav_pos provided), deduct TX energy from the node's battery using the
        first-order radio energy model E_tx = E_elec*b + E_amp*b*d^2.
        """
        generated = node.data_generation_rate * dt
        node.current_buffer = min(node.buffer_capacity, node.current_buffer + generated)

        if Config.ENABLE_AOI_EXPIRATION:
            node.aoi_timer += dt
            if node.aoi_timer >= Config.MAX_AOI_LIMIT:
                node.current_buffer = 0.0
                node.aoi_timer = 0.0

        # ---- Node TX energy depletion (first-order radio model) ----
        if Config.ENABLE_NODE_ENERGY_DRAIN and uav_pos is not None and node.node_battery_J > 0:
            import math
            dx = uav_pos[0] - node.x
            dy = uav_pos[1] - node.y
            dist_m = math.hypot(dx, dy)

            # Convert Mbits generated to bits
            bits_generated = generated * 1e6
            if bits_generated > 0 and dist_m > 0:
                e_tx = CommunicationEngine.compute_node_tx_energy(bits_generated, dist_m)
                node.tx_energy_consumed_J += e_tx
                node.node_battery_J = max(0.0, node.node_battery_J - e_tx)

