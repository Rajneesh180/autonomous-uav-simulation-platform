import math
from metrics.metric_engine import MetricEngine
from config.config import Config


class EnergyModel:
    """
    Phase-3 Energy & Safety Model

    Responsibilities:
    - Deterministic Rotary-Wing energy computation
    - Travel feasibility checks
    - Return-to-base safety validation
    - Preventive threshold detection
    """

    # ---------------------------------------------------------
    # Core Energy Calculations
    # ---------------------------------------------------------

    @staticmethod
    def propulsion_power(v: float) -> float:
        """
        Computes mechanical power P_p(v) in Watts required for a speed v.
        Incorporates blade profile, induced, and parasite power.
        """
        P_0 = Config.PROFILE_POWER_HOVER
        P_i = Config.INDUCED_POWER_HOVER
        U_tip = Config.ROTOR_TIP_SPEED
        v_0 = Config.MEAN_ROTOR_VELOCITY
        d_0 = Config.FUSELAGE_DRAG_RATIO
        rho = Config.AIR_DENSITY
        s = Config.ROTOR_SOLIDITY
        A = Config.ROTOR_DISC_AREA

        # 1. Profile power (friction on rotor blades)
        profile = P_0 * (1.0 + (3.0 * v**2) / (U_tip**2))

        # 2. Induced power (downward thrust)
        if v < 0.01:
            induced = P_i
        else:
            term1 = math.sqrt(1.0 + (v**4) / (4.0 * v_0**4))
            term2 = (v**2) / (2.0 * v_0**2)
            inner_val = max(0.0, term1 - term2)
            induced = P_i * math.sqrt(inner_val)

        # 3. Parasite power (fuselage drag)
        parasite = 0.5 * d_0 * rho * s * A * (v**3)

        return profile + induced + parasite

    @staticmethod
    def energy_for_distance(node, distance: float) -> float:
        """
        Energy = Power(v) * dt
        Assumes constant speed v over the temporal step dt.
        """
        dt = float(Config.TIME_STEP)
        # Handle zero division safeguard
        v = distance / dt if dt > 0 else 0.0
        
        # Energy = Power * Time
        power = EnergyModel.propulsion_power(v)
        return power * dt

    @staticmethod
    def hover_energy(node, hover_time: float) -> float:
        """
        Hover cost: zero velocity power evaluation over hover_time.
        """
        power_hover = EnergyModel.propulsion_power(0.0)
        return max(0.0, hover_time) * power_hover

    @staticmethod
    def total_energy(node, distance: float, hover_time: float = 0.0) -> float:
        return EnergyModel.energy_for_distance(
            node, distance
        ) + EnergyModel.hover_energy(node, hover_time)

    @staticmethod
    def mechanical_energy(node, acceleration_vec: tuple) -> float:
        """
        Acceleration-based mechanical energy model.
        E_ME = \\Delta \\sum wc || a[i] - ag ||^2
        """
        ax, ay, az = acceleration_vec
        ag_x, ag_y, ag_z = 0.0, 0.0, -Config.GRAVITY
        
        # w: weather resistance, c: mass/inertia coefficient
        w = 1.0  
        c = Config.UAV_MASS
        
        norm_sq = (ax - ag_x)**2 + (ay - ag_y)**2 + (az - ag_z)**2
        power = w * c * norm_sq
        
        return power * float(Config.TIME_STEP)

    # ---------------------------------------------------------
    # Feasibility Checks
    # ---------------------------------------------------------

    @staticmethod
    def can_travel(node, distance: float) -> bool:
        """
        Checks if UAV can afford forward movement.
        """
        required = EnergyModel.energy_for_distance(node, distance)
        return node.current_battery >= required

    @staticmethod
    def consume(node, energy: float):
        """
        Deduct energy with floor clamp.
        """
        node.current_battery -= max(0.0, energy)
        if node.current_battery < 0:
            node.current_battery = 0.0

    # ---------------------------------------------------------
    # Return-to-Base Safety
    # ---------------------------------------------------------

    @staticmethod
    def can_return_to_base(
        node,
        current_pos,
        base_pos,
        risk_multiplier: float = 1.0,
    ) -> bool:
        """
        Conservative return check.

        Includes:
        - Euclidean distance
        - Risk multiplier
        - 5% battery safety buffer
        """

        distance_back = MetricEngine.euclidean_distance(current_pos, base_pos)

        adjusted_distance = distance_back * max(1.0, risk_multiplier)

        required = EnergyModel.energy_for_distance(node, adjusted_distance)

        safety_buffer = node.battery_capacity * 0.05

        return node.current_battery >= (required + safety_buffer)

    @staticmethod
    def should_return(node) -> bool:
        """
        Preventive return trigger.
        """
        threshold = node.battery_capacity * node.return_threshold
        return node.current_battery <= threshold
