from metrics.metric_engine import MetricEngine


class EnergyModel:
    """
    Phase-3 Energy & Safety Model

    Responsibilities:
    - Deterministic energy computation
    - Travel feasibility checks
    - Return-to-base safety validation
    - Preventive threshold detection
    """

    # ---------------------------------------------------------
    # Core Energy Calculations
    # ---------------------------------------------------------

    @staticmethod
    def energy_for_distance(node, distance: float) -> float:
        """
        Linear propulsion model.
        """
        return max(0.0, distance) * node.energy_per_meter

    @staticmethod
    def hover_energy(node, hover_time: float) -> float:
        """
        Hover cost (currently optional).
        """
        return max(0.0, hover_time) * node.hover_cost

    @staticmethod
    def total_energy(node, distance: float, hover_time: float = 0.0) -> float:
        return EnergyModel.energy_for_distance(
            node, distance
        ) + EnergyModel.hover_energy(node, hover_time)

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
