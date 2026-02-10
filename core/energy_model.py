from metrics.metric_engine import MetricEngine


class EnergyModel:

    @staticmethod
    def energy_for_distance(node, distance: float) -> float:
        return distance * node.energy_per_meter

    @staticmethod
    def hover_energy(node, hover_time: float) -> float:
        return hover_time * node.hover_cost

    @staticmethod
    def total_energy(node, distance: float, hover_time: float = 0.0) -> float:
        return (
            EnergyModel.energy_for_distance(node, distance)
            + EnergyModel.hover_energy(node, hover_time)
        )

    @staticmethod
    def can_travel(node, distance: float) -> bool:
        required = EnergyModel.energy_for_distance(node, distance)
        return node.current_battery >= required

    @staticmethod
    def consume(node, energy: float):
        node.current_battery -= energy
        if node.current_battery < 0:
            node.current_battery = 0
