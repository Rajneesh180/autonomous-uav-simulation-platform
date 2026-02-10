from config.config import Config
from core.seed_manager import set_global_seed
from core.environment_model import Environment
from core.dataset_generator import generate_nodes
from metrics.metric_engine import MetricEngine
from metrics.logger import Logger
from core.energy_model import EnergyModel


def main():
    print("=== Autonomous UAV Simulation Platform ===")

    # -------- Seed --------
    set_global_seed(Config.RANDOM_SEED)

    # -------- Environment --------
    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)
    env.dataset_mode = Config.DATASET_MODE

    # -------- Node Generation --------
    nodes = generate_nodes(
        Config.DATASET_MODE,
        Config.NODE_COUNT,
        Config.MAP_WIDTH,
        Config.MAP_HEIGHT,
        Config.RANDOM_SEED
    )

    for node in nodes:
        env.add_node(node)

    print("Environment Summary:", env.summary())

    # -------- Energy Simulation --------
    print("\n--- Energy Simulation ---")

    uav = env.nodes[0]
    base_position = uav.position()
    visited = 0

    for target in env.nodes[1:]:

        # Check return threshold first
        if EnergyModel.should_return(uav):
            print("Battery threshold reached. Returning to base.")
            break

        distance = MetricEngine.euclidean_distance(
            uav.position(),
            target.position()
        )

        # Can reach next node?
        if not EnergyModel.can_travel(uav, distance):
            print("Cannot reach next node safely.")
            break

        # Simulate travel
        energy_needed = EnergyModel.energy_for_distance(uav, distance)
        EnergyModel.consume(uav, energy_needed)

        # Check return feasibility
        if not EnergyModel.can_return_to_base(
            uav,
            target.position(),
            base_position
        ):
            print("Return-to-base unsafe. Mission halted.")
            break

        visited += 1
        print(
            f"Visited Node {target.id} | "
            f"Battery Left: {round(uav.current_battery, 2)}"
        )

    # <-- OUTSIDE LOOP
    print(f"Total Visited: {visited}")

    # -------- Metrics Test --------
    timer_start = MetricEngine.start_timer()

    sample_positions = [node.position() for node in env.nodes[:10]]
    sample_path_length = MetricEngine.path_length(sample_positions)

    runtime = MetricEngine.end_timer(timer_start)

    print(f"Sample Path Length: {round(sample_path_length, 2)}")
    print(f"Runtime: {runtime}s")

    # -------- Logging --------
    if Config.ENABLE_LOGGING:
        payload = {
            "timestamp": Logger.timestamp(),
            "node_count": env.get_node_count(),
            "sample_path_length": round(sample_path_length, 2),
            "runtime": runtime
        }

        Logger.log_json("run_log.json", payload)
        Logger.log_csv(
            "run_metrics.csv",
            ["timestamp", "node_count", "sample_path_length", "runtime"],
            [
                payload["timestamp"],
                payload["node_count"],
                payload["sample_path_length"],
                payload["runtime"]
            ]
        )

        print("Logs written to /logs directory")


if __name__ == "__main__":
    main()
