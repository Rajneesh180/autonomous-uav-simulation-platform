from config.config import Config
from core.seed_manager import set_global_seed
from core.environment_model import Environment
from core.dataset_generator import generate_nodes
from metrics.metric_engine import MetricEngine
from metrics.logger import Logger


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
