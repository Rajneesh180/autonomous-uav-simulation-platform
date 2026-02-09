from config.config import Config
from core.seed_manager import set_global_seed
from core.node_model import Node
from core.environment_model import Environment
from metrics.metric_engine import MetricEngine
from metrics.logger import Logger

import random


def main():
    print("=== Autonomous UAV Simulation Platform ===")

    # ---------------- Seed ----------------
    set_global_seed(Config.RANDOM_SEED)

    # ---------------- Environment ----------------
    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)

    # ---------------- Node Generation ----------------
    for i in range(Config.NODE_COUNT):
        x = random.randint(0, Config.MAP_WIDTH)
        y = random.randint(0, Config.MAP_HEIGHT)
        node = Node(id=i, x=x, y=y)
        env.add_node(node)

    print("Environment Summary:", env.summary())

    # ---------------- Metrics Test ----------------
    timer_start = MetricEngine.start_timer()

    # Take first 10 nodes as sample path
    sample_positions = [node.position() for node in env.nodes[:10]]
    sample_path_length = MetricEngine.path_length(sample_positions)

    runtime = MetricEngine.end_timer(timer_start)

    print(f"Sample Path Length: {round(sample_path_length, 2)}")
    print(f"Runtime: {runtime}s")

    # ---------------- Logging ----------------
    log_payload = {
        "timestamp": Logger.timestamp(),
        "node_count": env.get_node_count(),
        "sample_path_length": round(sample_path_length, 2),
        "runtime": runtime
    }

    # JSON Log
    Logger.log_json("run_log.json", log_payload)

    # CSV Log
    Logger.log_csv(
        "run_metrics.csv",
        ["timestamp", "node_count", "sample_path_length", "runtime"],
        [
            log_payload["timestamp"],
            log_payload["node_count"],
            log_payload["sample_path_length"],
            log_payload["runtime"]
        ]
    )

    print("Logs written to /logs directory")


if __name__ == "__main__":
    main()
