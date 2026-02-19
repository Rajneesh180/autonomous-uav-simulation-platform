import time

from config.config import Config
from core.seed_manager import set_global_seed
from core.environment_model import Environment
from core.dataset_generator import generate_nodes
from core.obstacle_model import Obstacle
from core.risk_zone_model import RiskZone
from core.temporal_engine import TemporalEngine
from core.run_manager import RunManager
from clustering.cluster_manager import ClusterManager
from core.stability_monitor import StabilityMonitor
from core.mission_controller import MissionController


def main():
    print("=== Autonomous UAV Simulation Platform ===")

    # ---------------------------------------------------------
    # Seed Initialization
    # ---------------------------------------------------------
    if Config.RANDOMIZE_SEED:
        active_seed = int(time.time())
    else:
        active_seed = Config.RANDOM_SEED

    set_global_seed(active_seed)
    print(f"[SeedManager] Active seed: {active_seed}")

    # ---------------------------------------------------------
    # Apply Hostility Profile (Phase-3 Control)
    # ---------------------------------------------------------
    Config.apply_hostility_profile()

    print(f"[Config] Hostility Level = {Config.HOSTILITY_LEVEL}")
    print(f"[Config] Moving Obstacles = {Config.ENABLE_MOVING_OBSTACLES}")
    print(f"[Config] Obstacle Velocity Scale = {Config.OBSTACLE_VELOCITY_SCALE}")
    print(f"[Config] Dynamic Node Interval = {Config.DYNAMIC_NODE_INTERVAL}")
    print(f"[Config] Node Removal Probability = {Config.NODE_REMOVAL_PROBABILITY}")

    # ---------------------------------------------------------
    # Run Manager
    # ---------------------------------------------------------
    run_manager = RunManager()
    print(f"[RunManager] Run ID: {run_manager.get_run_id()}")

    # ---------------------------------------------------------
    # Core Managers
    # ---------------------------------------------------------
    cluster_manager = ClusterManager()
    stability_monitor = StabilityMonitor()

    # ---------------------------------------------------------
    # Environment Setup
    # ---------------------------------------------------------
    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)
    env.dataset_mode = Config.DATASET_MODE

    # Temporal Engine
    temporal = TemporalEngine(Config.TIME_STEP, Config.MAX_TIME_STEPS)
    env.temporal_engine = temporal

    print(f"[Config] MAX_TIME_STEPS = {Config.MAX_TIME_STEPS}")
    print(f"[Config] TIME_STEP = {Config.TIME_STEP}")

    # ---------------------------------------------------------
    # Node Generation
    # ---------------------------------------------------------
    nodes = generate_nodes(
        Config.DATASET_MODE,
        Config.NODE_COUNT,
        Config.MAP_WIDTH,
        Config.MAP_HEIGHT,
        active_seed,
    )

    for node in nodes:
        env.add_node(node)

    # ---------------------------------------------------------
    # Obstacles
    # ---------------------------------------------------------
    if Config.ENABLE_OBSTACLES:
        env.add_obstacle(Obstacle(200, 200, 350, 350))
        env.add_obstacle(Obstacle(500, 100, 650, 250))

    # ---------------------------------------------------------
    # Risk Zones
    # ---------------------------------------------------------
    if Config.ENABLE_RISK_ZONES:
        env.add_risk_zone(RiskZone(100, 400, 300, 550, multiplier=1.8))

    print("Environment Summary:", env.summary())

    # ---------------------------------------------------------
    # Mission Execution
    # ---------------------------------------------------------
    print("\n--- Unified Mission Execution ---")

    mission = MissionController(env, temporal, run_manager)

    if not mission.is_active():
        print("[Warning] Mission inactive at start.")
        print("Battery:", mission.uav.current_battery)
        print("Temporal active:", temporal.active)
        return

    step_counter = 0

    while mission.is_active():
        mission.step()
        step_counter += 1

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    print("\nMission execution completed.")
    print(f"Total simulation steps executed: {step_counter}")
    print(f"Final battery: {round(mission.uav.current_battery, 2)}")
    print(f"Visited nodes: {len(mission.visited)}")
    print(f"Replan count: {temporal.replan_count}")
    print(f"Collision count: {mission.collision_count}")
    print(f"Unsafe return count: {mission.unsafe_return_count}")


if __name__ == "__main__":
    main()
