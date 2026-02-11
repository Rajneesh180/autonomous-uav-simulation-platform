from config.config import Config
from core.seed_manager import set_global_seed
from core.environment_model import Environment
from core.dataset_generator import generate_nodes
from core.obstacle_model import Obstacle
from metrics.metric_engine import MetricEngine
from metrics.logger import Logger
from core.energy_model import EnergyModel
from core.risk_zone_model import RiskZone
from visualization.plot_renderer import PlotRenderer
from core.temporal_engine import TemporalEngine 
from core.dataset_generator import spawn_single_node


def main():
    print("=== Autonomous UAV Simulation Platform ===")

    # -------- Seed --------
    set_global_seed(Config.RANDOM_SEED)

    # -------- Environment --------
    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)
    env.dataset_mode = Config.DATASET_MODE

    # -------- Temporal Engine --------
    if Config.ENABLE_TEMPORAL:
        temporal = TemporalEngine(
            Config.TIME_STEP,
            Config.MAX_TIME_STEPS
        )


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

    # -------- Obstacles --------
    if Config.ENABLE_OBSTACLES:
        env.add_obstacle(Obstacle(200, 200, 350, 350))
        env.add_obstacle(Obstacle(500, 100, 650, 250))

    # -------- Risk Zones --------
    if Config.ENABLE_RISK_ZONES:
        env.add_risk_zone(RiskZone(100, 400, 300, 550, multiplier=1.8))

    print("Environment Summary:", env.summary())

    # -------- Energy Simulation --------

    print("\n--- Temporal Simulation ---")

    while temporal.tick():

        print(f"[Time Step] {temporal.current_step}")

    if (
        Config.ENABLE_DYNAMIC_NODES and
        temporal.current_step % Config.DYNAMIC_NODE_INTERVAL == 0 and
        len(env.nodes) < Config.NODE_COUNT + Config.MAX_DYNAMIC_NODES
    ):
        new_id = len(env.nodes)

        new_node = spawn_single_node(
            Config.MAP_WIDTH,
            Config.MAP_HEIGHT,
            new_id,
            env
        )

        if new_node:
            env.add_node(new_node)
            print(f"[Dynamic] Node Spawned: {new_id}")
        else:
            print("[Dynamic] Spawn Skipped (Obstacle Collision)")





    # Existing energy simulation logic goes here
    print("\n--- Energy Simulation ---")

    uav = env.nodes[0]
    base_position = uav.position()

    visited = 0
    attempted = 0
    energy_consumed_total = 0
    abort_reason = None
    return_triggered = False

    # NEW COUNTERS
    collision_count = 0
    unsafe_return_count = 0

    for target in env.nodes[1:]:

        attempted += 1

        # -------- Collision Check --------
        if env.has_collision(uav.position(), target.position()):
            collision_count += 1
            print(f"Path blocked by obstacle to Node {target.id}")
            continue

        # -------- Threshold Check --------
        if EnergyModel.should_return(uav):
            abort_reason = "threshold"
            return_triggered = True
            print("Battery threshold reached. Returning to base.")
            break

        distance = MetricEngine.euclidean_distance(
            uav.position(),
            target.position()
        )

        # -------- Risk Multiplier --------
        risk_mult = env.risk_multiplier(target.position())
        distance *= risk_mult

        # -------- Feasibility Check --------
        if not EnergyModel.can_travel(uav, distance):
            abort_reason = "insufficient_energy"
            print("Cannot reach next node safely.")
            break

        # -------- Consume Energy --------
        energy_needed = EnergyModel.energy_for_distance(uav, distance)
        EnergyModel.consume(uav, energy_needed)
        energy_consumed_total += energy_needed

        # -------- Return Safety --------
        if not EnergyModel.can_return_to_base(
            uav,
            target.position(),
            base_position
        ):
            unsafe_return_count += 1
            abort_reason = "unsafe_return"
            print("Return-to-base unsafe. Mission halted.")
            break

        visited += 1
        print(
            f"Visited Node {target.id} | "
            f"Battery Left: {round(uav.current_battery, 2)}"
        )

    # -------- MetricEngine Derived Metrics --------
    mission_completion = MetricEngine.mission_completion(
        visited,
        attempted
    )

    energy_efficiency = MetricEngine.energy_efficiency(
        energy_consumed_total,
        visited
    )

    abort_flag = MetricEngine.abort_flag(abort_reason)
    return_flag = MetricEngine.return_flag(return_triggered)

    coverage_ratio = MetricEngine.coverage_ratio(
        visited,
        len(env.nodes) - 1
    )

    constraint_flag = MetricEngine.constraint_violation_flag(
        collision_count,
        unsafe_return_count
    )

    print(f"Total Visited: {visited}")
    print(f"Energy Consumed: {round(energy_consumed_total, 2)}")
    print(f"Mission Completion %: {mission_completion}")
    print(f"Energy Efficiency: {energy_efficiency}")
    print(f"Coverage Ratio: {coverage_ratio}")
    print(f"Constraint Violations: {constraint_flag}")

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
            "runtime": runtime,
            "energy_consumed": round(energy_consumed_total, 2),
            "visited_nodes": visited,
            "attempted_nodes": attempted,
            "mission_completion_pct": mission_completion,
            "energy_efficiency": energy_efficiency,
            "coverage_ratio": coverage_ratio,
            "constraint_flag": constraint_flag,
            "abort_flag": abort_flag,
            "return_flag": return_flag
        }

        Logger.log_json("run_log.json", payload)
        Logger.log_csv(
            "run_metrics.csv",
            list(payload.keys()),
            list(payload.values())
        )

        print("Logs written to /logs directory")

    # -------- Visualization --------
    if Config.ENABLE_VISUALS:
        PlotRenderer.render_environment(env)
        PlotRenderer.render_energy_plots(visited, energy_consumed_total)
        PlotRenderer.render_metrics_snapshot(
            mission_completion,
            energy_consumed_total
        )



if __name__ == "__main__":
    main()
