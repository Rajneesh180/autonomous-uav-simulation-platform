#!/usr/bin/env python3
"""Quick probe: sensing parameters and data rates."""
import sys, os, math, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
Config.apply_hostility_profile()
Config.validate()

print("SENSING_TAU:", Config.SENSING_TAU)
print("SENSING_OMEGA:", Config.SENSING_OMEGA)
print("ENABLE_PROBABILISTIC_SENSING:", Config.ENABLE_PROBABILISTIC_SENSING)
print("MIN_SENSING_PROB_THRESH:", Config.MIN_SENSING_PROB_THRESH)
print("ISAC_SENSING_RADIUS:", Config.ISAC_SENSING_RADIUS)
print("ENABLE_TDMA_SCHEDULING:", Config.ENABLE_TDMA_SCHEDULING)
print("UAV_FLIGHT_ALTITUDE:", Config.UAV_FLIGHT_ALTITUDE)

print("\n--- Sensing Probability vs Distance ---")
for d in [0, 1, 5, 10, 20, 50, 100, 150, 200]:
    p = math.exp(-Config.SENSING_TAU * d)
    print(f"  dist={d:4d}m -> p_sense={p:.6f}  pass={p >= Config.MIN_SENSING_PROB_THRESH}")

from core.comms.communication import CommunicationEngine
from core.models.environment_model import Environment
env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)

print("\n--- Data Rate vs Horizontal Distance ---")
node_pos = (100, 100, 0)
for d in [0, 1, 5, 10, 50, 100, 250, 500]:
    uav_pos = (100 + d, 100, Config.UAV_FLIGHT_ALTITUDE)
    rate = CommunicationEngine.achievable_data_rate(node_pos, uav_pos, env)
    print(f"  horiz_dist={d:4d}m -> rate={rate:.4f} Mbps")

# Check: what is chord-fly collecting?
# Does process_data_collection actually work?
print("\n--- process_data_collection test ---")
from core.comms.buffer_aware_manager import BufferAwareManager
from core.models.node_model import SensorNode
test_node = SensorNode(id=99, x=100, y=100, z=0, buffer_capacity=50.0,
                       data_generation_rate=0.5, priority=5, risk=0.3)
test_node.current_buffer = 50.0
for d in [0, 1, 5, 10, 50, 100, 200, 500]:
    test_node.current_buffer = 50.0
    uav_pos = (100+d, 100, Config.UAV_FLIGHT_ALTITUDE)
    collected = BufferAwareManager.process_data_collection(uav_pos, test_node, 1.0, env, active_node_id=99)
    print(f"  horiz_dist={d:4d}m -> collected={collected:.4f} Mb")

# Check: what does simulation_runner pass for total_collected_mbits?
print("\n--- Simulation Data Collection trace ---")
from core.simulation_runner import run_simulation
from core.mission_controller import MissionController

# Patch _move_one_step to log data
_orig = MissionController._move_one_step
data_log = []
def _patched(self):
    before = self.collected_data_mbits
    _orig(self)
    after = self.collected_data_mbits
    if after > before:
        data_log.append({"step": self.temporal.current_step, "delta": after-before, "total": after})
MissionController._move_one_step = _patched

results = run_simulation(verbose=False, render=False, seed_override=42)

print(f"  mission.collected_data_mbits: {results.get('total_collected_mbits', 'MISSING')}")
print(f"  mission.total_uplinked_mbits: {results.get('total_uplinked_mbits', 'MISSING')}")
print(f"  data_collection_rate_percent: {results.get('data_collection_rate_percent', 'MISSING')}")
print(f"  nodes_visited: {results.get('nodes_visited', 'MISSING')}")
print(f"  Data log entries: {len(data_log)}")
for entry in data_log[:20]:
    print(f"    step={entry['step']} delta={entry['delta']:.4f} total={entry['total']:.4f}")
