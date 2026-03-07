#!/usr/bin/env python3
"""Behavioral execution audit: instrument subsystems and dump metrics."""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
Config.apply_hostility_profile()
Config.validate()

counters = {}

# --- Patch Environment ---
from core.models.environment_model import Environment
_orig_uo = Environment.update_obstacles
def _p_uo(self):
    counters["update_obstacles"] = counters.get("update_obstacles", 0) + 1
    return _orig_uo(self)
Environment.update_obstacles = _p_uo

_orig_ur = Environment.update_risk_zones
def _p_ur(self, s):
    counters["update_risk_zones"] = counters.get("update_risk_zones", 0) + 1
    return _orig_ur(self, s)
Environment.update_risk_zones = _p_ur

# --- Patch ClusterManager ---
from core.clustering.cluster_manager import ClusterManager
_orig_pc = ClusterManager.perform_clustering
def _p_pc(self, *a, **k):
    counters["perform_clustering"] = counters.get("perform_clustering", 0) + 1
    return _orig_pc(self, *a, **k)
ClusterManager.perform_clustering = _p_pc

_orig_sr = ClusterManager.should_recluster
def _p_sr(self, *a, **k):
    counters["should_recluster"] = counters.get("should_recluster", 0) + 1
    return _orig_sr(self, *a, **k)
ClusterManager.should_recluster = _p_sr

# --- Patch BufferAwareManager ---
from core.comms.buffer_aware_manager import BufferAwareManager
_orig_es = BufferAwareManager.execute_service
@staticmethod
def _p_es(*a, **k):
    counters["execute_service"] = counters.get("execute_service", 0) + 1
    return _orig_es(*a, **k)
BufferAwareManager.execute_service = _p_es

# --- Patch PhysicsEngine ---
from core.physics_engine import PhysicsEngine
_orig_em = PhysicsEngine.execute_movement
@staticmethod
def _p_em(*a, **k):
    counters["execute_movement"] = counters.get("execute_movement", 0) + 1
    return _orig_em(*a, **k)
PhysicsEngine.execute_movement = _p_em

# --- Patch CommunicationEngine ---
from core.comms.communication import CommunicationEngine
_orig_fb = CommunicationEngine.fill_buffer
@staticmethod
def _p_fb(*a, **k):
    counters["fill_buffer"] = counters.get("fill_buffer", 0) + 1
    return _orig_fb(*a, **k)
CommunicationEngine.fill_buffer = _p_fb

# --- Patch MissionController ---
from core.mission_controller import MissionController
_orig_rp = MissionController._recompute_plan
def _p_rp(self, *a, **k):
    counters["recompute_plan"] = counters.get("recompute_plan", 0) + 1
    return _orig_rp(self, *a, **k)
MissionController._recompute_plan = _p_rp

_orig_tr = MissionController._trigger_replan
def _p_tr(self, reason):
    counters["trigger_replan"] = counters.get("trigger_replan", 0) + 1
    counters[f"replan:{reason}"] = counters.get(f"replan:{reason}", 0) + 1
    return _orig_tr(self, reason)
MissionController._trigger_replan = _p_tr

_orig_mo = MissionController._move_one_step
def _p_mo(self):
    counters["move_one_step"] = counters.get("move_one_step", 0) + 1
    return _orig_mo(self)
MissionController._move_one_step = _p_mo

# --- Run simulation ---
from core.simulation_runner import run_simulation
print("Running simulation with seed=42 ...")
results = run_simulation(verbose=False, render=False, seed_override=42)

print("\n" + "=" * 60)
print("BEHAVIORAL EXECUTION AUDIT — seed=42")
print("=" * 60)

print("\n--- SUBSYSTEM CALL COUNTS ---")
for k in sorted(counters.keys()):
    print(f"  {k:30s}: {counters[k]}")

print("\n--- KEY METRICS ---")
keys = [
    "steps", "replans", "collision_count", "visited_nodes",
    "coverage_ratio_percent", "energy_consumed_total_J",
    "adaptation_latency", "path_stability_index",
    "node_churn_impact", "replan_frequency", "collision_rate",
    "energy_prediction_error", "event_count",
    "total_collected_mbits", "mean_achievable_rate_mbps",
    "mean_service_time_s", "throughput_mbps",
]
for k in keys:
    v = results.get(k, "MISSING")
    print(f"  {k:30s}: {v}")

print("\n--- EVENT/REPLAN TIMESTAMPS ---")
print(f"  event_timestamps: {results.get('event_timestamps', [])}")
print(f"  replan_timestamps: {results.get('replan_timestamps', [])}")

print("\n--- NODE VISIT DETAILS ---")
visited = results.get("visited_nodes", 0)
total = results.get("total_nodes", Config.NODE_COUNT)
print(f"  visited: {visited} / {total}")

# Also run with seed=123 for comparison
print("\n" + "=" * 60)
print("SEED COMPARISON: running seed=123 ...")
print("=" * 60)
counters2 = {}
# reset counters
for k in list(counters.keys()):
    counters[k] = 0

results2 = run_simulation(verbose=False, render=False, seed_override=123)
print("\n--- seed=123 CALL COUNTS ---")
for k in sorted(counters.keys()):
    if counters[k] > 0:
        print(f"  {k:30s}: {counters[k]}")

print("\n--- seed=123 KEY METRICS ---")
for k in keys:
    v = results2.get(k, "MISSING")
    print(f"  {k:30s}: {v}")

print("\n--- SEED DIFF ---")
for k in keys:
    v1 = results.get(k, 0)
    v2 = results2.get(k, 0)
    if v1 != v2:
        print(f"  {k:30s}: seed42={v1} vs seed123={v2}")
    else:
        print(f"  {k:30s}: SAME ({v1})")

print("\nAudit complete.")
