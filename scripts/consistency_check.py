"""Consistency cross-check script for Phase-4 validation."""
import json
import csv
import sys

LATEST = "2026-03-07_09-12-42"
BASE = f"visualization/runs/{LATEST}"

with open(f"{BASE}/logs/run_summary.json") as f:
    summary = json.load(f)

rows = []
with open(f"{BASE}/telemetry/step_telemetry.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

errors = []

# nodes_visited consistency
telem_visited = int(rows[-1]["nodes_visited"])
summary_visited = summary.get("nodes_visited", 0)
match = "MATCH" if summary_visited == telem_visited else "MISMATCH"
print(f"nodes_visited: summary={summary_visited}, telemetry_final={telem_visited} -> {match}")
if match == "MISMATCH":
    errors.append("nodes_visited mismatch")

# steps consistency
summary_steps = summary.get("steps", 0)
telem_steps = len(rows)
match = "MATCH" if summary_steps == telem_steps else "MISMATCH"
print(f"steps: summary={summary_steps}, telemetry_rows={telem_steps} -> {match}")
if match == "MISMATCH":
    errors.append("steps mismatch")

# coverage_ratio consistency
total_nodes = summary.get("total_nodes", 49)
expected_cov = round(summary_visited / total_nodes * 100, 2)
reported_cov = round(summary.get("coverage_ratio_percent", 0), 2)
match = "MATCH" if abs(expected_cov - reported_cov) < 0.1 else "MISMATCH"
print(f"coverage_ratio: expected={expected_cov}%, reported={reported_cov}% -> {match}")
if match == "MISMATCH":
    errors.append("coverage_ratio mismatch")

# final_battery_J consistency
telem_final_bat = float(rows[-1]["battery_J"])
summary_bat = summary.get("final_battery_J", 0)
discrepancy = abs(telem_final_bat - summary_bat)
status = "OK" if discrepancy < 1000 else "LARGE_DISCREPANCY"
print(f"final_battery_J: telemetry={telem_final_bat:.2f}, summary={summary_bat:.2f}, discrepancy={discrepancy:.2f} J -> {status}")
if status != "OK":
    errors.append("final_battery discrepancy > 1000 J")

# replans consistency
telem_replans = int(rows[-1]["replan_count"])
summary_replans = summary.get("replans", 0)
match = "MATCH" if telem_replans == summary_replans else "MISMATCH"
print(f"replans: summary={summary_replans}, telemetry_final={telem_replans} -> {match}")
if match == "MISMATCH":
    errors.append("replans mismatch")

# collision_count consistency
telem_collisions = int(rows[-1]["collision_count"])
summary_collision_rate = summary.get("collision_rate", 0)
print(f"collision_count (telemetry final): {telem_collisions}, collision_rate (summary): {summary_collision_rate}")

print(f"\nseed: {summary.get('seed')}")
print(f"run_id: {summary.get('run_id')}")

if errors:
    print(f"\nFAILED: {errors}")
    sys.exit(1)
else:
    print("\n=== All consistency checks PASSED ===")
