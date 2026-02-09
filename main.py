from config.config import Config
from core.seed_manager import set_global_seed

def main():
    print("=== Autonomous UAV Simulation Platform ===")
    set_global_seed(Config.RANDOM_SEED)

    print("Configuration Loaded:")
    print(f"Nodes: {Config.NODE_COUNT}")
    print(f"Dataset Mode: {Config.DATASET_MODE}")
    print(f"Visuals Enabled: {Config.ENABLE_VISUALS}")

if __name__ == "__main__":
    main()
