from config.config import Config
from core.seed_manager import set_global_seed
from core.node_model import Node
from core.environment_model import Environment
import random

def main():
    print("=== Autonomous UAV Simulation Platform ===")
    set_global_seed(Config.RANDOM_SEED)

    # Create Environment
    env = Environment(Config.MAP_WIDTH, Config.MAP_HEIGHT)

    # Generate Sample Nodes
    for i in range(Config.NODE_COUNT):
        x = random.randint(0, Config.MAP_WIDTH)
        y = random.randint(0, Config.MAP_HEIGHT)
        node = Node(id=i, x=x, y=y)
        env.add_node(node)

    print("Environment Summary:", env.summary())

if __name__ == "__main__":
    main()
