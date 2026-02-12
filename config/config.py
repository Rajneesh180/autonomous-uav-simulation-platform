class Config:
    # -------- Simulation --------
    MAP_WIDTH = 800
    MAP_HEIGHT = 600
    NODE_COUNT = 50
    CLUSTER_COUNT = 5
    DATASET_MODE = "random"
    RANDOM_SEED = 42
    RANDOMIZE_SEED = True

    # -------- Visualization --------
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    FPS = 60
    GRID_ON = True

    # -------- Metrics / Logging --------
    ENABLE_LOGGING = True
    LOG_FORMAT = "json"

    # -------- Energy & Constraint Placeholders --------
    ENERGY_PER_METER = 0.12
    BATTERY_CAPACITY = 300.0
    HOVER_COST = 0.0
    RETURN_THRESHOLD = 0.2

    ENABLE_ENERGY = True
    ENABLE_OBSTACLES = True
    ENABLE_RISK_ZONES = True
    ENABLE_VISUALS = True

    # -------- Temporal Engine --------
    TIME_STEP = 1
    MAX_TIME_STEPS = 50
    ENABLE_TEMPORAL = True

    # -------- Phase 3 Dynamic Controls --------
    ENABLE_DYNAMIC_NODES = True
    DYNAMIC_NODE_INTERVAL = 10  # every 10 ticks
    MAX_DYNAMIC_NODES = 30

    # -------- Dynamic Node Removal --------
    ENABLE_NODE_REMOVAL = True
    NODE_REMOVAL_INTERVAL = 10
    NODE_REMOVAL_PROBABILITY = 0.25
    MIN_NODE_FLOOR = 10
