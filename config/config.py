class Config:
    # -------- Simulation --------
    MAP_WIDTH = 800
    MAP_HEIGHT = 600
    NODE_COUNT = 50
    CLUSTER_COUNT = 5
    DATASET_MODE = "random"
    RANDOM_SEED = 42

    # -------- Visualization --------
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    FPS = 60
    GRID_ON = True

    # -------- Metrics / Logging --------
    ENABLE_LOGGING = True
    LOG_FORMAT = "json"

    # -------- Energy & Constraint Placeholders --------
    ENERGY_PER_METER = 0.1
    BATTERY_CAPACITY = 100.0
    HOVER_COST = 0.0
    RETURN_THRESHOLD = 0.2

    ENABLE_ENERGY = False
    ENABLE_OBSTACLES = False
    ENABLE_RISK_ZONES = False
