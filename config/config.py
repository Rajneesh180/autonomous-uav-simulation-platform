class Config:
    # =========================================================
    # Core Map Configuration
    # =========================================================
    MAP_WIDTH = 800
    MAP_HEIGHT = 600

    NODE_COUNT = 50
    CLUSTER_COUNT = 5
    DATASET_MODE = "random"

    # =========================================================
    # Randomness / Reproducibility
    # =========================================================
    RANDOM_SEED = 42
    RANDOMIZE_SEED = False  # Research default

    # =========================================================
    # Visualization
    # =========================================================
    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 800
    FPS = 60
    GRID_ON = True
    ENABLE_VISUALS = True
    FRAME_SAVE_INTERVAL = 5

    # =========================================================
    # Logging
    # =========================================================
    ENABLE_LOGGING = True
    LOG_FORMAT = "json"

    # =========================================================
    # Energy Model
    # =========================================================
    BATTERY_CAPACITY = 600.0
    ENERGY_PER_METER = 0.12
    HOVER_COST = 0.02
    RETURN_THRESHOLD = 0.15

    ENABLE_ENERGY = True
    ENABLE_RETURN_TO_BASE = True
    TERMINATE_ON_ENERGY_RISK = False

    # =========================================================
    # Temporal Engine
    # =========================================================
    TIME_STEP = 1
    MAX_TIME_STEPS = 400
    ENABLE_TEMPORAL = True

    # =========================================================
    # Hostility Level (Phase-3 Control Spectrum)
    # =========================================================
    HOSTILITY_LEVEL = "medium"  # low | medium | high | extreme

    # =========================================================
    # Obstacles
    # =========================================================
    ENABLE_OBSTACLES = True
    ENABLE_MOVING_OBSTACLES = True
    OBSTACLE_MOTION_MODE = "linear"  # linear | random_walk
    OBSTACLE_VELOCITY_SCALE = 0.6

    # =========================================================
    # Risk Zones
    # =========================================================
    ENABLE_RISK_ZONES = True

    # =========================================================
    # Dynamic Node Behavior
    # =========================================================
    ENABLE_DYNAMIC_NODES = True
    DYNAMIC_NODE_INTERVAL = 15
    MAX_DYNAMIC_NODES = 20

    ENABLE_NODE_REMOVAL = True
    NODE_REMOVAL_INTERVAL = 20
    NODE_REMOVAL_PROBABILITY = 0.15
    MIN_NODE_FLOOR = 15

    # =========================================================
    # UAV Motion Model
    # =========================================================
    UAV_STEP_SIZE = 5.0
    COLLISION_MARGIN = 3.0

    # =========================================================
    # Predictive Avoidance Layer (NEW)
    # =========================================================
    ENABLE_PREDICTIVE_AVOIDANCE = True
    ENABLE_LOCAL_STEERING = True
    REPLAN_COOLDOWN_STEPS = 5

    # These will be derived from hostility
    PREDICTION_HORIZON = 3
    STEERING_ANGLES = [15, -15, 30, -30]

    # =========================================================
    # Hostility Policy Application
    # =========================================================
    @staticmethod
    def apply_hostility_profile():
        level = Config.HOSTILITY_LEVEL.lower()

        if level == "low":
            Config.OBSTACLE_VELOCITY_SCALE = 0.3
            Config.DYNAMIC_NODE_INTERVAL = 25
            Config.NODE_REMOVAL_PROBABILITY = 0.05
            Config.PREDICTION_HORIZON = 2
            Config.STEERING_ANGLES = [15, -15]

        elif level == "medium":
            Config.OBSTACLE_VELOCITY_SCALE = 0.6
            Config.DYNAMIC_NODE_INTERVAL = 15
            Config.NODE_REMOVAL_PROBABILITY = 0.15
            Config.PREDICTION_HORIZON = 3
            Config.STEERING_ANGLES = [15, -15, 30, -30]

        elif level == "high":
            Config.OBSTACLE_VELOCITY_SCALE = 1.0
            Config.DYNAMIC_NODE_INTERVAL = 10
            Config.NODE_REMOVAL_PROBABILITY = 0.25
            Config.PREDICTION_HORIZON = 4
            Config.STEERING_ANGLES = [15, -15, 30, -30, 45, -45]

        elif level == "extreme":
            Config.OBSTACLE_VELOCITY_SCALE = 1.5
            Config.DYNAMIC_NODE_INTERVAL = 5
            Config.NODE_REMOVAL_PROBABILITY = 0.4
            Config.PREDICTION_HORIZON = 5
            Config.STEERING_ANGLES = [15, -15, 30, -30, 45, -45, 60, -60]
