class Config:
    # =========================================================
    # Core Map Configuration
    # =========================================================
    MAP_WIDTH = 800
    MAP_HEIGHT = 600

    NODE_COUNT = 50
    CLUSTER_COUNT = 5
    # Supported modes: random, priority_heavy, deadline_critical, risk_dense, mixed_feature
    DATASET_MODE = "random"

    # =========================================================
    # Phase 4: Semantic Intelligence & Latent Clustering
    # =========================================================
    ENABLE_SEMANTIC_CLUSTERING = True
    SCALING_METHOD = "minmax"          # minmax | zscore
    REDUCTION_DIMS = 3                 # Target components for PCA
    CLUSTER_ALGO_MODE = "dbscan"       # kmeans | dbscan
    DBSCAN_EPS = 0.2                   # Calibrated for normalized 800x600 latent space
    DBSCAN_MIN_SAMPLES = 4             # Minimum dense neighbors to form a Routing Centroid

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
    # Energy Model & Aerodynamics
    # =========================================================
    BATTERY_CAPACITY = 600000.0  # Joules (scaled up for realistic Watt-seconds)
    
    # Rotary-Wing Aerodynamic Constants for Propulsion Power P_p(v(t))
    UAV_MASS = 2.0                 # m (kg)
    GRAVITY = 9.8                  # g (m/s^2)
    AIR_DENSITY = 1.225            # rho (kg/m^3)
    ROTOR_DISC_AREA = 0.503        # A (m^2)
    ROTOR_SOLIDITY = 0.05          # s
    ROTOR_TIP_SPEED = 120.0        # U_tip (m/s)
    FUSELAGE_DRAG_RATIO = 0.6      # d_0
    
    # Derived Hovering Power Constants
    PROFILE_POWER_HOVER = 79.856   # P_0 (W)
    INDUCED_POWER_HOVER = 88.627   # P_i (W)
    MEAN_ROTOR_VELOCITY = 4.03     # v_0 (m/s)

    # Legacy basic constants (phased out for continuous modeling where applicable)
    ENERGY_PER_METER = 0.12
    HOVER_COST = PROFILE_POWER_HOVER + INDUCED_POWER_HOVER
    RETURN_THRESHOLD = 0.15

    ENABLE_ENERGY = True
    ENABLE_RETURN_TO_BASE = True
    TERMINATE_ON_ENERGY_RISK = False

    # =========================================================
    # 3D Environment & Communication Modeling (Rician Fading)
    # =========================================================
    UAV_FLIGHT_ALTITUDE = 100.0  # meters (h)
    
    # Rician K-factor (dB)
    RICIAN_K_FACTOR_DB = 15.0
    
    # Environmental constants for LoS probability (Urban profile)
    LOS_PARAM_A = 9.61
    LOS_PARAM_B = 0.16
    
    # Path loss exponents
    PATH_LOSS_LOS = 2.0
    PATH_LOSS_NLOS = 2.8
    
    # Transmission properties
    CARRIER_FREQUENCY = 2e9        # Carrier freq in Hz (2 GHz)
    NOISE_POWER_DENSITY = -174.0   # dBm/Hz
    BANDWIDTH = 1e6                # 1 MHz
    NODE_TX_POWER = 0.1            # 100 mW

    # =========================================================
    # Buffer & Data Collection Settings (DST-BA)
    # =========================================================
    DEFAULT_BUFFER_CAP_MBITS = 50.0 
    DEFAULT_DATA_RATE_MBPS = 0.5    

    # =========================================================
    # Phase 3.5: Probabilistic Sensing & Age of Information
    # =========================================================
    ENABLE_PROBABILISTIC_SENSING = True
    SENSING_TAU = 0.05             # Decay parameter for Prob(success)
    MIN_SENSING_PROB_THRESH = 0.7  # Threshold below which sensing is invalid
    MAX_AOI_LIMIT = 200            # Time-steps until data expires
    ENABLE_AOI_EXPIRATION = True

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

    ENABLE_ROLLING_METRICS = False

    # =========================================================
    # Dynamic Node Behavior
    # =========================================================
    ENABLE_DYNAMIC_NODES = False
    DYNAMIC_NODE_INTERVAL = 15
    MAX_DYNAMIC_NODES = 20

    ENABLE_NODE_REMOVAL = False
    NODE_REMOVAL_INTERVAL = 20
    NODE_REMOVAL_PROBABILITY = 0.15
    MIN_NODE_FLOOR = 15

    # =========================================================
    # UAV Motion Model (Agent-Centric Kinematics)
    # =========================================================
    UAV_STEP_SIZE = 5.0
    COLLISION_MARGIN = 3.0
    MAX_YAW_RATE = 30.0    # degrees per second
    MAX_PITCH_RATE = 15.0  # degrees per second
    MAX_ACCELERATION = 2.0 # m/s^2

    # =========================================================
    # Predictive Avoidance Layer (NEW)
    # =========================================================
    ENABLE_PREDICTIVE_AVOIDANCE = True
    ENABLE_LOCAL_STEERING = True
    REPLAN_COOLDOWN_STEPS = 5
    
    # Phase 3.5: ISAC and Digital Twin Obstacle Mapping
    ENABLE_ISAC_DIGITAL_TWIN = True
    ISAC_SENSING_RADIUS = 150.0

    # These will be derived from hostility
    PREDICTION_HORIZON = 3
    STEERING_ANGLES = [15, -15, 30, -30]

    # =========================================================
    # Motion Primitive Scoring (Research-Clean Layer)
    # =========================================================

    # Weight balance for motion primitive evaluation
    ALIGNMENT_WEIGHT = 1.0  # Attraction to target
    OBSTACLE_PENALTY_WEIGHT = 3.0  # Proximity to obstacles
    RISK_PENALTY_WEIGHT = 2.0  # Risk zone multiplier penalty
    ENERGY_PENALTY_WEIGHT = 0.5  # Energy efficiency consideration

    # Obstacle proximity radius (soft penalty zone)
    OBSTACLE_INFLUENCE_RADIUS = 40.0

    # Normalization epsilon
    SCORE_EPS = 1e-6

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
