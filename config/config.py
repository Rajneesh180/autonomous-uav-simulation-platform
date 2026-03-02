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
    # Base Station Uplink Model (Gap 10 — Zheng & Liu, IEEE TVT 2025)
    # R_BS = B log2(1 + γ₀ / d_3d^α)
    # =========================================================
    BS_GAMMA_0_DB = -10.0           # Reference SNR / channel gain (γ₀) in dB
    BS_PATH_LOSS_EXP = 2.5          # Path-loss exponent α for UAV-BS link
    BS_HEIGHT_M = 5.0               # Base station antenna height (metres)
    BS_DATA_AGE_LIMIT = 300         # T_data_limit: max steps before forced uplink (was 100)
    BS_UPLINK_CHECK_INTERVAL = 10   # Check uplink urgency every N steps
    ENABLE_BS_UPLINK_MODEL = True   # Toggle BS uplink urgency + offload

    # =========================================================
    # Buffer & Data Collection Settings (DST-BA)
    # =========================================================
    DEFAULT_BUFFER_CAP_MBITS = 50.0 
    DEFAULT_DATA_RATE_MBPS = 0.5    

    # =========================================================
    # Rendezvous Point (RP) Selection (Gap 1 — Donipati et al.)
    # Greedy neighbourhood algorithm: compresses N nodes → |R| << N RPs.
    # =========================================================
    ENABLE_RENDEZVOUS_SELECTION = True
    RP_COVERAGE_RADIUS = 120.0          # R_max: ground node TX range (metres)
    RP_OBSTACLE_BUFFER = 35.0           # Min distance from obstacle footprint for RP eligibility

    # =========================================================
    # TDMA Single-Node Scheduling (Gap 7 — Wang et al., IEEE IoT 2022)
    # =========================================================
    ENABLE_TDMA_SCHEDULING = True       # enforce single-node-per-slot discipline

    # =========================================================
    # SCA Hover Position Optimizer (Gap 9)
    # Successive Convex Approximation: refine hover xyz minimising path cost
    # =========================================================
    ENABLE_SCA_HOVER = True
    SCA_MAX_ITERATIONS = 15             # max SCA refinement steps per hover point
    SCA_STEP_SIZE = 3.0                 # initial gradient step (metres)
    SCA_CONVERGENCE_TOL = 0.5           # distance convergence threshold (metres)

    # =========================================================
    # IoT Node First-Order Radio Energy Model (Gap 2 — DST-BA)
    # Based on Heinzelman et al. first-order model:
    #   E_tx(b, d) = E_elec * b + E_amp * b * d^2  (Joules)
    # where b = number of bits transmitted, d = distance (m).
    # Aligned with: Donipati et al. (DST-BA, IEEE TNSM 2025)
    # =========================================================
    NODE_E_ELEC_J_PER_BIT = 50e-9          # Electronics energy: 50 nJ/bit
    NODE_E_AMP_J_PER_BIT_M2 = 10e-12       # Amplifier energy: 10 pJ/bit/m^2
    NODE_BATTERY_CAPACITY_J = 0.5          # Ground node battery: 0.5 J (typical IoT sensor)
    ENABLE_NODE_ENERGY_DRAIN = True         # Toggle node TX energy depletion

    # =========================================================
    # Phase 3.5: Probabilistic Sensing & Age of Information
    # =========================================================
    ENABLE_PROBABILISTIC_SENSING = True
    SENSING_TAU = 0.05             # Decay parameter for Prob(success)
    MIN_SENSING_PROB_THRESH = 0.7  # Threshold below which sensing is invalid
    MAX_AOI_LIMIT = 200            # Time-steps until data expires
    ENABLE_AOI_EXPIRATION = True
    AOI_URGENCY_WEIGHT = 1.5        # AoI-to-priority boost multiplier for semantic clustering
    SENSING_OMEGA = 0.95           # Target cumulative sensing success probability (Gap 3)
    SENSING_SLOT_DURATION = 1.0    # Duration of each sensing trial slot T_s (seconds)

    # =========================================================
    # Temporal Engine
    # =========================================================
    TIME_STEP = 1
    MAX_TIME_STEPS = 800

    ENABLE_TEMPORAL = True
    FRAME_SUBSAMPLE_INTERVAL = 20   # render every Nth step as a keyframe (always-on)

    # =========================================================
    # GA Visiting Sequence Optimizer (Gap 4 — Zheng & Liu, IEEE TVT 2025)
    # =========================================================
    ENABLE_GA_SEQUENCE = True
    GA_POPULATION_SIZE = 30
    GA_MAX_GENERATIONS = 50
    GA_CROSSOVER_RATE = 0.85
    GA_MUTATION_RATE = 0.15
    GA_TW_PENALTY_WEIGHT = 5.0

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

    # 3D Gaussian Obstacle Height Model (Gap 5 — Zheng & Liu, IEEE TVT 2025)
    # z_obs(x,y) = Σ hᵢ * exp(-((x-xᵢ)/aˣ)² - ((y-yᵢ)/aʸ)²)
    GAUSSIAN_SPREAD_X = 40.0        # aˣ: horizontal spread in x (metres)
    GAUSSIAN_SPREAD_Y = 40.0        # aʸ: horizontal spread in y (metres)
    VERTICAL_CLEARANCE = 10.0       # Δz: minimum clearance above obstacle peak (m)
    ENABLE_GAUSSIAN_HEIGHT = True   # Toggle altitude constraint enforcement

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

    # =========================================================
    # Configuration Validation (Research-Grade Assertions)
    # =========================================================
    @classmethod
    def validate(cls):
        """
        Pre-flight configuration sanity checks.

        Raises ValueError with a descriptive message for any parameter that
        violates physical, algorithmic, or domain-specific constraints.
        This method should be called once at startup (before any simulation
        run) to surface misconfigurations early.
        """
        errors: list = []

        def _check(condition: bool, msg: str):
            if not condition:
                errors.append(msg)

        # --- Map / Geometry ---
        _check(cls.MAP_WIDTH > 0 and cls.MAP_HEIGHT > 0,
               "MAP_WIDTH and MAP_HEIGHT must be positive integers")
        _check(cls.NODE_COUNT >= 2,
               "NODE_COUNT must be >= 2 (at least UAV + 1 ground node)")
        _check(cls.CLUSTER_COUNT >= 1,
               "CLUSTER_COUNT must be >= 1")
        _check(cls.DATASET_MODE in ("random", "priority_heavy", "deadline_critical",
                                     "risk_dense", "mixed_feature"),
               f"Unknown DATASET_MODE: '{cls.DATASET_MODE}'")

        # --- Energy / Aerodynamics ---
        _check(cls.BATTERY_CAPACITY > 0,
               "BATTERY_CAPACITY must be positive (Joules)")
        _check(cls.UAV_MASS > 0,
               "UAV_MASS must be positive (kg)")
        _check(0 < cls.RETURN_THRESHOLD < 1,
               "RETURN_THRESHOLD must be in (0, 1) — fraction of battery")
        _check(cls.ENERGY_PER_METER > 0,
               "ENERGY_PER_METER must be positive")

        # --- Communication ---
        _check(cls.UAV_FLIGHT_ALTITUDE > 0,
               "UAV_FLIGHT_ALTITUDE must be positive (metres)")
        _check(cls.BANDWIDTH > 0,
               "BANDWIDTH must be positive (Hz)")
        _check(cls.NODE_TX_POWER > 0,
               "NODE_TX_POWER must be positive (Watts)")
        _check(cls.CARRIER_FREQUENCY > 0,
               "CARRIER_FREQUENCY must be positive (Hz)")

        # --- SCA Hover Optimizer ---
        _check(cls.SCA_MAX_ITERATIONS >= 1,
               "SCA_MAX_ITERATIONS must be >= 1")
        _check(cls.SCA_STEP_SIZE > 0,
               "SCA_STEP_SIZE must be positive (metres)")
        _check(cls.SCA_CONVERGENCE_TOL > 0,
               "SCA_CONVERGENCE_TOL must be positive (metres)")

        # --- GA Sequence Optimizer ---
        _check(cls.GA_POPULATION_SIZE >= 4,
               "GA_POPULATION_SIZE must be >= 4 for crossover to operate")
        _check(cls.GA_MAX_GENERATIONS >= 1,
               "GA_MAX_GENERATIONS must be >= 1")
        _check(0 <= cls.GA_CROSSOVER_RATE <= 1,
               "GA_CROSSOVER_RATE must be in [0, 1]")
        _check(0 <= cls.GA_MUTATION_RATE <= 1,
               "GA_MUTATION_RATE must be in [0, 1]")

        # --- Temporal ---
        _check(cls.TIME_STEP > 0,
               "TIME_STEP must be positive")
        _check(cls.MAX_TIME_STEPS >= 1,
               "MAX_TIME_STEPS must be >= 1")

        # --- AoI / Sensing ---
        _check(cls.MAX_AOI_LIMIT >= 1,
               "MAX_AOI_LIMIT must be >= 1")
        _check(cls.AOI_URGENCY_WEIGHT >= 0,
               "AOI_URGENCY_WEIGHT must be non-negative")
        _check(0 < cls.SENSING_OMEGA <= 1,
               "SENSING_OMEGA must be in (0, 1]")

        # --- Clustering ---
        _check(cls.REDUCTION_DIMS >= 1,
               "REDUCTION_DIMS must be >= 1")
        _check(cls.CLUSTER_ALGO_MODE in ("kmeans", "dbscan"),
               f"Unknown CLUSTER_ALGO_MODE: '{cls.CLUSTER_ALGO_MODE}'")
        _check(cls.DBSCAN_EPS > 0,
               "DBSCAN_EPS must be positive")
        _check(cls.DBSCAN_MIN_SAMPLES >= 1,
               "DBSCAN_MIN_SAMPLES must be >= 1")
        _check(cls.SCALING_METHOD in ("minmax", "zscore"),
               f"Unknown SCALING_METHOD: '{cls.SCALING_METHOD}'")

        # --- Motion Model ---
        _check(cls.UAV_STEP_SIZE > 0,
               "UAV_STEP_SIZE must be positive (m/step)")
        _check(cls.MAX_YAW_RATE > 0,
               "MAX_YAW_RATE must be positive (deg/s)")
        _check(cls.MAX_PITCH_RATE > 0,
               "MAX_PITCH_RATE must be positive (deg/s)")
        _check(cls.MAX_ACCELERATION > 0,
               "MAX_ACCELERATION must be positive (m/s^2)")
        _check(cls.COLLISION_MARGIN >= 0,
               "COLLISION_MARGIN must be non-negative (metres)")

        # --- Hostility ---
        _check(cls.HOSTILITY_LEVEL in ("low", "medium", "high", "extreme"),
               f"Unknown HOSTILITY_LEVEL: '{cls.HOSTILITY_LEVEL}'")

        # --- RP Selection ---
        _check(cls.RP_COVERAGE_RADIUS > 0,
               "RP_COVERAGE_RADIUS must be positive (metres)")

        # --- Node Energy ---
        _check(cls.NODE_BATTERY_CAPACITY_J > 0,
               "NODE_BATTERY_CAPACITY_J must be positive (Joules)")

        # --- Base Station Uplink ---
        _check(cls.BS_DATA_AGE_LIMIT >= 1,
               "BS_DATA_AGE_LIMIT must be >= 1 (steps)")

        if errors:
            msg = "Configuration validation failed:\n" + "\n".join(
                f"  [{i+1}] {e}" for i, e in enumerate(errors)
            )
            raise ValueError(msg)
