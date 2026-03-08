class FeatureToggles:
    """
    Centralized controller for enabling or disabling major simulation features.
    These can be overridden via CLI arguments in main.py.

    Toggle Categories for Academic Research:
      Physics     — obstacles, movement, risk zones, 3D heights
      Energy      — UAV propulsion, node TX drain, return-to-base
      Communication — Rician fading, BS uplink, TDMA
      Intelligence  — GA, GLS, SCA, semantic clustering, RP selection
      Sensing     — probabilistic sensing, AoI, digital twin
      Environment — dynamic nodes, predictive avoidance
      Visualization — render mode, frame saving
    """

    # --- Physics ---
    RENDER_MODE = "2D"
    ENABLE_OBSTACLES = True
    MOVING_OBSTACLES = False        # simple preset default
    ENABLE_RISK_ZONES = False
    ENABLE_GAUSSIAN_HEIGHT = True

    # --- Energy ---
    ENABLE_ENERGY = True
    ENABLE_RETURN_TO_BASE = True
    ENABLE_NODE_ENERGY_DRAIN = True

    # --- Communication ---
    ENABLE_BS_UPLINK = False        # simple preset default
    ENABLE_TDMA = True

    # --- Intelligence ---
    ENABLE_GA_SEQUENCE = True
    ENABLE_SEMANTIC_CLUSTERING = True
    ENABLE_RENDEZVOUS_SELECTION = True
    ENABLE_SCA_HOVER = True

    # --- Sensing ---
    ENABLE_PROBABILISTIC_SENSING = True
    ENABLE_AOI_EXPIRATION = True
    ENABLE_ISAC_DIGITAL_TWIN = False

    # --- Environment ---
    ENABLE_DYNAMIC_NODES = False
    ENABLE_NODE_REMOVAL = False
    ENABLE_PREDICTIVE_AVOIDANCE = False

    # --- Visualization ---
    ENABLE_VISUALS = True

    @classmethod
    def apply_overrides(cls, args):
        """Inject CLI arguments into the toggle state unconditionally."""
        if hasattr(args, "render_mode") and args.render_mode:
            cls.RENDER_MODE = str(args.render_mode).upper()

        _bool_flags = {
            "obstacles": "ENABLE_OBSTACLES",
            "moving_obstacles": "MOVING_OBSTACLES",
            "risk_zones": "ENABLE_RISK_ZONES",
            "energy": "ENABLE_ENERGY",
            "bs_uplink": "ENABLE_BS_UPLINK",
            "tdma": "ENABLE_TDMA",
            "ga": "ENABLE_GA_SEQUENCE",
            "clustering": "ENABLE_SEMANTIC_CLUSTERING",
            "rendezvous": "ENABLE_RENDEZVOUS_SELECTION",
            "sca": "ENABLE_SCA_HOVER",
            "sensing": "ENABLE_PROBABILISTIC_SENSING",
            "dynamic_nodes": "ENABLE_DYNAMIC_NODES",
            "predictive_avoidance": "ENABLE_PREDICTIVE_AVOIDANCE",
        }
        for arg_name, toggle_attr in _bool_flags.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                val = str(getattr(args, arg_name)).lower() == "true"
                setattr(cls, toggle_attr, val)

        # Logical guard: no moving obstacles if obstacles are off
        if not cls.ENABLE_OBSTACLES:
            cls.MOVING_OBSTACLES = False

    @classmethod
    def sync_to_config(cls):
        """Push toggle values into Config class attributes."""
        from config.config import Config
        Config.ENABLE_OBSTACLES = cls.ENABLE_OBSTACLES
        Config.ENABLE_MOVING_OBSTACLES = cls.MOVING_OBSTACLES
        Config.ENABLE_RISK_ZONES = cls.ENABLE_RISK_ZONES
        Config.ENABLE_GAUSSIAN_HEIGHT = cls.ENABLE_GAUSSIAN_HEIGHT
        Config.ENABLE_ENERGY = cls.ENABLE_ENERGY
        Config.ENABLE_RETURN_TO_BASE = cls.ENABLE_RETURN_TO_BASE
        Config.ENABLE_NODE_ENERGY_DRAIN = cls.ENABLE_NODE_ENERGY_DRAIN
        Config.ENABLE_BS_UPLINK_MODEL = cls.ENABLE_BS_UPLINK
        Config.ENABLE_TDMA_SCHEDULING = cls.ENABLE_TDMA
        Config.ENABLE_GA_SEQUENCE = cls.ENABLE_GA_SEQUENCE
        Config.ENABLE_SEMANTIC_CLUSTERING = cls.ENABLE_SEMANTIC_CLUSTERING
        Config.ENABLE_RENDEZVOUS_SELECTION = cls.ENABLE_RENDEZVOUS_SELECTION
        Config.ENABLE_SCA_HOVER = cls.ENABLE_SCA_HOVER
        Config.ENABLE_PROBABILISTIC_SENSING = cls.ENABLE_PROBABILISTIC_SENSING
        Config.ENABLE_AOI_EXPIRATION = cls.ENABLE_AOI_EXPIRATION
        Config.ENABLE_ISAC_DIGITAL_TWIN = cls.ENABLE_ISAC_DIGITAL_TWIN
        Config.ENABLE_DYNAMIC_NODES = cls.ENABLE_DYNAMIC_NODES
        Config.ENABLE_NODE_REMOVAL = cls.ENABLE_NODE_REMOVAL
        Config.ENABLE_PREDICTIVE_AVOIDANCE = cls.ENABLE_PREDICTIVE_AVOIDANCE
        Config.ENABLE_VISUALS = cls.ENABLE_VISUALS
