class FeatureToggles:
    """
    Centralized controller for enabling or disabling major simulation features.
    These can be overridden via CLI arguments in main.py.
    """

    # --- Core Mechanics ---
    # Physics always runs in 3D.  RENDER_MODE controls visualisation only.
    RENDER_MODE = "2D"          # "2D", "3D", or "both"
    ENABLE_OBSTACLES = True     # Apply obstacle constraints
    MOVING_OBSTACLES = True     # Evolve obstacles temporally

    # --- Metrics & Output ---
    ENABLE_VISUALS = True       # Save rendering artifacts
    ENABLE_SEMANTIC_CLUSTERING = True 

    @classmethod
    def apply_overrides(cls, args):
        """Inject CLI arguments into the toggle state unconditionally."""
        if hasattr(args, "render_mode") and args.render_mode:
            cls.RENDER_MODE = str(args.render_mode).upper()
            
        if hasattr(args, "obstacles"):
            cls.ENABLE_OBSTACLES = str(args.obstacles).lower() == 'true'
            
        if hasattr(args, "moving_obstacles"):
            cls.MOVING_OBSTACLES = str(args.moving_obstacles).lower() == 'true'
            if not cls.ENABLE_OBSTACLES and cls.MOVING_OBSTACLES:
               cls.MOVING_OBSTACLES = False # Cannot have moving obstacles if obstacles are off
