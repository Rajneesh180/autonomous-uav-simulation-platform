class TemporalEngine:
    def __init__(self, time_step=1, max_steps=100):
        self.time_step = time_step
        self.max_steps = max_steps
        self.max_time = float(max_steps * time_step)

        self.current_step = 0
        self.current_time = 0.0  # continuous clock (seconds)
        self.active = True

        # --- Replan State ---
        self.replan_count = 0
        self.replan_required = False
        self.replan_reason = None

        # --- Visual Flash Buffer ---
        self.replan_flash_frames = 0

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    def reset(self):
        self.current_step = 0
        self.current_time = 0.0
        self.active = True
        self.replan_count = 0
        self.replan_required = False
        self.replan_reason = None
        self.replan_flash_frames = 0

    # ---------------------------------------------------------
    # Time Advancement
    # ---------------------------------------------------------

    def advance(self, dt: float):
        """
        Advances the continuous clock by dt seconds.
        Returns True if the system remains active after advancement.
        """
        if not self.active:
            return False

        self.current_time += dt

        if self.current_time > self.max_time:
            self.active = False
            return False

        return True

    def tick(self):
        """
        Advances time by one discrete step (backward-compatible wrapper).
        Returns True if system remains active.
        """
        if not self.active:
            return False

        # Advance discrete step counter
        self.current_step += self.time_step

        # Advance continuous clock
        result = self.advance(float(self.time_step))

        # Check termination via legacy step counter as well
        if self.current_step > self.max_steps:
            self.active = False
            return False

        return result

    # ---------------------------------------------------------
    # Replan Management
    # ---------------------------------------------------------

    def trigger_replan(self, reason: str):
        """
        Marks replan as required.
        """
        if not self.replan_required:
            self.replan_required = True
            self.replan_reason = reason

        self.replan_count += 1
        self.replan_flash_frames = 5

    def reset_replan(self):
        self.replan_required = False
        self.replan_reason = None

    def consume_replan_flash(self):
        if self.replan_flash_frames > 0:
            self.replan_flash_frames -= 1
            return True
        return False

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(self):
        return {
            "current_step": self.current_step,
            "current_time": self.current_time,
            "max_steps": self.max_steps,
            "max_time": self.max_time,
            "active": self.active,
            "replan_count": self.replan_count,
        }
