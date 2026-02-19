class TemporalEngine:
    def __init__(self, time_step=1, max_steps=100):
        self.time_step = time_step
        self.max_steps = max_steps

        self.current_step = 0
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
        self.active = True
        self.replan_count = 0
        self.replan_required = False
        self.replan_reason = None
        self.replan_flash_frames = 0

    # ---------------------------------------------------------
    # Time Advancement
    # ---------------------------------------------------------

    def tick(self):
        """
        Advances time by one step.
        Returns True if system remains active.
        """

        if not self.active:
            return False

        # Advance time
        self.current_step += self.time_step

        # Check termination AFTER advancing
        if self.current_step > self.max_steps:
            self.active = False
            return False

        return True

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
            "max_steps": self.max_steps,
            "active": self.active,
            "replan_count": self.replan_count,
        }
