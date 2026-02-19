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

    def reset(self):
        self.current_step = 0
        self.active = True

    def tick(self):
        if not self.active:
            return False

        self.current_step += self.time_step

        if self.current_step >= self.max_steps:
            self.active = False

        return self.active

    def summary(self):
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "active": self.active,
        }

    def trigger_replan(self, reason):
        if not self.replan_required:
            self.replan_required = True
            self.replan_reason = reason

        self.replan_count += 1

        # Activate visual flash for 5 frames
        self.replan_flash_frames = 5

    def consume_replan_flash(self):
        """
        Called by renderer.
        Returns True if flash should be shown.
        """
        if self.replan_flash_frames > 0:
            self.replan_flash_frames -= 1
            return True
        return False

    def reset_replan(self):
        self.replan_required = False
        self.replan_reason = None
