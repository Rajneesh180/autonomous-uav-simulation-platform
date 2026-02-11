class TemporalEngine:
    def __init__(self, time_step=1, max_steps=100):
        self.time_step = time_step
        self.max_steps = max_steps
        self.current_step = 0
        self.active = True

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
