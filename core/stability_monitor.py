class StabilityMonitor:
    def __init__(self):
        self.window = 10
        self.replans = []
        self.churn = []
        self.energy_error = []

    def record(self, replan_count, churn_rate, energy_error):
        self.replans.append(replan_count)
        self.churn.append(churn_rate)
        self.energy_error.append(energy_error)

        if len(self.replans) > self.window:
            self.replans.pop(0)
            self.churn.pop(0)
            self.energy_error.pop(0)

    def stability_score(self):
        if not self.replans:
            return 1.0

        r = sum(self.replans) / len(self.replans)
        c = sum(self.churn) / len(self.churn)
        e = sum(self.energy_error) / len(self.energy_error)

        score = 1 / (1 + r * 0.2 + c * 5 + e * 0.01)
        return round(score, 3)

    def is_unstable(self):
        return self.stability_score() < 0.4
