import os
from datetime import datetime


class RunManager:
    def __init__(self, base_dir="artifacts/runs"):
        self.base_dir = base_dir
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_path = os.path.join(self.base_dir, self.run_id)

        self.figures_path = os.path.join(self.run_path, "figures")
        self.plots_path = os.path.join(self.run_path, "plots")
        self.logs_path = os.path.join(self.run_path, "logs")

        self._create_dirs()

    def _create_dirs(self):
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

    def get_figures_path(self):
        return self.figures_path

    def get_plots_path(self):
        return self.plots_path

    def get_logs_path(self):
        return self.logs_path

    def get_run_path(self):
        return self.run_path

    def get_run_id(self):
        return self.run_id
