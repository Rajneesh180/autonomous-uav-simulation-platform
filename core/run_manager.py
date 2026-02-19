import os
from datetime import datetime


class RunManager:
    """
    Central filesystem authority for a simulation run.
    All runtime artifacts must be stored under:
        results/runs/<RUN_ID>/
    """

    def __init__(self):
        self.run_id = self._generate_run_id()
        self.base_path = os.path.join("results", "runs", self.run_id)

        self.paths = {
            "logs": os.path.join(self.base_path, "logs"),
            "frames": os.path.join(self.base_path, "frames"),
            "plots": os.path.join(self.base_path, "plots"),
            "figures": os.path.join(self.base_path, "figures"),
        }

        self._create_directories()

    # ----------------------------
    # Internal
    # ----------------------------

    def _generate_run_id(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def _create_directories(self):
        os.makedirs(self.base_path, exist_ok=True)
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    # ----------------------------
    # Public API
    # ----------------------------

    def get_path(self, category: str):
        if category not in self.paths:
            raise ValueError(f"Invalid run category: {category}")
        return self.paths[category]

    def get_run_id(self):
        return self.run_id
