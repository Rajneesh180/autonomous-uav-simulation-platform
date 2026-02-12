import csv
import json
from datetime import datetime
from pathlib import Path


class Logger:
    @staticmethod
    def log_json(filepath, data):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            f.write(json.dumps(data) + "\n")

    @staticmethod
    def log_csv(filepath, headers, row):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_exists = path.exists()

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
