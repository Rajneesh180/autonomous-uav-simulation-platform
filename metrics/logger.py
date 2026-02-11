import csv
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class Logger:
    @staticmethod
    def log_json(filename, data):
        filepath = LOG_DIR / filename
        with open(filepath, "a") as f:
            f.write(json.dumps(data) + "\n")

    @staticmethod
    def log_csv(filename, headers, row):
        filepath = LOG_DIR / filename
        file_exists = filepath.exists()

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

    @staticmethod
    def timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
