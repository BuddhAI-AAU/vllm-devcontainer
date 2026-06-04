import csv
import os
import time
from threading import Lock

LOG_PATH = "performance_log.csv"
_lock = Lock()

# Ensure header exists
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "user_id",
            "stage",
            "duration_sec",
            "extra"
        ])

def log_perf(user_id: str, stage: str, duration: float, extra: str = ""):
    with _lock:
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                user_id,
                stage,
                f"{duration:.6f}",
                extra
            ])
