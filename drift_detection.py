import numpy as np
from scipy.stats import ks_2samp
import time

class DriftDetector:
    def __init__(self):
        self.reference = None

    def detect(self, new_data):
        """
        Detects drift between the reference data and new data using KS test.
        Returns:
            drift (bool): True if drift detected, False otherwise
            p_value (float): p-value from KS test
        """
        new_data = np.array(new_data)

        # Set reference if first call
        if self.reference is None:
            self.reference = new_data
            return False, 1.0

        stat, p_value = ks_2samp(self.reference, new_data)
        drift = p_value < 0.05  # significance level
        return drift, p_value

def fetch_new_data():
    """
    Simulate fetching new incoming data.
    Replace this with your real data source or API call.
    """
    return np.random.randn(100)  # 100 random numbers as example

# ----------------- Configuration -----------------
interval_minutes = 2  # time interval in minutes
# -------------------------------------------------

detector = DriftDetector()

print(f"Starting drift detection every {interval_minutes} minute(s)...\n")

while True:
    new_data = fetch_new_data()  # fetch new data

    drift, p_value = detector.detect(new_data)
    if drift:
        print(f"[ALERT] Drift detected! p-value = {p_value:.4f}")
    else:
        print(f"No drift detected. p-value = {p_value:.4f}")

    time.sleep(interval_minutes * 60)  # wait for specified minutes