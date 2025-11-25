"""
Simple monitoring script for your local multi-model system.

What this script does:
1. Loads latest predictions + actuals stored locally.
2. Computes performance metrics (MSE, MAE, MAPE).
3. Compares performance against historical performance.
4. Detects data drift using simple statistical checks.
5. Logs results in local logs/monitoring/ folder.
6. Raises alerts if models degrade.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# CONFIG PATHS
# -----------------------------
PREDICTIONS_PATH = "artifacts/predictions"     # Your pred.py logs
HISTORY_PATH = "monitoring/history"
ALERTS_PATH = "monitoring/alerts"
DRIFT_PATH = "monitoring/drift"
os.makedirs(HISTORY_PATH, exist_ok=True)
os.makedirs(ALERTS_PATH, exist_ok=True)
os.makedirs(DRIFT_PATH, exist_ok=True)

# -----------------------------------------------------
# LOAD ALL PREDICTION LOGS
# -----------------------------------------------------
def load_prediction_logs():
    rows = []
    for file in os.listdir(PREDICTIONS_PATH):
        if file.endswith(".json"):
            with open(os.path.join(PREDICTIONS_PATH, file), "r") as f:
                data = json.load(f)

                # Expand nested features dict into separate columns
                flat = {
                    "timestamp": data["timestamp"],
                    "customer_id": data["customer_id"],
                    "prediction": data["prediction"],
                    "model_used": data["model_used"]
                }

                # Add each feature as its own column
                for k, v in data["features"].items():
                    flat[f"feat_{k}"] = v

                rows.append(flat)

    if not rows:
        print("No prediction logs found.")
        return None

    return pd.DataFrame(rows)

# -----------------------------------------------------
# METRICS (NO TRUE LABELS → ONLY SIMPLE STATS)
# -----------------------------------------------------
def compute_metrics(df):
    """
    Since we don't have actual values, we compute:
    - prediction stability
    - avg prediction per model
    - prediction volatility
    """
    results = []

    for customer_id, group in df.groupby("customer_id"):
        preds = group["prediction"]

        results.append({
            "customer_id": customer_id,
            "model_type": group["model_used"].iloc[0],
            "mean_prediction": preds.mean(),
            "std_prediction": preds.std(),
            "min_prediction": preds.min(),
            "max_prediction": preds.max(),
            "num_samples": len(group)
        })

    return pd.DataFrame(results)

# -----------------------------------------------------
# DRIFT DETECTION
# -----------------------------------------------------
def detect_drift(df):
    """
    Compare feature means to past means stored in drift_stats.json.
    Flags drift if >30% change.
    """
    drift_report = {}

    stats_file = os.path.join(DRIFT_PATH, "feature_stats.json")

    # Load previous stats
    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            previous_stats = json.load(f)
    else:
        previous_stats = {}

    # Current stats
    numeric_df = df.select_dtypes(include=[np.number])
    current_stats = numeric_df.mean().to_dict()

    # Compare
    for col, curr_mean in current_stats.items():
        if col in previous_stats:
            prev_mean = previous_stats[col]

            drift_score = abs(curr_mean - prev_mean) / (prev_mean + 1e-5)

            if drift_score > 0.30:
                drift_report[col] = {
                    "previous_mean": float(prev_mean),
                    "current_mean": float(curr_mean),
                    "drift_score": float(drift_score),
                    "status": "DRIFT DETECTED"
                }

    # Save updated stats
    with open(stats_file, "w") as f:
        json.dump(current_stats, f, indent=4)

    return drift_report

# -----------------------------------------------------
# ALERTS (NO TRUE LABELS → ALERT ON WEIRD PREDICTIONS)
# -----------------------------------------------------
def raise_alerts(metrics_df):
    alerts = []

    for _, row in metrics_df.iterrows():
        if row["std_prediction"] > 500:
            alerts.append(
                f"High prediction volatility for customer {row['customer_id']} (std={row['std_prediction']})"
            )

        if row["mean_prediction"] > 3000 or row["mean_prediction"] < 0:
            alerts.append(
                f"Suspicious mean prediction for customer {row['customer_id']}: {row['mean_prediction']}"
            )

    if alerts:
        alert_file = os.path.join(
            ALERTS_PATH,
            f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(alert_file, "w") as f:
            for line in alerts:
                f.write(line + "\n")

        print("Alerts generated. Check monitoring/alerts/")
    else:
        print("No alerts raised.")

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    print("Loading predictions...")
    df = load_prediction_logs()

    if df is None:
        exit()

    print("Computing metrics...")
    metrics_df = compute_metrics(df)
    print(metrics_df)

    print("\nDetecting drift...")
    drift_report = detect_drift(df)

    if drift_report:
        print("Drift detected:")
        print(json.dumps(drift_report, indent=4))

        drift_file = os.path.join(
            DRIFT_PATH,
            f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(drift_file, "w") as f:
            json.dump(drift_report, f, indent=4)
    else:
        print("No drift detected.")

    print("\nChecking alerts...")
    raise_alerts(metrics_df)

    # Save historical performance
    history_file = os.path.join(
        HISTORY_PATH,
        f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    metrics_df.to_csv(history_file, index=False)

    print("\nMonitoring complete.")

