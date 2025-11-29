"""
Creates a unified model registry for all customer-level models stored locally.

Why this file is important:
---------------------------
When you have 10, 50, or 500 models, you need a SINGLE place that tells your prediction script:
- Which customers have a model?
- Where is the model file located?
- What features were used?
- When was it trained?
- What is the model quality (MSE)?
- Is the model valid/usable?

This registry is the source of truth.

What this script does:
----------------------
1. Scans ./pipeline_models/local/<customer_id> folders
2. Reads metadata.json for each customer
3. Verifies the model file exists
4. Creates registry.json summarizing everything
"""
import json
import os
from pathlib import Path
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
# ------------------------------------------
# Directory Setup
# ------------------------------------------
MODEL_DIR = Path("./pipeline_models/local")
CUSTOMER_DIR = MODEL_DIR / "customer"
REGISTRY_PATH = MODEL_DIR / "registry.json"
METRICS_PATH = MODEL_DIR / "local_model_metrics.csv"

# ------------------------------------------
# Load Metrics (Optional, but nice to have)
# ------------------------------------------
def load_metrics():
    """
    Loads the CSV produced in train_per_customer.py. If missing, returns empty DataFrame.
    """
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    else:
        return pd.DataFrame(columns=["customer_id", "mse", "status"])

# ------------------------------------------
# Build Registry
# ------------------------------------------
if __name__ == "__main__":
    registry = {}
    metrics_df = load_metrics()

    print("Building model registry...")

    # Loop through model folders
    for customer_dir in CUSTOMER_DIR.iterdir():

        # Skip root files like registry.json or CSV
        if not customer_dir.is_dir():
            continue

        customer_id = customer_dir.name
        metadata_path = customer_dir / "metadata.json"

        # ---- No metadata found → skip ----
        if not metadata_path.exists():
            print(f"Skipping {customer_id}: metadata.json not found")
            continue

        # ---- Load metadata ----
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # ---- Validate model path ----
        model_path = metadata.get("model_path")
        model_exists = os.path.exists(model_path)

        if not model_exists:
            print(f"Model file missing for customer {customer_id}")
            metadata["status"] = "model_missing"
        else:
            metadata["status"] = "active"

        # ---- Add MSE from metrics.csv if available ----
        row = metrics_df[metrics_df["customer_id"] == int(customer_id)]
        if len(row) > 0:
            metadata["mse"] = float(row.iloc[0]["mse"]) if pd.notna(row.iloc[0]["mse"]) else None
            metadata["training_status"] = row.iloc[0]["status"]
        else:
            metadata["mse"] = None
            metadata["training_status"] = "unknown"

        # ---- Add to registry ----
        registry[customer_id] = metadata

    # ------------------------------------------
    # Save registry.json
    # ------------------------------------------
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

    print(f"\nRegistry created successfully at {REGISTRY_PATH}")
    print(f"Total models registered: {len(registry)}")