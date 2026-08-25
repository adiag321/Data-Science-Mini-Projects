"""
Loads the model registry, picks the correct model for a given customer, and generates predictions.

This script covers:
1. Customer already has a trained model → use it.
2. Customer is NEW → use the global model.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ----------------------------
# Paths
# ----------------------------
MODEL_DIR = Path("./pipeline_models/local")
REGISTRY_PATH = MODEL_DIR / "registry.json"
GLOBAL_MODEL_PATH = Path("./pipeline_models/global/global_model.pkl")
PREDICTION_LOG_DIR = Path("artifacts/predictions")
PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Helper: Load registry
# ----------------------------
def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError("Registry not found. Run build_registry.py first.")
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

# ----------------------------
# Helper: Load model safely
# ----------------------------
def load_model(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")
    return load(model_path)

# ----------------------------
# Core Function: Predict
# ----------------------------
def predict(customer_id, input_data):
    """
    input_data = dict or DataFrame row
    customer_id = customer account ID
    """
    registry = load_registry()

    # Convert input_data to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    # ----------------------------
    # Case 1: Customer-specific model exists
    # ----------------------------
    if str(customer_id) in registry:
        entry = registry[str(customer_id)]
        model_path = entry["model_path"]
        status = entry.get("status", "unknown")

        if status == "active":
            model = load_model(model_path)
            preds = model.predict(input_df)
            prediction_value = float(preds[0])
            model_used = f"customer_{customer_id}_model"

            _log_prediction(customer_id, input_data, prediction_value, model_used)
            return {
                "model_used": model_used,
                "prediction": prediction_value
            }

    # ----------------------------
    # Case 2: Customer is NEW → use global model
    # ----------------------------
    if GLOBAL_MODEL_PATH.exists():
        global_model = load_model(GLOBAL_MODEL_PATH)
        preds = global_model.predict(input_df)
        prediction_value = float(preds[0])
        model_used = "global_model"

        _log_prediction(customer_id, input_data, prediction_value, model_used)
        return {
            "model_used": model_used,
            "prediction": prediction_value
        }

    # ----------------------------
    # No model available at all
    # ----------------------------
    raise Exception("No model found for prediction.")

# ----------------------------
# NEW: Logging function
# ----------------------------
def _log_prediction(customer_id, features, prediction, model_used):
    """
    Saves each prediction as a timestamped JSON file for monitoring.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_id": customer_id,
        "features": features,
        "prediction": prediction,
        "model_used": model_used
    }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"pred_{customer_id}_{timestamp}.json"
    filepath = PREDICTION_LOG_DIR / filename

    with open(filepath, "w") as f:
        json.dump(log_entry, f, indent=4)

# ----------------------------
# Demo (Real-world examples)
# ----------------------------
if __name__ == "__main__":

    # Example 1: Customer with an existing model
    example_existing_customer = {
        "day_of_week": 2,
        "week_of_year": 45,
        "month": 11,
        "avg_target": 410,
        "max_target": 700,
        "min_target": 100,
        "count_records": 240,
        "lag_1": 400,
        "rolling_3_mean": 390
    }
    try:
        result = predict(customer_id=12437, input_data=example_existing_customer)
        print("\nPrediction for EXISTING customer (12437):")
        print(result)
    except Exception as e:
        print("Error:", e)

    # Example 2: New customer → fallback to global model
    example_new_customer = {
        "day_of_week": 5,
        "week_of_year": 45,
        "month": 11,
        "avg_target": 130,
        "max_target": 230,
        "min_target": 10,
        "count_records": 25,
        "lag_1": 118,
        "rolling_3_mean": 110
    }
    try:
        result = predict(customer_id=9999, input_data=example_new_customer)
        print("\nPrediction for NEW customer (9999):")
        print(result)
    except Exception as e:
        print("Error:", e)
