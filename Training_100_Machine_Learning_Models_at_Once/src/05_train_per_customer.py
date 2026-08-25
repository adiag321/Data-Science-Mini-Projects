"""
Trains one ML model per customer.

Why do this?
------------
Some customers behave differently from the overall population. Training a separate model for each customer allows:
- Better personalization
- Lower error for customers with unique patterns
- A fair comparison against the global model

What this script does:
----------------------
1. Loads the feature dataset
2. Groups data by customer_id
3. Trains a simple model for each customer with enough data
4. Saves the model + metadata locally
5. Stores training metrics (like MSE) in a CSV file
"""
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# Directory Setup
# ---------------------------------------------------
FEATURE_DIR = Path("./data/features")
MODEL_DIR = Path("./pipeline_models/local/customer")
METRICS_PATH = MODEL_DIR / "local_model_metrics.csv"
FEATURE_PATH = FEATURE_DIR / "features.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
def load_data():
    print("Loading feature dataset...")
    df = pd.read_csv(FEATURE_PATH)

    # Required columns
    assert "customer_id" in df.columns, "customer_id column missing"
    assert "target" in df.columns, "Target column 'y' missing"

    return df

# ---------------------------------------------------
# Train a Local Model for One Customer
# ---------------------------------------------------
def train_single_customer(customer_id, df):
    """
    df : DataFrame filtered for a single customer.
    Returns:
        model, mse_score
    """
    df = df.sort_values("invoice_month").reset_index(drop = True)

    # Columns to train on
    target = "target"
    drop_cols = ["target", "customer_id", "invoice_month"]

    feature_cols = [col for col in df.columns if col not in drop_cols]

    # Not enough data → skip
    if len(df) < 10:
        return None, None, "not_enough_data"

    # Split: last 20% is test data
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train, y_train = train_df[feature_cols], train_df[target]
    X_test, y_test = test_df[feature_cols], test_df[target]

    # Train model (simple for beginners)
    model = RandomForestRegressor(
        n_estimators = 200,
        max_depth = 6,
        random_state = 42
    )
    model.fit(X_train, y_train)

    # Evaluate: MSE on test
    preds = model.predict(X_test)
    mse_score = mean_squared_error(y_test, preds)

    return model, mse_score, "trained"

# ---------------------------------------------------
# Save Model + Metadata
# ---------------------------------------------------
def save_local_model(customer_id, model, mse_score, status, feature_cols):
    """
    Creates a folder per customer: ./pipleine_models/local/customer/<customer_id>/
    Stores:
       - model.pkl
       - metadata.json
    """
    customer_folder = MODEL_DIR / str(customer_id)
    customer_folder.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = customer_folder / "model.pkl"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "customer_id": customer_id,
        "model_path": str(model_path),
        "status": status,
        "mse": mse_score,
        "features_used": feature_cols
    }

    meta_path = customer_folder / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

# ---------------------------------------------------
# Main Loop to Train All Models
# ---------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print("Starting local model training...")

    results = []

    # Train model for each unique customer
    for customer_id, group_df in df.groupby("customer_id"):
        print(f"Training model for customer {customer_id}...")

        model, mse, status = train_single_customer(customer_id, group_df)

        # Not enough data → skip saving
        if status != "trained":
            results.append([customer_id, None, status])
            print(f"Skipping customer {customer_id}: {status}")
            continue

        feature_cols = [col for col in group_df.columns if col not in ["target", "customer_id", "invoice_month"]]

        save_local_model(customer_id, model, mse, status, feature_cols)

        results.append([customer_id, mse, status])
        print(f"→ Done. MSE = {mse:.4f}")

    # Save summary metrics
    metrics_df = pd.DataFrame(results, columns=["customer_id", "mse", "status"])
    trained_count = metrics_df[metrics_df['status'] == 'trained'].shape[0]
    print(f"\nTrained models for {trained_count} customers.")
    not_trained_count = metrics_df[metrics_df['status'] != 'trained'].shape[0]
    print(f"skipped {not_trained_count} customers due to insufficient data.")
    
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\nAll local models trained!")
    print(f"Metrics saved to {METRICS_PATH}")
