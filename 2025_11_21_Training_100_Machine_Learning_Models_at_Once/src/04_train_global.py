"""
Trains a single global model on the complete dataset.

Assumes:
- Features are stored in ./data/features/features.csv
- Output global model will be saved in ./models/global/
- Metadata saved in ./models/global/metadata.json

This step ensures:
1. A baseline model trained on ALL customers
2. A reference point that local models can be compared against
3. Model + metadata are stored locally for reproducibility
"""
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------
# Local directories
# ---------------------------------------------------
FEATURE_DIR = Path("./data/features")
MODEL_DIR = Path("./pipeline_models/global")

INPUT_PATH = FEATURE_DIR / "features.csv"
MODEL_PATH = MODEL_DIR / "global_model.pkl"
META_PATH = MODEL_DIR / "metadata.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
def load_data():
    print("Loading feature dataset...")
    df = pd.read_csv(INPUT_PATH)
    target = "target"
    id_col = "customer_id"  # not used for training, only for debugging
    feature_cols = [col for col in df.columns if col not in [target, id_col, "invoice_month"]]
    X = df[feature_cols]
    y = df[target]

    return X, y, feature_cols

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------
def train_global_model(X, y):
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training global model...")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Evaluating global model...")
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Global Model MSE: {mse:.4f}")

    return model, mse

# ---------------------------------------------------
# Save Model + Metadata
# ---------------------------------------------------
def save_global_model(model, feature_cols, mse):
    print("Saving global model locally...")
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_path": str(MODEL_PATH),
        "type": "global_model",
        "metric": {"mse": mse},
        "features_used": feature_cols,
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Model + metadata saved in {MODEL_DIR}")


# ---------------------------------------------------
# Run Complete Pipeline
# ---------------------------------------------------
if __name__ == "__main__":
    X, y, feature_cols = load_data()
    model, mse = train_global_model(X, y)
    save_global_model(model, feature_cols, mse)
