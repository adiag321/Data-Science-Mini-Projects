"""
This step assumes:
- Preprocessing already handled missing values, types, and filtering.
- Input file is stored in ./data/processed/clean_data.csv
- Output file will be stored in ./data/features/features.csv

The goal here is to create:
1. Aggregate features
2. Behavior-based features
3. Time-based features
4. Encoded categorical features
"""
import pandas as pd
import numpy as np
from pathlib import Path        # for file paths
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
import os
os.chdir(Path(__file__).resolve().parents[1])  # set working dir to project root

# Get the most recent processed file
def get_latest_raw_file():
    files = list(PROCESSED_DIR.glob("monthly_*.csv"))
    if not files:
        raise FileNotFoundError("No raw files found in data/raw/. Run ingest_data.py first.")
    return max(files, key=lambda f: f.stat().st_mtime)

# ---------------------------------------------------
#  Where files are stored locally

PROCESSED_DIR = Path("./data/processed")
FEATURE_DIR = Path("./data/features")
OUTPUT_PATH = FEATURE_DIR / "features.csv"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PATH = get_latest_raw_file()

# ---------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------
def add_customer_aggregates(df, id_col, target_col):
    """Create simple customer-level aggregates."""
    agg = df.groupby(id_col).agg(
        avg_target=(target_col, "mean"),
        max_target=(target_col, "max"),
        min_target=(target_col, "min"),
        count_records=(target_col, "count")
    ).reset_index()

    return df.merge(agg, on=id_col, how="left")

def add_time_features(df, date_col):
    """Extract time-based features."""
    df[date_col] = pd.to_datetime(df[date_col])

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["month"] = df[date_col].dt.month

    return df

def add_interaction_features(df):
    """Create non-linear relationships."""
    if "feature1" in df.columns and "feature2" in df.columns:
        df["feat1_x_feat2"] = df["feature1"] * df["feature2"]

    return df

def encode_categoricals(df):
    """Simple one-hot encoding."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns = cat_cols, drop_first=True)
    return df

def add_statistical_features(df, id_col, target_col):
    """
    Rolling or lag features per customer.
    This is where personalization becomes powerful.
    """
    df = df.sort_values([id_col, "invoice_month"]).reset_index(drop = True)

    df["lag_1"] = df.groupby(id_col)[target_col].shift(1)
    df["rolling_3_mean"] = df.groupby(id_col)[target_col].shift(1).rolling(3).mean()

    return df

# ---------------------------------------------------
# Main Function
# ---------------------------------------------------
if __name__ == "__main__":
    print("Loading cleaned data...")
    df = pd.read_csv(INPUT_PATH)

    print("Creating time features...")
    df = add_time_features(df, date_col="invoice_month")

    print("Adding aggregate customer features...")
    df = add_customer_aggregates(df, id_col="customer_id", target_col="target")

    print("Adding interaction features...")
    df = add_interaction_features(df, )

    print("Encoding categoricals...")
    df = encode_categoricals(df)

    print("Adding lag & rolling features...")
    df = add_statistical_features(df, id_col="customer_id", target_col="target")

    df.fillna(0, inplace = True)
    print("Saving feature dataset...")
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Feature engineering complete! Saved to {OUTPUT_PATH}")
