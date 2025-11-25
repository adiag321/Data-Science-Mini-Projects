"""
Example: compare a single global model vs many per-customer models
Dataset: Online Retail (monthly aggregation by CustomerID)
"""
import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

# ---------------------------
# Configuration
# ---------------------------
# set working dir to script location
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)  
DATA_PATH = "data/Online Retail.csv"  # put the Kaggle CSV here
OUTPUT_DIR = Path("models")
N_JOBS = 6  # parallel workers for per-customer training
MIN_MONTHS_FOR_CUSTOMER_MODEL = 12  # only build customer model if they have >= this many months
RANDOM_SEED = 42

# create output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1. Load dataset
# ---------------------------
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Download Online Retail CSV and place at this path.")
    df = pd.read_csv(path)
    print(f"shape of data: {df.shape}")
    return df

raw = load_data(DATA_PATH)

# ---------------------------
# Helpers: feature engineering
# ---------------------------
def prepare_monthly_aggregate(df):
    """
    Given raw invoice-level df with columns: CustomerID, InvoiceDate, Quantity, UnitPrice
    Return customer-month level df with 'customer_id', 'invoice_month' (period), 'total_spend'
    """
    df = df.copy()
    # ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # total price per row
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    # month period
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
    monthly = (df
               .groupby(['CustomerID', 'InvoiceMonth'], as_index=False)
               ['TotalPrice']
               .sum()
               .rename(columns={'CustomerID': 'customer_id', 'InvoiceMonth': 'invoice_month', 'TotalPrice': 'target'}))
    # sort
    monthly = monthly.sort_values(['customer_id', 'invoice_month']).reset_index(drop=True)
    return monthly

monthly = prepare_monthly_aggregate(raw)
print("Customer-month rows:", len(monthly))


def add_time_series_features(monthly_df, lags=(1, 2, 3), rolling_windows=(3, 6)):
    """
    For each customer, add lag and rolling mean features.
    monthly_df expected to have: customer_id, invoice_month, target
    """
    frames = []
    for cust, group in monthly_df.groupby('customer_id'):
        g = group.copy().set_index('invoice_month').asfreq('MS')  # monthly start freq; fills missing months
        # fill missing target months with 0 (or could use forward/backfill depending on domain)
        g['target'] = g['target'].fillna(0.0)
        # lags
        for lag in lags:
            g[f'lag_{lag}'] = g['target'].shift(lag)
        # rolling means
        for w in rolling_windows:
            g[f'roll_mean_{w}'] = g['target'].shift(1).rolling(window=w, min_periods=1).mean()
        g['month'] = g.index.month
        g['customer_id'] = cust
        frames.append(g.reset_index())
    feat_df = pd.concat(frames, ignore_index=True).sort_values(['customer_id', 'invoice_month'])
    # drop rows where lag_1 is NaN (not enough history for features)
    feat_df = feat_df.dropna(subset=['lag_1']).reset_index(drop=True)
    return feat_df

feat = add_time_series_features(monthly)
print("Feature rows after adding time series features:", len(feat))


# ---------------------------
# Train global model
# ---------------------------
def train_global_model(feature_df, save_path=OUTPUT_DIR / "global_model"):
    os.makedirs(save_path, exist_ok=True)
    df = feature_df.copy()
    # encode customer id (or drop if you prefer purely pooled)
    le = LabelEncoder()
    df['cust_enc'] = le.fit_transform(df['customer_id'].astype(str))
    # features & target
    feature_cols = [c for c in df.columns if c not in ('customer_id', 'invoice_month', 'target')]
    X = df[feature_cols]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=RANDOM_SEED)

    model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    # Save
    model.booster_.save_model(str(save_path / "model.txt"))
    with open(save_path / "metrics.json", "w") as f:
        json.dump({"mse": mse}, f, indent=4)
    print(f"[GLOBAL] MSE: {mse:.4f}")
    return {"mse": mse, "model_path": str(save_path / "model.txt")}

global_results = train_global_model(feat, save_path=OUTPUT_DIR / "global_model")


# ---------------------------
# Train per-customer models (parallel)
# ---------------------------
def train_customer_model(customer_id, customer_df, out_root=OUTPUT_DIR):
    """
    customer_df is a small dataframe for this customer with features and target
    """
    res = {"customer_id": customer_id, "trained": False, "mse": None, "reason": None}
    try:
        # require enough rows
        if len(customer_df) < MIN_MONTHS_FOR_CUSTOMER_MODEL:
            res.update({"reason": f"insufficient_months({len(customer_df)})"})
            return res

        # sort by date and split last 20% as test (time-aware)
        customer_df = customer_df.sort_values('invoice_month')
        # features
        feature_cols = [c for c in customer_df.columns if c not in ('customer_id', 'invoice_month', 'target')]
        X = customer_df[feature_cols]
        y = customer_df['target']
        # time-split: last 20% as test
        n_test = max(1, int(0.2 * len(customer_df)))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]

        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        # save artifacts
        cust_dir = out_root / str(customer_id)
        os.makedirs(cust_dir, exist_ok=True)
        model.booster_.save_model(str(cust_dir / "model.txt"))
        with open(cust_dir / "metrics.json", "w") as f:
            json.dump({"mse": mse, "trained_at": datetime.utcnow().isoformat()}, f, indent=4)

        res.update({"trained": True, "mse": mse, "model_path": str(cust_dir / "model.txt")})
    except Exception as e:
        res.update({"reason": f"error:{str(e)}"})
    return res

grouped = {cust: g.reset_index(drop=True) for cust, g in feat.groupby('customer_id')}
candidate_customers = [c for c, g in grouped.items() if len(g) >= MIN_MONTHS_FOR_CUSTOMER_MODEL]
print(f"Eligible customers for local models (>= {MIN_MONTHS_FOR_CUSTOMER_MODEL} months): {len(candidate_customers)}")
# If there are many customers, limit to 100 for the experiment (or run all)
LIMIT = 100
selected = candidate_customers[:LIMIT]
print(f"Training per-customer models for {len(selected)} customers (limit={LIMIT}) ...")


results = Parallel(n_jobs=N_JOBS)(
        delayed(train_customer_model)(cust, grouped[cust], OUTPUT_DIR / "customers")
        for cust in tqdm(selected)
    )

trained = [r for r in results if r.get("trained")]
untrained = [r for r in results if not r.get("trained")]
print(f"Trained customer models: {len(trained)}")
print(f"Skipped / failed: {len(untrained)}")
if trained:
    mses = [r['mse'] for r in trained if r.get('mse') is not None]
    print(f"Per-customer MSE median: {np.median(mses):.4f}, mean: {np.mean(mses):.4f}")


# ---------------------------
# Registry builder
# ---------------------------
def build_registry(out_root=OUTPUT_DIR):
    registry = {}
    for cust_dir in os.listdir(out_root):
        path = out_root / cust_dir
        if not path.is_dir():
            continue
        metrics_file = path / "metrics.json"
        model_file = path / "model.txt"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            registry[cust_dir] = {"model_path": str(model_file), "mse": metrics.get("mse"), "trained_at": metrics.get("trained_at")}
    return registry

registry = build_registry(OUTPUT_DIR / "customers")
print("Registry sample:", list(registry.items())[:3])


# # ---------------------------
# # Main experiment flow
# # ---------------------------
# # 1) Load raw data
# raw = load_data(DATA_PATH)

# # 2) Preprocess -> monthly aggregates
# monthly = prepare_monthly_aggregate(raw)
# print("Customer-month rows:", len(monthly))

# # 3) Feature engineering (lags/rolling)
# feat = add_time_series_features(monthly)
# print("Feature rows after adding time series features:", len(feat))

# # 4) Global model training
# global_results = train_global_model(feat, save_path=OUTPUT_DIR / "global_model")

# # 5) Prepare per-customer datasets (only those with enough months)
# grouped = {cust: g.reset_index(drop=True) for cust, g in feat.groupby('customer_id')}
# candidate_customers = [c for c, g in grouped.items() if len(g) >= MIN_MONTHS_FOR_CUSTOMER_MODEL]
# print(f"Eligible customers for local models (>= {MIN_MONTHS_FOR_CUSTOMER_MODEL} months): {len(candidate_customers)}")

# # If there are many customers, limit to 100 for the experiment (or run all)
# LIMIT = 100
# selected = candidate_customers[:LIMIT]
# print(f"Training per-customer models for {len(selected)} customers (limit={LIMIT}) ...")

# # 6) Parallel training
# results = Parallel(n_jobs=N_JOBS)(
#     delayed(train_customer_model)(cust, grouped[cust], OUTPUT_DIR / "customers")
#     for cust in tqdm(selected)
# )

# # 7) Compare metrics
# trained = [r for r in results if r.get("trained")]
# untrained = [r for r in results if not r.get("trained")]
# print(f"Trained customer models: {len(trained)}")
# print(f"Skipped / failed: {len(untrained)}")
# if trained:
#     mses = [r['mse'] for r in trained if r.get('mse') is not None]
#     print(f"Per-customer MSE median: {np.median(mses):.4f}, mean: {np.mean(mses):.4f}")

# # 8) Registry
# registry = build_registry(OUTPUT_DIR / "customers")
# print("Registry sample:", list(registry.items())[:3])

# 9) Quick comparison: fraction of customers beating global model on their holdouts
# For a proper comparison we'd evaluate the global model on each customer's holdout periods. Quick heuristic:
global_mse = global_results['mse']
better_count = sum(1 for r in trained if r['mse'] < global_mse)
worse_count = sum(1 for r in trained if r['mse'] > global_mse)
print(f"Out of {len(trained)} customer models: {better_count} better than global, {worse_count} worse (global_mse={global_mse:.4f})")

# save summary
with open(OUTPUT_DIR / "experiment_summary.json", "w") as f:
    json.dump({
        "global": global_results,
        "customer_summary": {
            "trained_count": len(trained),
            "trained_mse_mean": float(np.mean(mses)) if trained else None,
            "trained_mse_median": float(np.median(mses)) if trained else None,
            "better_than_global": int(better_count),
            "worse_than_global": int(worse_count)
        }
    }, f, indent=4)

print("Experiment complete. Artifacts saved to:", OUTPUT_DIR)

