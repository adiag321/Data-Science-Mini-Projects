import os
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
VERSION_DIR = ROOT_DIR / "data" / "versions"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VERSION_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Pick most recent ingested file
# ----------------------------------------------------
def get_latest_raw_file():
    files = list(RAW_DIR.glob("online_retail_*.csv"))
    if not files:
        raise FileNotFoundError("No raw files found in data/raw/. Run ingest_data.py first.")
    return max(files, key=lambda f: f.stat().st_mtime)

# ----------------------------------------------------
# Raw → monthly aggregates
# ----------------------------------------------------
def preprocess_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw invoice-level data into customer-month aggregates."""
    df = df.copy()

    # ensure proper columns exist
    required_cols = {"CustomerID", "InvoiceDate", "Quantity", "UnitPrice"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # total row price
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # group into months
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["CustomerID", "InvoiceMonth"], as_index=False)["TotalPrice"]
        .sum()
        .rename(columns={
            "CustomerID": "customer_id",
            "InvoiceMonth": "invoice_month",
            "TotalPrice": "target"
        })
    )

    monthly = monthly.sort_values(["customer_id", "invoice_month"]).reset_index(drop=True)
    return monthly

# ----------------------------------------------------
# Main function
# ----------------------------------------------------
if __name__ == "__main__":
    latest_raw = get_latest_raw_file()
    print(f"Using latest raw file: {latest_raw}")

    df = pd.read_csv(latest_raw)
    monthly = preprocess_monthly(df)

    # Versioned filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = PROCESSED_DIR / f"monthly_{timestamp}.csv"
    monthly.to_csv(out_path, index=False)

    # write metadata
    metadata = {
        "stage": "preprocess",
        "created_at": datetime.now().isoformat(),
        "source_raw_file": str(latest_raw),
        "output_file": str(out_path),
        "row_count": len(monthly),
        "columns": monthly.columns.tolist(),
    }

    meta_file = VERSION_DIR / f"preprocess_{timestamp}.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing complete.")
    print(f"Processed file:   {out_path}")
    print(f"Metadata stored:  {meta_file}")

