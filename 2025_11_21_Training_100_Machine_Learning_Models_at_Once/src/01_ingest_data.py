import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]    # project-root/
RAW_DIR = ROOT_DIR / "data" / "raw"              # data/raw/
VERSION_DIR = ROOT_DIR / "data" / "versions"     # data/versions/

RAW_DIR.mkdir(parents=True, exist_ok=True)
VERSION_DIR.mkdir(parents=True, exist_ok=True)

def validate_csv(path: Path):
    """Try loading the CSV to ensure it's readable."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV is empty.")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load CSV: {e}")

def ingest_data(source_path: str):
    """
    Copies source CSV → data/raw, validates it, writes metadata.
    """
    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    print(f"Loading and validating: {source}")
    df = validate_csv(source)

    # create versioned filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest_filename = f"online_retail_{timestamp}.csv"
    dest_path = RAW_DIR / dest_filename

    # copy file
    shutil.copy2(source, dest_path)

    # write metadata
    metadata = {
        "ingested_at": datetime.utcnow().isoformat(),
        "original_path": str(source),
        "stored_path": str(dest_path),
        "row_count": len(df),
        "columns": df.columns.tolist(),
    }

    meta_file = VERSION_DIR / f"version_{timestamp}.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Ingestion complete.")
    print(f"Stored file:     {dest_path}")
    print(f"Metadata stored: {meta_file}")


if __name__ == "__main__":
    # Example usage:
    # python ingest_data.py --source "../some_path/Online Retail.csv"
    import argparse

    parser = argparse.ArgumentParser(description="Ingest raw CSV into data/raw.")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to raw CSV file.")
    args = parser.parse_args()

    ingest_data(args.source)
