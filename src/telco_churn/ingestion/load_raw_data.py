import os
import glob
import sqlite3
import kagglehub
import pandas as pd

from telco_churn.config import DB_PATH

def load_raw_kaggle_to_sqlite():
    print("1. Downloading Telco Customer Churn dataset via kagglehub...")
    # Downloads the latest version and returns the local cache directory path
    download_dir = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print(f"Dataset cached at: {download_dir}")

    # Find the CSV file inside the downloaded directory
    csv_files = glob.glob(os.path.join(download_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {download_dir}")
    csv_path = csv_files[0]

    print(f"2. Reading raw CSV: {os.path.basename(csv_path)}...")
    df_raw = pd.read_csv(csv_path)

    # Ensure the database directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"3. Writing 'raw_telco_churn' table to SQLite ({DB_PATH})...")

    conn = sqlite3.connect(DB_PATH)

    try:
        df_raw.to_sql(
            "raw_telco_churn",
            conn,
            if_exists="replace",
            index=False
        )
    finally:
        conn.close()

    print(
        f"Success! Loaded {len(df_raw):,} rows "
        f"into {DB_PATH} (table: raw_telco_churn)."
    )

if __name__ == "__main__":
    load_raw_kaggle_to_sqlite()