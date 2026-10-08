
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/reddit_raw/data_raw")

files = sorted(DATA_DIR.glob("*.parquet"))

print("=" * 60)
print("REDDIT DATASET INSPECTION")
print("=" * 60)

print(f"\nFiles found: {len(files)}")

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {DATA_DIR.resolve()}"
    )

# Read all downloaded daily files
df = pd.concat(
    (pd.read_parquet(file) for file in files),
    ignore_index=True
)

print(f"\nTotal records: {len(df):,}")

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nSUBREDDITS:")
if "subreddit" in df.columns:
    print(df["subreddit"].value_counts(dropna=False))

print("\nRECORD TYPES:")
if "type" in df.columns:
    print(df["type"].value_counts(dropna=False))

print("\nMISSING VALUES:")
print(df.isna().sum())

print("\nSAMPLE RECORDS:")
print(df.head(3).to_string())

print("\nDATE RANGE:")
if "created_at" in df.columns:
    dates = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    print("Earliest:", dates.min())
    print("Latest:", dates.max())

print("\nINSPECTION COMPLETE")
