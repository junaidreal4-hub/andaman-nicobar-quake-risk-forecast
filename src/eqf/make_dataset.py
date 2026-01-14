import os
import pandas as pd
from .config import Config

REQUIRED = ["time", "latitude", "longitude", "depth", "mag", "type"]

def main():
    cfg = Config()
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(cfg.raw_csv_path)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "latitude", "longitude", "mag"])
    df = df[df["type"].astype(str).str.lower().eq("earthquake")]  # 'type' is in USGS CSV [web:41]
    df = df.sort_values("time").reset_index(drop=True)

    df.to_parquet("data/processed/events.parquet", index=False)
    print("Saved data/processed/events.parquet")
    print("Rows:", len(df))
    print("Time range:", df["time"].min(), "→", df["time"].max())

if __name__ == "__main__":
    main()
