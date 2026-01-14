import os
import pandas as pd
from .features import make_region_daily_ml_table

def main():
    os.makedirs("data/processed", exist_ok=True)

    events = pd.read_parquet("data/processed/events.parquet")
    events["time"] = pd.to_datetime(events["time"], utc=True)

    ml = make_region_daily_ml_table(
        events,
        feature_min_mag=2.5,
        target_mag=5.0,      # <-- CHANGED
        horizon_days=7
    )

    ml.to_parquet("data/processed/ml_table.parquet", index=False)

    print("Saved data/processed/ml_table.parquet")
    print("Rows:", len(ml), "Cols:", len(ml.columns))
    print("Positive rate (label mean):", float(ml["label"].mean()))

if __name__ == "__main__":
    main()
