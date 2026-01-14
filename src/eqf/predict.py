import joblib
import pandas as pd

def main(as_of: str | None = None):
    bundle = joblib.load("data/models/model.joblib")
    model = bundle["model"]
    features = bundle["features"]
    target_mag = bundle.get("target_mag", 5.0)
    horizon_days = bundle.get("horizon_days", 7)

    df = pd.read_parquet("data/processed/ml_table.parquet")
    df["day"] = pd.to_datetime(df["day"], utc=True)

    as_of_day = df["day"].max() if as_of is None else pd.to_datetime(as_of, utc=True)
    row = df.loc[df["day"] == as_of_day]
    if row.empty:
        raise ValueError(f"No row found for day={as_of_day}")

    p = model.predict_proba(row[features])[:, 1][0]
    print("As of day:", as_of_day)
    print(f"Predicted probability of >={target_mag} quake in next {horizon_days} days (region):", float(p))

if __name__ == "__main__":
    main()
