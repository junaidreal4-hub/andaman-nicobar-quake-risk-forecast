import pandas as pd

def make_region_daily_ml_table(
    events: pd.DataFrame,
    freq: str = "D",
    feature_min_mag: float = 2.5,
    target_mag: float = 5.0,      # <-- CHANGED: predict M>=5.0
    horizon_days: int = 7,
    lookbacks=(1, 7, 30),
) -> pd.DataFrame:
    # Use events above feature_min_mag to build features
    ev = events.loc[events["mag"] >= feature_min_mag].copy()
    ev["day"] = ev["time"].dt.floor("D")

    daily = ev.groupby("day", as_index=False).agg(
        count=("mag", "size"),
        maxmag=("mag", "max"),
        mean_depth=("depth", "mean"),
    )

    # Full daily index for region only
    all_days = pd.date_range(daily["day"].min(), daily["day"].max(), freq=freq, tz="UTC")
    df = (
        pd.DataFrame({"day": all_days})
        .merge(daily, on="day", how="left")
        .fillna({"count": 0, "maxmag": 0.0, "mean_depth": 0.0})
        .sort_values("day")
        .reset_index(drop=True)
    )

    # Rolling features (past-only)
    for w in lookbacks:
        df[f"count_{w}d"] = df["count"].rolling(w, min_periods=1).sum()
        df[f"maxmag_{w}d"] = df["maxmag"].rolling(w, min_periods=1).max()

    # Target event today (M>=target_mag)
    df["is_target_event_today"] = (df["maxmag"] >= target_mag).astype(int)

    # Label: any target event in next horizon_days
    df["label"] = (
        df["is_target_event_today"]
        .shift(-1)
        .rolling(horizon_days, min_periods=1)
        .max()
        .fillna(0)
        .astype(int)
    )

    # Drop tail where label is incomplete
    cutoff = df["day"].max() - pd.Timedelta(days=horizon_days)
    df = df[df["day"] <= cutoff].reset_index(drop=True)

    return df
