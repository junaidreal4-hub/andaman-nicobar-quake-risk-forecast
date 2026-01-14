import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

FEATURES = [
    "count_1d", "count_7d", "count_30d",
    "maxmag_1d", "maxmag_7d", "maxmag_30d",
    "mean_depth",
]

def main():
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_parquet("data/processed/ml_table.parquet")
    df["day"] = pd.to_datetime(df["day"], utc=True)

    # Time-based split (no shuffling)
    test_start = pd.to_datetime("2022-01-01", utc=True)
    train = df[df["day"] < test_start].copy()
    test = df[df["day"] >= test_start].copy()

    X_train, y_train = train[FEATURES], train["label"].astype(int)
    X_test, y_test = test[FEATURES], test["label"].astype(int)

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    yhat = (p >= 0.5).astype(int)

    print("Test ROC-AUC:", roc_auc_score(y_test, p))
    print("Brier score loss:", brier_score_loss(y_test, p))  # probability quality [web:204]
    print(classification_report(y_test, yhat, digits=3))     # summary table [web:185]

    # Calibration table (10 bins) [web:210]
    prob_true, prob_pred = calibration_curve(y_test, p, n_bins=10, strategy="uniform")
    cal = pd.DataFrame({"mean_predicted_p": prob_pred, "fraction_positive": prob_true})
    cal.to_csv("data/processed/calibration_table.csv", index=False)
    print("Saved data/processed/calibration_table.csv")

    # Save test predictions for plots/report
    test_out = test[["day", "label"]].copy()
    test_out["p"] = p
    test_out.to_csv("data/processed/test_predictions.csv", index=False)
    print("Saved data/processed/test_predictions.csv")

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "test_start": "2022-01-01",
            "target_mag": 5.0,
            "horizon_days": 7
        },
        "data/models/model.joblib",
    )
    print("Saved data/models/model.joblib")

if __name__ == "__main__":
    main()
