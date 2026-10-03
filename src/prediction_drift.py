import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split


PRODUCTION_DATA_PATH = "data/processed/production_data.csv"
BASELINE_DATA_PATH = "data/processed/cleaned_data.csv"
REPORT_PATH = "models/prediction_drift_report.json"

PREDICTION_DRIFT_THRESHOLD = 0.05


def calculate_prediction_rate(model, preprocessor, X):
    X_processed = preprocessor.transform(X)
    predictions = model.predict(X_processed)

    return predictions.mean()


if __name__ == "__main__":

    print("Loading data and model...")

    production_df = pd.read_csv(PRODUCTION_DATA_PATH)
    baseline_df = pd.read_csv(BASELINE_DATA_PATH)

    # Separate features and target
    X = baseline_df.drop(columns=["Churn"])
    y = baseline_df["Churn"]

    # Recreate baseline test split
    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Load model and preprocessor
    import joblib

    model = joblib.load("models/churn_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")

    # Baseline predictions
    baseline_rate = calculate_prediction_rate(
        model,
        preprocessor,
        X_test
    )

    # Production predictions
    production_X = production_df.drop(columns=["Churn"])

    production_rate = calculate_prediction_rate(
        model,
        preprocessor,
        production_X
    )

    difference = abs(
        production_rate - baseline_rate
    )

    drift_detected = (
        difference >= PREDICTION_DRIFT_THRESHOLD
    )

    status = "DRIFT" if drift_detected else "NO DRIFT"

    print("\n========== PREDICTION DRIFT ==========")

    print(
        f"Baseline prediction rate   : "
        f"{baseline_rate:.2%}"
    )

    print(
        f"Production prediction rate : "
        f"{production_rate:.2%}"
    )

    print(
        f"Difference                  : "
        f"{difference:.2%}"
    )

    print(
        f"Threshold                   : "
        f"{PREDICTION_DRIFT_THRESHOLD:.2%}"
    )

    print(f"Status                      : {status}")

    # Save report
    report = {
    "baseline_prediction_rate": float(baseline_rate),
    "production_prediction_rate": float(production_rate),
    "difference": float(difference),
    "threshold": float(PREDICTION_DRIFT_THRESHOLD),
    "drift_detected": bool(drift_detected),
    "status": status
    }

    os.makedirs("models", exist_ok=True)

    with open(REPORT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print(
        f"\nPrediction drift report saved to: "
        f"{REPORT_PATH}"
    )