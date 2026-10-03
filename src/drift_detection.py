import os

import pandas as pd
from scipy.stats import ks_2samp


TRAINING_DATA_PATH = "data/processed/cleaned_data.csv"
PRODUCTION_DATA_PATH = "data/processed/production_data.csv"

DRIFT_REPORT_PATH = "logs/drift_report.csv"


def detect_drift(training_df, production_df, column):
    """Detect distribution drift using the Kolmogorov-Smirnov test."""

    training_values = training_df[column].dropna()
    production_values = production_df[column].dropna()

    statistic, p_value = ks_2samp(
        training_values,
        production_values
    )

    drift_detected = p_value < 0.05

    return statistic, p_value, drift_detected


if __name__ == "__main__":

    print("Loading training and production data...")

    training_df = pd.read_csv(TRAINING_DATA_PATH)
    production_df = pd.read_csv(PRODUCTION_DATA_PATH)

    # Select numerical features
    numerical_features = training_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Churn is the target variable, not an input feature
    numerical_features.remove("Churn")

    results = []

    print("\n========== DRIFT DETECTION ==========")

    for column in numerical_features:

        statistic, p_value, drift_detected = detect_drift(
            training_df,
            production_df,
            column
        )

        status = "DRIFT" if drift_detected else "NO DRIFT"

        print(
            f"{column:<20} "
            f"KS={statistic:.4f} "
            f"P-value={p_value:.6f} "
            f"{status}"
        )

        results.append({
            "feature": column,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected
        })

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Create drift report
    report_df = pd.DataFrame(results)

    report_df.to_csv(
        DRIFT_REPORT_PATH,
        index=False
    )

    total_drift = report_df["drift_detected"].sum()

    print("\n========== SUMMARY ==========")
    print(f"Features checked : {len(numerical_features)}")
    print(f"Drift detected   : {total_drift}")

    print(f"\nDrift report saved to: {DRIFT_REPORT_PATH}")