import json
import os

import pandas as pd


def check_file(path):
    assert os.path.exists(path), f"Missing file: {path}"
    print(f"✅ Found: {path}")


def main():

    print("=" * 60)
    print("          ML MODEL MONITORING SYSTEM TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Check important files
    # --------------------------------------------------

    files = [
        "data/processed/cleaned_data.csv",
        "data/processed/production_data.csv",
        "logs/drift_report.csv",
        "models/churn_model.pkl",
        "models/preprocessor.pkl",
        "models/baseline_metrics.json",
        "models/model_comparison.json",
        "models/model_registry.json",
        "models/prediction_drift_report.json",
    ]

    print("\nChecking project files...")

    for file in files:
        check_file(file)

    # --------------------------------------------------
    # Check processed data
    # --------------------------------------------------

    print("\nChecking processed data...")

    df = pd.read_csv(
        "data/processed/cleaned_data.csv"
    )

    assert len(df) > 0
    assert "Churn" in df.columns
    assert df.isnull().sum().sum() == 0

    print(f"✅ Processed dataset: {df.shape}")
    print("✅ No missing values")
    print("✅ Churn target exists")

    # --------------------------------------------------
    # Check data drift report
    # --------------------------------------------------

    print("\nChecking data drift report...")

    drift = pd.read_csv(
        "logs/drift_report.csv"
    )

    assert len(drift) > 0
    assert "drift_detected" in drift.columns

    drift_count = int(
        drift["drift_detected"].sum()
    )

    print(
        f"✅ Features monitored: {len(drift)}"
    )

    print(
        f"✅ Drifted features: {drift_count}"
    )

    # --------------------------------------------------
    # Check prediction drift
    # --------------------------------------------------

    print("\nChecking prediction drift...")

    with open(
        "models/prediction_drift_report.json",
        "r"
    ) as file:
        prediction_drift = json.load(file)

    assert "baseline_prediction_rate" in prediction_drift
    assert "production_prediction_rate" in prediction_drift
    assert "difference" in prediction_drift
    assert "drift_detected" in prediction_drift

    print(
        f"✅ Baseline rate: "
        f"{prediction_drift['baseline_prediction_rate']:.2%}"
    )

    print(
        f"✅ Production rate: "
        f"{prediction_drift['production_prediction_rate']:.2%}"
    )

    print(
        f"✅ Prediction drift: "
        f"{prediction_drift['status']}"
    )

    # --------------------------------------------------
    # Check model comparison
    # --------------------------------------------------

    print("\nChecking model comparison...")

    with open(
        "models/model_comparison.json",
        "r"
    ) as file:
        comparison = json.load(file)

    assert "current" in comparison
    assert "retrained" in comparison

    current_f1 = comparison["current"]["f1_score"]
    retrained_f1 = comparison["retrained"]["f1_score"]

    print(
        f"✅ Current model F1: {current_f1:.4f}"
    )

    print(
        f"✅ Retrained model F1: {retrained_f1:.4f}"
    )

    # --------------------------------------------------
    # Check model registry
    # --------------------------------------------------

    print("\nChecking model registry...")

    with open(
        "models/model_registry.json",
        "r"
    ) as file:
        registry = json.load(file)

    assert "current_version" in registry
    assert "history" in registry

    print(
        f"✅ Current model: "
        f"{registry['current_version']}"
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("             ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()