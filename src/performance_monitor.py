import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


BASELINE_METRICS_PATH = "models/baseline_metrics.json"

MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

PRODUCTION_DATA_PATH = "data/processed/production_data.csv"


def load_baseline_metrics():
    """Load baseline model performance metrics."""

    with open(BASELINE_METRICS_PATH, "r") as file:
        return json.load(file)


def evaluate_production_model():
    """Evaluate the model on production data."""

    production_df = pd.read_csv(PRODUCTION_DATA_PATH)

    X = production_df.drop(columns=["Churn"])
    y = production_df["Churn"]

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    X_processed = preprocessor.transform(X)

    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1_score": f1_score(y, predictions),
        "roc_auc": roc_auc_score(y, probabilities),
    }

    return metrics


if __name__ == "__main__":

    print("Loading baseline metrics...")

    baseline_metrics = load_baseline_metrics()

    print("Evaluating model on production data...")

    production_metrics = evaluate_production_model()

    print("\n========== MODEL PERFORMANCE ==========")

    for metric in baseline_metrics:

        baseline = baseline_metrics[metric]
        production = production_metrics[metric]

        change = production - baseline

        print(
            f"\n{metric.upper()}"
        )

        print(
            f"Baseline    : {baseline:.4f}"
        )

        print(
            f"Production  : {production:.4f}"
        )

        print(
            f"Change      : {change:+.4f}"
        )