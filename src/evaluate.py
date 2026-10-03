import json
import os

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

BASELINE_PATH = "models/baseline_metrics.json"


def load_data():
    """Load the processed dataset."""
    return pd.read_csv(DATA_PATH)


def evaluate_baseline():
    """Evaluate the trained baseline model."""

    df = load_data()

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Use exactly the same split as training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Load trained model and preprocessor
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Transform test data
    X_test_processed = preprocessor.transform(X_test)

    # Predictions
    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return metrics


def save_baseline_metrics(metrics):
    """Save baseline metrics as JSON."""

    os.makedirs("models", exist_ok=True)

    with open(BASELINE_PATH, "w") as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":

    print("Evaluating baseline model...")

    metrics = evaluate_baseline()

    print("\n========== BASELINE METRICS ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    save_baseline_metrics(metrics)

    print("\nBaseline metrics saved to:")
    print(BASELINE_PATH)