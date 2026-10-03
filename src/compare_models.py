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
from sklearn.model_selection import train_test_split


DATA_PATH = "data/processed/production_data.csv"

CURRENT_MODEL_PATH = "models/churn_model.pkl"
CURRENT_PREPROCESSOR_PATH = "models/preprocessor.pkl"

RETRAINED_MODEL_PATH = "models/churn_model_retrained.pkl"
RETRAINED_PREPROCESSOR_PATH = "models/preprocessor_retrained.pkl"

COMPARISON_PATH = "models/model_comparison.json"


def load_data():
    """Load production data."""
    return pd.read_csv(DATA_PATH)


def prepare_test_data(df):
    """Create a fixed test dataset."""

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


def evaluate_model(
    model_path,
    preprocessor_path,
    X_test,
    y_test
):
    """Evaluate a model on the same test data."""

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    X_test_processed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_processed)

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions
        ),
        "recall": recall_score(
            y_test,
            predictions
        ),
        "f1_score": f1_score(
            y_test,
            predictions
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
    }

    return metrics


def save_comparison(current_metrics, retrained_metrics):
    """Save model comparison results."""

    comparison = {
        "current": current_metrics,
        "retrained": retrained_metrics
    }

    with open(COMPARISON_PATH, "w") as file:
        json.dump(
            comparison,
            file,
            indent=4
        )

    print(
        f"\nComparison saved to: {COMPARISON_PATH}"
    )


if __name__ == "__main__":

    print("Loading production data...")

    df = load_data()

    X_test, y_test = prepare_test_data(df)

    print(
        f"Evaluation samples: {len(X_test)}"
    )

    print("\nEvaluating current model...")

    current_metrics = evaluate_model(
        CURRENT_MODEL_PATH,
        CURRENT_PREPROCESSOR_PATH,
        X_test,
        y_test
    )

    print("Evaluating retrained model...")

    retrained_metrics = evaluate_model(
        RETRAINED_MODEL_PATH,
        RETRAINED_PREPROCESSOR_PATH,
        X_test,
        y_test
    )

    print("\n========== MODEL COMPARISON ==========")

    for metric in current_metrics:

        current = current_metrics[metric]
        retrained = retrained_metrics[metric]

        change = retrained - current

        print(f"\n{metric.upper()}")

        print(
            f"Current model   : {current:.4f}"
        )

        print(
            f"Retrained model : {retrained:.4f}"
        )

        print(
            f"Change          : {change:+.4f}"
        )

    save_comparison(
        current_metrics,
        retrained_metrics
    )

    # Basic comparison result
    if retrained_metrics["f1_score"] > current_metrics["f1_score"]:

        print(
            "\nInitial result   : Retrained model has "
            "higher F1 score."
        )

    else:

        print(
            "\nInitial result   : Current model has "
            "higher or equal F1 score."
        )