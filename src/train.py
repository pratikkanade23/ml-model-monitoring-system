import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


# Paths
DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")


def load_data():
    """Load the processed dataset."""
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    """Separate features and target and create preprocessing pipeline."""

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features,
            ),
            (
                "numerical",
                "passthrough",
                numerical_features,
            ),
        ]
    )

    return X, y, preprocessor


def train_model(X_train, y_train, preprocessor):
    """Preprocess training data and train XGBoost model."""

    X_train_processed = preprocessor.fit_transform(X_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train_processed, y_train)

    return model


def evaluate_model(model, preprocessor, X_test, y_test):
    """Evaluate the trained model."""

    X_test_processed = preprocessor.transform(X_test)

    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n========== MODEL PERFORMANCE ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


def save_model(model, preprocessor):
    """Save model and preprocessing pipeline."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("\n========== MODEL SAVED ==========")
    print(f"Model       : {MODEL_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining XGBoost model...")

    model = train_model(
        X_train,
        y_train,
        preprocessor,
    )

    evaluate_model(
        model,
        preprocessor,
        X_test,
        y_test,
    )

    save_model(
        model,
        preprocessor,
    )

    print("\nTraining pipeline completed successfully!")