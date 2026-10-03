import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = "data/processed/production_data.csv"

MODEL_DIR = "models"

NEW_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model_retrained.pkl"
)

NEW_PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "preprocessor_retrained.pkl"
)


def load_data():
    """Load production data for retraining."""

    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    """Prepare features and target."""

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
    """Train a new XGBoost model."""

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

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

    model.fit(
        X_train_processed,
        y_train
    )

    return model


def save_model(model, preprocessor):
    """Save the retrained model."""

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        NEW_MODEL_PATH
    )

    joblib.dump(
        preprocessor,
        NEW_PREPROCESSOR_PATH
    )

    print("\n========== NEW MODEL SAVED ==========")

    print(
        f"Model       : {NEW_MODEL_PATH}"
    )

    print(
        f"Preprocessor: {NEW_PREPROCESSOR_PATH}"
    )


if __name__ == "__main__":

    print("Loading production data...")

    df = load_data()

    print(
        f"Production dataset shape: {df.shape}"
    )

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    print("\nTraining new XGBoost model...")

    model = train_model(
        X_train,
        y_train,
        preprocessor
    )

    save_model(
        model,
        preprocessor
    )

    print(
        "\nRetraining completed successfully!"
    )