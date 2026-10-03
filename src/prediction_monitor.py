import joblib
import pandas as pd


PRODUCTION_DATA_PATH = "data/processed/production_data.csv"

MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    return model, preprocessor


def generate_predictions(model, preprocessor, df):
    """Generate model predictions for a dataset."""

    X = df.drop(columns=["Churn"])

    X_processed = preprocessor.transform(X)

    predictions = model.predict(X_processed)

    return predictions


def analyze_predictions(predictions):
    """Analyze the distribution of model predictions."""

    total = len(predictions)

    predicted_no_churn = (predictions == 0).sum()
    predicted_churn = (predictions == 1).sum()

    churn_rate = predicted_churn / total

    return {
        "total_predictions": total,
        "predicted_no_churn": predicted_no_churn,
        "predicted_churn": predicted_churn,
        "predicted_churn_rate": churn_rate,
    }


if __name__ == "__main__":

    print("Loading production data...")

    production_df = pd.read_csv(PRODUCTION_DATA_PATH)

    print(f"Production samples: {len(production_df)}")

    print("\nLoading trained model...")

    model, preprocessor = load_model()

    print("Generating predictions...")

    predictions = generate_predictions(
        model,
        preprocessor,
        production_df
    )

    results = analyze_predictions(predictions)

    print("\n========== PREDICTION MONITORING ==========")

    print(
        f"Total predictions     : "
        f"{results['total_predictions']}"
    )

    print(
        f"Predicted No Churn    : "
        f"{results['predicted_no_churn']}"
    )

    print(
        f"Predicted Churn       : "
        f"{results['predicted_churn']}"
    )

    print(
        f"Predicted Churn Rate  : "
        f"{results['predicted_churn_rate']:.2%}"
    )