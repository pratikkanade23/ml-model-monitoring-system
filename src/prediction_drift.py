import joblib
import pandas as pd


DATA_PATH = "data/processed/cleaned_data.csv"

MODEL_PATH = "models/churn_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"


def calculate_churn_rate(model, preprocessor, df):
    """Calculate the percentage of records predicted as churn."""

    X = df.drop(columns=["Churn"])

    X_processed = preprocessor.transform(X)

    predictions = model.predict(X_processed)

    churn_rate = (predictions == 1).mean()

    return churn_rate


if __name__ == "__main__":

    print("Loading baseline data...")

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Use the same test split logic as our baseline
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    baseline_df = X_test.copy()
    baseline_df["Churn"] = y_test

    # Load production data
    production_df = pd.read_csv(
        "data/processed/production_data.csv"
    )

    baseline_churn_rate = calculate_churn_rate(
        model,
        preprocessor,
        baseline_df
    )

    production_churn_rate = calculate_churn_rate(
        model,
        preprocessor,
        production_df
    )

    difference = abs(
        production_churn_rate - baseline_churn_rate
    )

    threshold = 0.05

    drift_detected = difference > threshold

    print("\n========== PREDICTION DRIFT ==========")

    print(
        f"Baseline churn rate    : "
        f"{baseline_churn_rate:.2%}"
    )

    print(
        f"Production churn rate  : "
        f"{production_churn_rate:.2%}"
    )

    print(
        f"Difference             : "
        f"{difference:.2%}"
    )

    print(
        f"Threshold              : "
        f"{threshold:.2%}"
    )

    if drift_detected:
        print("Status                 : DRIFT DETECTED")
    else:
        print("Status                 : NO DRIFT")