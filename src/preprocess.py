import pandas as pd


RAW_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_data.csv"


def load_data():
    """Load the raw Telco Customer Churn dataset."""
    df = pd.read_csv(RAW_DATA_PATH)
    return df


def preprocess_data(df):
    """Clean and preprocess the raw dataset."""

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Remove customer ID because it is an identifier
    df = df.drop(columns=["customerID"])

    # Convert target variable to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df


def save_data(df):
    """Save the processed dataset."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data()

    print(f"Original shape: {df.shape}")

    df = preprocess_data(df)

    print(f"Processed shape: {df.shape}")

    save_data(df)

    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")