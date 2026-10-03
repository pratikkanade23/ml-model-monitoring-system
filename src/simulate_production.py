import os

import numpy as np
import pandas as pd


SOURCE_PATH = "data/processed/cleaned_data.csv"
PRODUCTION_PATH = "data/processed/production_data.csv"


def simulate_production_data(df):
    """Create simulated production data with intentional distribution changes."""

    production_df = df.sample(
        n=2000,
        random_state=42
    ).copy()

    # Simulate a change in customer spending behavior.
    # Increase MonthlyCharges for a portion of production customers.
    np.random.seed(42)

    drift_mask = np.random.rand(len(production_df)) < 0.60

    production_df.loc[drift_mask, "MonthlyCharges"] = (
        production_df.loc[drift_mask, "MonthlyCharges"] * 1.25
    )

    # Add small random noise to make the production data realistic.
    noise = np.random.normal(
        loc=0,
        scale=2,
        size=len(production_df)
    )

    production_df["MonthlyCharges"] = (
        production_df["MonthlyCharges"] + noise
    ).clip(lower=0)

    return production_df


def save_production_data(df):
    """Save simulated production data."""

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        PRODUCTION_PATH,
        index=False
    )


if __name__ == "__main__":

    print("Loading processed training data...")

    df = pd.read_csv(SOURCE_PATH)

    print(f"Source dataset shape: {df.shape}")

    production_df = simulate_production_data(df)

    print(
        f"Production dataset shape: {production_df.shape}"
    )

    print("\nMonthlyCharges comparison:")

    print(
        f"Training mean    : "
        f"{df['MonthlyCharges'].mean():.2f}"
    )

    print(
        f"Production mean  : "
        f"{production_df['MonthlyCharges'].mean():.2f}"
    )

    print(
        f"\nTraining median  : "
        f"{df['MonthlyCharges'].median():.2f}"
    )

    print(
        f"Production median: "
        f"{production_df['MonthlyCharges'].median():.2f}"
    )

    save_production_data(production_df)

    print(
        f"\nProduction data saved to: {PRODUCTION_PATH}"
    )