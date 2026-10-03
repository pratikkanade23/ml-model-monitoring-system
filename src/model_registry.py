import json
import os
import shutil
from datetime import datetime


MODEL_DIR = "models"

CURRENT_MODEL = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

CURRENT_PREPROCESSOR = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

RETRAINED_MODEL = os.path.join(
    MODEL_DIR,
    "churn_model_retrained.pkl"
)

RETRAINED_PREPROCESSOR = os.path.join(
    MODEL_DIR,
    "preprocessor_retrained.pkl"
)

REGISTRY_PATH = os.path.join(
    MODEL_DIR,
    "model_registry.json"
)

COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "model_comparison.json"
)


def create_registry():
    """Create the model registry if it does not exist."""

    if not os.path.exists(REGISTRY_PATH):

        registry = {
            "current_version": "v1.0",
            "history": [
                {
                    "version": "v1.0",
                    "status": "production",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }

        with open(REGISTRY_PATH, "w") as file:
            json.dump(
                registry,
                file,
                indent=4
            )

        print("Model registry created.")


def load_registry():
    """Load the model registry."""

    with open(REGISTRY_PATH, "r") as file:
        return json.load(file)


def load_promotion_decision():
    """Determine whether the retrained model should be promoted."""

    with open(COMPARISON_PATH, "r") as file:
        comparison = json.load(file)

    current_f1 = comparison["current"]["f1_score"]
    retrained_f1 = comparison["retrained"]["f1_score"]

    improvement = retrained_f1 - current_f1

    threshold = 0.01

    if improvement >= threshold:
        return "PROMOTE"

    return "KEEP_CURRENT"


def promote_model():
    """Promote the retrained model to production."""

    if not os.path.exists(RETRAINED_MODEL):
        print("Retrained model not found.")
        return

    if not os.path.exists(RETRAINED_PREPROCESSOR):
        print("Retrained preprocessor not found.")
        return

    registry = load_registry()

    current_version = registry["current_version"]

    # Create backup of current model
    backup_model = os.path.join(
        MODEL_DIR,
        f"churn_model_{current_version}_backup.pkl"
    )

    backup_preprocessor = os.path.join(
        MODEL_DIR,
        f"preprocessor_{current_version}_backup.pkl"
    )

    shutil.copy2(
        CURRENT_MODEL,
        backup_model
    )

    shutil.copy2(
        CURRENT_PREPROCESSOR,
        backup_preprocessor
    )

    # Replace production model
    shutil.copy2(
        RETRAINED_MODEL,
        CURRENT_MODEL
    )

    shutil.copy2(
        RETRAINED_PREPROCESSOR,
        CURRENT_PREPROCESSOR
    )

    # Create new version
    new_version = "v2.0"

    registry["current_version"] = new_version

    registry["history"].append(
        {
            "version": new_version,
            "status": "production",
            "timestamp": datetime.now().isoformat()
        }
    )

    with open(REGISTRY_PATH, "w") as file:
        json.dump(
            registry,
            file,
            indent=4
        )

    print("\n========== MODEL PROMOTED ==========")
    print(f"Previous version : {current_version}")
    print(f"New version      : {new_version}")
    print("Previous model   : Backed up")
    print("Production model : Updated")


if __name__ == "__main__":

    print("Loading model registry...")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    create_registry()

    decision = load_promotion_decision()

    print("\n========== MODEL REGISTRY ==========")
    print(f"Promotion decision : {decision}")

    if decision == "PROMOTE":

        promote_model()

    else:

        print(
            "\nCurrent model will remain in production."
        )