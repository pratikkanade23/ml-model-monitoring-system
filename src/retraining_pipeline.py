import subprocess
import sys


def run_script(script_path):
    """Run a Python script and stop if it fails."""

    print("\n" + "=" * 60)
    print(f"Running: {script_path}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path} failed."
        )


def main():

    print("=" * 60)
    print("       AUTOMATIC RETRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Train candidate model
    run_script(
        "src/retrain_model.py"
    )

    # Step 2: Compare candidate with current model
    run_script(
        "src/compare_models.py"
    )

    # Step 3: Decide whether candidate should be promoted
    run_script(
        "src/promotion_decision.py"
    )

    # Step 4: Promote only if approved
    run_script(
        "src/model_registry.py"
    )

    print("\n" + "=" * 60)
    print("Retraining pipeline completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()