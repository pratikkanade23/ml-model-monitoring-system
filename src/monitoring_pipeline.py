import json
import subprocess
import sys


def run_script(script_path):
    """Run a monitoring script and return its output."""

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(
            f"{script_path} failed."
        )


def main():

    print("=" * 60)
    print("       ML MODEL MONITORING PIPELINE")
    print("=" * 60)

    print("\n[1/3] Running data drift detection...")
    run_script("src/drift_detection.py")

    print("\n[2/3] Running prediction drift detection...")
    run_script("src/prediction_drift.py")

    print("\n[3/3] Running performance monitoring...")
    run_script("src/performance_monitor.py")

    print("\n" + "=" * 60)
    print("Monitoring pipeline completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()