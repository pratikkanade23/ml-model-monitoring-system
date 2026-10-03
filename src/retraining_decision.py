import json


BASELINE_METRICS_PATH = "models/baseline_metrics.json"


# Maximum acceptable performance drop
PERFORMANCE_THRESHOLD = 0.05


def load_baseline_metrics():
    """Load baseline model metrics."""

    with open(BASELINE_METRICS_PATH, "r") as file:
        return json.load(file)


def check_performance_degradation(
    baseline_metrics,
    production_metrics
):
    """Check whether model performance has degraded."""

    degraded_metrics = []

    for metric in baseline_metrics:

        baseline = baseline_metrics[metric]
        production = production_metrics[metric]

        change = production - baseline

        # Performance degradation
        if change < -PERFORMANCE_THRESHOLD:
            degraded_metrics.append(metric)

    return degraded_metrics


def make_retraining_decision(
    drift_detected,
    prediction_drift_detected,
    degraded_metrics
):
    """Decide whether model retraining is required."""

    if degraded_metrics:
        return "RETRAIN"

    if drift_detected and prediction_drift_detected:
        return "RETRAIN"

    return "NO RETRAIN"


if __name__ == "__main__":

    print("Loading baseline metrics...")

    baseline_metrics = load_baseline_metrics()

    # Example production metrics.
    # These will later come directly from
    # performance_monitor.py.
    production_metrics = baseline_metrics.copy()

    degraded_metrics = check_performance_degradation(
        baseline_metrics,
        production_metrics
    )

    # Current monitoring results
    data_drift_detected = True
    prediction_drift_detected = False

    decision = make_retraining_decision(
        data_drift_detected,
        prediction_drift_detected,
        degraded_metrics
    )

    print("\n========== RETRAINING DECISION ==========")

    print(
        f"Data drift detected       : "
        f"{data_drift_detected}"
    )

    print(
        f"Prediction drift detected : "
        f"{prediction_drift_detected}"
    )

    print(
        f"Degraded metrics          : "
        f"{degraded_metrics}"
    )

    print(
        f"\nDecision                  : "
        f"{decision}"
    )