import json


COMPARISON_PATH = "models/model_comparison.json"

F1_IMPROVEMENT_THRESHOLD = 0.01


def load_comparison():
    """Load model comparison results."""

    with open(COMPARISON_PATH, "r") as file:
        return json.load(file)


def make_promotion_decision(comparison):
    """Decide whether the retrained model should be promoted."""

    current_f1 = comparison["current"]["f1_score"]
    retrained_f1 = comparison["retrained"]["f1_score"]

    improvement = retrained_f1 - current_f1

    if improvement >= F1_IMPROVEMENT_THRESHOLD:
        return "PROMOTE"

    return "KEEP_CURRENT"


if __name__ == "__main__":

    print("Loading model comparison...")

    comparison = load_comparison()

    current_f1 = comparison["current"]["f1_score"]
    retrained_f1 = comparison["retrained"]["f1_score"]

    improvement = retrained_f1 - current_f1

    decision = make_promotion_decision(
        comparison
    )

    print("\n========== PROMOTION DECISION ==========")

    print(
        f"Current F1       : {current_f1:.4f}"
    )

    print(
        f"Retrained F1     : {retrained_f1:.4f}"
    )

    print(
        f"F1 improvement   : {improvement:+.4f}"
    )

    print(
        f"Required         : "
        f"+{F1_IMPROVEMENT_THRESHOLD:.2f}"
    )

    print(
        f"\nDecision         : {decision}"
    )