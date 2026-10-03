import json
import os

import pandas as pd
import streamlit as st


DRIFT_REPORT_PATH = "logs/drift_report.csv"
PREDICTION_DRIFT_PATH = "models/prediction_drift_report.json"
MODEL_COMPARISON_PATH = "models/model_comparison.json"
MODEL_REGISTRY_PATH = "models/model_registry.json"
PRODUCTION_DATA_PATH = "data/processed/production_data.csv"


st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="📊",
    layout="wide"
)


st.title("📊 ML Model Monitoring Dashboard")

st.markdown(
    "Monitor data drift, prediction drift, "
    "model performance, and retraining status."
)


def load_json(path):
    if not os.path.exists(path):
        return None

    with open(path, "r") as file:
        return json.load(file)


def load_drift_report():
    if not os.path.exists(DRIFT_REPORT_PATH):
        return None

    return pd.read_csv(DRIFT_REPORT_PATH)


def load_production_data():
    if not os.path.exists(PRODUCTION_DATA_PATH):
        return None

    return pd.read_csv(PRODUCTION_DATA_PATH)


# Load reports
drift_report = load_drift_report()
prediction_drift = load_json(PREDICTION_DRIFT_PATH)
comparison = load_json(MODEL_COMPARISON_PATH)
registry = load_json(MODEL_REGISTRY_PATH)
production_data = load_production_data()


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.header("Model Overview")

col1, col2, col3, col4 = st.columns(4)


if registry:
    current_version = registry.get(
        "current_version",
        "Unknown"
    )
else:
    current_version = "Unknown"


with col1:
    st.metric(
        "Model Version",
        current_version
    )


with col2:
    if production_data is not None:
        st.metric(
            "Production Records",
            len(production_data)
        )
    else:
        st.metric(
            "Production Records",
            "N/A"
        )


with col3:
    if drift_report is not None:

        drift_count = int(
            drift_report["drift_detected"].sum()
        )

        st.metric(
            "Drifted Features",
            drift_count
        )

    else:
        st.metric(
            "Drifted Features",
            "N/A"
        )


with col4:

    if comparison:

        current_f1 = comparison["current"]["f1_score"]
        retrained_f1 = comparison["retrained"]["f1_score"]

        if retrained_f1 > current_f1:
            status = "Candidate Better"
        else:
            status = "Current Model"

    else:
        status = "Unknown"

    st.metric(
        "Model Status",
        status
    )


# ============================================================
# DATA DRIFT
# ============================================================

st.header("🔍 Data Drift")


if drift_report is not None:

    display_report = drift_report.copy()

    display_report["status"] = (
        display_report["drift_detected"]
        .apply(
            lambda x:
            "⚠️ DRIFT"
            if x
            else "✅ NO DRIFT"
        )
    )

    st.dataframe(
        display_report,
        width="stretch"
    )

else:

    st.warning(
        "Drift report not found."
    )


# ============================================================
# PREDICTION DRIFT
# ============================================================

st.header("📡 Prediction Drift")


if prediction_drift:

    baseline_rate = (
        prediction_drift[
            "baseline_prediction_rate"
        ]
    )

    production_rate = (
        prediction_drift[
            "production_prediction_rate"
        ]
    )

    difference = (
        prediction_drift[
            "difference"
        ]
    )

    threshold = (
        prediction_drift[
            "threshold"
        ]
    )

    drift_detected = (
        prediction_drift[
            "drift_detected"
        ]
    )

    status = (
        "⚠️ DRIFT DETECTED"
        if drift_detected
        else "✅ NO DRIFT"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Baseline Prediction Rate",
            f"{baseline_rate:.2%}"
        )


    with col2:

        st.metric(
            "Production Prediction Rate",
            f"{production_rate:.2%}"
        )


    with col3:

        st.metric(
            "Difference",
            f"{difference:.2%}"
        )


    with col4:

        st.metric(
            "Threshold",
            f"{threshold:.2%}"
        )


    if drift_detected:

        st.error(status)

    else:

        st.success(status)


else:

    st.warning(
        "Prediction drift report not found."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📈 Model Performance")


if comparison:

    current_metrics = comparison["current"]


    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        st.metric(
            "Accuracy",
            f"{current_metrics['accuracy']:.2%}"
        )


    with col2:

        st.metric(
            "Precision",
            f"{current_metrics['precision']:.2%}"
        )


    with col3:

        st.metric(
            "Recall",
            f"{current_metrics['recall']:.2%}"
        )


    with col4:

        st.metric(
            "F1 Score",
            f"{current_metrics['f1_score']:.2%}"
        )


    with col5:

        st.metric(
            "ROC-AUC",
            f"{current_metrics['roc_auc']:.2%}"
        )


else:

    st.warning(
        "Model comparison data not found."
    )


# ============================================================
# CURRENT VS RETRAINED MODEL
# ============================================================

st.header("🏆 Current vs Retrained Model")


if comparison:

    comparison_table = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],

        "Current Model": [
            comparison["current"]["accuracy"],
            comparison["current"]["precision"],
            comparison["current"]["recall"],
            comparison["current"]["f1_score"],
            comparison["current"]["roc_auc"]
        ],

        "Retrained Model": [
            comparison["retrained"]["accuracy"],
            comparison["retrained"]["precision"],
            comparison["retrained"]["recall"],
            comparison["retrained"]["f1_score"],
            comparison["retrained"]["roc_auc"]
        ]
    })


    st.dataframe(
        comparison_table,
        width="stretch"
    )


else:

    st.warning(
        "Model comparison data not found."
    )


# ============================================================
# MODEL REGISTRY
# ============================================================

st.header("📦 Model Registry")


if registry:

    history = registry.get(
        "history",
        []
    )

    registry_table = pd.DataFrame(
        history
    )

    st.dataframe(
        registry_table,
        width="stretch"
    )

else:

    st.warning(
        "Model registry not found."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Monitoring System"
)

st.sidebar.markdown(
    """
    **Components**

    - Data Drift Detection
    - Prediction Drift
    - Performance Monitoring
    - Automatic Retraining
    - Model Comparison
    - Model Registry
    """
)

st.sidebar.success(
    "Monitoring system loaded"
)