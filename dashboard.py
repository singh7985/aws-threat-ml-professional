import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="AWS Threat Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar styling
st.sidebar.title("🛡️ Threat Detection Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Investigate Incident", "Model Performance"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "batch_prediction" / "predictions.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "batch_prediction" / "summary.json"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


@st.cache_data
def load_data():
    try:
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
        summary = json.loads(SUMMARY_PATH.read_text())
        metrics = json.loads(METRICS_PATH.read_text())
        return predictions_df, summary, metrics
    except Exception as e:
        LOGGER.error(f"Failed to load data: {e}")
        return None, None, None


data_tuple = load_data()

if data_tuple[0] is None:
    st.error(
        "Failed to load local prediction outputs. Did you run `python -m threat_ml.batch_predict`?"
    )
    st.stop()

predictions_df, summary, metrics = data_tuple

if page == "Overview":
    st.header("Security Event Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events Analyzed", summary.get("total_events_processed", 0))
    with col2:
        st.metric("Suspicious Events", summary.get("suspicious_predictions", 0))
    with col3:
        high_risk = len(predictions_df[predictions_df["risk_level"] == "HIGH"])
        st.metric("High Risk Incidents", high_risk, delta_color="inverse")
    with col4:
        st.metric("Detection Recall", f"{metrics.get('recall', 0) * 100:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Distribution")
        risk_counts = predictions_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]  # Rename for clarity
        fig1 = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"},
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Machine Learning vs Rule Scores")
        fig2 = px.scatter(
            predictions_df,
            x="anomaly_score",
            y="rule_score",
            color="risk_level",
            hover_data=["event_name", "principal_id"],
            color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"},
            labels={"anomaly_score": "ML Anomaly Score", "rule_score": "Security Rules Score"},
        )
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Investigate Incident":
    st.header("Incident Investigation")

    risk_filter = st.selectbox("Filter Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])

    filtered_df = predictions_df
    if risk_filter != "All":
        filtered_df = predictions_df[predictions_df["risk_level"] == risk_filter]

    st.dataframe(
        filtered_df[["timestamp", "event_name", "principal_id", "service", "risk_level"]].head(50),
        use_container_width=True,
    )

    st.subheader("Select Incident to Analyze")
    # For a real app you might use ag-grid or a selectbox, but indexing works for a fast prototype
    incident_index = st.number_input(
        "Enter row index to investigate:", min_value=0, max_value=len(filtered_df) - 1, value=0
    )

    if not filtered_df.empty:
        incident = filtered_df.iloc[incident_index]
        st.json(incident.to_dict())


elif page == "Model Performance":
    st.header("Machine Learning Evaluation")

    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{metrics.get('precision', 0):.2f}")
    col2.metric("Recall", f"{metrics.get('recall', 0):.2f}")
    col3.metric("F1 Score", f"{metrics.get('f1_score', 0):.2f}")

    st.markdown("### Confusion Matrix")
    matrix = metrics.get("confusion_matrix", [])
    if matrix:
        matrix_df = pd.DataFrame(
            matrix,
            columns=["Predicted Normal", "Predicted Suspicious"],
            index=["Actual Normal", "Actual Suspicious"],
        )
        st.table(matrix_df)
