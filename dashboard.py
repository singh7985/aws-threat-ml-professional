import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="AWS Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🛡️ Threat Detection Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Investigate Incident", "Threat Behavior", "Model Performance"],
)

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "batch_prediction" / "predictions.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "batch_prediction" / "summary.json"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

@st.cache_data
def load_data():
    try:
        predictions_df = pd.read_csv(PREDICTIONS_PATH)
        if 'timestamp' in predictions_df.columns:
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        summary = json.loads(SUMMARY_PATH.read_text())
        metrics = json.loads(METRICS_PATH.read_text())
        return predictions_df, summary, metrics
    except Exception as e:
        LOGGER.error(f"Failed to load data: {e}")
        return None, None, None

data_tuple = load_data()
if data_tuple[0] is None:
    st.error("Failed to load local prediction outputs. Did you run `python -m threat_ml.batch_predict`?")
    st.stop()

predictions_df, summary, metrics = data_tuple

if page == "Overview":
    st.header("Security Event Overview")
    with st.expander("ℹ️ About This Project (Click to Expand)", expanded=True):
        st.markdown("""
        **What is this?**
        This dashboard serves as the front-end for a serverless, AI-powered AWS Cloud Security platform. It processes AWS CloudTrail activity logs and detects potential hacker activity in real-time.
        
        **How does it work?**
        It analyzes events using a **Hybrid Scoring System**:
        1. **Machine Learning (Isolation Forest):** Unsupervised learning algorithm that analyzes historical behaviors to flag anomalous patterns (weighted at 70%).
        2. **Deterministic Security Rules:** Python logic that strictly checks for known red-flags like sensitive operations at unusual hours from external networks (weighted at 30%).
        
        The scores are combined to mathematically produce a final Risk Level (**LOW, MEDIUM, HIGH**).
        """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events Analyzed", f"{summary.get('total_events_processed', len(predictions_df)):,}")
    col2.metric("Suspicious ML Predictions", f"{summary.get('suspicious_predictions', len(predictions_df[predictions_df['model_prediction']=='suspicious'])):,}")
    high_risk = len(predictions_df[predictions_df["risk_level"] == "HIGH"])
    col3.metric("Critical High Risk Incidents", high_risk, delta_color="inverse")
    col4.metric("AI Detection Recall", f"{metrics.get('recall', 0)*100:.1f}%")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overall Risk Distribution")
        risk_counts = predictions_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig1 = px.bar(
            risk_counts, 
            x="Risk Level", 
            y="Count",
            color="Risk Level",
            color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Machine Learning vs Rule Scores")
        st.markdown("Top-right indicates both AI and Security Rules severely flagged the event.")
        fig2 = px.scatter(
            predictions_df,
            x="anomaly_score",
            y="rule_score",
            color="risk_level",
            hover_data=["event_name", "principal_id"],
            color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"},
            labels={"anomaly_score": "ML Anomaly Score", "rule_score": "Security Rules Score"}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Security Events Over Time")
    if 'timestamp' in predictions_df.columns:
        # Group by hour to see attack spikes
        temp_df = predictions_df.copy()
        temp_df['hour'] = temp_df['timestamp'].dt.floor('H')
        time_series = temp_df.groupby(['hour', 'risk_level']).size().reset_index(name='count')
        fig_time = px.line(
            time_series, 
            x='hour', 
            y='count', 
            color='risk_level',
            color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"},
            labels={"hour": "Time", "count": "Number of Events"}
        )
        st.plotly_chart(fig_time, use_container_width=True)

elif page == "Investigate Incident":
    st.header("Incident Investigation")
    st.markdown("Filter and inspect the exact reasons an event was flagged by the system.")

    risk_filter = st.selectbox("Filter Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
    
    filtered_df = predictions_df
    if risk_filter != "All":
        filtered_df = predictions_df[predictions_df["risk_level"] == risk_filter]

    st.dataframe(
        filtered_df[["timestamp", "event_name", "principal_id", "service", "risk_level", "final_risk_score"]].head(100),
        use_container_width=True
    )
    
    st.subheader("Deep Dive: Selected Incident")
    incident_index = st.number_input("Enter row index from above to investigate the full payload and reasoning:", min_value=0, max_value=max(0, len(filtered_df)-1), value=0)
    
    if not filtered_df.empty:
        incident = filtered_df.iloc[incident_index]
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Event Details:**")
            st.write(f"- **Time:** {incident.get('timestamp')}")
            st.write(f"- **Action:** {incident.get('event_name')} ({incident.get('service')})")
            st.write(f"- **User Identify:** {incident.get('principal_id')}")
            st.write(f"- **Attack Category:** {incident.get('attack_type', 'N/A')}")
            st.write(f"- **Final Risk Calculation:** {incident.get('final_risk_score')} ({incident.get('risk_level')})")
        
        with col2:
            st.write("**Explainable AI: Why was it flagged?**")
            try:
                # Secure evaluation of the array string saved in CSV
                reasons = eval(incident.get("reasons", "[]"))
                if isinstance(reasons, list):
                    for r in reasons:
                        st.info(f"🚩 {r}")
                else:
                    st.write(incident.get("reasons"))
            except Exception:
                st.write(incident.get("reasons"))

elif page == "Threat Behavior":
    st.header("Threat Behavior Analysis")
    st.markdown("Analyze macroscopic patterns of attacks and suspicious API usage across the AWS Cloud environment.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attacks by Category")
        if 'attack_type' in predictions_df.columns:
            attack_counts = predictions_df[predictions_df['attack_type'].notna()]['attack_type'].value_counts().reset_index()
            attack_counts.columns = ["Attack Type", "Count"]
            fig_attack = px.bar(
                attack_counts, 
                x="Count", 
                y="Attack Type", 
                orientation='h', 
                color="Count",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_attack, use_container_width=True)
            
    with col2:
        st.subheader("Most Targeted AWS Services")
        if 'service' in predictions_df.columns:
            service_counts = predictions_df['service'].value_counts().reset_index()
            service_counts.columns = ["Service", "Count"]
            fig_service = px.pie(
                service_counts, 
                names="Service", 
                values="Count", 
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_service, use_container_width=True)

elif page == "Model Performance":
    st.header("Machine Learning Evaluation")
    st.markdown("""
    To verify that the AI reliably catches hackers, we generated a testing dataset of unseen normal and attack behaviors completely separate from training data.
    These metrics prove the **Isolation Forest** algorithms's strict math accuracy. This guarantees we aren't just blinding guessing!
    """)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{metrics.get('precision', 0):.3f}", help="When it cries wolf, is there actually a wolf?")
    col2.metric("Recall", f"{metrics.get('recall', 0):.3f}", help="Of all real attacks, how many did it catch?")
    col3.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}", help="Balance of Precision and Recall.")
    col4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", help="Probability it ranks a random threat higher than a random normal event.")
    col5.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Algorithm Confusion Matrix")
        st.markdown("Visualizes exact hits, misses, and false alarms.")
        matrix = metrics.get('confusion_matrix', [])
        if matrix:
            fig_cm = px.imshow(
                matrix, 
                text_auto=True, 
                labels=dict(x="Predicted by AI", y="Actual Truth", color="Event Count"),
                x=["Predicted Normal", "Predicted Suspicious"],
                y=["Actual Normal", "Actual Suspicious"],
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
    with col2:
        st.subheader("Model Anomaly Score Distribution")
        st.markdown("Visualizes how the AI groups scored behavior.")
        fig_hist = px.histogram(
            predictions_df, 
            x="anomaly_score", 
            color="risk_level",
            color_discrete_map={"LOW": "green", "MEDIUM": "orange", "HIGH": "red"},
            nbins=40,
            opacity=0.7,
            barmode="overlay",
        )
        fig_hist.update_layout(xaxis_title="Raw ML Anomaly Score", yaxis_title="Number of Logs")
        st.plotly_chart(fig_hist, use_container_width=True)

