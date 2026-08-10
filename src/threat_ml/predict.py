from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from threat_ml.predict_event import (
    calculate_rule_score,
    convert_decision_to_anomaly_score,
    determine_risk_level,
    is_anomalous,
    read_anomaly_threshold,
)
from threat_ml.schemas import IncidentPrediction, SecurityEvent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
MANIFEST_PATH = PROJECT_ROOT / "models" / "feature_manifest.json"


class ThreatPredictor:
    """Loads the trained ML model and scores incoming AWS events."""

    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
            raise FileNotFoundError("Model or manifest not found. Run train_model.py first.")

        self.model: IsolationForest = joblib.load(MODEL_PATH)
        self.manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text("utf-8"))
        self.feature_columns: list[str] = self.manifest["feature_columns"]
        self.model_version: str = self.manifest["model_version"]
        # Same calibrated cutoff the CLI and the Lambda use.
        self.anomaly_threshold: float = read_anomaly_threshold(self.manifest)

    def _extract_features(self, event: SecurityEvent) -> pd.DataFrame:
        """Converts raw AWS event JSON into ML numeric structure."""

        # In a real environment, you look up historical context (calls_last_5m) from
        # Redis/DynamoDB. For simplicity in the prediction demo, we stub normal values
        # and extract what we can from the pure event.
        extracted = {
            "hour_of_day": event.timestamp.hour,
            "is_weekend": int(event.timestamp.weekday() >= 5),
            "failed_call": int(not event.success),
            "sensitive_operation": int(event.sensitive_operation),
            "unusual_hour": int(event.timestamp.hour < 6 or event.timestamp.hour > 19),
            "external_ip": 1,  # Simplified (assuming 1 if outside VPC)
            "outside_home_region": int(event.region != "us-east-1"),  # Assuming us-east-1 is home
            # The following are historical rolling counters, usually stored externally.
            # We mock normal behavior here so the ML can judge the core event properties.
            "calls_last_5m": 5,
            "failed_calls_last_5m": 0 if event.success else 1,
            "unique_services_last_1h": 2,
            "iam_calls_last_1h": 1,
            "api_risk_score": 0.5,
        }

        # Select only the correct columns in the exact trained order
        features = {col: extracted.get(col, 0) for col in self.feature_columns}

        # Convert dictionary to DataFrame with exactly one row
        return pd.DataFrame([features])

    def predict(self, raw_event_payload: dict[str, Any]) -> IncidentPrediction:
        """Takes an raw AWS JSON event and returns a Risk Level."""

        # 1. Validation: Ensure the pure JSON matches our strict AWS SecurityEvent schema
        event = SecurityEvent(**raw_event_payload)

        # 2. Extract ML features
        features = self._extract_features(event)

        # 3. Score through the shared pipeline.
        #
        # This deliberately delegates to threat_ml.predict_event rather than
        # implementing its own maths. When this class carried a private scoring
        # formula it disagreed with both the CLI and the Lambda for the same
        # event -- three different answers from one system.
        decision_value = float(self.model.decision_function(features)[0])

        anomaly_score = convert_decision_to_anomaly_score(
            decision_value,
            self.anomaly_threshold,
        )

        payload = {column: float(features.iloc[0][column]) for column in self.feature_columns}
        rule_score, reasons = calculate_rule_score(payload)

        risk_score = min(1.0, 0.70 * anomaly_score + 0.30 * rule_score)

        if is_anomalous(decision_value, self.anomaly_threshold):
            reasons.insert(0, "The machine-learning model marked the behavior as unusual.")

        if not reasons:
            reasons.append("No major security-warning rules were triggered.")

        # 4. Classify the final output
        risk_level = determine_risk_level(risk_score)

        # 5. Return the fully-validated Pydantic result object
        return IncidentPrediction(
            event_id=event.event_id,
            anomaly_score=anomaly_score,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=tuple(reasons),
            model_version=self.model_version,
        )
