from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

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

        # 3. Model outputs decision function (higher is more normal, negative is anomalous)
        raw_score: float = self.model.decision_function(features)[0]

        # Convert IsolationForest [-0.5, 0.5] score into a standard [0.0 - 1.0] anomaly score
        # (Where 1.0 means highly anomalous)
        anomaly_score = float(max(0.0, min(1.0, 0.5 - raw_score)))

        # 4. Apply hardcoded Security Rules
        reasons = []
        risk_score = anomaly_score

        if anomaly_score > 0.6:
            reasons.append(f"High ML Anomaly detected (Score: {anomaly_score:.2f})")

        if not event.success and event.sensitive_operation:
            reasons.append("Failed sensitive operation (e.g. Delete, Create, Modify IAM)")
            risk_score += 0.3

        if event.region != "us-east-1" and event.sensitive_operation:
            reasons.append(f"Sensitive Operation performed outside typical region ({event.region})")
            risk_score += 0.2

        if event.timestamp.hour < 5:
            reasons.append(
                f"Activity occurred during very early hours ({event.timestamp.hour} AM)."
            )
            risk_score += 0.1

        # Cap Risk Score at 1.0
        risk_score = min(1.0, risk_score)

        # 5. Classify the final output
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 6. Return the fully-validated Pydantic result object
        return IncidentPrediction(
            event_id=event.event_id,
            anomaly_score=anomaly_score,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=tuple(reasons),
            model_version=self.model_version,
        )
