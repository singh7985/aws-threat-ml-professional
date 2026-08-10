from __future__ import annotations

import json
import os
from typing import Any

import boto3

# The client is created on first use, not at import time. Building it at import
# requires a resolvable region and would crash the module on any machine or test
# run without AWS configuration.
_SNS_CLIENT: Any = None


def _get_sns_client() -> Any:
    global _SNS_CLIENT
    if _SNS_CLIENT is None:
        _SNS_CLIENT = boto3.client("sns")
    return _SNS_CLIENT


def publish_high_risk_alert(
    payload: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    """Publish one high-risk incident to Amazon SNS."""

    topic_arn = os.getenv("ALERTS_TOPIC_ARN")

    if not topic_arn:
        raise RuntimeError("ALERTS_TOPIC_ARN is not configured.")

    incident_id = str(payload.get("event_id", "unknown-event"))

    message = {
        "incident_id": incident_id,
        "risk_level": prediction["risk_level"],
        "final_risk_score": prediction["final_risk_score"],
        "model_prediction": prediction["model_prediction"],
        "principal_id": payload.get("principal_id", "unknown"),
        "event_name": payload.get("event_name", "unknown"),
        "service": payload.get("service", "unknown"),
        "region": payload.get("region", "unknown"),
        "reasons": prediction.get("reasons", []),
    }

    _get_sns_client().publish(
        TopicArn=topic_arn,
        Subject=(f"HIGH AWS threat detected: {incident_id}")[:100],
        Message=json.dumps(message, indent=2),
    )
