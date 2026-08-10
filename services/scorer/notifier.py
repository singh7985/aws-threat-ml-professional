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
    event: dict[str, Any],
    prediction: dict[str, Any],
) -> None:
    """Publish one high-risk incident to Amazon SNS.

    Takes the envelope's event metadata, not the feature vector. Reading
    identity off a feature vector meant every alert said "unknown" for
    principal, event name, service and Region -- the four things a responder
    needs first.
    """

    topic_arn = os.getenv("ALERTS_TOPIC_ARN")

    if not topic_arn:
        raise RuntimeError("ALERTS_TOPIC_ARN is not configured.")

    incident_id = str(event.get("event_id", "unknown-event"))

    message = {
        "incident_id": incident_id,
        "risk_level": prediction["risk_level"],
        "final_risk_score": prediction["final_risk_score"],
        "model_prediction": prediction["model_prediction"],
        "principal_id": event.get("principal_id", "unknown"),
        "source_ip": event.get("source_ip", "unknown"),
        "event_name": event.get("event_name", "unknown"),
        "service": event.get("service", "unknown"),
        "region": event.get("region", "unknown"),
        "event_time": event.get("timestamp", "unknown"),
        "reasons": prediction.get("reasons", []),
    }

    _get_sns_client().publish(
        TopicArn=topic_arn,
        Subject=(f"HIGH AWS threat detected: {incident_id}")[:100],
        Message=json.dumps(message, indent=2),
    )
