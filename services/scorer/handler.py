from __future__ import annotations

import json
import math
import os
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
import joblib
import pandas as pd
from aws_lambda_powertools import Logger, Metrics
from aws_lambda_powertools.metrics import MetricUnit
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest

from threat_ml.contracts import ScoringMessage
from threat_ml.predict_event import (
    read_anomaly_threshold,
    score_feature_vector,
)

# Metadata fields promoted to top-level DynamoDB attributes; anything else the
# producer supplies is preserved under event_context.
EVENT_CORE_FIELDS = (
    "event_id",
    "timestamp",
    "principal_id",
    "source_ip",
    "event_name",
    "service",
    "region",
)

try:
    from notifier import publish_high_risk_alert
except ImportError:
    from services.scorer.notifier import publish_high_risk_alert

LOGGER = Logger(service="threat-ml-scorer")
METRICS = Metrics(namespace="IntelligentAwsThreatResponse", service="scorer")

# Environment variables injected via AWS CDK.
# Alerting goes through notifier.publish_high_risk_alert, which reads
# ALERTS_TOPIC_ARN. There used to be a second, near-duplicate publisher here
# reading SNS_TOPIC_ARN, so the same incident produced a different alert format
# depending on whether it arrived via SQS or a direct invoke.
INCIDENTS_TABLE = os.getenv("INCIDENTS_TABLE_NAME")

# Client lazily initialized for performance across executions
_dynamodb = None


def _default_model_directory() -> Path:
    """Find model artifacts locally or inside Lambda."""

    container_directory = Path("/var/task/model")

    if container_directory.exists():
        return container_directory

    # Local path:
    # project/services/scorer/handler.py -> project/
    project_root = Path(__file__).resolve().parents[2]

    return project_root / "models"


MODEL_DIRECTORY = Path(
    os.getenv(
        "MODEL_DIRECTORY",
        str(_default_model_directory()),
    )
)

MODEL_PATH = MODEL_DIRECTORY / "model.joblib"
FEATURE_MANIFEST_PATH = MODEL_DIRECTORY / "feature_manifest.json"


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[
    IsolationForest,
    tuple[str, ...],
    dict[str, Any],
]:
    """
    Load model artifacts once per Lambda execution environment.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")
    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Feature manifest was not found: {FEATURE_MANIFEST_PATH}")

    loaded_model = joblib.load(MODEL_PATH)
    if not isinstance(loaded_model, IsolationForest):
        raise TypeError("The saved model is not an IsolationForest.")

    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    feature_columns = manifest.get("feature_columns")
    if not isinstance(feature_columns, list):
        raise ValueError("The feature manifest does not contain a valid feature_columns list.")

    return (
        loaded_model,
        tuple(str(column) for column in feature_columns),
        manifest,
    )


def prepare_features(
    payload: dict[str, Any],
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    missing_features = [f for f in feature_columns if f not in payload]
    if missing_features:
        raise ValueError(f"Event is missing required features: {missing_features}")

    prepared: dict[str, float] = {}
    for feature in feature_columns:
        raw_value = payload[feature]
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Feature '{feature}' must be numeric. Received: {raw_value!r}"
            ) from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"Feature '{feature}' must contain a finite number.")
        prepared[feature] = numeric_value

    return pd.DataFrame([prepared], columns=list(feature_columns))


def parse_message(body: str | dict[str, Any]) -> ScoringMessage:
    """Validate one queue message against the contract, before anything else.

    Called on the raw SQS body so a malformed message is rejected at the
    boundary and routed to the DLQ, rather than being partially processed. The
    bare feature vector used by the previous contract is refused with an
    actionable error: accepting it would keep producing incidents that carry no
    identity and therefore can never be correlated.
    """
    if isinstance(body, str):
        try:
            message = json.loads(body)
        except json.JSONDecodeError as error:
            raise ValueError(f"Message body is not valid JSON: {error}") from error
    else:
        message = body

    if not isinstance(message, dict):
        raise ValueError(f"Message body must be a JSON object, got {type(message).__name__}.")

    try:
        return ScoringMessage.model_validate(message)
    except ValidationError as error:
        if "event" not in message and "features" not in message:
            raise ValueError(
                "Message is not a scoring message. Expected "
                '{"schema_version", "event", "features"}; received a bare '
                f"feature vector with keys {sorted(message)[:6]}. Producers must "
                "send identity under 'event' so incidents can be correlated."
            ) from error
        raise ValueError(f"Invalid scoring message: {error}") from error


def score_features(features: dict[str, float]) -> dict[str, Any]:
    """Score one numeric feature vector. Identity never reaches this function."""
    model, feature_columns, manifest = load_artifacts()
    feature_frame = prepare_features(payload=features, feature_columns=feature_columns)

    # The calibrated cutoff travels with the model in feature_manifest.json, so
    # serving uses the same boundary the model was evaluated against. A manifest
    # without it raises rather than silently scoring uncalibrated.
    anomaly_threshold = read_anomaly_threshold(manifest)

    decision_value = float(model.decision_function(feature_frame)[0])

    # One scoring implementation, shared with the CLI, batch scorer and
    # ThreatPredictor. The Lambda adds telemetry and artifact provenance around
    # it but does not re-derive any of the arithmetic.
    result = score_feature_vector(
        decision_value=decision_value,
        features=features,
        anomaly_threshold=anomaly_threshold,
    )

    # Record metrics for CloudWatch
    METRICS.add_metric(name="EventsScored", unit=MetricUnit.Count, value=1)
    risk_level = result["risk_level"]
    if risk_level == "HIGH":
        METRICS.add_metric(name="HighRiskEvents", unit=MetricUnit.Count, value=1)
    elif risk_level == "MEDIUM":
        METRICS.add_metric(name="MediumRiskEvents", unit=MetricUnit.Count, value=1)
    else:
        METRICS.add_metric(name="LowRiskEvents", unit=MetricUnit.Count, value=1)

    return {
        "model_name": manifest.get("model_name", "AWS Threat Isolation Forest"),
        "model_version": manifest.get("model_version", "unknown"),
        **result,
    }


def _to_dynamo_number(value: Any) -> Decimal:
    """DynamoDB rejects float; Decimal(str(x)) avoids binary-float artefacts."""
    return Decimal(str(value))


def build_incident(message: ScoringMessage, prediction: dict[str, Any]) -> dict[str, Any]:
    """Assemble the stored incident from the message and the score.

    Identity is written as **top-level attributes**, not buried in a serialised
    blob. That is what lets the agent correlate incidents, and what a global
    secondary index on principal_id or source_ip would key off later.
    """
    event = message.event
    extra = event.model_dump(mode="json", exclude=set(EVENT_CORE_FIELDS))

    item: dict[str, Any] = {
        "incident_id": str(uuid.uuid4()),
        "schema_version": message.schema_version,
        # When the incident was scored, distinct from when the event occurred.
        "timestamp": datetime.now(UTC).isoformat(),
        "event_time": event.timestamp.isoformat(),
        "event_id": event.event_id,
        # --- scoring outcome ---
        "risk_level": prediction["risk_level"],
        "final_risk_score": _to_dynamo_number(prediction["final_risk_score"]),
        "anomaly_score": _to_dynamo_number(prediction["anomaly_score"]),
        "rule_score": _to_dynamo_number(prediction["rule_score"]),
        "model_flagged_anomaly": prediction["model_flagged_anomaly"],
        "model_version": prediction["model_version"],
        "reasons": prediction["reasons"],
        # The exact vector that produced the score, kept for explainability.
        "features": {k: _to_dynamo_number(v) for k, v in message.features.items()},
    }

    # Only populated identity fields are written. Absent ones are omitted rather
    # than stored as null, so a correlation scan never matches two incidents on
    # the shared absence of a field.
    item.update(event.correlation_dimensions())

    if extra:
        # Any additional CloudTrail context the producer supplied.
        item["event_context"] = json.loads(json.dumps(extra), parse_float=Decimal)

    return item


def save_to_dynamodb(item: dict[str, Any]) -> None:
    global _dynamodb
    if not INCIDENTS_TABLE:
        LOGGER.warning("INCIDENTS_TABLE_NAME not set; skipping database save.")
        return
    if not _dynamodb:
        _dynamodb = boto3.resource("dynamodb")

    table = _dynamodb.Table(INCIDENTS_TABLE)

    try:
        table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(incident_id)",
        )
        LOGGER.info("Saved incident to DynamoDB", extra={"incident_id": item["incident_id"]})
    except Exception as e:
        LOGGER.error(f"Failed to save incident to DynamoDB: {e!s}")


def process_sqs_event(event: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    failed_messages: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = str(record.get("messageId", "unknown"))

        try:
            body = record.get("body")
            if not isinstance(body, str):
                raise ValueError("SQS record body must be a JSON string.")

            # Validated at the boundary: nothing downstream sees an unvalidated
            # message, and a malformed one becomes a batch item failure -> DLQ.
            message = parse_message(body)
            result = score_features(message.numeric_features())

            LOGGER.info(
                json.dumps(
                    {
                        "message_id": message_id,
                        "event_id": message.event.event_id,
                        "principal_id": message.event.principal_id,
                        "event_name": message.event.event_name,
                        "prediction": result,
                    }
                )
            )

            # Step 1: Save every prediction to DynamoDB
            save_to_dynamodb(build_incident(message, result))

            METRICS.add_metric(
                name="EventsProcessed",
                unit=MetricUnit.Count,
                value=1,
            )

            risk_level = str(result["risk_level"])

            if risk_level == "HIGH":
                METRICS.add_metric(
                    name="HighRiskIncidents",
                    unit=MetricUnit.Count,
                    value=1,
                )

                try:
                    publish_high_risk_alert(
                        event=message.event.model_dump(mode="json"),
                        prediction=result,
                    )
                except Exception as sns_err:
                    LOGGER.error(f"Failed to publish SNS alert: {sns_err}")

            elif risk_level == "MEDIUM":
                METRICS.add_metric(
                    name="MediumRiskIncidents",
                    unit=MetricUnit.Count,
                    value=1,
                )

            else:
                METRICS.add_metric(
                    name="LowRiskEvents",
                    unit=MetricUnit.Count,
                    value=1,
                )

        except Exception:
            METRICS.add_metric(
                name="ProcessingFailures",
                unit=MetricUnit.Count,
                value=1,
            )
            LOGGER.exception("Failed to process SQS message %s", message_id)
            failed_messages.append({"itemIdentifier": message_id})

    return {"batchItemFailures": failed_messages}


@METRICS.log_metrics(capture_cold_start_metric=True)
@LOGGER.inject_lambda_context(clear_state=True)
def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Main Lambda entry point."""
    request_id = getattr(context, "aws_request_id", "local-request")
    LOGGER.info("Processing request %s", request_id)

    if event.get("health_check") is True:
        return {
            "status": "healthy",
            "model_path": str(MODEL_PATH),
            "feature_manifest_path": str(FEATURE_MANIFEST_PATH),
        }

    if isinstance(event.get("Records"), list):
        return process_sqs_event(event)

    # Direct synchronous calls go through the same contract, scoring, storage
    # and alerting as the SQS path, so both produce identical incidents.
    message = parse_message(event)
    result = score_features(message.numeric_features())
    save_to_dynamodb(build_incident(message, result))

    if result.get("risk_level") == "HIGH":
        try:
            publish_high_risk_alert(
                event=message.event.model_dump(mode="json"),
                prediction=result,
            )
        except Exception as sns_error:
            LOGGER.error(f"Failed to publish SNS alert: {sns_error}")

    return {
        "request_id": request_id,
        "event_id": message.event.event_id,
        "prediction": result,
    }
