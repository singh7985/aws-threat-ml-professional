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
from sklearn.ensemble import IsolationForest

from threat_ml.predict_event import (
    calculate_rule_score,
    convert_decision_to_anomaly_score,
    determine_risk_level,
    is_anomalous,
    read_anomaly_threshold,
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


def score_event(payload: dict[str, Any]) -> dict[str, Any]:
    model, feature_columns, manifest = load_artifacts()
    feature_frame = prepare_features(payload=payload, feature_columns=feature_columns)

    # The calibrated cutoff travels with the model in feature_manifest.json, so
    # serving uses the same boundary the model was evaluated against. Older
    # manifests without the key fall back to the previous behaviour.
    anomaly_threshold = read_anomaly_threshold(manifest)

    decision_value = float(model.decision_function(feature_frame)[0])
    model_flagged_anomaly = is_anomalous(decision_value, anomaly_threshold)
    anomaly_score = convert_decision_to_anomaly_score(decision_value, anomaly_threshold)
    rule_score, reasons = calculate_rule_score(payload)

    final_risk_score = min(1.0, 0.70 * anomaly_score + 0.30 * rule_score)
    risk_level = determine_risk_level(final_risk_score)

    if model_flagged_anomaly:
        reasons.insert(0, "The machine-learning model marked the behavior as unusual.")
    if not reasons:
        reasons.append("No major security warning was triggered.")

    # Record metrics for CloudWatch
    METRICS.add_metric(name="EventsScored", unit=MetricUnit.Count, value=1)
    if risk_level == "HIGH":
        METRICS.add_metric(name="HighRiskEvents", unit=MetricUnit.Count, value=1)
    elif risk_level == "MEDIUM":
        METRICS.add_metric(name="MediumRiskEvents", unit=MetricUnit.Count, value=1)
    else:
        METRICS.add_metric(name="LowRiskEvents", unit=MetricUnit.Count, value=1)

    return {
        "model_name": manifest.get("model_name", "AWS Threat Isolation Forest"),
        "model_version": manifest.get("model_version", "unknown"),
        "model_prediction": "suspicious" if model_flagged_anomaly else "normal",
        "model_flagged_anomaly": model_flagged_anomaly,
        "model_decision_value": round(decision_value, 6),
        "anomaly_score": round(anomaly_score, 4),
        "rule_score": round(rule_score, 4),
        "final_risk_score": round(final_risk_score, 4),
        "risk_level": risk_level,
        "reasons": reasons,
    }


def save_to_dynamodb(prediction: dict[str, Any], payload: dict[str, Any]) -> None:
    global _dynamodb
    if not INCIDENTS_TABLE:
        LOGGER.warning("INCIDENTS_TABLE_NAME not set; skipping database save.")
        return
    if not _dynamodb:
        _dynamodb = boto3.resource("dynamodb")

    table = _dynamodb.Table(INCIDENTS_TABLE)
    now = datetime.now(UTC).isoformat()

    item = {
        "incident_id": str(uuid.uuid4()),
        "timestamp": now,
        "risk_level": prediction["risk_level"],
        "final_risk_score": Decimal(str(prediction["final_risk_score"])),
        "anomaly_score": Decimal(str(prediction["anomaly_score"])),
        "rule_score": Decimal(str(prediction["rule_score"])),
        "reasons": prediction["reasons"],
        # Save a serialized version of what we analyzed
        "original_payload": json.dumps(payload),
    }

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

            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("SQS body must contain one JSON object.")

            result = score_event(payload)

            LOGGER.info(json.dumps({"message_id": message_id, "prediction": result}))

            # Step 1: Save every prediction to DynamoDB
            save_to_dynamodb(result, payload)

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
                        payload=payload,
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

    # For direct synchronous calls, we still score, save, and alert. Uses the
    # same publisher as the SQS path so both produce identical alerts.
    result = score_event(event)
    save_to_dynamodb(result, event)
    if result.get("risk_level") == "HIGH":
        try:
            publish_high_risk_alert(payload=event, prediction=result)
        except Exception as sns_error:
            LOGGER.error(f"Failed to publish SNS alert: {sns_error}")

    return {"request_id": request_id, "prediction": result}
