from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aws_lambda_powertools import Logger

import threat_ml.predict_event
from threat_ml.predict_event import predict_event

logger = Logger(service="threat-ml-scorer")

# Retarget absolute paths for Lambda container expectations
threat_ml.predict_event.MODEL_PATH = Path("/var/task/models/model.joblib")
threat_ml.predict_event.FEATURE_MANIFEST_PATH = Path("/var/task/models/feature_manifest.json")


@logger.inject_lambda_context(clear_state=True)
def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Starter handler; ML added securely."""

    if event.get("health_check") is True:
        return {"statusCode": 200, "body": json.dumps({"status": "healthy"})}

    try:
        if "body" in event and isinstance(event["body"], str):
            payload = json.loads(event["body"])
        else:
            payload = event

        prediction_result = predict_event(payload)

        logger.info("Executed successfully", extra={"prediction_result": prediction_result})

        return {"statusCode": 200, "body": json.dumps(prediction_result)}

    except Exception as e:
        logger.exception("Error scoring the threat.")
        return {"statusCode": 400, "body": json.dumps({"error": str(e)})}
