from __future__ import annotations

from typing import Any

from aws_lambda_powertools import Logger

logger = Logger(service="threat-ml-scorer")


@logger.inject_lambda_context(clear_state=True)
def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Starter handler; add validation, feature engineering, and ML inference next."""
    if event.get("health_check") is True:
        return {"statusCode": 200, "body": {"status": "healthy"}}

    logger.info("Received event", extra={"event": event})
    return {
        "statusCode": 202,
        "body": {
            "message": "Starter scorer received the event",
            "next_step": "Add Pydantic validation, features, and model inference",
        },
    }
