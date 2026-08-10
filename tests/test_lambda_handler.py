from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# These must be set before the handler is imported: it reads configuration at
# module scope, and boto3 needs a region before any client can be constructed.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

from services.scorer.handler import (
    lambda_handler,
    score_event,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_test_event(name: str) -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "test" / name

    return dict(json.loads(path.read_text(encoding="utf-8")))


class MockContext:
    def __init__(self) -> None:
        self.function_name = "test_function"
        self.memory_limit_in_mb = 128
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test_function"
        self.aws_request_id = "test-request-id"


def test_lambda_health_check() -> None:
    result = lambda_handler(
        {"health_check": True},
        MockContext(),
    )

    assert result["status"] == "healthy"


def test_high_risk_scores_above_normal() -> None:
    high_risk = score_event(load_test_event("high_risk_event.json"))

    normal = score_event(load_test_event("normal_event.json"))

    assert high_risk["final_risk_score"] > normal["final_risk_score"]


def test_sqs_success_returns_no_failures() -> None:
    payload = load_test_event("high_risk_event.json")

    event = {
        "Records": [
            {
                "messageId": "message-001",
                "body": json.dumps(payload),
            }
        ]
    }

    result = lambda_handler(event, MockContext())

    assert result == {"batchItemFailures": []}


def test_invalid_sqs_record_returns_failure() -> None:
    event = {
        "Records": [
            {
                "messageId": "bad-message",
                "body": json.dumps({"unknown_feature": 1}),
            }
        ]
    }

    result = lambda_handler(event, MockContext())

    assert result == {"batchItemFailures": [{"itemIdentifier": "bad-message"}]}
