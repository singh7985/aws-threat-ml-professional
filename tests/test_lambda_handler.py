from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = [
    "hour_of_day",
    "is_weekend",
    "failed_call",
    "sensitive_operation",
    "unusual_hour",
    "external_ip",
    "outside_home_region",
    "calls_last_5m",
    "failed_calls_last_5m",
    "unique_services_last_1h",
    "iam_calls_last_1h",
    "api_risk_score",
]


def _build_model_fixture() -> str:
    """Create a self-contained calibrated artifact set for the handler.

    models/model.joblib is deliberately not in source control, so the suite must
    not depend on a locally trained artifact: on a clean checkout there is none,
    and these tests would fail for a reason unrelated to the code under test.
    The forest is fitted on benign-looking traffic so an unusual event still
    scores as more anomalous than a normal one.
    """
    directory = Path(tempfile.mkdtemp(prefix="scorer-model-"))

    rng = np.random.default_rng(42)
    benign = rng.normal(0.0, 0.5, size=(200, len(FEATURE_COLUMNS)))

    model = IsolationForest(n_estimators=100, random_state=42)
    model.fit(benign)
    joblib.dump(model, directory / "model.joblib")

    (directory / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_name": "AWS Threat Isolation Forest",
                "model_version": "0.2.0-test",
                "algorithm": "IsolationForest",
                "feature_columns": FEATURE_COLUMNS,
                "anomaly_threshold": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return str(directory)


# These must be set before the handler is imported: it resolves the model
# directory and its configuration at module scope, and boto3 needs a region
# before any client can be constructed.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ["MODEL_DIRECTORY"] = _build_model_fixture()

from services.scorer.handler import (  # noqa: E402
    lambda_handler,
    score_event,
)


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
