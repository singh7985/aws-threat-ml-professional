from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
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
    build_incident,
    lambda_handler,
    parse_message,
    score_features,
)


def load_test_event(name: str) -> dict[str, Any]:
    """Load one scoring message fixture."""
    path = PROJECT_ROOT / "data" / "test" / name

    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_features(name: str) -> dict[str, float]:
    return dict(load_test_event(name)["features"])


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
    high_risk = score_features(load_features("high_risk_event.json"))

    normal = score_features(load_features("normal_event.json"))

    assert high_risk["final_risk_score"] > normal["final_risk_score"]


def test_message_metadata_reaches_the_stored_incident() -> None:
    """The whole point of the contract: identity survives into DynamoDB."""
    message = parse_message(load_test_event("high_risk_event.json"))
    incident = build_incident(
        event=message.event,
        prediction=score_features(message.numeric_features()),
        features=message.numeric_features(),
    )

    assert incident["principal_id"] == "developer-04"
    assert incident["source_ip"] == "203.0.113.77"
    assert incident["event_name"] == "CreateAccessKey"
    assert incident["service"] == "iam"
    assert incident["region"] == "eu-west-1"
    assert incident["event_id"] == "evt-high-0001"
    assert incident["schema_version"] == "1.0"
    # Scoring outcome and the exact vector that produced it.
    assert incident["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert set(incident["features"]) == set(message.features)


def test_incident_carries_the_full_documented_shape() -> None:
    message = parse_message(load_test_event("high_risk_event.json"))
    incident = build_incident(
        event=message.event,
        prediction=score_features(message.numeric_features()),
        features=message.numeric_features(),
    )

    required = {
        "incident_id",
        "event_id",
        "timestamp",
        "principal_id",
        "source_ip",
        "event_name",
        "service",
        "region",
        "risk_level",
        "final_risk_score",
        "anomaly_score",
        "rule_score",
        "reasons",
        "status",
        "model_version",
    }
    assert required <= set(incident), f"missing: {sorted(required - set(incident))}"
    assert incident["status"] == "OPEN"
    assert incident["model_version"]


def test_incident_id_is_the_event_id_so_writes_are_idempotent() -> None:
    """SQS delivers at least once; a redelivery must not create a second incident."""
    message = parse_message(load_test_event("high_risk_event.json"))
    prediction = score_features(message.numeric_features())

    first = build_incident(event=message.event, prediction=prediction)
    second = build_incident(event=message.event, prediction=prediction)

    assert first["incident_id"] == first["event_id"] == "evt-high-0001"
    assert first["incident_id"] == second["incident_id"]


def test_naked_feature_vector_is_rejected_with_guidance() -> None:
    """The old contract must fail loudly, not score without identity."""
    with pytest.raises(ValueError, match="not a scoring message"):
        parse_message(load_features("high_risk_event.json"))


def test_individual_identity_fields_may_be_absent() -> None:
    """Real CloudTrail records do not always carry every field."""
    payload = load_test_event("high_risk_event.json")
    del payload["event"]["principal_id"]
    del payload["event"]["service"]

    message = parse_message(payload)

    assert message.event.principal_id is None
    assert message.event.source_ip == "203.0.113.77"
    # Absent fields are omitted from storage, never written as null, so a scan
    # cannot match two incidents on a shared missing value.
    incident = build_incident(
        event=message.event,
        prediction=score_features(message.numeric_features()),
        features=message.numeric_features(),
    )
    assert "principal_id" not in incident
    assert "service" not in incident
    assert incident["source_ip"] == "203.0.113.77"


def test_message_requires_at_least_one_identity_field() -> None:
    """An event with no correlatable field can never be linked to anything."""
    payload = load_test_event("high_risk_event.json")
    for field in ("principal_id", "source_ip", "event_name", "service", "region"):
        payload["event"].pop(field, None)

    with pytest.raises(ValueError, match="at least one of"):
        parse_message(payload)


def test_unknown_top_level_key_is_rejected() -> None:
    """A typo such as 'feature' must fail at the boundary, not half-process."""
    payload = load_test_event("high_risk_event.json")
    payload["featurez"] = {}

    with pytest.raises(ValueError, match="Invalid scoring message"):
        parse_message(payload)


def test_malformed_json_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_message("{not json")


def test_non_object_body_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_message(json.dumps([1, 2, 3]))


def test_non_finite_feature_is_rejected() -> None:
    payload = load_test_event("high_risk_event.json")
    payload["features"]["api_risk_score"] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        parse_message(payload)


def test_invalid_source_ip_is_rejected() -> None:
    payload = load_test_event("high_risk_event.json")
    payload["event"]["source_ip"] = "not-an-ip"

    with pytest.raises(ValueError, match="valid IP address"):
        parse_message(payload)


def test_model_never_receives_identity() -> None:
    """score_features must work from features alone."""
    features = load_features("high_risk_event.json")
    assert not {"principal_id", "source_ip", "event_name"} & set(features)

    result = score_features(features)
    assert 0.0 <= result["final_risk_score"] <= 1.0


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
