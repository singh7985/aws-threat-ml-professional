"""Contract regression tests for the scoring message and incident shape.

One test per guarantee this phase depends on. If any of these fail, the
producer/consumer contract has moved and incidents will either stop being
scored or stop being correlatable -- both silent failures in production, which
is why they are pinned here rather than left to integration testing.

Numbered to match the contract checklist:

  1  valid message parses
  2  missing `features` fails
  3  wrong feature type fails
  4  the Lambda scores only `features`
  5  principal_id reaches DynamoDB unchanged
  6  source_ip reaches DynamoDB unchanged
  7  event_name reaches DynamoDB unchanged
  8  service and region reach DynamoDB unchanged
  9  the calibrated threshold is still applied
 10  a malformed SQS record enters partial-batch failure handling
 11  a duplicate event_id stays idempotent
 12  no destructive AWS call is issued by the agent
"""

from __future__ import annotations

import json
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import joblib
import numpy as np
import pytest
from moto import mock_aws
from sklearn.ensemble import IsolationForest

from services.scorer import handler
from threat_ml.contracts import ScoringMessage

TABLE_NAME = "ThreatML-Contract-Test"

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

# Deliberately distinctive values, so "reaches DynamoDB unchanged" means the
# exact string and not something coincidentally equal.
EVENT = {
    "event_id": "evt-contract-0001",
    "timestamp": "2026-08-11T03:15:00Z",
    "principal_id": "AIDAEXAMPLE:alice@example.com",
    "source_ip": "203.0.113.25",
    "event_name": "CreateAccessKey",
    "service": "iam",
    "region": "ap-southeast-1",
}

FEATURES: dict[str, float | int] = {
    "hour_of_day": 3,
    "is_weekend": 0,
    "failed_call": 1,
    "sensitive_operation": 1,
    "unusual_hour": 1,
    "external_ip": 1,
    "outside_home_region": 1,
    "calls_last_5m": 12,
    "failed_calls_last_5m": 5,
    "unique_services_last_1h": 7,
    "iam_calls_last_1h": 6,
    "api_risk_score": 0.98,
}

CALIBRATED_THRESHOLD = 0.0824


def valid_message() -> dict[str, Any]:
    return {"schema_version": "1.0", "event": dict(EVENT), "features": dict(FEATURES)}


@pytest.fixture
def scorer(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the handler at a throwaway artifact set with a known threshold."""
    directory = Path(tempfile.mkdtemp(prefix="contract-model-"))

    rng = np.random.default_rng(42)
    model = IsolationForest(n_estimators=100, random_state=42)
    model.fit(rng.normal(0.0, 0.5, size=(200, len(FEATURE_COLUMNS))))
    joblib.dump(model, directory / "model.joblib")

    (directory / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_name": "AWS Threat Isolation Forest",
                "model_version": "0.2.0-contract",
                "feature_columns": FEATURE_COLUMNS,
                "anomaly_threshold": CALIBRATED_THRESHOLD,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(handler, "MODEL_PATH", directory / "model.joblib")
    monkeypatch.setattr(handler, "FEATURE_MANIFEST_PATH", directory / "feature_manifest.json")
    handler.load_artifacts.cache_clear()
    yield directory
    handler.load_artifacts.cache_clear()


@pytest.fixture
def incidents_table(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real (mocked) DynamoDB table wired into the handler."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "incident_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "incident_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        monkeypatch.setattr(handler, "INCIDENTS_TABLE", TABLE_NAME)
        monkeypatch.setattr(handler, "_dynamodb", None)
        yield dynamodb.Table(TABLE_NAME)


class MockContext:
    function_name = "contract-test"
    memory_limit_in_mb = 128
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:contract-test"
    aws_request_id = "contract-request"


def store(table: Any) -> dict[str, Any]:
    """Score the canonical message through the handler and return the stored row."""
    handler.lambda_handler(
        {"Records": [{"messageId": "m-1", "body": json.dumps(valid_message())}]},
        MockContext(),
    )
    return table.scan()["Items"][0]


# ---------------------------------------------------------------------------
# 1. Valid message parses
# ---------------------------------------------------------------------------


def test_01_valid_message_parses() -> None:
    message = handler.parse_message(json.dumps(valid_message()))

    assert isinstance(message, ScoringMessage)
    assert message.schema_version == "1.0"
    assert message.event.event_id == EVENT["event_id"]
    assert set(message.features) == set(FEATURES)


def test_01b_schema_version_defaults_when_absent() -> None:
    payload = valid_message()
    del payload["schema_version"]

    assert handler.parse_message(payload).schema_version == "1.0"


# ---------------------------------------------------------------------------
# 2. Missing features fails
# ---------------------------------------------------------------------------


def test_02_missing_features_key_fails() -> None:
    payload = valid_message()
    del payload["features"]

    with pytest.raises(ValueError, match="Invalid scoring message"):
        handler.parse_message(payload)


def test_02b_empty_features_fails() -> None:
    payload = valid_message()
    payload["features"] = {}

    with pytest.raises(ValueError, match="Invalid scoring message"):
        handler.parse_message(payload)


def test_02c_missing_event_fails() -> None:
    payload = valid_message()
    del payload["event"]

    with pytest.raises(ValueError, match="Invalid scoring message"):
        handler.parse_message(payload)


# ---------------------------------------------------------------------------
# 3. Wrong feature type fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    ["not-a-number", None, [1, 2], {"nested": 1}, float("nan"), float("inf")],
)
def test_03_wrong_feature_type_fails(bad_value: Any) -> None:
    payload = valid_message()
    payload["features"]["api_risk_score"] = bad_value

    with pytest.raises(ValueError):
        handler.parse_message(payload)


def test_03b_numeric_strings_are_coerced_not_silently_dropped() -> None:
    """A numeric string is a producer bug worth coercing, not discarding."""
    payload = valid_message()
    payload["features"]["api_risk_score"] = "0.5"

    message = handler.parse_message(payload)

    assert message.numeric_features()["api_risk_score"] == 0.5


# ---------------------------------------------------------------------------
# 4. The Lambda scores only features
# ---------------------------------------------------------------------------


def test_04_scorer_receives_only_features(
    scorer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture what the scoring implementation is actually handed."""
    captured: dict[str, Any] = {}
    original = handler.score_feature_vector

    def spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(handler, "score_feature_vector", spy)

    message = handler.parse_message(valid_message())
    handler.score_features(message.numeric_features())

    assert set(captured["features"]) == set(FEATURE_COLUMNS)
    for identity in ("principal_id", "source_ip", "event_name", "service", "region", "event_id"):
        assert identity not in captured["features"]


def test_04b_score_features_signature_takes_features_alone() -> None:
    import inspect

    assert list(inspect.signature(handler.score_features).parameters) == ["features"]


# ---------------------------------------------------------------------------
# 5-8. Identity reaches DynamoDB unchanged
# ---------------------------------------------------------------------------


def test_05_principal_id_reaches_dynamodb_unchanged(scorer: Path, incidents_table: Any) -> None:
    assert store(incidents_table)["principal_id"] == "AIDAEXAMPLE:alice@example.com"


def test_06_source_ip_reaches_dynamodb_unchanged(scorer: Path, incidents_table: Any) -> None:
    assert store(incidents_table)["source_ip"] == "203.0.113.25"


def test_07_event_name_reaches_dynamodb_unchanged(scorer: Path, incidents_table: Any) -> None:
    assert store(incidents_table)["event_name"] == "CreateAccessKey"


def test_08_service_and_region_reach_dynamodb_unchanged(
    scorer: Path, incidents_table: Any
) -> None:
    item = store(incidents_table)

    assert item["service"] == "iam"
    assert item["region"] == "ap-southeast-1"


def test_08b_full_incident_shape_is_stored(scorer: Path, incidents_table: Any) -> None:
    item = store(incidents_table)

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
    assert required <= set(item), f"missing: {sorted(required - set(item))}"
    assert item["status"] == "OPEN"
    assert item["incident_id"] == item["event_id"] == EVENT["event_id"]


# ---------------------------------------------------------------------------
# 9. The calibrated threshold is still applied
# ---------------------------------------------------------------------------


def test_09_calibrated_threshold_from_manifest_is_used(
    scorer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    original = handler.score_feature_vector

    def spy(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(handler, "score_feature_vector", spy)
    handler.score_features(dict(FEATURES))

    assert captured["anomaly_threshold"] == CALIBRATED_THRESHOLD


def test_09b_threshold_actually_changes_the_verdict(scorer: Path) -> None:
    """Guards against the threshold being read but ignored."""
    from threat_ml.predict_event import score_feature_vector

    permissive = score_feature_vector(
        decision_value=-0.01, features=dict(FEATURES), anomaly_threshold=-1.0
    )
    strict = score_feature_vector(
        decision_value=-0.01, features=dict(FEATURES), anomaly_threshold=1.0
    )

    assert permissive["model_flagged_anomaly"] is True
    assert strict["model_flagged_anomaly"] is False
    assert permissive["anomaly_score"] > strict["anomaly_score"]


def test_09c_manifest_without_threshold_refuses_to_score(
    scorer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No silent fallback to the uncalibrated contamination boundary."""
    manifest = scorer / "feature_manifest.json"
    data = json.loads(manifest.read_text())
    del data["anomaly_threshold"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    handler.load_artifacts.cache_clear()

    with pytest.raises(ValueError, match="anomaly_threshold"):
        handler.score_features(dict(FEATURES))


# ---------------------------------------------------------------------------
# 10. A malformed SQS record enters partial-batch failure handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "label"),
    [
        (json.dumps({"unknown_feature": 1}), "bare feature vector"),
        ("{not json", "malformed JSON"),
        (json.dumps([1, 2, 3]), "JSON array"),
        (json.dumps({"schema_version": "1.0", "event": dict(EVENT)}), "no features"),
    ],
)
def test_10_malformed_record_becomes_a_batch_item_failure(
    scorer: Path, incidents_table: Any, body: str, label: str
) -> None:
    result = handler.lambda_handler(
        {"Records": [{"messageId": "bad-1", "body": body}]}, MockContext()
    )

    assert result == {"batchItemFailures": [{"itemIdentifier": "bad-1"}]}, label
    assert incidents_table.scan()["Count"] == 0, f"{label} must not be stored"


def test_10b_one_bad_record_does_not_fail_the_whole_batch(
    scorer: Path, incidents_table: Any
) -> None:
    """Partial-batch failure: the good record must still be processed."""
    result = handler.lambda_handler(
        {
            "Records": [
                {"messageId": "good-1", "body": json.dumps(valid_message())},
                {"messageId": "bad-1", "body": "{not json"},
            ]
        },
        MockContext(),
    )

    assert result == {"batchItemFailures": [{"itemIdentifier": "bad-1"}]}
    assert incidents_table.scan()["Count"] == 1


# ---------------------------------------------------------------------------
# 11. A duplicate event_id stays idempotent
# ---------------------------------------------------------------------------


def test_11_duplicate_event_id_is_idempotent(scorer: Path, incidents_table: Any) -> None:
    """SQS delivers at least once; a redelivery must not create a second row."""
    records = [{"messageId": "m-1", "body": json.dumps(valid_message())}]

    first = handler.lambda_handler({"Records": records}, MockContext())
    count_after_first = incidents_table.scan()["Count"]

    second = handler.lambda_handler({"Records": records}, MockContext())
    count_after_second = incidents_table.scan()["Count"]

    assert first == {"batchItemFailures": []}
    # A redelivery is normal traffic, not a failure to retry.
    assert second == {"batchItemFailures": []}
    assert count_after_first == count_after_second == 1


def test_11b_redelivery_does_not_overwrite_triage_state(
    scorer: Path, incidents_table: Any
) -> None:
    """A responder's status change must survive a duplicate delivery."""
    records = [{"messageId": "m-1", "body": json.dumps(valid_message())}]
    handler.lambda_handler({"Records": records}, MockContext())

    incidents_table.update_item(
        Key={"incident_id": EVENT["event_id"]},
        UpdateExpression="SET #s = :v",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":v": "INVESTIGATING"},
    )

    handler.lambda_handler({"Records": records}, MockContext())

    item = incidents_table.get_item(Key={"incident_id": EVENT["event_id"]})["Item"]
    assert item["status"] == "INVESTIGATING"


def test_11c_distinct_event_ids_produce_distinct_incidents(
    scorer: Path, incidents_table: Any
) -> None:
    second = valid_message()
    second["event"]["event_id"] = "evt-contract-0002"

    handler.lambda_handler(
        {
            "Records": [
                {"messageId": "m-1", "body": json.dumps(valid_message())},
                {"messageId": "m-2", "body": json.dumps(second)},
            ]
        },
        MockContext(),
    )

    assert incidents_table.scan()["Count"] == 2


# ---------------------------------------------------------------------------
# 12. No destructive AWS call is issued by the agent
# ---------------------------------------------------------------------------

DESTRUCTIVE_OPERATIONS = frozenset(
    {
        "PutItem",
        "UpdateItem",
        "DeleteItem",
        "BatchWriteItem",
        "CreateTable",
        "DeleteTable",
        "UpdateTable",
        "DeleteUser",
        "DeleteRole",
        "DeleteAccessKey",
        "UpdateAccessKey",
        "AttachUserPolicy",
        "AttachRolePolicy",
        "DetachUserPolicy",
        "PutUserPolicy",
        "PutRolePolicy",
        "DeleteLoginProfile",
        "TerminateInstances",
        "StopInstances",
        "RevokeSecurityGroupIngress",
        "PutBucketPolicy",
        "DeleteObject",
    }
)


def test_12_agent_issues_no_destructive_aws_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Record every API call the investigation makes and assert all are reads.

    Hooks botocore's event system rather than inspecting source, so a
    destructive call made indirectly through any helper is still caught.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("HOME_AWS_REGION", "us-east-1")
    monkeypatch.setenv("INCIDENTS_TABLE_NAME", TABLE_NAME)

    from agent.contracts import InvestigationRequest
    from agent.orchestrator import investigate

    called: list[str] = []

    with mock_aws():
        session = boto3.Session(region_name="us-east-1")

        def record(model: Any = None, **_kwargs: Any) -> None:
            if model is not None:
                called.append(model.name)

        session.events.register("before-call", record)
        monkeypatch.setattr(boto3, "resource", session.resource)
        monkeypatch.setattr(boto3, "client", session.client)

        dynamodb = session.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "incident_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "incident_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        dynamodb.Table(TABLE_NAME).put_item(
            Item={
                "incident_id": EVENT["event_id"],
                "event_id": EVENT["event_id"],
                "timestamp": "2026-08-11T03:15:00+00:00",
                "principal_id": EVENT["principal_id"],
                "source_ip": EVENT["source_ip"],
                "event_name": EVENT["event_name"],
                "service": EVENT["service"],
                "region": EVENT["region"],
                "risk_level": "HIGH",
                "final_risk_score": Decimal("0.7978"),
                "anomaly_score": Decimal("0.7133"),
                "rule_score": Decimal("0.995"),
                "reasons": ["seeded"],
                "status": "OPEN",
                "model_version": "0.2.0",
            }
        )

        called.clear()  # only the agent's calls from here on
        report = investigate(InvestigationRequest(incident_id=EVENT["event_id"]))

    assert report.severity == "HIGH"
    assert called, "no AWS calls recorded -- the probe is not wired up"

    destructive = sorted(set(called) & DESTRUCTIVE_OPERATIONS)
    assert not destructive, f"agent issued destructive call(s): {destructive}"
    assert set(called) <= {"GetItem", "Scan", "DescribeTable"}, (
        f"unexpected AWS call(s): {sorted(set(called))}"
    )
