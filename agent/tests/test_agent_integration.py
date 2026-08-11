"""End-to-end integration test for the investigation agent.

Unlike the unit tests, nothing here is hand-seeded into DynamoDB. Three scoring
messages go through the real Lambda handler -- contract validation, model
scoring, incident assembly, storage -- and the agent is then run against what
the handler actually wrote.

That distinction matters. Seeding rows by hand tests the agent against the shape
someone *believes* the scorer produces. The bug this whole change exists to fix
was precisely that the two disagreed: incidents reached storage with no identity
at all, and correlation silently returned nothing.

The scenario:

    A  principal=alice  ip=203.0.113.10  CreateAccessKey    iam  us-east-1
    B  principal=alice  ip=203.0.113.50  AttachUserPolicy   iam  us-east-1
    C  principal=bob    ip=198.51.100.20 ListBuckets        s3   us-east-1

Investigating A must return B (same principal, same service, same Region) and
must rank B above C, which shares only the Region.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import boto3
import joblib
import numpy as np
import pytest
from moto import mock_aws
from sklearn.ensemble import IsolationForest

from agent.contracts import InvestigationRequest
from agent.orchestrator import investigate
from agent.tools.incident_tools import find_related_incidents, get_incident
from services.scorer import handler
from threat_ml.contracts import build_message

TABLE_NAME = "ThreatML-Incidents-integration"
HOME_REGION = "us-east-1"

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

SUSPICIOUS_FEATURES: dict[str, float | int] = {
    "hour_of_day": 3,
    "is_weekend": 0,
    "failed_call": 1,
    "sensitive_operation": 1,
    "unusual_hour": 1,
    "external_ip": 1,
    "outside_home_region": 0,
    "calls_last_5m": 12,
    "failed_calls_last_5m": 5,
    "unique_services_last_1h": 7,
    "iam_calls_last_1h": 6,
    "api_risk_score": 0.98,
}

BENIGN_FEATURES: dict[str, float | int] = {
    "hour_of_day": 11,
    "is_weekend": 0,
    "failed_call": 0,
    "sensitive_operation": 0,
    "unusual_hour": 0,
    "external_ip": 0,
    "outside_home_region": 0,
    "calls_last_5m": 2,
    "failed_calls_last_5m": 0,
    "unique_services_last_1h": 2,
    "iam_calls_last_1h": 0,
    "api_risk_score": 0.10,
}

# A and B share a principal, service and Region but differ by source IP -- the
# realistic case of one actor working from two addresses. C shares only the
# Region, which on its own must not make it "related".
INCIDENT_A = {
    "event_id": "evt-A",
    "timestamp": "2026-08-11T03:15:00Z",
    "principal_id": "alice",
    "source_ip": "203.0.113.10",
    "event_name": "CreateAccessKey",
    "service": "iam",
    "region": HOME_REGION,
}
INCIDENT_B = {
    "event_id": "evt-B",
    "timestamp": "2026-08-11T03:22:00Z",
    "principal_id": "alice",
    "source_ip": "203.0.113.50",
    "event_name": "AttachUserPolicy",
    "service": "iam",
    "region": HOME_REGION,
}
INCIDENT_C = {
    "event_id": "evt-C",
    "timestamp": "2026-08-11T11:40:00Z",
    "principal_id": "bob",
    "source_ip": "198.51.100.20",
    "event_name": "ListBuckets",
    "service": "s3",
    "region": HOME_REGION,
}


class MockContext:
    function_name = "integration"
    memory_limit_in_mb = 128
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:integration"
    aws_request_id = "integration-request"


@pytest.fixture
def live_pipeline(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run the three messages through the real handler into a real table."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", HOME_REGION)
    monkeypatch.setenv("AWS_REGION", HOME_REGION)
    monkeypatch.setenv("HOME_AWS_REGION", HOME_REGION)
    monkeypatch.setenv("INCIDENTS_TABLE_NAME", TABLE_NAME)

    directory = Path(tempfile.mkdtemp(prefix="integration-model-"))
    rng = np.random.default_rng(42)
    model = IsolationForest(n_estimators=100, random_state=42)
    model.fit(rng.normal(0.0, 0.5, size=(200, len(FEATURE_COLUMNS))))
    joblib.dump(model, directory / "model.joblib")
    (directory / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_name": "AWS Threat Isolation Forest",
                "model_version": "0.2.0-integration",
                "feature_columns": FEATURE_COLUMNS,
                "anomaly_threshold": 0.0824,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(handler, "MODEL_PATH", directory / "model.joblib")
    monkeypatch.setattr(handler, "FEATURE_MANIFEST_PATH", directory / "feature_manifest.json")
    handler.load_artifacts.cache_clear()

    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name=HOME_REGION)
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "incident_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "incident_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        monkeypatch.setattr(handler, "INCIDENTS_TABLE", TABLE_NAME)
        monkeypatch.setattr(handler, "_dynamodb", None)

        messages = [
            build_message(event=INCIDENT_A, features=SUSPICIOUS_FEATURES),
            build_message(event=INCIDENT_B, features=SUSPICIOUS_FEATURES),
            build_message(event=INCIDENT_C, features=BENIGN_FEATURES),
        ]
        result = handler.lambda_handler(
            {
                "Records": [
                    {"messageId": f"msg-{i}", "body": message.model_dump_json()}
                    for i, message in enumerate(messages)
                ]
            },
            MockContext(),
        )
        assert result == {"batchItemFailures": []}, "the handler rejected a valid message"

        yield dynamodb.Table(TABLE_NAME)

    handler.load_artifacts.cache_clear()


# ---------------------------------------------------------------------------
# The handler wrote what the agent needs
# ---------------------------------------------------------------------------


def test_all_three_incidents_are_stored(live_pipeline: Any) -> None:
    assert live_pipeline.scan()["Count"] == 3


def test_identity_survives_the_whole_pipeline(live_pipeline: Any) -> None:
    """Message -> handler -> DynamoDB, with identity intact at every hop."""
    incident = get_incident("evt-A")

    assert incident["principal_id"] == "alice"
    assert incident["source_ip"] == "203.0.113.10"
    assert incident["event_name"] == "CreateAccessKey"
    assert incident["service"] == "iam"
    assert incident["region"] == HOME_REGION


# ---------------------------------------------------------------------------
# The correlation this change exists to enable
# ---------------------------------------------------------------------------


def test_investigating_a_returns_b_as_related(live_pipeline: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="evt-A"))

    related_ids = [item["incident_id"] for item in report.related_incidents]

    assert "evt-B" in related_ids, "B shares alice, iam and the Region with A"
    assert report.related_incidents, "an empty list is the bug this change fixes"


def test_b_outranks_c(live_pipeline: Any) -> None:
    """B must score above C, which shares only the Region."""
    related = find_related_incidents(get_incident("evt-A"), min_score=0)

    scores = {item["incident_id"]: item["correlation_score"] for item in related}

    assert scores["evt-B"] > scores.get("evt-C", 0), (
        f"B must outrank C; got {scores}"
    )
    # Ordering is part of the contract: strongest first.
    ordered = [item["incident_id"] for item in related]
    assert ordered.index("evt-B") < ordered.index("evt-C")


def test_c_is_not_related_by_default(live_pipeline: Any) -> None:
    """A shared Region alone must not make two incidents related.

    Nearly all activity sits in one Region, so treating that as a match would
    relate every incident to every other and make the feature useless.
    """
    report = investigate(InvestigationRequest(incident_id="evt-A"))

    assert "evt-C" not in [item["incident_id"] for item in report.related_incidents]


def test_correlation_explains_itself(live_pipeline: Any) -> None:
    related = find_related_incidents(get_incident("evt-A"))
    match = next(item for item in related if item["incident_id"] == "evt-B")

    reasons = " | ".join(match["match_reasons"])

    assert "alice" in reasons
    assert "iam" in reasons
    # Different IPs, so that must NOT be claimed as a match.
    assert "203.0.113.50" not in reasons


def test_correlation_is_symmetric(live_pipeline: Any) -> None:
    """Investigating B must find A, not only the other way round."""
    report = investigate(InvestigationRequest(incident_id="evt-B"))

    assert "evt-A" in [item["incident_id"] for item in report.related_incidents]


def test_investigating_c_finds_nothing(live_pipeline: Any) -> None:
    """bob's benign S3 read shares nothing meaningful with alice's IAM activity."""
    report = investigate(InvestigationRequest(incident_id="evt-C"))

    assert report.related_incidents == []


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


def test_report_uses_risk_score_not_confidence(live_pipeline: Any) -> None:
    """risk_score answers "how dangerous"; no confidence figure is invented."""
    report = investigate(InvestigationRequest(incident_id="evt-A"))
    payload = json.loads(report.model_dump_json())

    assert "risk_score" in payload
    assert "confidence" not in payload
    assert 0.0 <= payload["risk_score"] <= 1.0


def test_report_is_actionable(live_pipeline: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="evt-A"))

    assert report.severity in {"LOW", "MEDIUM", "HIGH"}
    assert report.suspicious_reasons
    assert report.recommended_actions
    # CreateAccessKey is credential compromise; the playbook must reflect that.
    assert any("access key" in action.lower() for action in report.recommended_actions)
