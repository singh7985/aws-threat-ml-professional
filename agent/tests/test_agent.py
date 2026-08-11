"""Tests for the deterministic investigation agent.

Run with:

    python -m pytest agent/tests -v

These cover the agent before any language model is connected: retrieval,
explanation, correlation, response actions, report validation, and the
guarantee that no destructive AWS call is ever issued.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import boto3
import pytest
from moto import mock_aws

from agent.contracts import InvestigationReport, InvestigationRequest
from agent.orchestrator import investigate
from agent.tools.incident_tools import (
    explain_detection,
    find_related_incidents,
    get_incident,
    recommend_response_actions,
)

TABLE_NAME = "ThreatML-Incidents-test"
HOME_REGION = "us-east-1"

# The feature vector the scorer actually sends to the model.
HIGH_RISK_FEATURES = {
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

LOW_RISK_FEATURES = {
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

# DynamoDB write APIs, plus the IAM/EC2 calls a containment action would use.
# The agent must never issue any of these.
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
        "DetachRolePolicy",
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

SEED_INCIDENTS: list[dict[str, Any]] = [
    # HIGH, credential compromise, full identity.
    {
        "incident_id": "inc-high-001",
        "timestamp": "2026-08-07T03:15:00+00:00",
        "risk_level": "HIGH",
        "final_risk_score": Decimal("0.88"),
        "anomaly_score": Decimal("0.93"),
        "rule_score": Decimal("0.80"),
        "principal_id": "developer-04",
        "source_ip": "203.0.113.77",
        "event_name": "CreateAccessKey",
        "service": "iam",
        "region": "eu-west-1",
        "attack_type": "credential-compromise",
        "reasons": ["The machine-learning model marked the behavior as unusual."],
        "original_payload": json.dumps(HIGH_RISK_FEATURES),
    },
    # Same principal and source IP -> should correlate strongly.
    {
        "incident_id": "inc-high-002",
        "timestamp": "2026-08-07T03:20:00+00:00",
        "risk_level": "HIGH",
        "final_risk_score": Decimal("0.91"),
        "anomaly_score": Decimal("0.95"),
        "principal_id": "developer-04",
        "source_ip": "203.0.113.77",
        "event_name": "StopLogging",
        "service": "cloudtrail",
        "region": "eu-west-1",
        "attack_type": "defense-evasion",
        "original_payload": json.dumps(HIGH_RISK_FEATURES),
    },
    # Different principal, different IP, home Region -> must not correlate.
    {
        "incident_id": "inc-low-001",
        "timestamp": "2026-08-07T11:00:00+00:00",
        "risk_level": "LOW",
        "final_risk_score": Decimal("0.12"),
        "anomaly_score": Decimal("0.15"),
        "principal_id": "developer-99",
        "source_ip": "10.0.0.5",
        "event_name": "GetMetricData",
        "service": "cloudwatch",
        "region": "us-east-1",
        "original_payload": json.dumps(LOW_RISK_FEATURES),
    },
    # The shape services/scorer/handler.py writes today: scores and a feature
    # vector, but no identity fields.
    {
        "incident_id": "inc-scorer-shape",
        "timestamp": "2026-08-07T03:14:00+00:00",
        "risk_level": "HIGH",
        "final_risk_score": Decimal("0.87"),
        "anomaly_score": Decimal("0.91"),
        "rule_score": Decimal("0.80"),
        "reasons": ["The machine-learning model marked the behavior as unusual."],
        "original_payload": json.dumps(HIGH_RISK_FEATURES),
    },
]


@pytest.fixture
def aws_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point boto3 at fake credentials and the test table."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", HOME_REGION)
    monkeypatch.setenv("AWS_REGION", HOME_REGION)
    monkeypatch.setenv("HOME_AWS_REGION", HOME_REGION)
    monkeypatch.setenv("INCIDENTS_TABLE_NAME", TABLE_NAME)


@pytest.fixture
def incidents_table(aws_environment: None) -> Any:
    """Create and seed an in-memory incidents table."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name=HOME_REGION)
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "incident_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "incident_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table = dynamodb.Table(TABLE_NAME)
        for item in SEED_INCIDENTS:
            table.put_item(Item=item)

        yield table


# ---------------------------------------------------------------------------
# A valid incident produces an investigation report
# ---------------------------------------------------------------------------


def test_valid_incident_produces_report(incidents_table: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))

    assert isinstance(report, InvestigationReport)
    assert report.incident_id == "inc-high-001"
    assert report.severity == "HIGH"
    assert report.summary
    assert report.suspicious_reasons
    assert report.recommended_actions


def test_report_summary_states_severity_and_score(incidents_table: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))

    assert "HIGH" in report.summary
    assert "0.88" in report.summary


def test_report_is_valid_structured_json(incidents_table: Any) -> None:
    """The report must survive serialisation -- DynamoDB Decimals break json.dumps."""
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))

    payload = json.loads(report.model_dump_json())

    assert payload["incident_id"] == "inc-high-001"
    assert payload["severity"] == "HIGH"
    assert isinstance(payload["suspicious_reasons"], list)
    assert isinstance(payload["related_incidents"], list)
    assert isinstance(payload["recommended_actions"], list)
    assert isinstance(payload["risk_score"], float)

    # Round-trips back through validation unchanged.
    assert InvestigationReport.model_validate(payload) == report


def test_incident_without_identity_fields_still_reports(incidents_table: Any) -> None:
    """The shape the scorer writes today must still yield a usable report."""
    report = investigate(InvestigationRequest(incident_id="inc-scorer-shape"))

    assert report.severity == "HIGH"
    assert report.suspicious_reasons
    assert report.recommended_actions
    # No identity fields are stored, so there is nothing to correlate on.
    assert report.related_incidents == []


# ---------------------------------------------------------------------------
# A missing incident raises a clear error
# ---------------------------------------------------------------------------


def test_missing_incident_raises_clear_error(incidents_table: Any) -> None:
    with pytest.raises(ValueError, match="Incident not found: inc-does-not-exist"):
        investigate(InvestigationRequest(incident_id="inc-does-not-exist"))


def test_empty_incident_id_is_rejected(incidents_table: Any) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        get_incident("")


# ---------------------------------------------------------------------------
# HIGH incidents receive urgent response actions
# ---------------------------------------------------------------------------


def test_high_incident_receives_urgent_actions(incidents_table: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))
    actions = report.recommended_actions

    # The credential-compromise playbook.
    assert "Disable or rotate the affected access key" in actions
    assert "Review recent activity by the principal" in actions
    assert "Invalidate active sessions" in actions
    assert "Check for unauthorized IAM changes" in actions
    assert "Preserve CloudTrail evidence" in actions

    # Plus escalation, because the incident is HIGH.
    assert any("Escalate" in action for action in actions)


def test_low_incident_is_not_escalated(incidents_table: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="inc-low-001"))

    assert report.severity == "LOW"
    assert not any("Escalate" in action for action in report.recommended_actions)


def test_actions_are_advisory_never_imperative_automation(incidents_table: Any) -> None:
    """Recommendations go to a human; the agent must not claim it acted."""
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))
    joined = " ".join(report.recommended_actions).lower()

    for claim in ("disabled the", "deleted the", "revoked the", "automatically"):
        assert claim not in joined


# ---------------------------------------------------------------------------
# Related incidents are returned correctly
# ---------------------------------------------------------------------------


def test_related_incidents_are_returned(incidents_table: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))
    related_ids = [item["incident_id"] for item in report.related_incidents]

    assert "inc-high-002" in related_ids


def test_related_excludes_self_and_weak_matches(incidents_table: Any) -> None:
    report = investigate(InvestigationRequest(incident_id="inc-high-001"))
    related_ids = [item["incident_id"] for item in report.related_incidents]

    assert "inc-high-001" not in related_ids, "an incident must not relate to itself"
    assert "inc-low-001" not in related_ids, "unrelated incident was correlated"


def test_related_incidents_explain_why_they_matched(incidents_table: Any) -> None:
    related = find_related_incidents(get_incident("inc-high-001"))

    match = next(item for item in related if item["incident_id"] == "inc-high-002")

    assert match["correlation_score"] >= 2
    reasons = " ".join(match["match_reasons"])
    assert "developer-04" in reasons
    assert "203.0.113.77" in reasons


def test_related_incidents_sorted_strongest_first(incidents_table: Any) -> None:
    related = find_related_incidents(get_incident("inc-high-001"))
    scores = [item["correlation_score"] for item in related]

    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Confidence remains between 0 and 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "incident_id",
    ["inc-high-001", "inc-high-002", "inc-low-001", "inc-scorer-shape"],
)
def test_risk_score_within_bounds(incidents_table: Any, incident_id: str) -> None:
    report = investigate(InvestigationRequest(incident_id=incident_id))

    assert 0.0 <= report.risk_score <= 1.0


def test_risk_score_survives_out_of_range_stored_value(incidents_table: Any) -> None:
    """A corrupt stored score must be clamped, not propagated into the report."""
    incidents_table.put_item(
        Item={
            "incident_id": "inc-corrupt",
            "risk_level": "HIGH",
            "final_risk_score": Decimal("9.99"),
        }
    )

    report = investigate(InvestigationRequest(incident_id="inc-corrupt"))

    assert report.risk_score == 1.0


def test_risk_score_handles_missing_value(incidents_table: Any) -> None:
    incidents_table.put_item(Item={"incident_id": "inc-no-score", "risk_level": "LOW"})

    report = investigate(InvestigationRequest(incident_id="inc-no-score"))

    assert report.risk_score == 0.0


# ---------------------------------------------------------------------------
# Detection reasons
# ---------------------------------------------------------------------------


def test_explain_detection_covers_expected_reasons(incidents_table: Any) -> None:
    reasons = " | ".join(explain_detection(get_incident("inc-high-001")))

    assert "High anomaly score" in reasons
    assert "Sensitive IAM operation" in reasons
    assert "External source IP" in reasons
    assert "Unusual execution hour" in reasons
    assert "Activity outside the normal AWS Region" in reasons
    assert "Multiple recent failed calls" in reasons


def test_explain_detection_never_returns_empty_for_real_incident(
    incidents_table: Any,
) -> None:
    assert explain_detection(get_incident("inc-low-001"))


def test_tools_tolerate_empty_incident() -> None:
    assert explain_detection({}) == []
    assert find_related_incidents({}) == []
    assert recommend_response_actions({}) == [
        "Investigate the incident context and escalate appropriately."
    ]


# ---------------------------------------------------------------------------
# No destructive AWS action is executed
# ---------------------------------------------------------------------------


def test_no_destructive_aws_action_is_executed(
    aws_environment: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Record every AWS API call the agent issues and assert all are read-only.

    This hooks botocore's event system rather than inspecting source, so it also
    catches a destructive call made indirectly through any helper.
    """
    called_operations: list[str] = []

    with mock_aws():
        session = boto3.Session(region_name=HOME_REGION)

        def record(model: Any = None, **_kwargs: Any) -> None:
            if model is not None:
                called_operations.append(model.name)

        session.events.register("before-call", record)

        # Force every boto3.resource()/client() call inside the tools through
        # the instrumented session.
        monkeypatch.setattr(boto3, "resource", session.resource)
        monkeypatch.setattr(boto3, "client", session.client)

        dynamodb = session.resource("dynamodb", region_name=HOME_REGION)
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "incident_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "incident_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table = dynamodb.Table(TABLE_NAME)
        for item in SEED_INCIDENTS:
            table.put_item(Item=item)

        # Only calls made from here on are the agent's.
        called_operations.clear()

        report = investigate(InvestigationRequest(incident_id="inc-high-001"))

    assert report.incident_id == "inc-high-001"
    assert called_operations, "no AWS calls recorded -- the probe is not wired up"

    destructive = sorted(set(called_operations) & DESTRUCTIVE_OPERATIONS)
    assert not destructive, f"agent issued destructive AWS call(s): {destructive}"

    assert set(called_operations) <= {"GetItem", "Scan", "DescribeTable"}, (
        f"agent issued unexpected AWS call(s): {sorted(set(called_operations))}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_investigation_is_deterministic(incidents_table: Any) -> None:
    reports = [
        investigate(InvestigationRequest(incident_id="inc-high-001")) for _ in range(3)
    ]

    assert reports[0] == reports[1] == reports[2]
