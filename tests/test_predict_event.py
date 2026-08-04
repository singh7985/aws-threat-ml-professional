from __future__ import annotations

import pytest

from threat_ml.predict_event import (
    calculate_rule_score,
    determine_risk_level,
    prepare_input,
)

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


def create_normal_payload() -> dict[str, float]:
    """Create one normal AWS activity example."""

    return {
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


def create_suspicious_payload() -> dict[str, float]:
    """Create one suspicious AWS activity example."""

    return {
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


def test_low_risk_level() -> None:
    assert determine_risk_level(0.20) == "LOW"


def test_medium_risk_level() -> None:
    assert determine_risk_level(0.50) == "MEDIUM"


def test_high_risk_level() -> None:
    assert determine_risk_level(0.85) == "HIGH"


def test_risk_level_boundary_values() -> None:
    assert determine_risk_level(0.00) == "LOW"
    assert determine_risk_level(0.39) == "LOW"
    assert determine_risk_level(0.40) == "MEDIUM"
    assert determine_risk_level(0.69) == "MEDIUM"
    assert determine_risk_level(0.70) == "HIGH"
    assert determine_risk_level(1.00) == "HIGH"


def test_prepare_input_returns_correct_feature_order() -> None:
    payload = create_normal_payload()

    prepared = prepare_input(
        payload=payload,
        feature_columns=FEATURE_COLUMNS,
    )

    assert prepared.columns.tolist() == FEATURE_COLUMNS
    assert len(prepared) == 1


def test_prepare_input_converts_values_to_float() -> None:
    payload = create_normal_payload()

    prepared = prepare_input(
        payload=payload,
        feature_columns=FEATURE_COLUMNS,
    )

    for column in FEATURE_COLUMNS:
        assert prepared[column].dtype.kind == "f"


def test_prepare_input_rejects_missing_feature() -> None:
    payload = create_normal_payload()
    payload.pop("external_ip")

    with pytest.raises(
        ValueError,
        match="missing required features",
    ):
        prepare_input(
            payload=payload,
            feature_columns=FEATURE_COLUMNS,
        )


def test_prepare_input_rejects_non_numeric_value() -> None:
    payload = create_normal_payload()
    payload["hour_of_day"] = "three"  # type: ignore[assignment]

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        prepare_input(
            payload=payload,
            feature_columns=FEATURE_COLUMNS,
        )


def test_normal_event_has_low_rule_score() -> None:
    payload = create_normal_payload()

    score, reasons = calculate_rule_score(payload)

    assert 0.0 <= score <= 1.0
    assert score < 0.40
    assert isinstance(reasons, list)


def test_suspicious_event_has_high_rule_score() -> None:
    payload = create_suspicious_payload()

    score, reasons = calculate_rule_score(payload)

    assert 0.0 <= score <= 1.0
    assert score >= 0.70
    assert len(reasons) >= 5


def test_suspicious_event_includes_expected_reasons() -> None:
    payload = create_suspicious_payload()

    _, reasons = calculate_rule_score(payload)

    combined_reasons = " ".join(reasons).lower()

    assert "sensitive operation" in combined_reasons
    assert "unusual time" in combined_reasons
    assert "external ip" in combined_reasons
    assert "outside the home region" in combined_reasons
    assert "high security risk" in combined_reasons


def test_rule_score_never_exceeds_one() -> None:
    payload = create_suspicious_payload()
    payload["api_risk_score"] = 10.0

    score, _ = calculate_rule_score(payload)

    assert score == 1.0
