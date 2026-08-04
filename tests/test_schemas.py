from datetime import UTC, datetime
from ipaddress import ip_address

import pytest
from pydantic import ValidationError

from threat_ml.schemas import SecurityEvent


def valid_event() -> dict[str, object]:
    return {
        "event_id": "evt-1001",
        "timestamp": datetime.now(UTC),
        "account_id": "111111111111",
        "principal_id": "developer-01",
        "event_name": "ListBuckets",
        "service": "s3",
        "region": "us-east-1",
        "source_ip": ip_address("198.51.100.10"),
        "user_agent": "boto3/1",
        "success": True,
        "sensitive_operation": False,
        "label": "normal",
    }


def test_valid_event() -> None:
    event = SecurityEvent.model_validate(valid_event())
    assert event.event_name == "ListBuckets"


def test_invalid_account_id_is_rejected() -> None:
    payload = valid_event()
    payload["account_id"] = "123"
    with pytest.raises(ValidationError):
        SecurityEvent.model_validate(payload)


def test_unknown_fields_are_rejected() -> None:
    payload = valid_event()
    payload["unexpected"] = "value"
    with pytest.raises(ValidationError):
        SecurityEvent.model_validate(payload)
