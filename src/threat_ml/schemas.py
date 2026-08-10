from __future__ import annotations

import math
from datetime import datetime
from ipaddress import IPv4Address, IPv6Address
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SecurityEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    account_id: str = Field(pattern=r"^\d{12}$")
    principal_id: str = Field(min_length=1, max_length=256)
    event_name: str = Field(min_length=1, max_length=256)
    service: str = Field(min_length=1, max_length=128)
    region: str = Field(pattern=r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
    source_ip: IPv4Address | IPv6Address
    user_agent: str = Field(min_length=1, max_length=512)
    success: bool
    sensitive_operation: bool
    label: Literal["normal", "suspicious"] | None = None
    attack_type: str | None = Field(default=None, max_length=128)


class IncidentPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    anomaly_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    reasons: tuple[str, ...] = ()
    model_version: str


# ---------------------------------------------------------------------------
# Scoring envelope -- the contract between the producer and the scorer
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"


class EventMetadata(BaseModel):
    """Who did what, where, and when.

    None of this reaches the model. It exists so an incident stays
    investigable: the agent correlates on principal, source IP, service, event
    name and Region, and an alert that says "CreateAccessKey by developer-04
    from 203.0.113.77" is actionable where a bare risk score is not.

    Extra fields are preserved rather than rejected, so richer CloudTrail
    context (account_id, user_agent, request_id, …) can ride along and reach
    the investigator without a schema change.
    """

    model_config = ConfigDict(extra="allow")

    event_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    principal_id: str = Field(min_length=1, max_length=256)
    source_ip: IPv4Address | IPv6Address
    event_name: str = Field(min_length=1, max_length=256)
    service: str = Field(min_length=1, max_length=128)
    region: str = Field(min_length=1, max_length=64)


class ScoringEnvelope(BaseModel):
    """One message on the scoring queue.

    Deliberately two-part:

        event    -- identity and context, preserved and never scored
        features -- the numeric vector the model consumes

    Keeping them apart means the model never receives `principal_id` or
    `source_ip` as an input merely because the investigator needs them later,
    and it stops identity being discarded on the way to storage. The previous
    contract sent a naked feature vector, so incidents reached DynamoDB with
    nothing to correlate on.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    event: EventMetadata
    features: dict[str, float] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def _features_must_be_finite(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject NaN/inf early rather than letting them reach the model."""
        bad = [name for name, number in value.items() if not math.isfinite(number)]
        if bad:
            raise ValueError(f"Feature values must be finite numbers. Offending: {sorted(bad)}")
        return value
