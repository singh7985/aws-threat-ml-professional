from __future__ import annotations

from datetime import datetime
from ipaddress import IPv4Address, IPv6Address
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


# The scoring message contract lives in threat_ml.contracts. It is deliberately
# not duplicated here: two definitions of the same contract is how producer and
# consumer drift apart.
