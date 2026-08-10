"""The scoring message contract.

Shared by every producer and consumer: the Lambda scorer validates against it,
tests build fixtures from it, and future producers (a CloudTrail ingester, a
replay tool) should construct it rather than hand-rolling a dict.

The message is deliberately two-part:

    event     who / what / where / when -- context, never scored
    features  the numeric vector the model consumes

Keeping them apart means the model never receives ``principal_id`` or
``source_ip`` as an input merely because the investigator needs them later, and
identity is no longer discarded on the way to storage. The previous contract
sent a bare feature vector, so incidents reached DynamoDB with nothing to
correlate on and every alert read "unknown".
"""

from __future__ import annotations

import math
from datetime import datetime
from ipaddress import ip_address
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

# Fields the investigation agent can correlate incidents on. Each is optional
# individually -- real CloudTrail records do not always carry every one -- but a
# message with none of them produces an incident that can never be linked to
# anything, so at least one is required.
CORRELATION_FIELDS = (
    "principal_id",
    "source_ip",
    "event_name",
    "service",
    "region",
)


class EventContext(BaseModel):
    """Identity and context for one observed event.

    Extra fields are kept rather than rejected, so richer CloudTrail context
    (account_id, user_agent, request_id, …) can ride along to the investigator
    without a schema change.
    """

    model_config = ConfigDict(extra="allow")

    event_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    principal_id: str | None = Field(default=None, max_length=256)
    source_ip: str | None = Field(default=None, max_length=64)
    event_name: str | None = Field(default=None, max_length=256)
    service: str | None = Field(default=None, max_length=128)
    region: str | None = Field(default=None, max_length=64)

    @field_validator("source_ip")
    @classmethod
    def _source_ip_must_parse(cls, value: str | None) -> str | None:
        """A malformed address is a broken producer, not a low-quality event."""
        if value is None:
            return None
        try:
            ip_address(value)
        except ValueError as error:
            raise ValueError(f"source_ip must be a valid IP address. Received: {value!r}") from error
        return value

    @model_validator(mode="after")
    def _needs_one_correlation_field(self) -> EventContext:
        if not any(getattr(self, name) for name in CORRELATION_FIELDS):
            raise ValueError(
                "event must carry at least one of "
                f"{list(CORRELATION_FIELDS)}. An event with none of them yields "
                "an incident that can never be correlated with any other."
            )
        return self

    def correlation_dimensions(self) -> dict[str, str]:
        """The identity fields that are actually populated."""
        return {
            name: str(getattr(self, name))
            for name in CORRELATION_FIELDS
            if getattr(self, name)
        }


class ScoringMessage(BaseModel):
    """One message on the scoring queue.

    ``extra="forbid"`` is deliberate: a typo such as ``feature`` or ``events``
    should fail loudly at the queue boundary rather than silently produce a
    message missing half its content.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=SCHEMA_VERSION)
    event: EventContext
    features: dict[str, float | int] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def _features_must_be_finite(
        cls, value: dict[str, float | int]
    ) -> dict[str, float | int]:
        """NaN and infinity must not reach the model."""
        bad = sorted(name for name, number in value.items() if not math.isfinite(number))
        if bad:
            raise ValueError(f"Feature values must be finite numbers. Offending: {bad}")
        return value

    def numeric_features(self) -> dict[str, float]:
        """Features as plain floats, ready for the model."""
        return {name: float(number) for name, number in self.features.items()}


def build_message(
    *,
    event: dict[str, Any],
    features: dict[str, float | int],
    schema_version: str = SCHEMA_VERSION,
) -> ScoringMessage:
    """Construct a validated scoring message.

    Producers should call this rather than assembling a dict, so a malformed
    message fails where it is created instead of on the queue.
    """
    return ScoringMessage(
        schema_version=schema_version,
        event=EventContext.model_validate(event),
        features=features,
    )
