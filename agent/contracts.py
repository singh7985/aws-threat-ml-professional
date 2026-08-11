from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    incident_id: str


class InvestigationReport(BaseModel):
    """The result of one deterministic investigation.

    ``risk_score`` is the detector's assessment of how dangerous the event is.
    It was previously exposed as ``confidence``, which answers a different
    question -- how certain the system is about that assessment -- and the two
    are not interchangeable: a LOW incident scoring 0.12 reported
    ``confidence: 0.12``, reading as "12% sure" when the detector is in fact
    confident the event is benign.

    No confidence figure is published, because none is calculated. If Bedrock
    later needs one, it should come from something real -- evidence strength,
    correlation strength -- rather than a risk score wearing a different name.
    """

    incident_id: str
    severity: str
    risk_score: float = Field(ge=0.0, le=1.0)
    summary: str
    suspicious_reasons: list[str]
    related_incidents: list[dict[str, Any]] = Field(default_factory=list)
    recommended_actions: list[str]
