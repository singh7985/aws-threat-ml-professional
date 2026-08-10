from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    incident_id: str

class InvestigationReport(BaseModel):
    incident_id: str
    severity: str
    summary: str
    suspicious_reasons: list[str]
    related_incidents: list[dict[str, Any]] = Field(
        default_factory=list
    )
    recommended_actions: list[str]
    confidence: float
