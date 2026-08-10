"""Deterministic investigation orchestrator.

Runs a full investigation for one incident by calling the four deterministic
tools in order, then assembles the result into a validated
:class:`~agent.contracts.InvestigationReport`.

There is no language model in this path. Retrieval, scoring, severity,
correlation, and the allowed response actions are all decided by code, so the
same incident always produces the same report. When the Bedrock layer is added
it should rewrite and organise this evidence -- not replace it.
"""

from __future__ import annotations

import math
from typing import Any

from agent.contracts import (
    InvestigationReport,
    InvestigationRequest,
)
from agent.tools.incident_tools import (
    explain_detection,
    find_related_incidents,
    get_incident,
    recommend_response_actions,
)

UNKNOWN_SEVERITY = "UNKNOWN"


def _as_confidence(value: Any) -> float:
    """Read the stored risk score as a confidence in the range 0.0-1.0.

    Guards the three ways the stored value can be unusable: absent, non-numeric,
    or NaN/infinite. Any of those would otherwise propagate into the report.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(score):
        return 0.0

    return min(max(score, 0.0), 1.0)


def investigate(
    request: InvestigationRequest,
) -> InvestigationReport:
    """Investigate one incident and return a structured report.

    Raises:
        ValueError: if no incident exists with the requested id.
    """
    incident = get_incident(request.incident_id)

    if not incident:
        raise ValueError(
            f"Incident not found: {request.incident_id}"
        )

    reasons = explain_detection(incident)
    related = find_related_incidents(incident)
    actions = recommend_response_actions(incident)

    # `or` rather than a get() default: the key may be present but null, in
    # which case a plain default would render the severity as "None".
    risk_level = str(
        incident.get("risk_level") or UNKNOWN_SEVERITY
    )

    score = _as_confidence(
        incident.get("final_risk_score", 0)
    )

    reason_text = (
        f"The main detection reasons were: {', '.join(reasons)}."
        if reasons
        else "No specific detection rule was recorded for this incident."
    )

    summary = (
        f"This incident was classified as {risk_level} "
        f"with a final risk score of {score:.2f}. "
        f"{reason_text}"
    )

    return InvestigationReport(
        incident_id=request.incident_id,
        severity=risk_level,
        summary=summary,
        suspicious_reasons=reasons,
        related_incidents=related,
        recommended_actions=actions,
        confidence=score,
    )
