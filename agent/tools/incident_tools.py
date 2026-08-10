"""Deterministic investigation tools for the incident agent.

These tools do the factual work of an investigation so the language model does
not have to invent it: they read an incident, correlate it against other
incidents, explain why the detector fired, and propose a response plan.

Every function here is **read-only and advisory**. The tools call only
DynamoDB read APIs (``get_item`` / ``scan``). Nothing in this module disables a
principal, revokes a session, deletes a resource, or modifies IAM -- containment
stays a human decision, and the agent's job is to recommend, not to act.

The functions are deterministic: the same incident always yields the same
reasons, the same ordering, and the same recommended actions.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from typing import Any

import boto3

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_TABLE_NAME = "ThreatML-Incidents-dev"
DEFAULT_REGION = "us-east-1"

# The Region the account normally operates in. Activity outside it is a signal,
# not a verdict.
HOME_REGION_ENV = "HOME_AWS_REGION"

# An anomaly score at or above this is called out as high.
HIGH_ANOMALY_SCORE = 0.70

# "Multiple" failed calls, for the recent-failures reason.
FAILED_CALL_THRESHOLD = 2

# Hours considered outside normal working activity (UTC).
BUSINESS_HOURS = range(8, 20)

# Ceiling on how many items a correlation scan will read. The development table
# is small; this stops a runaway scan if it is pointed at a large one.
MAX_SCAN_ITEMS = 5_000

# RFC 1918 / link-local prefixes treated as internal.
INTERNAL_IP_PREFIXES = ("10.", "192.168.", "127.", "169.254.", "172.16.", "172.31.")

# --------------------------------------------------------------------------
# Attack taxonomy
#
# Categories and event names mirror the labels in the training dataset, so the
# agent's vocabulary matches what the model was actually trained on.
# --------------------------------------------------------------------------

CREDENTIAL_COMPROMISE = "credential-compromise"
PRIVILEGE_ESCALATION = "privilege-escalation"
RECONNAISSANCE = "reconnaissance"
DATA_EXFILTRATION = "data-exfiltration"
DEFENSE_EVASION = "defense-evasion"
PUBLIC_EXPOSURE = "public-exposure"

EVENT_NAME_TO_CATEGORY: dict[str, str] = {
    # Credential compromise
    "createaccesskey": CREDENTIAL_COMPROMISE,
    "consolelogin": CREDENTIAL_COMPROMISE,
    "getsessiontoken": CREDENTIAL_COMPROMISE,
    "assumerole": CREDENTIAL_COMPROMISE,
    # Privilege escalation
    "attachuserpolicy": PRIVILEGE_ESCALATION,
    "attachrolepolicy": PRIVILEGE_ESCALATION,
    "putrolepolicy": PRIVILEGE_ESCALATION,
    "putuserpolicy": PRIVILEGE_ESCALATION,
    "updateassumerolepolicy": PRIVILEGE_ESCALATION,
    "createpolicyversion": PRIVILEGE_ESCALATION,
    # Reconnaissance
    "listusers": RECONNAISSANCE,
    "listroles": RECONNAISSANCE,
    "describesecuritygroups": RECONNAISSANCE,
    "getcalleridentity": RECONNAISSANCE,
    "describeinstances": RECONNAISSANCE,
    # Data exfiltration
    "getobject": DATA_EXFILTRATION,
    "selectobjectcontent": DATA_EXFILTRATION,
    "batchgetitem": DATA_EXFILTRATION,
    "createdbsnapshot": DATA_EXFILTRATION,
    "copysnapshot": DATA_EXFILTRATION,
    # Defense evasion
    "stoplogging": DEFENSE_EVASION,
    "deletetrail": DEFENSE_EVASION,
    "puteventselectors": DEFENSE_EVASION,
    "deleteflowlogs": DEFENSE_EVASION,
    "deleteloggroup": DEFENSE_EVASION,
    # Public exposure
    "putbucketpolicy": PUBLIC_EXPOSURE,
    "putpublicaccessblock": PUBLIC_EXPOSURE,
    "deletepublicaccessblock": PUBLIC_EXPOSURE,
    "authorizesecuritygroupingress": PUBLIC_EXPOSURE,
    "putbucketacl": PUBLIC_EXPOSURE,
}

# Operations that are sensitive regardless of which category they fall into.
SENSITIVE_EVENT_NAMES = frozenset(
    {
        "createaccesskey",
        "deleteaccesskey",
        "updateaccesskey",
        "attachuserpolicy",
        "attachrolepolicy",
        "putrolepolicy",
        "putuserpolicy",
        "updateassumerolepolicy",
        "createpolicyversion",
        "createuser",
        "createrole",
        "deleteuser",
        "deleterole",
        "stoplogging",
        "deletetrail",
        "puteventselectors",
        "putbucketpolicy",
        "putpublicaccessblock",
        "deletepublicaccessblock",
    }
)

SENSITIVE_SERVICES = frozenset({"iam", "sts", "cloudtrail", "organizations", "kms"})

# Enumeration verbs. An unrecognised List*/Describe* call is discovery, which is
# a better guess than whatever its service would otherwise imply.
ENUMERATION_PREFIXES = ("list", "describe")

# ...except these, which read data rather than metadata.
DATA_ACCESS_EVENT_NAMES = frozenset({"query", "scan", "getobject", "batchgetitem"})

# --------------------------------------------------------------------------
# Correlation weights
#
# A shared principal or source IP is strong evidence two incidents belong to the
# same episode. A shared Region is weak on its own -- most activity sits in the
# home Region, so matching on it alone would relate an incident to everything.
# --------------------------------------------------------------------------

CORRELATION_WEIGHTS: dict[str, int] = {
    "principal_id": 5,
    "source_ip": 5,
    "attack_category": 3,
    "event_name": 2,
    "service": 1,
    "region": 1,
}

# Minimum weight for two incidents to count as related. At 2, a shared Region or
# service alone is not enough, but service + Region together is.
DEFAULT_MIN_CORRELATION_SCORE = 2


# --------------------------------------------------------------------------
# AWS helpers
# --------------------------------------------------------------------------


def get_boto_client(service: str) -> Any:
    """Factory helper to obtain a boto3 client."""
    return boto3.client(service, region_name=os.getenv("AWS_REGION", DEFAULT_REGION))


def get_dynamo_resource() -> Any:
    """Factory helper to obtain a connected boto3 dynamodb resource."""
    return boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", DEFAULT_REGION))


def _get_table_name() -> str:
    """Resolve the incidents table name from the environment, or fall back."""
    return os.getenv("INCIDENTS_TABLE_NAME", DEFAULT_TABLE_NAME)


def _get_home_region() -> str:
    return os.getenv(HOME_REGION_ENV) or os.getenv("AWS_REGION") or DEFAULT_REGION


# --------------------------------------------------------------------------
# Value coercion
# --------------------------------------------------------------------------


def _decode(value: Any) -> Any:
    """Convert DynamoDB types into plain JSON-serialisable Python values.

    DynamoDB returns every number as ``Decimal``, which ``json.dumps`` refuses to
    serialise. Anything handed to the language model has to survive that call.
    """
    if isinstance(value, Decimal):
        as_float = float(value)
        return int(as_float) if as_float.is_integer() else as_float
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_decode(item) for item in value]
    return value


def _as_number(value: Any, default: float = 0.0) -> float:
    """Best-effort numeric read; returns ``default`` when the value is unusable."""
    if value is None or isinstance(value, bool):
        return float(value) if isinstance(value, bool) else default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first(*values: Any) -> Any:
    """Return the first value that is neither None nor an empty string."""
    for value in values:
        if value not in (None, ""):
            return value
    return None


# --------------------------------------------------------------------------
# Incident normalisation
# --------------------------------------------------------------------------


def _payload(incident: dict[str, Any]) -> dict[str, Any]:
    """Return the scored payload stored alongside the incident.

    The scorer persists what it analysed under ``original_payload`` as a JSON
    string. Older or hand-written rows may store it as a dict, or omit it.
    """
    # "features" is the current contract; the older keys are still read so a
    # hand-written or externally produced row is not silently ignored.
    raw = (
        incident.get("features")
        or incident.get("original_payload")
        or incident.get("payload")
    )

    if isinstance(raw, dict):
        decoded: dict[str, Any] = _decode(raw)
        return decoded

    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
        if isinstance(parsed, dict):
            decoded_payload: dict[str, Any] = _decode(parsed)
            return decoded_payload

    return {}


def normalize_incident(incident: dict[str, Any]) -> dict[str, Any]:
    """Flatten an incident into one canonical shape.

    Identity fields may live in three different places depending on how the row
    was written: as top-level DynamoDB attributes, inside the scored
    ``original_payload``, or in raw CloudTrail casing (``eventName``,
    ``sourceIPAddress``, ``userIdentity.principalId``). Reading through one
    canonical view keeps the rest of this module free of that branching.
    """
    if not incident:
        return {}

    incident = _decode(incident)
    payload = _payload(incident)
    identity = incident.get("userIdentity") or payload.get("userIdentity") or {}
    if not isinstance(identity, dict):
        identity = {}

    def field(*names: str) -> Any:
        """Look a field up by any of its aliases, across incident then payload."""
        candidates = [source.get(name) for source in (incident, payload) for name in names]
        return _first(*candidates)

    event_name = field("event_name", "eventName")
    service = field("service", "eventSource", "event_source")
    attack_category = field("attack_category", "attack_type", "attackType")

    # `service` may arrive as a CloudTrail event source such as "s3.amazonaws.com".
    if isinstance(service, str) and service.endswith(".amazonaws.com"):
        service = service.removesuffix(".amazonaws.com")

    if not attack_category and isinstance(event_name, str):
        attack_category = EVENT_NAME_TO_CATEGORY.get(event_name.lower())

    return {
        "incident_id": field("incident_id", "incidentId", "id"),
        "timestamp": field("timestamp", "eventTime", "event_time"),
        "principal_id": _first(
            field("principal_id", "principalId", "user_name", "userName"),
            identity.get("principalId"),
            identity.get("userName"),
            identity.get("arn"),
        ),
        "source_ip": field("source_ip", "sourceIPAddress", "sourceIp"),
        "event_name": event_name,
        "service": service,
        "region": field("region", "awsRegion", "aws_region"),
        "attack_category": attack_category,
        "risk_level": field("risk_level", "riskLevel"),
        "anomaly_score": _as_number(field("anomaly_score", "anomalyScore")),
        "rule_score": _as_number(field("rule_score", "ruleScore")),
        "final_risk_score": _as_number(field("final_risk_score", "finalRiskScore")),
        "reasons": incident.get("reasons") or payload.get("reasons") or [],
        "score_metadata": incident.get("score_metadata") or payload.get("score_metadata") or {},
        "features": payload,
    }


# --------------------------------------------------------------------------
# Tool 1: read one incident
# --------------------------------------------------------------------------


def get_incident(incident_id: str) -> dict[str, Any]:
    """Read one incident from DynamoDB by its ``incident_id``.

    Returns the stored item with DynamoDB ``Decimal`` values converted to plain
    numbers, or an empty dict when no such incident exists.
    """
    if not incident_id or not str(incident_id).strip():
        raise ValueError("incident_id must be a non-empty string.")

    table = get_dynamo_resource().Table(_get_table_name())
    response = table.get_item(Key={"incident_id": str(incident_id)})

    item: dict[str, Any] = _decode(response.get("Item") or {})
    return item


# --------------------------------------------------------------------------
# Tool 2: correlate against other incidents
# --------------------------------------------------------------------------


def _scan_all_incidents(max_items: int = MAX_SCAN_ITEMS) -> list[dict[str, Any]]:
    """Read the incidents table, following pagination.

    A single ``scan`` call returns at most 1 MB, so without the pagination loop
    correlation would silently miss everything past the first page. Scanning is
    acceptable for the small development table; the intended replacement is a
    global secondary index on ``principal_id`` and ``source_ip``.
    """
    table = get_dynamo_resource().Table(_get_table_name())

    items: list[dict[str, Any]] = []
    scan_kwargs: dict[str, Any] = {}

    while True:
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))

        last_key = response.get("LastEvaluatedKey")
        if not last_key or len(items) >= max_items:
            break
        scan_kwargs["ExclusiveStartKey"] = last_key

    return items[:max_items]


def _correlate(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[int, list[str]]:
    """Score how strongly two normalised incidents relate, and say why."""
    score = 0
    reasons: list[str] = []

    for dimension, weight in CORRELATION_WEIGHTS.items():
        left = source.get(dimension)
        right = candidate.get(dimension)

        if not left or not right:
            continue

        if str(left).casefold() != str(right).casefold():
            continue

        score += weight
        reasons.append(f"Same {dimension.replace('_', ' ')}: {left}")

    return score, reasons


def find_related_incidents(
    incident: dict[str, Any],
    min_score: int = DEFAULT_MIN_CORRELATION_SCORE,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Find incidents that share identifying dimensions with ``incident``.

    Correlates on principal, source IP, AWS service, event name, Region, and
    attack category. Each dimension carries a weight -- a shared principal or
    source IP is strong evidence, a shared Region on its own is not -- and only
    candidates reaching ``min_score`` are returned.

    Each result is the stored incident plus two added keys: ``correlation_score``
    and ``match_reasons``. Results are ordered strongest first, then newest
    first, so the ordering is stable for a given table.
    """
    source = normalize_incident(incident)
    if not source:
        return []

    # With no identifying dimension there is nothing to correlate on, and a scan
    # would be pure waste.
    if not any(source.get(dimension) for dimension in CORRELATION_WEIGHTS):
        return []

    source_id = source.get("incident_id")
    related: list[dict[str, Any]] = []

    for item in _scan_all_incidents():
        candidate = normalize_incident(item)

        if not candidate or candidate.get("incident_id") == source_id:
            continue

        score, reasons = _correlate(source, candidate)
        if score < min_score:
            continue

        enriched = _decode(item)
        enriched["correlation_score"] = score
        enriched["match_reasons"] = reasons
        related.append(enriched)

    related.sort(
        key=lambda item: (item["correlation_score"], str(item.get("timestamp") or "")),
        reverse=True,
    )

    return related[:limit]


# --------------------------------------------------------------------------
# Tool 3: explain why the detector fired
# --------------------------------------------------------------------------


def _triggered_rules(view: dict[str, Any]) -> set[str]:
    """Collect rule names the scorer recorded, normalised to lowercase."""
    metadata = view.get("score_metadata") or {}
    raw = metadata.get("rules_triggered") or metadata.get("rules") or []

    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        return set()

    return {str(rule).strip().lower() for rule in raw if str(rule).strip()}


def _hour_of_day(view: dict[str, Any]) -> float | None:
    """Recover the event hour from the feature vector or the timestamp."""
    features = view.get("features") or {}

    if "hour_of_day" in features:
        return _as_number(features["hour_of_day"], default=-1.0)

    timestamp = view.get("timestamp")
    if isinstance(timestamp, str) and "T" in timestamp:
        try:
            return float(timestamp.split("T", 1)[1][:2])
        except (ValueError, IndexError):
            return None

    return None


def explain_detection(incident: dict[str, Any]) -> list[str]:
    """Return human-readable reasons the incident was flagged.

    Evidence is drawn from whichever of these the row actually carries: the
    rules the scorer recorded, the numeric feature vector it scored, and the raw
    event fields. Reasons come back in a fixed order so the same incident always
    reads the same way.
    """
    view = normalize_incident(incident)
    if not view:
        return []

    features = view.get("features") or {}
    rules = _triggered_rules(view)
    reasons: list[str] = []

    def flag(name: str) -> bool:
        return _as_number(features.get(name)) >= 1

    # 1. High anomaly score
    anomaly_score = view["anomaly_score"]
    if anomaly_score >= HIGH_ANOMALY_SCORE or "anomaly" in rules:
        reasons.append(f"High anomaly score ({anomaly_score:.2f})")

    # 2. Sensitive IAM operation
    event_name = view.get("event_name")
    service = view.get("service")
    is_sensitive = (
        flag("sensitive_operation")
        or bool(rules & {"sensitive_action", "is_sensitive_action", "sensitive_operation"})
        or (isinstance(event_name, str) and event_name.lower() in SENSITIVE_EVENT_NAMES)
        or (isinstance(service, str) and service.lower() in SENSITIVE_SERVICES)
    )
    if is_sensitive:
        detail = f" ({event_name})" if event_name else ""
        reasons.append(f"Sensitive IAM operation{detail}")

    # 3. External source IP
    source_ip = view.get("source_ip")
    is_external = flag("external_ip") or bool(
        rules & {"is_external_ip", "external_ip", "external_source_ip"}
    )
    if not is_external and isinstance(source_ip, str) and source_ip:
        is_external = not source_ip.startswith(INTERNAL_IP_PREFIXES)
    if is_external:
        detail = f" ({source_ip})" if source_ip else ""
        reasons.append(f"External source IP{detail}")

    # 4. Unusual execution hour
    hour = _hour_of_day(view)
    is_unusual_hour = flag("unusual_hour") or bool(rules & {"hour_risk", "time_risk"})
    if not is_unusual_hour and hour is not None and hour >= 0:
        is_unusual_hour = int(hour) not in BUSINESS_HOURS
    if is_unusual_hour:
        detail = f" ({int(hour):02d}:00 UTC)" if hour is not None and hour >= 0 else ""
        reasons.append(f"Unusual execution hour{detail}")

    # 5. Activity outside the normal AWS Region
    region = view.get("region")
    home_region = _get_home_region()
    outside_region = flag("outside_home_region") or "region_risk" in rules
    if not outside_region and isinstance(region, str) and region:
        outside_region = region.casefold() != home_region.casefold()
    if outside_region:
        detail = f" ({region})" if region else ""
        reasons.append(f"Activity outside the normal AWS Region{detail}")

    # 6. Multiple recent failed calls
    recent_failures = _as_number(features.get("failed_calls_last_5m"))
    has_failures = recent_failures >= FAILED_CALL_THRESHOLD or "error_rate_risk" in rules
    if has_failures:
        detail = f" ({int(recent_failures)} in the last 5 minutes)" if recent_failures else ""
        reasons.append(f"Multiple recent failed calls{detail}")

    if reasons:
        return reasons

    # Fall back to whatever the scorer wrote, then to a generic explanation, so
    # the agent is never handed an empty rationale.
    stored = [str(reason) for reason in (view.get("reasons") or []) if str(reason).strip()]
    if stored:
        return stored

    return ["The machine-learning model marked the behavior as unusual."]


# --------------------------------------------------------------------------
# Tool 4: recommend response actions
# --------------------------------------------------------------------------

# Advisory playbooks. These are recommendations for a human responder; nothing
# here is executed by the agent.
RESPONSE_PLAYBOOKS: dict[str, list[str]] = {
    CREDENTIAL_COMPROMISE: [
        "Disable or rotate the affected access key",
        "Review recent activity by the principal",
        "Invalidate active sessions",
        "Check for unauthorized IAM changes",
        "Preserve CloudTrail evidence",
    ],
    PRIVILEGE_ESCALATION: [
        "Review the policy or role change and revert it if unauthorized",
        "Identify which principal made the change and on whose behalf",
        "Audit the principal's effective permissions for further escalation",
        "Invalidate active sessions for the affected role",
        "Preserve CloudTrail evidence",
    ],
    DATA_EXFILTRATION: [
        "Verify data access patterns for the principal",
        "Review the affected S3 bucket policy or database permissions",
        "Check CloudTrail and S3 access logs for large or unusual transfers",
        "Confirm whether the accessed data was sensitive or regulated",
        "Preserve CloudTrail evidence for the associated resources",
    ],
    DEFENSE_EVASION: [
        "Re-enable the disabled trail, flow log, or log group immediately",
        "Determine the logging gap and what activity it may have hidden",
        "Review all activity by the principal during the gap",
        "Restrict CloudTrail and logging permissions to break-glass roles",
        "Preserve any remaining CloudTrail evidence",
    ],
    PUBLIC_EXPOSURE: [
        "Confirm whether the resource is currently reachable from the internet",
        "Restore the public access block or restrictive resource policy",
        "Review the security group or bucket policy change that opened access",
        "Check access logs for requests during the exposure window",
        "Preserve CloudTrail evidence",
    ],
    RECONNAISSANCE: [
        "Review the full sequence of enumeration calls by the principal",
        "Establish whether the principal normally performs discovery calls",
        "Watch for follow-on privilege escalation or data access",
        "Confirm the credential used is still expected to be active",
        "Preserve CloudTrail evidence",
    ],
}

DEFAULT_RESPONSE_ACTIONS = [
    "Review recent activity by the principal",
    "Confirm the activity with the resource owner",
    "Inspect VPC Flow Logs for related network activity",
    "Preserve CloudTrail evidence",
]

# Appended when the detector was highly confident, regardless of category.
HIGH_RISK_ESCALATION = "Escalate to the on-call security responder for containment approval"


def _classify(view: dict[str, Any]) -> str | None:
    """Determine the attack category for an incident."""
    category = view.get("attack_category")
    if isinstance(category, str) and category.strip():
        normalized = category.strip().lower().replace("_", "-").replace(" ", "-")
        if normalized in RESPONSE_PLAYBOOKS:
            return normalized

    event_name = view.get("event_name")
    if isinstance(event_name, str) and event_name.strip():
        normalized_event = event_name.strip().lower()

        mapped = EVENT_NAME_TO_CATEGORY.get(normalized_event)
        if mapped:
            return mapped

        if (
            normalized_event.startswith(ENUMERATION_PREFIXES)
            and normalized_event not in DATA_ACCESS_EVENT_NAMES
        ):
            return RECONNAISSANCE

    # Fall back to the service when the specific API call is unknown to us.
    service = view.get("service")
    if isinstance(service, str):
        service = service.strip().lower()
        if service in {"iam", "sts", "signin"}:
            return CREDENTIAL_COMPROMISE
        if service in {"s3", "dynamodb", "rds"}:
            return DATA_EXFILTRATION
        if service == "cloudtrail":
            return DEFENSE_EVASION

    return None


def recommend_response_actions(incident: dict[str, Any]) -> list[str]:
    """Recommend response actions for an incident, based on its category.

    Returns advice only. The agent must not disable principals, delete
    resources, or modify IAM -- containment stays with a human responder, and
    high-risk incidents get an explicit escalation step instead.
    """
    view = normalize_incident(incident)
    if not view:
        return ["Investigate the incident context and escalate appropriately."]

    category = _classify(view)
    playbook = RESPONSE_PLAYBOOKS[category] if category else DEFAULT_RESPONSE_ACTIONS
    actions = list(playbook)

    is_high_risk = (
        str(view.get("risk_level") or "").upper() == "HIGH"
        or view["final_risk_score"] >= HIGH_ANOMALY_SCORE
    )
    if is_high_risk and HIGH_RISK_ESCALATION not in actions:
        actions.append(HIGH_RISK_ESCALATION)

    return actions
