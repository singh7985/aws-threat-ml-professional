from __future__ import annotations

import random
import uuid
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path

from threat_ml.schemas import SecurityEvent

NORMAL_EVENTS = ["ListBuckets", "GetObject", "DescribeInstances", "GetMetricData"]
SUSPICIOUS_EVENTS = ["CreateAccessKey", "AttachUserPolicy", "PutRolePolicy", "StopLogging"]


def build_event(*, suspicious: bool, index: int, rng: random.Random) -> SecurityEvent:
    now = datetime.now(UTC)

    if suspicious:
        event_name = rng.choice(SUSPICIOUS_EVENTS)
        timestamp = now.replace(
            hour=rng.choice([2, 3, 4, 22, 23]),
            minute=rng.randrange(60),
            second=0,
            microsecond=0,
        )
        return SecurityEvent(
            event_id=f"evt-{uuid.uuid4()}",
            timestamp=timestamp,
            account_id="111111111111",
            principal_id=f"developer-{rng.randrange(1, 6):02d}",
            event_name=event_name,
            service="iam" if event_name != "StopLogging" else "cloudtrail",
            region=rng.choice(["us-west-2", "eu-west-1"]),
            source_ip=ip_address(f"203.0.113.{rng.randrange(1, 255)}"),
            user_agent="aws-cli/2",
            success=True,
            sensitive_operation=True,
            label="suspicious",
            attack_type="privilege-escalation",
        )

    event_name = rng.choice(NORMAL_EVENTS)
    timestamp = (now - timedelta(minutes=index)).replace(
        hour=rng.randrange(9, 18),
        second=0,
        microsecond=0,
    )
    service_by_event = {
        "ListBuckets": "s3",
        "GetObject": "s3",
        "DescribeInstances": "ec2",
        "GetMetricData": "cloudwatch",
    }
    return SecurityEvent(
        event_id=f"evt-{uuid.uuid4()}",
        timestamp=timestamp,
        account_id="111111111111",
        principal_id=f"developer-{rng.randrange(1, 6):02d}",
        event_name=event_name,
        service=service_by_event[event_name],
        region="us-east-1",
        source_ip=ip_address(f"198.51.100.{rng.randrange(10, 20)}"),
        user_agent="boto3/1",
        success=rng.random() > 0.02,
        sensitive_operation=False,
        label="normal",
    )


def generate_events(*, output: Path, normal: int, suspicious: int, seed: int = 42) -> None:
    if normal < 0 or suspicious < 0:
        raise ValueError("Event counts must be non-negative")

    # A seeded, reproducible generator is the point: this produces synthetic
    # training events, never keys, tokens or anything security-bearing.
    rng = random.Random(seed)  # nosec B311
    events = [build_event(suspicious=False, index=i, rng=rng) for i in range(normal)]
    events.extend(build_event(suspicious=True, index=i, rng=rng) for i in range(suspicious))
    rng.shuffle(events)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(event.model_dump_json() + "\n")
