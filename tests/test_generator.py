from pathlib import Path

from threat_ml.generator import generate_events
from threat_ml.schemas import SecurityEvent


def test_generate_events(tmp_path: Path) -> None:
    output = tmp_path / "events.jsonl"
    generate_events(output=output, normal=5, suspicious=2, seed=42)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7

    events = [SecurityEvent.model_validate_json(line) for line in lines]
    assert sum(event.label == "suspicious" for event in events) == 2
