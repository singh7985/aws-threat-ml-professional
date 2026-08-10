"""Local command for running one investigation against DynamoDB.

    python -m agent.run_local --incident-id "YOUR_INCIDENT_ID"

Reads the incidents table named by INCIDENTS_TABLE_NAME (defaulting to
ThreatML-Incidents-dev) in the Region named by AWS_REGION.
"""

from __future__ import annotations

import argparse

from agent.contracts import InvestigationRequest
from agent.orchestrator import investigate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Investigate one incident and print a structured report.",
    )
    parser.add_argument(
        "--incident-id",
        required=True,
    )
    args = parser.parse_args()

    report = investigate(
        InvestigationRequest(
            incident_id=args.incident_id
        )
    )

    print(
        report.model_dump_json(indent=2)
    )


if __name__ == "__main__":
    main()
