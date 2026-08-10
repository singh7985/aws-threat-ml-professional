"""Manual smoke test against a locally running scorer container.

Start the container first:
    docker run --rm -p 9000:8080 threat-ml-scorer
"""

import json
import urllib.request
from pathlib import Path

req = urllib.request.Request("http://localhost:9000/2015-03-31/functions/function/invocations", data=json.dumps({"health_check": True}).encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    print(response.read().decode())

with open(Path(__file__).resolve().parents[1] / "data" / "test" / "high_risk_event.json") as f:
    high_risk = f.read()

req = urllib.request.Request("http://localhost:9000/2015-03-31/functions/function/invocations", data=high_risk.encode("utf-8"), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    print(json.dumps(json.loads(response.read().decode()), indent=2))
