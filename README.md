# Autonomous Cloud Threat ML

Serverless AWS threat-detection platform. It scores CloudTrail-style API activity
with a calibrated Isolation Forest, stores incidents, alerts on high risk, and
runs a deterministic agent that investigates them.

## Architecture

```
SQS ──► Lambda (container)  ──► DynamoDB incidents
        │  score_event()    ──► SNS alert when risk is HIGH
        │                   ──► CloudWatch EMF metrics
        └─ partial-batch failures ──► DLQ
```

| Layer | Location |
|---|---|
| Feature schema, generator, scoring, training | `src/threat_ml/` |
| Lambda scorer (container image) | `services/scorer/` |
| CDK stacks: Foundation, Data, Scorer, Monitoring | `infrastructure/` |
| SageMaker pipeline, evaluation, model promotion | `mlops/` |
| Deterministic investigation agent | `agent/` |
| Streamlit dashboard | `dashboard.py` |

## How scoring works

Each event is reduced to 12 numeric features (`hour_of_day`, `unusual_hour`,
`external_ip`, `sensitive_operation`, `outside_home_region`, rolling call
counters, `api_risk_score`, …) and scored two ways, then blended:

```
final_risk_score = 0.70 × anomaly_score + 0.30 × rule_score
LOW < 0.40 ≤ MEDIUM < 0.70 ≤ HIGH
```

**The decision threshold is calibrated, not taken from `contamination`.** Training
holds out 20% of the labelled data, picks the cutoff that maximises F1 on that
slice, and writes it to `models/feature_manifest.json` as `anomaly_threshold`.
Evaluation and serving both read that value, so the reported metrics describe the
model as it is actually served.

| | contamination cutoff | calibrated cutoff |
|---|---|---|
| precision | 0.484 | **0.800** |
| recall | 0.909 | 0.889 |
| false positives (of 901 normal) | 96 | **22** |

There is no fallback to an uncalibrated default: a manifest without
`anomaly_threshold` raises rather than silently scoring the old way.

Current model: **v0.2.0**, threshold `0.0824`.

## Prerequisites

Python 3.12, Git, Node.js 22+, AWS CLI v2, Docker, VS Code.

## Local setup

```bash
# macOS / Linux / WSL2
chmod +x scripts/setup.sh && ./scripts/setup.sh
source .venv/bin/activate
```

```powershell
# Windows PowerShell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Copy `.env.example` to `.env`, set `ALERT_EMAIL`, then validate:

```bash
python scripts/check_environment.py
make check        # lint + types + tests + security
```

## Common commands

| Command | Purpose |
|---|---|
| `make check` | ruff, mypy, pytest (with coverage gate), bandit, pip-audit |
| `make format` | ruff format and autofix |
| `make test` | pytest with coverage (fails below 78%) |
| `make security` | bandit + pip-audit — identical to the CI job |
| `make generate` | generate synthetic security events |
| `make synth` / `diff` / `deploy` / `destroy` | CDK lifecycle |

## Train and score

```bash
python -m threat_ml.train_model                     # calibrate + write artifacts
python -m threat_ml.batch_predict                   # score a dataset
python -m threat_ml.predict_event --input data/test/high_risk_event.json
```

Training writes three artifacts to `models/` — `model.joblib`,
`feature_manifest.json` (feature order **and** `anomaly_threshold`), and
`metrics.json`. The manifest is the train/serve contract: the container copies it
in, and the Lambda reads the threshold from it.

## Deploy

```bash
aws configure sso
python scripts/bootstrap_aws.py --profile threat-ml-dev --region us-east-1
python scripts/cdk.py synth
python scripts/cdk.py diff
python scripts/cdk.py deploy
```

The Lambda is a container image built for `linux/amd64` to match its `x86_64`
architecture. Model changes reach production only through
`cdk deploy ThreatMlScorer-dev`, which rebuilds and pushes the image.

### Container smoke test

```bash
docker build --platform linux/amd64 -f services/scorer/Dockerfile -t threat-ml-scorer .
docker run --rm -p 9000:8080 threat-ml-scorer
python scripts/smoke_container.py        # from another terminal
```

Reference results — the CLI, container, and deployed Lambda must all agree:

| fixture | score | risk |
|---|---|---|
| `data/test/high_risk_event.json` | 0.7978 | HIGH |
| `data/test/normal_event.json` | 0.3142 | LOW |

## MLOps pipeline

```bash
# Register/update the pipeline definition (no compute, no quota needed)
python -m mlops.pipeline.run_pipeline --bucket-name YOUR_BUCKET

# Add --execute to start a run (consumes SageMaker instance quota)
python -m mlops.pipeline.run_pipeline --bucket-name YOUR_BUCKET --execute
```

`ProcessData → TrainIsolationForest → EvaluateModel → CheckModelQuality → RegisterModel`.
Thresholds live in `mlops/quality_gates.py` and are shared by the pipeline's
condition step and `approve_model.py`, so the two cannot disagree:

| gate | minimum |
|---|---|
| recall | 0.70 |
| precision | 0.70 |
| PR-AUC | 0.65 |

After a model is registered: `python -m mlops.deployment.approve_model`, then
`python -m mlops.deployment.promote_model`, then redeploy the scorer stack.

## Investigation agent

`agent/` turns an incident id into a validated `InvestigationReport` — detection
reasons, correlated incidents, and recommended response actions:

```bash
python -m agent.run_local --incident-id "YOUR_INCIDENT_ID"
```

The tools are deterministic and **read-only**: they call only DynamoDB
`get_item` and `scan`. Nothing disables a principal, revokes a session, or
modifies IAM — containment stays a human decision, and high-risk incidents get an
explicit escalation action instead.

## Tests

```bash
pytest                        # full suite, coverage gate at 78%
pytest agent/tests -v         # agent only
```

`scripts/smoke_container.py` needs a running container and is deliberately kept
out of `tests/` so pytest does not collect it.

## CI

`.github/workflows/ci.yml` runs ruff, mypy (`src infrastructure agent`), pytest
with the coverage gate, **bandit**, **pip-audit**, and `cdk synth`. `make check`
plus `make security` runs the same checks locally.

## Cleanup

```bash
python scripts/cdk.py destroy
```

The `dev` environment uses destroy removal policies so teardown is clean. Do not
use those policies for production data.
