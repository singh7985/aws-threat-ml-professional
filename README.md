# Autonomous Cloud Threat ML — Professional Starter

A professional starter environment for an AWS, Python, machine-learning, and AI cloud-threat detection platform.

## Included

- Python 3.12 virtual environment setup
- Windows PowerShell and macOS/Linux/WSL scripts
- `src/` package layout
- Pydantic configuration and security-event schemas
- Synthetic event generator
- Pytest, Ruff, mypy, Bandit, pip-audit, and pre-commit
- VS Code settings, tasks, debugger, and extension recommendations
- AWS CDK v2 foundation stack
- S3, SQS + DLQ, DynamoDB, and SNS
- Lambda Python 3.12 container starter
- GitHub Actions CI

## Prerequisites

Install Python 3.12, Git, Node.js 22+, AWS CLI v2, Docker, and VS Code.

## Local setup

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

### macOS, Linux, or WSL2

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
source .venv/bin/activate
```

Edit `.env`, then validate:

```bash
python scripts/check_environment.py
python -m threat_ml.cli generate \
  --output data/raw/security_events.jsonl \
  --normal 100 \
  --suspicious 10
ruff check .
mypy src infrastructure
pytest -v
```

## AWS authentication and CDK bootstrap

```bash
aws configure sso
python scripts/bootstrap_aws.py \
  --profile threat-ml-dev \
  --region us-east-1
```

## Review and deploy

```bash
python scripts/cdk.py synth
python scripts/cdk.py diff
python scripts/cdk.py deploy
```

The foundation stack creates an encrypted/versioned S3 bucket, SQS processing queue, dead-letter queue, DynamoDB incidents table, and SNS alert topic.

## Cleanup

```bash
python scripts/cdk.py destroy
```

The `dev` environment uses destroy policies for learning and testing. Do not use those policies for production data.

## Lambda container smoke test

```bash
docker build -f services/scorer/Dockerfile -t threat-ml-scorer .
docker run --rm -p 9000:8080 threat-ml-scorer
```

From another terminal:

```bash
curl -X POST \
  "http://localhost:9000/2015-03-31/functions/function/invocations" \
  -d '{"health_check": true}'
```
