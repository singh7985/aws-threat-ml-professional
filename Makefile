.PHONY: help setup format lint type test security check generate synth diff deploy destroy

help:
	@echo "setup     Install project dependencies"
	@echo "format    Format and fix Python files"
	@echo "check     Run lint, typing, tests, and security"
	@echo "generate  Generate synthetic security events"
	@echo "synth     Synthesize AWS CDK"
	@echo "diff      Review AWS changes"
	@echo "deploy    Deploy AWS resources"
	@echo "destroy   Destroy AWS resources"

setup:
	python -m pip install --upgrade pip setuptools wheel
	python -m pip install -e ".[dev,infra,mlops]"
	npm install
	pre-commit install

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

type:
	mypy src services infrastructure agent mlops

test:
	pytest --cov=threat_ml --cov-report=term-missing

# Kept identical to the CI job so a local pass means a CI pass.
security:
	bandit -c pyproject.toml -r src services infrastructure mlops agent
	pip-audit --skip-editable

check: lint type test security

generate:
	python -m threat_ml.cli generate --output data/raw/security_events.jsonl --normal 100 --suspicious 10

synth:
	python scripts/cdk.py synth

diff:
	python scripts/cdk.py diff

deploy:
	python scripts/cdk.py deploy

destroy:
	python scripts/cdk.py destroy
