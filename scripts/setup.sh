#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Required command '$1' was not found." >&2
    exit 1
  fi
}

require_command "$PYTHON_BIN"
require_command node
require_command npm
require_command git
require_command docker
require_command aws

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
node -e 'const major = Number(process.versions.node.split(".")[0]); if (major < 22) { throw new Error(`Node.js 22+ required; found ${process.versions.node}`); }'
aws --version 2>&1 | grep -q '^aws-cli/2' || { echo "ERROR: AWS CLI v2 is required." >&2; exit 1; }

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is installed, but its daemon is not running." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,infra]"
npm install
if [[ ! -d .git ]]; then
  git init
fi
pre-commit install

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

mkdir -p data/raw data/processed models
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep

python scripts/check_environment.py

echo
echo "Ready. Activate with: source .venv/bin/activate"
echo "Edit .env before deploying."
