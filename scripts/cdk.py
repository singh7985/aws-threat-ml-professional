from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AWS CDK using values from .env")
    parser.add_argument("action", choices=["synth", "diff", "deploy", "destroy", "list"])
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ.copy()

    command = ["npx", "cdk", args.action, *args.extra]
    print("+", " ".join(command))
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
