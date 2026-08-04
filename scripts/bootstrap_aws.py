from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], env: dict[str, str]) -> str:
    print("+", " ".join(command))
    result = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout.strip())
    return result.stdout


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    env = os.environ.copy()
    env["AWS_PROFILE"] = args.profile
    env["AWS_REGION"] = args.region
    env["AWS_DEFAULT_REGION"] = args.region

    run(["aws", "sso", "login", "--profile", args.profile], env)
    identity = json.loads(
        run(
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--profile",
                args.profile,
                "--output",
                "json",
            ],
            env,
        )
    )
    account_id = identity["Account"]
    run(["npx", "cdk", "bootstrap", f"aws://{account_id}/{args.region}"], env)
    print(f"Bootstrapped account {account_id} in {args.region}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
