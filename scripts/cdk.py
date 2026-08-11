from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Every stack in infrastructure/app.py is named ThreatMl<Name>-<environment>.
STACK_PREFIX = "ThreatMl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AWS CDK using values from .env")
    parser.add_argument("action", choices=["synth", "diff", "deploy", "destroy", "list"])
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ.copy()

    # cdk.json declares the app as bare "python infrastructure/app.py", which
    # resolves to whatever python is first on PATH -- usually not the project
    # virtualenv, so synthesis fails with ModuleNotFoundError: aws_cdk. Passing
    # the interpreter running this script makes the command work whether or not
    # the virtualenv happens to be activated.
    app = f"{sys.executable} infrastructure/app.py"

    # This app defines four stacks, so deploy/destroy refuse to run without a
    # target. Default to all of them, and let the caller narrow it by naming
    # stacks explicitly, rather than failing with a usage message.
    #
    # Stack names are matched by prefix rather than "the first non-flag
    # argument": that would read the value of a flag such as
    # `--require-approval never` as a stack name.
    targets = list(args.extra)
    has_target = "--all" in targets or any(
        item.startswith(STACK_PREFIX) or "*" in item for item in targets
    )
    if args.action in {"deploy", "destroy"} and not has_target:
        targets.insert(0, "--all")

    command = ["npx", "cdk", "--app", app, args.action, *targets]
    print("+", " ".join(command))
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
