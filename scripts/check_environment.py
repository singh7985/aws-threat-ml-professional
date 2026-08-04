from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def command_version(command: list[str]) -> tuple[bool, str]:
    if shutil.which(command[0]) is None:
        return False, "not installed"
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip().splitlines()
    return True, output[0] if output else "installed"


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--aws", action="store_true")
    args = parser.parse_args()

    checks: list[tuple[str, bool, str]] = [
        ("Python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0]),
        ("Virtual environment", sys.prefix != sys.base_prefix, sys.prefix),
        (".env exists", (PROJECT_ROOT / ".env").exists(), str(PROJECT_ROOT / ".env")),
    ]

    for label, command in [
        ("Git", ["git", "--version"]),
        ("Node.js", ["node", "--version"]),
        ("npm", ["npm", "--version"]),
        ("AWS CLI", ["aws", "--version"]),
        ("Docker", ["docker", "--version"]),
        ("Project CDK", ["npx", "cdk", "--version"]),
    ]:
        ok, details = command_version(command)
        checks.append((label, ok, details))

    if args.aws:
        ok, details = command_version(["aws", "sts", "get-caller-identity", "--output", "text"])
        checks.append(("AWS identity", ok, details))

    width = max(len(name) for name, _, _ in checks)
    failed = False
    for name, ok, details in checks:
        print(f"{'PASS' if ok else 'FAIL':4}  {name:<{width}}  {details}")
        failed = failed or not ok

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
