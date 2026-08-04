from __future__ import annotations

import argparse
from pathlib import Path

from threat_ml.generator import generate_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="threat-ml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate synthetic security events")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--normal", type=int, default=100)
    generate.add_argument("--suspicious", type=int, default=10)
    generate.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "generate":
        generate_events(
            output=args.output,
            normal=args.normal,
            suspicious=args.suspicious,
            seed=args.seed,
        )
        print(f"Generated {args.normal + args.suspicious} events at {args.output}")
        return 0

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
