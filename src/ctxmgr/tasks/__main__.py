"""Generate/check official instances, run oracle self-checks, and score snapshots."""

import argparse
import json
from pathlib import Path

from contextbench import run
from contextbench.oracle import OracleAgent

from .core import BUDGET, FAMILIES, RESERVE, generate_instance, validate_instance
from .scoring import score_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("family", choices=FAMILIES)
    generate.add_argument("--level", required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--out", type=Path, required=True)
    for name in ("validate", "oracle", "score"):
        command = commands.add_parser(name)
        command.add_argument("instance", type=Path)
        if name != "validate":
            command.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        result = generate_instance(args.family, args.level, args.seed, args.out)
    elif args.command == "validate":
        result = validate_instance(args.instance)
    else:
        validate_instance(args.instance)
        if args.command == "oracle":
            result = run(
                OracleAgent(args.instance),
                args.instance,
                budget=BUDGET,
                reserve=RESERVE,
                out_dir=args.run_dir,
            )
            if result["reward"] != 1.0:
                raise RuntimeError(f"Oracle task self-check failed: {result}")
        result = score_run(args.instance, args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
