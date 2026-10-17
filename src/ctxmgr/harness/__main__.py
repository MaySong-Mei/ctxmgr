"""Run episodes against a served model: `python -m ctxmgr.harness run ...`."""

import argparse
import json
from pathlib import Path

import openai

from .agents import NoManagementAgent, Sampling
from .episode import run_episode

CONDITIONS = {"none": NoManagementAgent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("run", help="run one instance under one condition")
    cmd.add_argument("instance", type=Path)
    cmd.add_argument("--out", type=Path, required=True, help="new, empty run directory")
    cmd.add_argument("--condition", choices=sorted(CONDITIONS), default="none")
    cmd.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    cmd.add_argument("--model", required=True)
    cmd.add_argument("--served-max-len", type=int)
    cmd.add_argument("--precision")
    cmd.add_argument("--max-tokens", type=int, default=Sampling.max_tokens)
    cmd.add_argument("--thinking", action="store_true", help="enable Qwen3.5 thinking")
    cmd.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    agent = CONDITIONS[args.condition](
        client=client,
        model=args.model,
        sampling=Sampling(max_tokens=args.max_tokens, thinking=args.thinking),
    )
    outcome = run_episode(
        agent,
        args.instance,
        args.out,
        condition=args.condition,
        served_max_len=args.served_max_len,
        precision=args.precision,
    )
    score = outcome["score"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "terminated_by": outcome["episode"]["terminated_by"],
                "turns": len({c["turn"] for c in agent.calls}),
                "official_reward": score and score["official"]["reward"],
                "diagnostics": score and score["diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
