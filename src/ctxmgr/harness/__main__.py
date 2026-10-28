"""Run episodes against a served model: `python -m ctxmgr.harness run ...`."""

import argparse
import json
import re
from pathlib import Path

import openai

from .agents import (
    ManagedAgent,
    MaskAgent,
    NoManagementAgent,
    PeerManagedAgent,
    RuleNeedleAgent,
    Sampling,
    SelfManagedAgent,
)
from .episode import MAX_TURNS_PER_OP, run_episode
from .management import EDIT_VARIANTS

CONDITIONS = {
    "none": NoManagementAgent,
    "self": SelfManagedAgent,
    "peer_same": PeerManagedAgent,
    "peer_27b": PeerManagedAgent,
    # Manager-size ladder: a fixed external manager of another size (served as MANAGER).
    "peer_4b": PeerManagedAgent,
    "peer_9b": PeerManagedAgent,
    # Archive-with-recall arms: removed blocks go to an archive the actor can RECALL from.
    # `none_r` is `none` with the archive instructions (a prompt control; nothing is ever archived).
    "none_r": NoManagementAgent,
    "self_arc": SelfManagedAgent,
    "peer_same_arc": PeerManagedAgent,
    "peer_27b_arc": PeerManagedAgent,
    "mask_arc": MaskAgent,
    # Task-aware rule for Needle (copies needle lines, deletes the rest): an upper-bound control.
    "rule_needle": RuleNeedleAgent,
}
ARCHIVE_CONDITIONS = {"none_r", "self_arc", "peer_same_arc", "peer_27b_arc", "mask_arc"}
UNMANAGED = {"none", "none_r"}


def manager_size_matches(condition: str, manager_model: str | None) -> bool:
    """`peer_9b` must be served by a `...-9B` model (so a run is never labelled with the wrong size)."""
    size = re.match(r"peer_(\d+(?:\.\d+)?b)(?:_|$)", condition)
    return size is None or (manager_model or "").lower().endswith(f"-{size.group(1)}")


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
    cmd.add_argument("--temperature", type=float, default=Sampling.temperature)
    cmd.add_argument("--top-p", type=float, default=Sampling.top_p)
    cmd.add_argument("--top-k", type=int, help="vLLM top_k (default: not sent)")
    cmd.add_argument("--presence-penalty", type=float, help="default: not sent")
    cmd.add_argument("--sampling-seed", type=int, help="per-request seed (default: not sent)")
    cmd.add_argument("--timeout", type=float, default=600.0, help="seconds per request")
    cmd.add_argument("--trigger", type=int, default=ManagedAgent.trigger, help="o200k tokens")
    cmd.add_argument(
        "--manager-max-tokens",
        type=int,
        default=ManagedAgent.manager_max_tokens,
        help="output budget of edit requests (self and peer)",
    )
    cmd.add_argument(
        "--max-turns-per-op",
        type=int,
        default=MAX_TURNS_PER_OP,
        help="safety stop: max_turns = this x n_ops + slack",
    )
    cmd.add_argument(
        "--edit-variant",
        choices=sorted(EDIT_VARIANTS),
        help="manager-prompt variant added to edit-v8 (default: none)",
    )
    cmd.add_argument("--manager-base-url", help="peer manager server (default: --base-url)")
    cmd.add_argument("--manager-model", help="peer manager model (peer_same: same as --model)")
    args = parser.parse_args()

    client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    kwargs = {}
    if args.condition in ARCHIVE_CONDITIONS:
        kwargs["use_archive"] = True
    if args.condition not in UNMANAGED:
        kwargs["trigger"] = args.trigger
        kwargs["manager_max_tokens"] = args.manager_max_tokens
        if args.edit_variant:
            kwargs["edit_variant"] = args.edit_variant
    if args.condition.startswith("peer"):
        if not args.condition.startswith("peer_same") and not args.manager_model:
            parser.error(f"{args.condition} needs --manager-model")
        if not manager_size_matches(args.condition, args.manager_model):
            parser.error(f"{args.condition} does not match manager {args.manager_model}")
        manager_url = args.manager_base_url or args.base_url
        kwargs["manager_client"] = openai.OpenAI(
            base_url=manager_url, api_key="EMPTY", timeout=args.timeout
        )
        kwargs["manager_model"] = args.manager_model or args.model
    agent = CONDITIONS[args.condition](
        client=client,
        model=args.model,
        sampling=Sampling(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            thinking=args.thinking,
            top_k=args.top_k,
            presence_penalty=args.presence_penalty,
            seed=args.sampling_seed,
        ),
        **kwargs,
    )
    outcome = run_episode(
        agent,
        args.instance,
        args.out,
        condition=args.condition,
        manager_model=kwargs.get("manager_model"),
        served_max_len=args.served_max_len,
        precision=args.precision,
        max_turns_per_op=args.max_turns_per_op,
    )
    score = outcome["score"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "terminated_by": outcome["episode"]["terminated_by"],
                "turns": len({c["turn"] for c in agent.calls}),
                "edits": len(getattr(agent, "edits", [])),
                "official_reward": score and score["official"]["reward"],
                "diagnostics": score and score["diagnostics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
