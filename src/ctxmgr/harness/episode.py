"""Run one episode through the official ContextBench runner and write the run directory.

Layout follows `docs/log_schema.md`: ContextBench writes `context_snapshots/`, `result.json` and
`interactive_stream.json`; we add `episode.json`, `calls.jsonl`, `edits.jsonl` and `score.json`
(Track 1 scorer).
"""

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from contextbench import run

from ctxmgr.tasks import score_run, validate_instance
from ctxmgr.tasks.core import BUDGET, RESERVE, UPSTREAM_COMMIT

from .agents import ChatAgent, ServerRejected
from .management import EDIT_PROMPT_VERSION

MAX_TURNS_PER_OP = 2
MAX_TURNS_SLACK = 10


def _repo_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def run_episode(
    agent: ChatAgent,
    instance_dir: Path | str,
    out_dir: Path | str,
    *,
    condition: str,
    manager_model: str | None = None,
    served_max_len: int | None = None,
    precision: str | None = None,
    prompt_version: str = "contextbench-default",
    max_turns_per_op: int = MAX_TURNS_PER_OP,
) -> dict:
    """Run, log and score one episode. A server rejection is recorded, not raised."""
    instance_dir, out = Path(instance_dir), Path(out_dir)
    validate_instance(instance_dir)
    manifest = json.loads((instance_dir / "manifest.json").read_text(encoding="utf-8"))

    agent.wall = BUDGET - RESERVE
    # Safety stop. Without management the budget wall ends an actor that never writes the READY
    # marker; with management the context stays small, so such an actor loops forever (2B on KV R1000
    # under peer_27b ran 53,920 turns in job 30069167). Ends as `max_steps`, which is recorded.
    max_turns = max_turns_per_op * manifest["n_ops"] + MAX_TURNS_SLACK
    terminated_by, error = None, None
    try:
        result = run(
            agent, instance_dir, budget=BUDGET, reserve=RESERVE, out_dir=out, max_turns=max_turns
        )
        terminated_by = result["terminated_by"]
    except ServerRejected as rejected:
        # ContextBench writes no result.json when the agent raises; keep what we know.
        terminated_by, error = "server_rejected", str(rejected)[:2000]

    episode = {
        "family": manifest["family"],
        "level": manifest["level"],
        "seed": manifest["seed"],
        "instance_sha256": manifest["sha256"],
        "condition": condition,
        "actor_model": agent.model,
        "manager_model": manager_model,
        "precision": precision,
        "budget": BUDGET,
        "reserve": RESERVE,
        "manager_trigger_o200k": getattr(agent, "trigger", None),
        "manager_max_tokens": getattr(agent, "manager_max_tokens", None),
        "max_turns": max_turns,
        "served_max_len": served_max_len,
        "prompt_version": prompt_version,
        "edit_prompt_version": EDIT_PROMPT_VERSION if condition != "none" else None,
        "sampling": agent.sampling.__dict__,
        "terminated_by": terminated_by,
        "error": error,
        "repo_commit": _repo_commit(),
        "contextbench_commit": UPSTREAM_COMMIT,
        "openai_client_version": version("openai"),
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "episode.json", episode)
    with (out / "calls.jsonl").open("w", encoding="utf-8") as handle:
        for call in agent.calls:
            handle.write(json.dumps(call) + "\n")
    with (out / "edits.jsonl").open("w", encoding="utf-8") as handle:
        for edit in getattr(agent, "edits", []):
            handle.write(json.dumps(edit) + "\n")

    score = None
    if any((out / "context_snapshots").glob("turn-*.json")):
        score = score_run(instance_dir, out)
        _write_json(out / "score.json", score)
    return {"episode": episode, "score": score}
