"""BrowseComp-Plus agents and one-episode runner on the official ``ClmAgent`` (envs/clm).

    python -m ctxmgr.bcp.agent run --question-id Q --data DIR --out OUT --condition C \
        --model M --base-url URL --search-url URL --image SIF [--manager-model ...]

The search tools are exposed as shell commands, as in the CLM paper's BrowseComp-Plus
runs ("we expose the search tools as in-context skills"): ``/app/bin/bcp_search`` and
``/app/bin/bcp_get_document`` call a loopback search server that holds only the corpus.
The question is in the task instruction; the answer and gold documents never enter the
sandbox. The agent writes its final response to ``/app/answer.txt`` and submits.

Agent settings follow ``clm/clm_harness/configs/bcp.yaml`` (the released BrowseComp-Plus
config), except ``max_steps=100`` (the paper's 100-turn cap; editing turns stay free).
Conditions differ only in who decides when context is edited and who edits it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from clm_harness.clm_agent.harness import ClmAgent

from ctxmgr.harness.clm_whenhow import FixedTriggerMixin, ManagedMixin

APP = "/app"
ANSWER_FILE = f"{APP}/answer.txt"

BCP_KWARGS: dict[str, Any] = {
    "context_budget_tokens": 28672,
    "context_budget_reserve_tokens": 2048,
    "max_steps": 100,
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.95,
    "command_timeout": 120,
    "persistent_bash": True,
    "observation_max_chars": 60000,
    "nudge_ratios": "0.25,0.5,0.75",
    "persistent_nudge_ratio": 0.9,
    "max_num_retry_on_limit": 6,
    "retry_edit_margin_tokens": 2048,
    "finalize_nudge_turns": 3,
    "ctx_hint": "default",
    "task_template": "open_problems",
    "allow_edit_growth": "shrink",
    "enable_thinking": True,
    "send_chat_template_kwargs": True,
    "cost_metric": "usd",
}
NO_NUDGES = {"nudge_ratios": "", "persistent_nudge_ratio": None}

# condition -> (agent class name, extra kwargs, manager mode)
CONDITIONS: dict[str, tuple[str, dict, str | None]] = {
    "clm_self": ("BcpAgent", {}, None),
    "clm_self_nonudge": ("BcpAgent", NO_NUDGES, None),
    "clm_fixed": ("BcpFixed", NO_NUDGES, None),
    "mgr_fixed": ("BcpManaged", NO_NUDGES, "fixed"),
    "mgr_request": ("BcpManaged", NO_NUDGES, "request"),
    "mgr_watch": ("BcpManaged", NO_NUDGES, "watch"),
}

TOOL_SCRIPT = r'''#!/usr/bin/env python3
"""BrowseComp-Plus search tools (loopback corpus server)."""
import json, sys, urllib.parse, urllib.request
BASE = "__SEARCH_URL__"
def get(path, **params):
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return json.load(e)
if len(sys.argv) < 3 or sys.argv[1] not in ("search", "get_document"):
    sys.exit("usage: bcp_search QUERY | bcp_get_document DOCID")
if sys.argv[1] == "search":
    for i, hit in enumerate(get("/search", q=" ".join(sys.argv[2:])), 1):
        print(f"[{i}] docid={hit['docid']} score={hit['score']}\n{hit['snippet']}\n")
else:
    doc = get("/doc", docid=sys.argv[2].strip())
    print(doc.get("text") or doc.get("error"))
'''

INSTRUCTION = """You are a deep research agent. Answer the question below by searching a fixed
collection of web documents with these shell commands:

- `/app/bin/bcp_search "<query>"` prints the top 10 results: docid, score and the first
  512 tokens of each document.
- `/app/bin/bcp_get_document <docid>` prints one document (up to 8,192 tokens).

Reason and use the tools step by step, in an interleaved manner.

Question: {question}

When you have your answer, write your final response to `/app/answer.txt` in this format,
then submit:

Explanation: {{your explanation for your final answer, citing evidence docids in square
brackets [] at the end of sentences}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
"""


class BcpToolsMixin:
    """Installs the search commands and retrieves the answer file."""

    def __init__(self, *args: Any, search_url: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.search_url = search_url.rstrip("/")

    async def setup(self, environment: Any) -> None:
        await super().setup(environment)
        script = self.logs_dir / "bcp_tools.py"
        script.write_text(TOOL_SCRIPT.replace("__SEARCH_URL__", self.search_url))
        await environment.exec(command=f"mkdir -p {APP}/bin", timeout_sec=30)
        await environment.upload_file(str(script), f"{APP}/bin/bcp_tools.py")
        wrappers = (
            f"printf '#!/bin/sh\\nexec python3 {APP}/bin/bcp_tools.py search \"$@\"\\n' "
            f"> {APP}/bin/bcp_search && "
            f"printf '#!/bin/sh\\nexec python3 {APP}/bin/bcp_tools.py get_document \"$@\"\\n' "
            f"> {APP}/bin/bcp_get_document && chmod +x {APP}/bin/*"
        )
        await environment.exec(command=wrappers, timeout_sec=30)

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        try:
            await super().run(instruction, environment, context)
        finally:
            target = self.logs_dir / "answer.txt"
            try:
                await environment.download_file(ANSWER_FILE, str(target))
            except Exception:  # noqa: BLE001 - no answer file is a failed episode, not a crash
                target.write_text("")


class BcpAgent(BcpToolsMixin, ClmAgent):
    @staticmethod
    def name() -> str:
        return "clm-bcp"


class BcpFixed(BcpToolsMixin, FixedTriggerMixin, ClmAgent):
    @staticmethod
    def name() -> str:
        return "clm-bcp-fixed-trigger"


class BcpManaged(BcpToolsMixin, ManagedMixin, ClmAgent):
    @staticmethod
    def name() -> str:
        return "clm-bcp-managed"


def load_question(data: Path, query_id: str) -> dict:
    for line in (data / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row["query_id"]) == str(query_id):
            return row
    raise KeyError(f"query {query_id} is not in the prepared subset")


def build_task(question: str, out: Path, image: str) -> Path:
    task = out / "harbor-task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "instruction.md").write_text(INSTRUCTION.format(question=question.strip()))
    (task / "task.toml").write_text(
        "\n".join(
            [
                "[metadata]",
                'author_name = "ctxmgr"',
                'category = "deep-research"',
                'tags = ["browsecomp-plus"]',
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
                "[agent]",
                "timeout_sec = 21600.0",
                "",
                "[environment]",
                f'docker_image = "{image}"',
                f'workdir = "{APP}"',
                "build_timeout_sec = 600.0",
                "cpus = 1",
                "memory_mb = 2048",
                "storage_mb = 4096",
                "",
            ]
        )
    )
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    return task


def agent_kwargs(condition: str, args: argparse.Namespace) -> tuple[str, dict]:
    class_name, extra, mode = CONDITIONS[condition]
    kwargs = {**BCP_KWARGS, **extra, "api_base": args.base_url, "search_url": args.search_url}
    if mode is not None:
        kwargs.update(
            manager_mode=mode,
            manager_model=f"openai/{args.manager_model or args.model}",
            manager_api_base=args.manager_base_url or args.base_url,
        )
    return f"ctxmgr.bcp.agent:{class_name}", kwargs


async def run_episode(args: argparse.Namespace) -> dict:
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    row = load_question(args.data, args.question_id)
    import_path, kwargs = agent_kwargs(args.condition, args)
    metadata = {
        "benchmark": "browsecomp-plus",
        "query_id": row["query_id"],
        "condition": args.condition,
        "actor_model": args.model,
        "manager_model": kwargs.get("manager_model"),
        "agent": import_path,
        "agent_kwargs": {k: v for k, v in kwargs.items() if k not in ("api_base", "search_url")},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "terminated_by": "initializing",
    }
    (out / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
    task = build_task(row["query"], out, args.image)
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task)},
            "trial_name": "trial",
            "trials_dir": str(out / "harbor"),
            "agent": {
                "import_path": import_path,
                "model_name": f"openai/{args.model}",
                "kwargs": kwargs,
            },
            "environment": {
                "type": "singularity",
                "kwargs": {"singularity_no_mount": "home,tmp,bind-paths,cwd,hostfs"},
            },
            "verifier": {"disable": True},
        }
    )
    failure = None
    try:
        trial = await Trial.create(config)
        result = await trial.run()
        if result.exception_info:
            failure = result.exception_info.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - keep failed episodes in the denominator
        failure = {"exception_type": type(exc).__name__, "exception_message": str(exc)[:2000]}
    logs = out / "harbor" / "trial" / "agent"
    answer = next(iter(logs.rglob("answer.txt")), None) if logs.exists() else None
    metadata["answer_found"] = bool(answer and answer.read_text().strip())
    if answer is not None:
        (out / "answer.txt").write_text(answer.read_text())
    whenhow = next(iter(logs.rglob("whenhow.json")), None) if logs.exists() else None
    if whenhow is not None:
        metadata["whenhow"] = json.loads(whenhow.read_text()).get("summary")
    metadata["error"] = failure
    metadata["terminated_by"] = "error" if failure else "finished"
    (out / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--question-id", required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--search-url", required=True)
    run.add_argument("--image", required=True)
    run.add_argument("--manager-model")
    run.add_argument("--manager-base-url")
    args = parser.parse_args(argv)
    metadata = asyncio.run(run_episode(args))
    print(json.dumps({k: metadata[k] for k in ("query_id", "condition", "terminated_by")}))
    return 0 if not metadata["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
