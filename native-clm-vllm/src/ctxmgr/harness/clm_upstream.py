"""Run the unchanged official ClmStream with Track 1 tasks and Track 3 serving.

    python -m ctxmgr.harness.clm_upstream run INSTANCE --out RUN --model MODEL --image IMAGE

No agent loop, edit parser, prompt, or manager is implemented here.
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from importlib.metadata import distribution, version
from pathlib import Path

from ctxmgr.tasks import validate_instance
from ctxmgr.tasks.core import BUDGET, FAMILIES, RESERVE, UPSTREAM_COMMIT, require_exact_tokens

from .clm_settings import (
    PROFILE_OVERRIDES,  # noqa: F401 - preserve the existing public import
    add_sampling_arguments,
    profile_overrides,  # noqa: F401 - preserve the existing public import
    settings_from_args,
    settings_from_overrides,
)

ASSETS = Path(__file__).with_name("clm_assets")
CLM_COMMIT = "483198229ab7b90c0eed4651306ab3b3505aaf50"
AGENT_IMPORT = "clm_contextbench:ClmStream"
CONDITIONS = {
    "clm_self": (AGENT_IMPORT, "clm"),
    "none_same_harness": ("ctxmgr.harness.clm_control:NoManagementStream", "none"),
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def verify_upstream():
    """Fail before allocating a container if code, prompts or tokenizer drifted."""
    # Keep LiteLLM's price lookup offline; this does not change model requests.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    require_exact_tokens()
    lock = json.loads((ASSETS / "upstream.lock.json").read_text())
    for package, commit in (("clm", CLM_COMMIT), ("contextbench", UPSTREAM_COMMIT)):
        direct = json.loads(distribution(package).read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != commit:
            raise RuntimeError(f"{package}: use scripts/clm.sh or uv sync --project envs/clm")
    for package, expected in lock["packages"].items():
        spec = importlib.util.find_spec(package)
        root = Path(spec.origin).parent
        actual = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.suffix in (".py", ".yaml")
        }
        if actual != expected:
            raise RuntimeError(f"Official {package} source differs from {CLM_COMMIT}")
    for relative, expected in lock["assets"].items():
        if hashlib.sha256((ASSETS / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned ContextBench asset changed: {relative}")
    from clm_contextbench import load_skill
    from clm_harness.utils import tokens
    from clm_harness.utils.provenance import code_fingerprint

    if tokens.count_tokens([{"role": "user", "content": "check"}])[1] != "tiktoken:o200k_base":
        raise RuntimeError("CLM also needs exact o200k_base; token fallback is forbidden")
    for family in FAMILIES:
        load_skill(family, "clm", ASSETS / "skills")  # Official paper-skill hash check.
    return {
        "clm_commit": CLM_COMMIT,
        "contextbench_commit": UPSTREAM_COMMIT,
        "code_fingerprint": code_fingerprint(),
        "verified": True,
    }


def prepare_task(instance, out, image):
    """Package the SAME validated instance, using the official Harbor task builder."""
    from contextbench.harbor.package import build_task

    instance, out = Path(instance), Path(out)
    validate_instance(instance)
    manifest = json.loads((instance / "manifest.json").read_text())
    task = build_task(
        instance,
        out,
        name="task",
        image=image,
        budget=BUDGET,
        reserve=RESERVE,
        regime=manifest["regime"],
        agent_timeout_sec=21600,
    )
    # No gold, manifest, solution or host repository is uploaded to the actor.
    visible = sorted(
        p.relative_to(task / "environment").as_posix()
        for p in (task / "environment").rglob("*")
        if p.is_file()
    )
    if visible != ["stream/chunks.jsonl"]:
        raise RuntimeError(f"Unexpected agent-visible task files: {visible}")
    return task


def trial_config(
    task,
    out,
    *,
    model,
    base_url,
    environment="singularity",
    profile="official",
    condition="clm_self",
    sampling_overrides=None,
):
    from harbor.models.trial.config import TrialConfig

    agent_import, skill = CONDITIONS[condition]
    settings = settings_from_overrides(profile, sampling_overrides)
    env = {"type": environment}
    if environment == "singularity":
        # Keep course/project/home mounts and inherited cwd outside the agent sandbox.
        env["kwargs"] = {"singularity_no_mount": "home,tmp,bind-paths,cwd,hostfs"}
    return TrialConfig.model_validate(
        {
            "task": {"path": str(Path(task).resolve())},
            "trial_name": "trial",
            "trials_dir": str(Path(out).resolve() / "harbor"),
            "agent": {
                "import_path": agent_import,
                "model_name": f"openai/{model}",
                "kwargs": {
                    "api_base": base_url,
                    "skill": skill,
                    "skills_dir": str(ASSETS / "skills"),
                    **settings["profile_overrides"],
                },
            },
            "environment": env,
            # Score on the host with Ruthvik's strict snapshot-backed scorer below.
            # Also keeps tests/gold.json out of the container for its entire lifetime.
            "verifier": {"disable": True},
        }
    )


def repo_state():
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "repo_commit": git("rev-parse", "HEAD"),
            "repo_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"repo_commit": None, "repo_dirty": None}


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


async def run_episode(args):
    from .clm_logs import collect_run

    provenance = verify_upstream()
    from clm_contextbench import PAPER_KWARGS
    from harbor.trial.trial import Trial

    settings = settings_from_args(args)
    overrides = settings["profile_overrides"]
    effective_kwargs = settings["agent_kwargs"]
    condition = getattr(args, "condition", "clm_self")
    agent_import, skill = CONDITIONS[condition]
    instance, out = args.instance.resolve(), args.out.resolve()
    validate_instance(instance)
    minimum_window = BUDGET + effective_kwargs["max_tokens"]
    if args.served_max_len is not None and args.served_max_len < minimum_window:
        raise ValueError(
            f"Server window must be at least {minimum_window}; "
            "served-model tokenization may require additional headroom over o200k"
        )
    # Supported upstream setting; explicitly distinguish team accounting from upstream defaults.
    os.environ["CLM_TOK_RATIO"] = "0" if args.budget_ruler == "o200k" else "1"
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    manifest = json.loads((instance / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        **{k: manifest[k] for k in ("family", "level", "seed")},
        "instance_sha256": manifest["sha256"],
        "condition": condition,
        "implementation": agent_import,
        "management_enabled": condition == "clm_self",
        "method_skill": skill,
        "actor_model": args.model,
        "manager_model": None,
        "profile": settings["profile"],
        "backend": args.backend,
        "server_version": args.server_version
        or (os.environ.get("CTXMGR_VLLM_VERSION") if args.backend == "vllm" else None),
        "precision": args.precision,
        "actor_precision": args.precision,
        "model_revision": args.model_revision,
        "model_sha256": args.model_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "budget": BUDGET,
        "reserve": RESERVE,
        "budget_ruler": args.budget_ruler,
        "manager_trigger_o200k": None,
        "manager_max_tokens": None,
        "edit_prompt_version": f"official-clm:{CLM_COMMIT}" if condition == "clm_self" else None,
        "edit_trigger": "self_initiated_bash" if condition == "clm_self" else "disabled",
        "edits_schema": "native-clm-receipts-v1",
        # Both profiles retain the official extra final turn at the budget wall.
        "budget_final_turn_policy": "official:error_finalize",
        "served_max_len": args.served_max_len,
        # Keep the native spelling, while exposing the team's read-only metadata alias.
        # This does not add an agent kwarg or change any request.
        "sampling": {
            **settings["sampling"],
            "thinking": settings["sampling"]["enable_thinking"],
        },
        "prompt_version": (
            f"official-clm:{CLM_COMMIT}"
            if condition == "clm_self"
            else f"matched-native-none:v1:{CLM_COMMIT}"
        ),
        "upstream_kwargs": effective_kwargs,
        "official_paper_kwargs": PAPER_KWARGS,
        "profile_overrides": overrides,
        "environment": args.environment,
        "image": args.image,
        "image_sha256": None,
        "versions": {p: version(p) for p in ("clm", "contextbench", "harbor", "litellm")},
        "openai_client_version": version("openai"),
        **provenance,
        **repo_state(),
        "terminated_by": "initializing",
    }
    if args.backend == "vllm":
        metadata["vllm_version"] = metadata["server_version"]
    if Path(args.image).is_file():
        metadata["image_sha256"] = await asyncio.to_thread(file_digest, args.image)
    write_json(out / "episode.json", metadata)
    failure = None
    trial = None
    try:
        task = prepare_task(instance, out / "harbor-task", args.image)
        config = trial_config(
            task,
            out,
            model=args.model,
            base_url=args.base_url,
            environment=args.environment,
            profile=settings["profile"],
            condition=condition,
            sampling_overrides=overrides,
        )
        write_json(out / "harbor-config.json", config.model_dump(mode="json"))
        if args.prepare_only:
            metadata["terminated_by"] = "prepared_only"
            write_json(out / "episode.json", metadata)
            return {"episode": metadata, "score": None}
        binary = "singularity" if args.environment == "singularity" else "docker"
        if shutil.which(binary) is None:
            raise RuntimeError(f"{binary} is missing; see docs/clm-reproduction.md")
        trial = await Trial.create(config)
        result = await trial.run()
        if result.exception_info:
            failure = result.exception_info.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001 - retain failed trials for team accounting
        failure = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
    finally:
        if trial is not None:
            # The official loop catches server context overflow and calls it a
            # budget wall. Observe its flag without changing any agent behavior.
            overflow = getattr(trial.agent, "_cb_overflow", None)
            if isinstance(overflow, bool):
                metadata["native_server_context_overflow"] = overflow
            if failure is not None:
                metadata["error"] = failure
            write_json(out / "episode.json", metadata)
    return collect_run(instance, out, metadata=metadata, failure=failure)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="verify exact installed upstream code and bundled skills")
    collect = commands.add_parser("collect", help="recover logs/score from an interrupted trial")
    collect.add_argument("instance", type=Path)
    collect.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="run official ClmStream; needs a container runtime")
    run.add_argument("instance", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--model", required=True, help="exact served name, without openai/ prefix")
    run.add_argument("--condition", choices=tuple(CONDITIONS), default="clm_self")
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--image", required=True, help="absolute .sif path, or Docker image")
    run.add_argument("--environment", choices=("singularity", "docker"), default="singularity")
    add_sampling_arguments(run)
    run.add_argument(
        "--budget-ruler",
        choices=("o200k", "upstream"),
        default="o200k",
        help="o200k: team's fixed ruler; upstream: released served-token calibration",
    )
    run.add_argument("--served-max-len", type=int)
    run.add_argument("--precision", default="unknown")
    run.add_argument("--backend", default="vllm", help="serving backend, e.g. vllm or llama.cpp")
    run.add_argument("--server-version", help="exact serving build/version")
    run.add_argument("--model-revision")
    run.add_argument("--model-sha256", help="SHA-256 of the served model artifact (GGUF)")
    run.add_argument(
        "--prepare-only", action="store_true", help="write config/task; no model calls"
    )
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    if args.command == "check":
        print(json.dumps(verify_upstream(), indent=2))
        return 0
    if args.command == "collect":
        from .clm_logs import collect_run

        result = collect_run(args.instance, args.out)
    else:
        result = asyncio.run(run_episode(args))
    score = result["score"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "terminated_by": result["episode"]["terminated_by"],
                "official_reward": score and score["official"]["reward"],
                "error": result["episode"].get("error"),
            },
            indent=2,
        )
    )
    return 1 if result["episode"].get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
