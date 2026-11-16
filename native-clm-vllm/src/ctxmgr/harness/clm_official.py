"""Configurable evaluation of official CLM and the team's original chat baseline.

Only scheduling, supported constructor parameters and provenance live here.
The released CLM loop, original none agent, task generator and grader are unchanged.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

from ctxmgr.tasks import generate_instance, validate_instance
from ctxmgr.tasks.core import BUDGET, FAMILIES, RESERVE, paper_config

from . import clm_upstream
from .clm_grid import LEVELS, checked_config, write_json
from .clm_settings import add_sampling_arguments, resolve_settings, settings_from_args
from .clm_vllm_models import MODEL_SIZES, model_identity

MODEL = "Qwen3.5-0.8B-BF16"  # The plan selects the actual model.
CONDITIONS = ("none", "clm_self")
BACKEND = "vllm"
SERVER_VERSION = "0.30.0"
PRECISION = "bfloat16"
MODEL_SHA256_KIND = "hf-snapshot-tree-sha256"
CLIENT_TRANSPORT = "reasoning-alias-v1"
TERMINAL_REASONS = ("stream_exhausted", "budget_wall", "submit", "max_steps")
DEFAULT_LEVELS = tuple(f"{family}:{level}" for family, levels in LEVELS.items() for level in levels)
TEAM_SOURCE_COMMIT = "863dce2953805747dfcf30653c6d710a590a0b2c"
HISTORICAL_TEAM_COMMIT = "4e4d004b7160c73a3d1cefca819f433967a77aac"
_TASK_SOURCE_SHA256 = {
    "tasks/__init__.py": "edd149c8e90b3182261fbe604ed7d040836c95e41ff45f78616cec5a45dd1ac7",
    "tasks/core.py": "e6b89b6a85b35f75c7530873422d2a0291ed82dc11a97757f25c0fd93f14e498",
    "tasks/scoring.py": "6b566d533e9e9ca4a8cc0e7a6f874a9c450a897df4fe8e3e213f4c2fbb23deef",
}
TEAM_SOURCE_BUNDLES = {
    HISTORICAL_TEAM_COMMIT: {
        **_TASK_SOURCE_SHA256,
        "harness/agents.py": "d85fe2e3e7707fd8e61586ad3440d1e48803a61b8e2803b86c62d5eb3e3f3226",
        "harness/management.py": "2074242a583d1a402aaec8ce0fe1ebf0ac643aa80df2e3884bf630e942103e14",
        "harness/episode.py": "2d75a16a3de43c2966daaf9d291021ccfdee79e62ac2b4c83febb96a4f04eabd",
    },
    TEAM_SOURCE_COMMIT: {
        **_TASK_SOURCE_SHA256,
        "harness/agents.py": "ea0aa27318cb529ceae75b3497ccce3c27bcbf619c6e36c15fe8b56a2147eb3b",
        "harness/management.py": "0aac62e1bc8257999a43c0414aaae501d001f188b1266b5b32852d1a891cc7ba",
        "harness/episode.py": "0f927c459cd135b7d3649cfe46d9ff166699a2f2b13dfccf8e60b519fca7ab78",
    },
}
# Compatibility for callers inspecting the intended current source pin.
TEAM_SOURCE_SHA256 = TEAM_SOURCE_BUNDLES[TEAM_SOURCE_COMMIT]


def verify_team_sources(root=None):
    """Accept a complete audited bundle, never a per-file mixture of releases."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    observed = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in TEAM_SOURCE_SHA256
    }
    for commit, expected in TEAM_SOURCE_BUNDLES.items():
        if observed == expected:
            return {"commit": commit, "sha256": observed}
    raise RuntimeError("Team source changed or mixes releases; inspect before running")


def _unique(values, label):
    values = list(values)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be nonempty with no duplicate entries")
    return values


def validate_model_manifest(model, *, alias=None):
    """Check the immutable HF identity and serving contract, not just an alias.

    The model preparation/worker separately verifies every snapshot file. The
    aggregate SHA-256 here identifies that canonical inventory, not one weight
    shard and not a historical GGUF file.
    """
    if not isinstance(model, dict):
        raise TypeError("Model manifest must be an object")
    observed_alias = model.get("model_alias")
    identities = {model_identity(size)[0]: model_identity(size) for size in MODEL_SIZES}
    if observed_alias not in identities or (alias is not None and observed_alias != alias):
        raise ValueError("Wrong model manifest alias")
    _, repository, revision = identities[observed_alias]
    for key, expected in {
        "repository": repository,
        "revision": revision,
        "backend": BACKEND,
        "server_version": SERVER_VERSION,
        "precision": PRECISION,
        "model_sha256_kind": MODEL_SHA256_KIND,
    }.items():
        if model.get(key) != expected:
            raise ValueError(f"Model manifest mismatch: {key}")
    if not isinstance(model.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", model["sha256"]
    ):
        raise ValueError("Model manifest has no valid snapshot tree digest")
    return model


def configuration(
    *,
    sizes=MODEL_SIZES,
    levels=DEFAULT_LEVELS,
    seeds=(42, 43, 44, 45),
    conditions=CONDITIONS,
    served_max_len=81920,
    settings=None,
):
    """Validate every choice before generating inputs or starting a model server."""
    from contextbench.cli import resolve_level

    sizes = _unique(sizes, "sizes")
    if any(size not in MODEL_SIZES for size in sizes):
        raise ValueError(f"Supported sizes: {MODEL_SIZES}")
    levels = _unique(levels, "levels")
    available = paper_config()
    for spec in levels:
        if not isinstance(spec, str) or not re.fullmatch(r"[a-z_]+:[A-Za-z0-9-]+", spec):
            raise ValueError(f"Expected family:level, got {spec!r}")
        family, level = spec.split(":")
        if family not in FAMILIES:
            raise ValueError(f"Supported task families: {FAMILIES}")
        resolve_level(available, family, level)
    seeds = _unique(seeds, "seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Task seeds must be nonnegative integers")
    conditions = _unique(conditions, "conditions")
    if any(condition not in CONDITIONS for condition in conditions):
        raise ValueError(f"Supported conditions: {CONDITIONS}")
    settings = settings or resolve_settings("official")
    overrides = settings["profile_overrides"]
    allowed = {"temperature", "top_p", "max_tokens", "enable_thinking"}
    if set(overrides) - allowed:
        raise ValueError("Only supported sampling parameters may override the official agent")
    normalized = resolve_settings(
        settings["profile"],
        **{
            ("thinking" if key == "enable_thinking" else key): value
            for key, value in overrides.items()
        },
    )
    if settings != normalized:
        raise ValueError("Resolved sampling settings are inconsistent")
    minimum = BUDGET + settings["sampling"]["max_tokens"]
    if (
        isinstance(served_max_len, bool)
        or not isinstance(served_max_len, int)
        or served_max_len < minimum
    ):
        raise ValueError(
            f"Served context must be at least {minimum}; tokenizer headroom may need more"
        )
    return {
        "sizes": sizes,
        "levels": levels,
        "seeds": seeds,
        "conditions": conditions,
        "served_max_len": served_max_len,
        "settings": settings,
    }


def specs(config=None):
    config = config or configuration()
    for spec in config["levels"]:
        family, level = spec.split(":")
        for seed in config["seeds"]:
            yield family, level, seed, f"{family}-{level}-s{seed}"


def _rows(root, config):
    for size in config["sizes"]:
        model = model_identity(size)[0]
        for family, level, seed, name in specs(config):
            for condition in config["conditions"]:
                yield {
                    "size": size,
                    "model": model,
                    "family": family,
                    "level": level,
                    "seed": seed,
                    "name": name,
                    "condition": condition,
                    "instance": str(root / "instances" / name),
                    "out": str(root / model / condition / name),
                }


def prepare(root, *, reference_instances=None, **kwargs):
    root = Path(root).resolve()
    checked_config()
    team_sources = verify_team_sources()
    config = configuration(**kwargs)
    if (root / "official-run.json").exists() or (root / "instances").exists():
        raise FileExistsError("Use a new evaluation run directory")
    reference = Path(reference_instances).resolve() if reference_instances is not None else None
    reference_manifests = {}
    if reference is not None:
        # Validate the entire selection before copying any instance. These are
        # host-side task files; neither runner exposes the reference tree to tools.
        from contextbench.cli import resolve_level

        for family, level, seed, name in specs(config):
            source = reference / name
            validate_instance(source)
            manifest = json.loads((source / "manifest.json").read_text())
            expected = {
                "family": family,
                "level": level,
                "seed": seed,
                "budget": BUDGET,
                "reserve": RESERVE,
                "knobs": resolve_level(paper_config(), family, level),
            }
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Reference instance configuration mismatch: {name}")
            reference_manifests[name] = manifest
    root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for family, level, seed, name in specs(config):
        target = root / "instances" / name
        if reference is None:
            manifests[name] = generate_instance(family, level, seed, target)
        else:
            shutil.copytree(reference / name, target)
            validate_instance(target)
            manifest = json.loads((target / "manifest.json").read_text())
            if manifest != reference_manifests[name]:
                raise ValueError(f"Reference instance changed while copying: {name}")
            manifests[name] = manifest
    rows = list(_rows(root, config))
    for row in rows:
        row["instance_sha256"] = manifests[row["name"]]["sha256"]
    settings = config["settings"]
    record = {
        "schema_version": 3,
        "config": config,
        "models": [model_identity(size)[0] for size in config["sizes"]],
        "profile": settings["profile"],
        "sampling": settings["sampling"],
        "agent_kwargs": settings["agent_kwargs"],
        "conditions": config["conditions"],
        "implementation": clm_upstream.AGENT_IMPORT,
        "budget": BUDGET,
        "reserve": RESERVE,
        "budget_ruler": "o200k",
        "budget_note": "Raw o200k in both arms; CLM_TOK_RATIO=0 disables released served-token calibration.",
        "backend": BACKEND,
        "server_version": SERVER_VERSION,
        "precision": PRECISION,
        "model_sha256_kind": MODEL_SHA256_KIND,
        "client_transport": CLIENT_TRANSPORT,
        "comparison": "official_clm_vs_original_team_chat_none",
        "baseline_implementation": "ctxmgr.harness.agents:NoManagementAgent",
        "protocols_identical": False,
        "protocol_differences": [
            "Official CLM prompt, skill, Bash/file tools versus append-only chat.",
            "Official CLM budget final turn versus chat no-call wall.",
            "Original runners keep their own reasoning-history behavior and turn limits.",
        ],
        "server_sampling_stack": "vllm-defaults:--generation-config=vllm",
        "episodes": rows,
        "team_source_commit": team_sources["commit"],
        "team_source_sha256": team_sources["sha256"],
        "reference_instances": str(reference) if reference is not None else None,
    }
    write_json(root / "official-run.json", record)
    return record


def load_plan(root):
    root = Path(root).resolve()
    data = json.loads((root / "official-run.json").read_text())
    if data.get("schema_version") != 3:
        raise ValueError(
            "Legacy run: use its saved results.txt; this entry cannot resume or rewrite old runs"
        )
    config = configuration(**data["config"])
    if config != data["config"]:
        raise ValueError("Noncanonical run configuration")
    for key, expected in {
        "profile": config["settings"]["profile"],
        "sampling": config["settings"]["sampling"],
        "agent_kwargs": config["settings"]["agent_kwargs"],
        "models": [model_identity(size)[0] for size in config["sizes"]],
        "conditions": config["conditions"],
        "budget": BUDGET,
        "reserve": RESERVE,
        "budget_ruler": "o200k",
        "backend": BACKEND,
        "server_version": SERVER_VERSION,
        "precision": PRECISION,
        "model_sha256_kind": MODEL_SHA256_KIND,
        "server_sampling_stack": "vllm-defaults:--generation-config=vllm",
        "client_transport": CLIENT_TRANSPORT,
    }.items():
        if data.get(key) != expected:
            raise ValueError(f"Run configuration mismatch: {key}")
    if (
        data.get("team_source_commit") not in TEAM_SOURCE_BUNDLES
        or data.get("team_source_sha256") != TEAM_SOURCE_BUNDLES[data["team_source_commit"]]
    ):
        raise ValueError("Run plan does not identify a complete audited team source bundle")
    expected = list(_rows(root, config))
    rows = data["episodes"]
    if len(rows) != len(expected):
        raise ValueError("Episode count differs from selected grid")
    for row, target in zip(rows, expected, strict=True):
        if any(row.get(key) != value for key, value in target.items()):
            raise ValueError("Episode settings/path differ from selected grid")
        hashes = row.get("instance_sha256")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Missing shared task hashes")
    for name in {row["name"] for row in rows}:
        hashes = [row["instance_sha256"] for row in rows if row["name"] == name]
        if any(value != hashes[0] for value in hashes):
            raise ValueError("Shared task hashes differ across compared cells")
    return data


def episode_argv(row, *, config, base_url, image, revision, sha256):
    argv = [
        "run",
        row["instance"],
        "--out",
        row["out"],
        "--condition",
        "clm_self",
        "--profile",
        config["settings"]["profile"],
        "--model",
        row["model"],
        "--base-url",
        base_url,
        "--image",
        image,
        "--environment",
        "singularity",
        "--backend",
        BACKEND,
        "--server-version",
        SERVER_VERSION,
        "--precision",
        PRECISION,
        "--model-revision",
        revision,
        "--model-sha256",
        sha256,
        "--served-max-len",
        str(config["served_max_len"]),
        "--budget-ruler",
        "o200k",
    ]
    for key, value in config["settings"]["profile_overrides"].items():
        if key == "enable_thinking":
            argv.append("--thinking" if value else "--no-thinking")
        else:
            argv.extend([f"--{key.replace('_', '-')}", str(value)])
    return argv


def summary(root, *, write_outputs=True):
    from clm_contextbench import PAPER_SKILL_SHA16

    root = Path(root).resolve()
    plan = load_plan(root)
    config = plan["config"]
    settings = config["settings"]
    rows = []
    for item in plan["episodes"]:
        out = Path(item["out"])
        errors = []

        def read(name, out=out, errors=errors):
            try:
                path = out / name
                record = json.loads(path.read_text()) if path.exists() else {}
                if record is not None and not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                return record or {}
            except (ValueError, OSError) as exc:
                errors.append(f"{name}: {exc}")
                return {}

        episode, result, score = read("episode.json"), read("result.json"), read("score.json")
        for record in (episode, result):
            if record.get("error"):
                errors.append(str(record["error"]))
        if episode:
            expected = {
                "condition": item["condition"],
                "profile": settings["profile"],
                "actor_model": item["model"],
                "family": item["family"],
                "level": item["level"],
                "seed": item["seed"],
                "budget": plan["budget"],
                "reserve": plan["reserve"],
                "budget_ruler": "o200k",
                "instance_sha256": item["instance_sha256"],
                "served_max_len": config["served_max_len"],
                "precision": plan["precision"],
                "backend": plan["backend"],
                "server_version": plan["server_version"],
                "model_sha256_kind": plan["model_sha256_kind"],
                "client_transport": plan["client_transport"],
                "team_source_commit": plan["team_source_commit"],
                "team_source_sha256": plan["team_source_sha256"],
            }
            if item["condition"] == "clm_self":
                expected.update(
                    {
                        "implementation": clm_upstream.AGENT_IMPORT,
                        "profile_overrides": settings["profile_overrides"],
                        "upstream_kwargs": settings["agent_kwargs"],
                        "sampling": {
                            **settings["sampling"],
                            "thinking": settings["sampling"]["enable_thinking"],
                        },
                    }
                )
            else:
                expected.update(
                    {
                        "implementation": "ctxmgr.harness.agents:NoManagementAgent",
                        "sampling": {
                            ("thinking" if key == "enable_thinking" else key): value
                            for key, value in settings["sampling"].items()
                        },
                    }
                )
            for key, value in expected.items():
                if episode.get(key) != value:
                    errors.append(f"Configuration mismatch: {key}")
            manifest_path = root / "models" / item["model"] / "model.json"
            if manifest_path.exists():
                try:
                    model = json.loads(manifest_path.read_text())
                    validate_model_manifest(model, alias=item["model"])
                    for key, manifest_key in (
                        ("model_revision", "revision"),
                        ("model_sha256", "sha256"),
                    ):
                        if episode.get(key) != model[manifest_key]:
                            errors.append(f"Model identity mismatch: {key}")
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    errors.append(f"Model manifest: {exc}")
            else:
                errors.append("Missing verified model manifest")
        official = score.get("official") or {}
        reward = official.get("reward") if isinstance(official, dict) else None
        if reward is not None and (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            errors.append("Invalid reward")
            reward = None
        reason = result.get("terminated_by")
        complete = reason in TERMINAL_REASONS and reward is not None
        if (result or score) and not episode:
            errors.append("Missing episode.json provenance")
        if score and not result:
            errors.append("Missing result.json termination")
        if result and reason not in TERMINAL_REASONS:
            errors.append(f"Incomplete/failed termination: {reason}")
        if result and episode and episode.get("terminated_by") != reason:
            errors.append("Episode and result termination disagree")
        if episode.get("status") in ("failed", "error"):
            errors.append("Episode records a failed status")
        if complete and item["condition"] == "clm_self":
            native = read("harbor/trial/agent/clm_stream.json")
            kwargs = native.get("kwargs", {})
            if any(
                key not in kwargs or kwargs[key] != value
                for key, value in settings["agent_kwargs"].items()
            ):
                errors.append("Native agent kwargs do not confirm resolved settings")
            if kwargs.get("model_name") != f"openai/{item['model']}":
                errors.append("Native model differs from planned model")
            if (
                native.get("skill") != "clm"
                or native.get("skill_sha16") != PAPER_SKILL_SHA16[item["family"]]
            ):
                errors.append("Native skill differs from official skill")
        status = (
            "error"
            if errors
            else "completed"
            if complete and episode
            else "running"
            if episode
            else "unrun"
        )
        rows.append(
            {
                "model": item["model"],
                "task": f"{item['family']}:{item['level']}",
                "seed": item["seed"],
                "condition": item["condition"],
                "status": status,
                "reward": reward if status == "completed" else None,
                "termination": reason,
                "ops": f"{result.get('n_ops_delivered')}/{result.get('n_ops_total')}",
                "edits": (episode.get("native_usage") or {}).get("n_ctx_syncs_real")
                if item["condition"] == "clm_self"
                else (0 if complete else None),
                "error": " | ".join(errors) or None,
            }
        )
    if not write_outputs:
        return rows
    write_json(root / "results.json", rows)
    lines = [
        f"{len(plan['models'])} model(s) x {len(config['levels'])} levels x {len(config['seeds'])} seed(s) x {len(config['conditions'])} condition(s) = {len(rows)} episodes.",
        f"Profile: {settings['profile']}; effective sampling: {json.dumps(settings['sampling'], sort_keys=True)}",
        "Shared sampling and task inputs; native prompts/tools/stop protocols differ.",
        "Completed means graded; budget_wall is early termination. Missing scores are not zeros.",
        "model\ttask\tseed\tcondition\tstatus\treward\ttermination\tops\tedits\terror",
    ]
    lines.extend(
        "\t".join(str(value) if value is not None else "-" for value in row.values())
        for row in rows
    )
    (root / "results.txt").write_text("\n".join(lines) + "\n")
    return rows


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="resolve arguments once and create shared instances")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--sizes", nargs="+", choices=MODEL_SIZES, default=list(MODEL_SIZES))
    prep.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    prep.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    prep.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    prep.add_argument("--served-max-len", type=int, default=81920)
    prep.add_argument(
        "--reference-instances",
        type=Path,
        help="Validate and copy selected instances from this shared suite directory",
    )
    add_sampling_arguments(prep, default_profile="official")
    for command in ("models", "settings", "records", "summary", "run"):
        cmd = commands.add_parser(command)
        cmd.add_argument("--root", type=Path, required=True)
        if command in ("records", "run"):
            cmd.add_argument("--model", required=True)
        if command == "run":
            cmd.add_argument("--name", required=True)
            cmd.add_argument("--condition", choices=CONDITIONS, required=True)
            cmd.add_argument("--base-url", required=True)
            cmd.add_argument("--image", required=True)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.root,
            sizes=args.sizes,
            levels=args.levels,
            seeds=args.seeds,
            conditions=args.conditions,
            served_max_len=args.served_max_len,
            settings=settings_from_args(args),
            reference_instances=args.reference_instances,
        )
        return 0
    if args.command == "summary":
        return int(any(row["status"] != "completed" for row in summary(args.root)))
    plan = load_plan(args.root)
    config = plan["config"]
    if args.command == "models":
        print("\n".join(plan["models"]))
    elif args.command == "settings":
        values = config["settings"]["sampling"]
        print(
            "\t".join(
                str(values[key]).lower()
                for key in ("temperature", "top_p", "max_tokens", "enable_thinking")
            )
        )
    elif args.command == "records":
        if args.model not in plan["models"]:
            parser.error("Model is not in this run plan")
        for row in plan["episodes"]:
            if row["model"] == args.model:
                print(f"{row['name']}\t{row['condition']}")
    else:
        team_sources = verify_team_sources()
        if team_sources["sha256"] != plan["team_source_sha256"]:
            raise ValueError("Team source bundle differs from the prepared plan")
        candidates = [
            row
            for row in plan["episodes"]
            if row["model"] == args.model
            and row["name"] == args.name
            and row["condition"] == args.condition
        ]
        if len(candidates) != 1:
            parser.error("Requested episode is not in this run plan")
        row = candidates[0]
        validate_instance(row["instance"])
        manifest = json.loads((Path(row["instance"]) / "manifest.json").read_text())
        if manifest["sha256"] != row["instance_sha256"]:
            raise ValueError("Planned shared task hashes changed")
        model = json.loads((args.root / "models" / args.model / "model.json").read_text())
        validate_model_manifest(model, alias=args.model)
        if args.condition == "none":
            from .clm_baseline_eval import run_baseline

            return run_baseline(
                Path(row["instance"]),
                Path(row["out"]),
                model=model,
                base_url=args.base_url,
                settings=config["settings"],
                served_max_len=config["served_max_len"],
            )
        argv = episode_argv(
            row,
            config=config,
            base_url=args.base_url,
            image=args.image,
            revision=model["revision"],
            sha256=model["sha256"],
        )
        write_json(
            args.root / "models" / args.model / f"command-{args.condition}-{args.name}.json", argv
        )
        try:
            return clm_upstream.main(argv)
        finally:
            # Host-only addition after the official runner; no prompt, tool,
            # stopping, score or request content is changed.
            metadata_path = Path(row["out"]) / "episode.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                metadata["team_source_commit"] = team_sources["commit"]
                metadata["team_source_sha256"] = team_sources["sha256"]
                metadata["model_sha256_kind"] = MODEL_SHA256_KIND
                metadata["client_transport"] = CLIENT_TRANSPORT
                write_json(metadata_path, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
