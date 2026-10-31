"""One local 0.8B model, six seed-42 instances, native management on/off pairs."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from ctxmgr.tasks import generate_instance, validate_instance

from . import clm_upstream
from .clm_grid import LEVELS, checked_config, write_json

MODEL = "Qwen3.5-0.8B-Q8_0"
MODEL_REVISION = "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0"
MODEL_SHA = "0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"
CONDITIONS = ("none_same_harness", "clm_self")
COMMON_FIELDS = (
    "actor_model",
    "model_revision",
    "model_sha256",
    "backend",
    "server_version",
    "precision",
    "served_max_len",
    "profile",
    "sampling",
    "budget",
    "reserve",
    "budget_ruler",
    "budget_final_turn_policy",
    "upstream_kwargs",
    "clm_commit",
    "contextbench_commit",
    "code_fingerprint",
    "versions",
    "environment",
    "image_sha256",
    "instance_sha256",
)


def specs():
    for family, levels in LEVELS.items():
        for level in levels:
            yield family, level, f"{family}-{level}-s42"


def prepare(out):
    checked_config()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pilot.json").exists() or (out / "instances").exists():
        raise FileExistsError("Pilot already exists; use a new run directory")
    rows = []
    for index, (family, level, name) in enumerate(specs()):
        instance = out / "instances" / name
        manifest = generate_instance(family, level, 42, instance)
        # Alternate which condition goes first across levels; no claimed timing study.
        for condition in CONDITIONS if index % 2 == 0 else CONDITIONS[::-1]:
            rows.append(
                {
                    "family": family,
                    "level": level,
                    "seed": 42,
                    "name": name,
                    "condition": condition,
                    "instance": str(instance),
                    "instance_sha256": manifest["sha256"],
                    "out": str(out / condition / name),
                }
            )
    record = {
        "version": 1,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA,
        "seed": 42,
        "episodes": rows,
        "comparison": "same_native_harness_management_enabled_vs_disabled",
        "shared_stop_policy": "official:error_finalize",
        "historical_ruthvik_chat_none_equivalent": False,
        "treatment": [
            "management system-prompt section",
            "CLM method skill",
            "permission to synchronize context-file edits",
        ],
        "common_fields_checked": list(COMMON_FIELDS),
    }
    write_json(out / "pilot.json", record)
    return record


def load_pilot(out):
    root = Path(out).resolve()
    data = json.loads((root / "pilot.json").read_text())
    rows = data.get("episodes", [])
    expected = {(name, condition) for _, _, name in specs() for condition in CONDITIONS}
    if len(rows) != 12 or {(r["name"], r["condition"]) for r in rows} != expected:
        raise ValueError("Pilot must have exactly six seed-42 tasks x two conditions")
    if (data.get("model"), data.get("model_revision"), data.get("model_sha256")) != (
        MODEL,
        MODEL_REVISION,
        MODEL_SHA,
    ):
        raise ValueError("Pilot must use the downloaded, pinned 0.8B model")
    for row in rows:
        if Path(row["instance"]).resolve() != root / "instances" / row["name"]:
            raise ValueError("Instance path differs from pilot")
        if Path(row["out"]).resolve() != root / row["condition"] / row["name"]:
            raise ValueError("Episode path differs from pilot")
        if row["seed"] != 42:
            raise ValueError("Only seed 42 is allowed in this pilot")
    return data


def validate_run(args):
    root = args.out.resolve().parent.parent
    data = load_pilot(root)
    matching = [r for r in data["episodes"] if Path(r["out"]) == args.out.resolve()]
    if len(matching) != 1:
        raise ValueError("Output is not one of the 12 planned episodes")
    row = matching[0]
    if args.instance.resolve() != Path(row["instance"]) or args.condition != row["condition"]:
        raise ValueError("Task/condition differs from planned pair")
    validate_instance(args.instance)
    manifest = json.loads((args.instance / "manifest.json").read_text())
    if manifest["sha256"] != row["instance_sha256"]:
        raise ValueError("Shared task hashes changed")
    required = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA,
        "profile": "team-baseline",
        "backend": "llama.cpp",
        "server_version": "b11306",
        "precision": "gguf-Q8_0/llama.cpp-b11306",
        "served_max_len": 81920,
        "budget_ruler": "o200k",
        "environment": "singularity",
    }
    for key, expected in required.items():
        if getattr(args, key) != expected:
            raise ValueError(f"Paired pilot requires {key}={expected!r}")


def summary(out):
    root = Path(out).resolve()
    data = load_pilot(root)
    rows, metadata = [], {}
    for item in data["episodes"]:
        errors = []

        def read(name, item=item, errors=errors):
            path = Path(item["out"]) / name
            if not path.exists():
                return {}
            try:
                value = json.loads(path.read_text())
                if value is not None and not isinstance(value, dict):
                    raise ValueError("Expected object or null")
                return value or {}
            except (ValueError, OSError) as exc:
                errors.append(f"{name}: {exc}")
                return {}

        episode, result, score = read("episode.json"), read("result.json"), read("score.json")
        metadata[item["name"], item["condition"]] = episode
        for record in (episode, result):
            if record.get("error"):
                errors.append(json.dumps(record["error"], ensure_ascii=False))
        official = score.get("official") or {}
        if not isinstance(official, dict):
            errors.append("score.json: official must be an object")
            official = {}
        reward = official.get("reward")
        if reward is not None and (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            errors.append("Invalid official reward")
            reward = None
        reason = result.get("terminated_by") or episode.get("terminated_by")
        final = reason not in (None, "initializing", "prepared_only", "interrupted")
        status = (
            "error"
            if errors
            else (
                "completed"
                if result and final and reward is not None
                else "partial"
                if result
                else "running"
                if episode
                else "unrun"
            )
        )
        usage = episode.get("native_usage") or {}
        if not isinstance(usage, dict):
            errors.append("episode.json: native_usage must be an object")
            usage = {}
            status = "error"
        if item["condition"] == "none_same_harness":
            counters = ("n_ctx_syncs", "n_ctx_syncs_real")
            if any(usage.get(key, 0) != 0 for key in counters):
                errors.append("No-management invariant violated: context edit was applied")
                status = "error"
            elif status == "completed" and any(key not in usage for key in counters):
                errors.append("Cannot verify no-management invariant: edit counters missing")
                status = "error"
        rows.append(
            {
                "task": f"{item['family']}:{item['level']}",
                "seed": 42,
                "condition": item["condition"],
                "status": status,
                "official_reward": reward,
                "terminated_by": reason,
                "n_ops_delivered": result.get("n_ops_delivered"),
                "n_ops_total": result.get("n_ops_total"),
                "edits": usage.get("n_ctx_syncs_real"),
                "pair_settings": "pending",
                "partial_score": reward is not None and status != "completed",
                "error": " | ".join(dict.fromkeys(errors)) or None,
                "out": item["out"],
            }
        )
    parity = []
    for family, level, name in specs():
        left, right = [metadata[name, c] for c in CONDITIONS]
        comparable = bool(left and right)
        mismatches = (
            [
                key
                for key in COMMON_FIELDS
                if key not in left or key not in right or left[key] != right[key]
            ]
            if comparable
            else []
        )
        status = "mismatch" if mismatches else "matched" if comparable else "pending"
        parity.append({"task": f"{family}:{level}", "status": status, "mismatches": mismatches})
        for row in rows:
            if row["task"] == f"{family}:{level}":
                row["pair_settings"] = status
                if mismatches:
                    row["status"] = "error"
                    row["partial_score"] = row["official_reward"] is not None
                    row["error"] = (row["error"] or "") + f" Pair mismatch: {mismatches}"
    write_json(root / "pair-checks.json", parity)
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "0.8B native matched pilot: six tasks x seed 42 x two conditions = 12 episodes.",
        "No training. Missing/failed scores are not zeros; this is not the historical chat baseline.",
        "task\tcondition\tstatus\treward\ttermination\tops\tedits\tpair_settings\terror",
    ]
    for row in rows:
        score = "-" if row["official_reward"] is None else f"{row['official_reward']:.4f}"
        if row["partial_score"]:
            score += " (partial)"
        lines.append(
            "\t".join(
                str(v).replace("\n", " ")
                for v in (
                    row["task"],
                    row["condition"],
                    row["status"],
                    score,
                    row["terminated_by"] or "-",
                    f"{row['n_ops_delivered']}/{row['n_ops_total']}",
                    row["edits"],
                    row["pair_settings"],
                    row["error"] or "-",
                )
            )
        )
    (root / "results.txt").write_text("\n".join(lines) + "\n")
    return rows


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        args = clm_upstream.argument_parser().parse_args(argv)
        validate_run(args)
        return clm_upstream.main(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "records", "summary"):
        commands.add_parser(name).add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.out)
        print("Prepared 6 shared tasks and 12 paired episodes")
    elif args.command == "records":
        for row in load_pilot(args.out)["episodes"]:
            print("\t".join(row[key] for key in ("instance", "name", "condition", "out")))
    else:
        rows = summary(args.out)
        print(f"Results: {args.out / 'results.txt'}")
        return int(any(row["status"] == "error" for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
