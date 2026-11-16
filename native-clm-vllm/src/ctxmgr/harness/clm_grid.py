"""Prepare Ruthvik's fixed task grid and report every native CLM episode.

This is host-side orchestration only. It never changes the task generator, grader,
CLM agent, or model-generated edits. Model manifests pin the bytes actually served.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from ctxmgr.tasks import generate_instance, validate_instance
from ctxmgr.tasks.core import BUDGET, RESERVE, UPSTREAM_COMMIT

SIZES = ("0.8B", "2B", "4B")
# Model manifests can also serve Ruthvik's 9B cell; the legacy grid stays 72 episodes.
MODEL_SIZES = (*SIZES, "9B")
SEEDS = (42, 43, 44, 45)
LEVELS = {
    "kv_store": ("R130", "R1000", "R2000"),
    "needle_retention": ("K4-C4", "K4-C24", "K8-C48"),
}
ARTIFACTS = ("chunks.jsonl", "gold.json", "meta.json", "public/stream.json")
CONFIG = Path(__file__).resolve().parents[3] / "configs/tasks-pilot.yaml"
BASELINE_COMMIT = "4e4d004b7160c73a3d1cefca819f433967a77aac"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_identity(size):
    if size not in MODEL_SIZES:
        raise ValueError(f"Expected one of {MODEL_SIZES}, got {size!r}")
    alias = f"Qwen3.5-{size}-Q8_0"
    return alias, f"unsloth/Qwen3.5-{size}-GGUF", f"{alias}.gguf"


def verify_model(manifest_path):
    record = read_json(manifest_path)
    if not isinstance(record, dict):
        raise TypeError("Model manifest must be an object")
    matching = [
        size for size in MODEL_SIZES if record.get("model_alias") == model_identity(size)[0]
    ]
    if len(matching) != 1:
        raise ValueError("Unrecognized model alias in manifest")
    _alias, repo_id, filename = model_identity(matching[0])
    if record.get("repo_id") != repo_id or record.get("filename") != filename:
        raise ValueError("Model repository/filename does not match its alias")
    for key, length in (("revision", 40), ("sha256", 64)):
        if not isinstance(record.get(key), str) or not re.fullmatch(
            f"[0-9a-f]{{{length}}}", record[key]
        ):
            raise ValueError(f"Malformed model {key}")
    raw_path = record.get("gguf_path")
    if not isinstance(raw_path, str) or any(c in raw_path for c in "\t\n\r"):
        raise ValueError("Invalid GGUF path")
    path = Path(raw_path)
    if not path.is_absolute() or path.name != filename or not path.is_file():
        raise ValueError("GGUF path must be an existing absolute path with the expected filename")
    if digest(path) != record["sha256"]:
        raise ValueError(f"Model SHA-256 mismatch: {path}")
    return record


def prepare_model(size, gguf, manifest_path):
    from huggingface_hub import HfApi, hf_hub_download

    alias, repo_id, filename = model_identity(size)
    gguf, manifest_path = Path(gguf).absolute(), Path(manifest_path).absolute()
    if gguf.name != filename or any(c in str(gguf) for c in "\t\n\r"):
        raise ValueError(f"Expected GGUF filename {filename}")
    if manifest_path.exists():
        record = verify_model(manifest_path)
        if record["model_alias"] != alias or Path(record["gguf_path"]) != gguf:
            raise ValueError("Existing manifest belongs to a different model/path")
        return record
    info = HfApi().model_info(repo_id, files_metadata=True)
    revision = info.sha
    if not isinstance(revision, str) or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Hugging Face did not resolve an immutable model revision")
    siblings = [item for item in info.siblings or [] if item.rfilename == filename]
    if len(siblings) != 1:
        raise ValueError(f"Expected exactly one {filename} at {repo_id}@{revision}")
    lfs = siblings[0].lfs
    upstream_hash = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if upstream_hash and not re.fullmatch("[0-9a-f]{64}", upstream_hash):
        raise ValueError("Malformed remote LFS SHA-256")
    gguf.parent.mkdir(parents=True, exist_ok=True)
    if gguf.exists() and upstream_hash:
        if digest(gguf) != upstream_hash:
            raise ValueError(
                f"Existing GGUF differs from {repo_id}@{revision}; preserve or move it first: {gguf}"
            )
    elif gguf.exists():
        # Without a remote content hash, a preexisting local file is not evidence of
        # its revision. Independently download the resolved revision and compare it.
        with TemporaryDirectory(dir=gguf.parent, prefix=".clm-model-verify-") as temporary:
            downloaded = Path(
                hf_hub_download(repo_id, filename, revision=revision, local_dir=temporary)
            )
            if digest(downloaded) != digest(gguf):
                raise ValueError("Existing GGUF differs from the resolved remote revision")
    else:
        downloaded = Path(
            hf_hub_download(repo_id, filename, revision=revision, local_dir=str(gguf.parent))
        )
        if downloaded.absolute() != gguf:
            raise ValueError("Hub download returned an unexpected local GGUF path")
    actual_hash = digest(gguf)
    if upstream_hash and actual_hash != upstream_hash:
        raise ValueError("Downloaded GGUF does not match the remote LFS SHA-256")
    record = {
        "revision": revision,
        "sha256": actual_hash,
        "gguf_path": str(gguf),
        "model_alias": alias,
        "repo_id": repo_id,
        "filename": filename,
        "remote_lfs_sha256": upstream_hash,
    }
    write_json(manifest_path, record)
    return record


def checked_config(path=CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "upstream_commit": UPSTREAM_COMMIT,
        "budget": BUDGET,
        "reserve": RESERVE,
        "seeds": list(SEEDS),
        "families": {family: {"levels": list(levels)} for family, levels in LEVELS.items()},
        "primary_metric": "official.reward",
        "grading_source": "snapshots",
    }
    if not isinstance(config, dict) or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("tasks-pilot.yaml differs from Ruthvik's fixed 6-level/4-seed grid")
    return config


def task_specs():
    for family, levels in LEVELS.items():
        for level in levels:
            for seed in SEEDS:
                yield family, level, seed, f"{family}-{level}-s{seed}"


def prepare_grid(out, reference_instances=None):
    checked_config()
    out = Path(out).resolve()
    reference = Path(reference_instances).resolve() if reference_instances else None
    out.mkdir(parents=True, exist_ok=True)
    if (out / "grid.json").exists() or (out / "instances").exists():
        raise FileExistsError(f"Grid or instances already exist; use a new run root: {out}")
    rows, parity = [], []
    with TemporaryDirectory(dir=out, prefix=".clm-grid-") as temporary:
        stage = Path(temporary) / "instances"
        stage.mkdir()
        for family, level, seed, name in task_specs():
            target = stage / name
            generated = generate_instance(family, level, seed, target)
            if reference:
                source = reference / name
                validate_instance(source)
                source_manifest = read_json(source / "manifest.json")
                if (
                    any(
                        source_manifest.get(k) != generated[k]
                        for k in (
                            "family",
                            "level",
                            "seed",
                            "budget",
                            "reserve",
                            "token_method",
                            "knobs",
                        )
                    )
                    or source_manifest["sha256"] != generated["sha256"]
                ):
                    raise ValueError(f"Reference task/hash mismatch for {name}")
                shutil.rmtree(target)
                shutil.copytree(source, target)
                validate_instance(target)
                if any(digest(target / p) != generated["sha256"][p] for p in ARTIFACTS):
                    raise ValueError(f"Copied reference task changed: {name}")
            parity.append(
                {
                    "name": name,
                    "reference_status": "exact_match" if reference else "not_compared",
                    "reference_instance": str(reference / name) if reference else None,
                    "sha256": generated["sha256"],
                }
            )
        stage.rename(out / "instances")
    for size in SIZES:
        alias = model_identity(size)[0]
        for family, level, seed, name in task_specs():
            rows.append(
                {
                    "size": size,
                    "model": alias,
                    "family": family,
                    "level": level,
                    "seed": seed,
                    "name": name,
                    "instance": str(out / "instances" / name),
                    "out": str(out / alias / "clm_self" / name),
                }
            )
    grid = {
        "baseline_commit": BASELINE_COMMIT,
        "config_sha256": digest(CONFIG),
        "upstream_commit": UPSTREAM_COMMIT,
        "budget": BUDGET,
        "reserve": RESERVE,
        "sizes": list(SIZES),
        "seeds": list(SEEDS),
        "condition": "clm_self",
        "episodes": rows,
        "task_parity": parity,
        "reference_instances": str(reference) if reference else None,
    }
    write_json(out / "grid.json", grid)
    with (out / "task-parity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "reference_status", *ARTIFACTS])
        writer.writeheader()
        for item in parity:
            writer.writerow({k: item[k] for k in ("name", "reference_status")} | item["sha256"])
    return grid


def load_grid(out):
    grid = read_json(Path(out) / "grid.json")
    expected = {(size, name) for size in SIZES for *_, name in task_specs()}
    rows = grid.get("episodes", [])
    if len(rows) != 72 or {(r.get("size"), r.get("name")) for r in rows} != expected:
        raise ValueError("Grid must enumerate exactly the expected 72 episodes")
    return grid


def summary(out):
    out = Path(out).resolve()
    grid = load_grid(out)
    rows = []
    for item in grid["episodes"]:
        errors = []
        episode_out = Path(item["out"])

        def read_record(path, errors=errors):
            if not path.exists():
                return {}
            try:
                value = read_json(path)
                if value is not None and not isinstance(value, dict):
                    raise ValueError("Expected JSON object or null")
                return value or {}
            except (ValueError, OSError) as exc:
                errors.append(f"{path.name}: {exc}")
                return {}

        stage = read_record(out / item["model"] / "status.json")
        episode = read_record(episode_out / "episode.json")
        result = read_record(episode_out / "result.json")
        score = read_record(episode_out / "score.json")
        for record in (episode, result):
            if record.get("error"):
                errors.append(json.dumps(record["error"], ensure_ascii=False))
        if episode.get("score_error"):
            errors.append(json.dumps(episode["score_error"], ensure_ascii=False))
        termination = result.get("terminated_by") or episode.get("terminated_by")
        official = score.get("official") or {}
        reward = official.get("reward") if isinstance(official, dict) else None
        if reward is not None and (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or not 0 <= reward <= 1
        ):
            errors.append("Invalid official reward")
            reward = None
        final = termination not in (None, "initializing", "prepared_only", "interrupted")
        if errors:
            status = "error"
        elif final and result and reward is not None:
            status = "completed"
        elif stage.get("outcome") in ("failed", "error", "interrupted"):
            status = "error" if episode or result else "blocked"
            if stage.get("outcome") == "interrupted" and (episode or result):
                status = "interrupted"
            errors.append(str(stage.get("error") or f"Model stage failed: {stage.get('stage')}"))
        elif stage.get("outcome") in ("completed", "success", "completed_with_errors"):
            status = "incomplete" if episode or result else "missing"
            errors.append(
                str(
                    stage.get("error")
                    or "Model job ended without a complete episode and valid score"
                )
            )
        elif episode or result:
            status = "running"
        else:
            status = "unrun"
        usage = episode.get("native_usage") or {}
        rows.append(
            {
                "model": item["model"],
                "task": f"{item['family']}:{item['level']}",
                "seed": item["seed"],
                "status": status,
                "official_reward": reward,
                "partial_score": reward is not None and status != "completed",
                "terminated_by": termination,
                "n_ops_delivered": result.get("n_ops_delivered"),
                "n_ops_total": result.get("n_ops_total"),
                "edits": usage.get("n_ctx_syncs_real"),
                "error": " | ".join(dict.fromkeys(errors)) or None,
                "out": str(episode_out),
            }
        )
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "All 72 episodes; '-' means unavailable, never an implicit zero.",
        "model\ttask\tseed\tstatus\treward\ttermination\tops\tedits\terror",
    ]
    for row in rows:
        reward = "-" if row["official_reward"] is None else f"{row['official_reward']:.4f}"
        if row["partial_score"]:
            reward += " (partial)"
        values = [
            row["model"],
            row["task"],
            row["seed"],
            row["status"],
            reward,
            row["terminated_by"] or "-",
            f"{row['n_ops_delivered']}/{row['n_ops_total']}",
            row["edits"] if row["edits"] is not None else "-",
            row["error"] or "-",
        ]
        lines.append("\t".join(str(value).replace("\n", " ") for value in values))
    lines.append("\nPer-model/per-level means use completed, error-free episodes only;")
    lines.append("coverage is shown out of 4 seeds; partial/error scores are retained above.")
    for size in SIZES:
        alias = model_identity(size)[0]
        for family, levels in LEVELS.items():
            for level in levels:
                valid = [
                    row["official_reward"]
                    for row in rows
                    if row["model"] == alias
                    and row["task"] == f"{family}:{level}"
                    and row["status"] == "completed"
                ]
                mean = f"{sum(valid) / len(valid):.4f}" if valid else "-"
                lines.append(f"{alias}\t{family}:{level}\tmean={mean}\tcoverage={len(valid)}/4")
    (out / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    model.add_argument("--size", required=True, choices=MODEL_SIZES)
    model.add_argument("--gguf", type=Path, required=True)
    model.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify-model")
    verify.add_argument("--manifest", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--reference-instances", type=Path)
    records = commands.add_parser("records")
    records.add_argument("--out", type=Path, required=True)
    records.add_argument("--size", required=True, choices=SIZES)
    report = commands.add_parser("summary")
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "model":
            print(json.dumps(prepare_model(args.size, args.gguf, args.out), indent=2))
        elif args.command == "verify-model":
            record = verify_model(args.manifest)
            print("\t".join(record[k] for k in ("model_alias", "revision", "sha256", "gguf_path")))
        elif args.command == "prepare":
            prepare_grid(args.out, args.reference_instances)
            print(f"Prepared 24 shared task instances and 72 episodes: {args.out / 'grid.json'}")
        elif args.command == "records":
            for row in load_grid(args.out)["episodes"]:
                if row["size"] == args.size:
                    print("\t".join(row[k] for k in ("instance", "name", "out")))
        else:
            summary(args.out)
            print(f"Results: {args.out / 'results.txt'}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"clm_grid: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
