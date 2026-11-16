"""Aggregate the planned CLM/none suite without treating absent or invalid scores as zero.

This is host-side reporting, run on a CPU allocation after the experiment jobs.
It validates saved provenance and uses the existing official summary; it does not
change snapshots, regrade them, or infer independent repetitions from task seeds.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from . import clm_official
from .clm_grid import write_json


def _key(row):
    task = row.get("task") or f"{row['family']}:{row['level']}"
    return row["model"], task, row["seed"], row["condition"]


def _object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _instance_hashes(plan):
    """Check the files on disk, rather than only comparing manifest declarations."""
    seen = set()
    for row in plan["episodes"]:
        root = Path(row["instance"])
        if root in seen:
            continue
        seen.add(root)
        for relative, expected in row["instance_sha256"].items():
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Task manifest contains a path outside its instance")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Task file hash mismatch: {row['name']}/{relative}")


def _model_identity(path, model):
    record = _object(path)
    if record.get("model_alias") != model:
        raise ValueError("Model manifest alias mismatch")
    for key, length in (("revision", 40), ("sha256", 64)):
        value = record.get(key)
        if not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
            raise ValueError(f"Malformed model {key}")
    return record["revision"], record["sha256"]


def _jobs(suite, plan):
    selected = {
        (size, seed) for size in plan["config"]["sizes"] for seed in plan["config"]["seeds"]
    }
    jobs = {}
    identifiers = set()
    with (suite / "jobs.tsv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["role", "size", "seed", "job_id"]:
            raise ValueError("jobs.tsv must have role, size, seed, job_id columns")
        for row in reader:
            if row["role"] not in {"setup", "run", "report"}:
                raise ValueError("Unknown job role")
            if not re.fullmatch(r"[0-9]+", row["job_id"] or ""):
                raise ValueError("Invalid job ID")
            if row["job_id"] in identifiers:
                raise ValueError("Duplicate job ID")
            identifiers.add(row["job_id"])
            if row["role"] != "run":
                continue
            key = row["size"], int(row["seed"])
            if key not in selected or key in jobs:
                raise ValueError("Unexpected or duplicate model/seed job")
            jobs[key] = row["job_id"]
    return jobs


def _validate_job(shared, plan, size, seed, shared_root, job_root):
    expected_config = {**shared["config"], "sizes": [size], "seeds": [seed]}
    if plan["config"] != expected_config:
        raise ValueError("Job configuration differs from suite settings")
    for key, value in shared.items():
        if (
            key not in {"config", "models", "episodes", "reference_instances"}
            and plan.get(key) != value
        ):
            raise ValueError(f"Job provenance differs from suite: {key}")
    if plan.get("reference_instances") != str(shared_root / "instances"):
        raise ValueError("Job did not declare the suite's shared task instances")
    expected = {
        _key(row): row for row in shared["episodes"] if row["size"] == size and row["seed"] == seed
    }
    actual = {_key(row): row for row in plan["episodes"]}
    if actual.keys() != expected.keys():
        raise ValueError("Job episode selection differs from suite")
    for key, row in actual.items():
        if row["instance_sha256"] != expected[key]["instance_sha256"]:
            raise ValueError("Job task hashes differ from suite")
    _instance_hashes(plan)
    model = plan["models"][0]
    identity = _model_identity(job_root / "models" / model / "model.json", model)
    if identity != _model_identity(shared_root / "models" / model / "model.json", model):
        raise ValueError("Job model revision/SHA differs from suite's prepared model")
    return identity


def _tsv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in fields})


def aggregate(suite):
    suite = Path(suite).resolve()
    shared_root = suite / "shared"
    shared = clm_official.load_plan(shared_root)
    _instance_hashes(shared)
    raw_project = (suite / "project-directory.txt").read_text(encoding="utf-8").strip()
    project = Path(raw_project)
    if not project.is_absolute() or "\n" in raw_project:
        raise ValueError("project-directory.txt must contain one absolute project path")
    jobs = _jobs(suite, shared)
    rows = []
    by_cell = defaultdict(list)
    for item in shared["episodes"]:
        row = {
            "size": item["size"],
            "model": item["model"],
            "task": f"{item['family']}:{item['level']}",
            "seed": item["seed"],
            "condition": item["condition"],
            "job_id": jobs.get((item["size"], item["seed"])),
            "status": "unrun",
            "reward": None,
            "termination": None,
            "ops": None,
            "edits": None,
            "error": None,
            "model_revision": None,
            "model_sha256": None,
        }
        rows.append(row)
        by_cell[item["size"], item["seed"]].append(row)
    for (size, seed), targets in by_cell.items():
        job_id = jobs.get((size, seed))
        if job_id is None:
            for row in targets:
                row["error"] = "No submitted job recorded for this model/seed"
            continue
        root = project / "runs" / f"clm-official-{job_id}"
        if not (root / "official-run.json").exists():
            error = "Job has not produced an episode plan"
            failed = False
            if (root / "job-status.json").exists():
                try:
                    state = _object(root / "job-status.json")
                    failed = state.get("status") == "failed"
                    error = state.get("error") or error
                except (ValueError, TypeError, OSError) as exc:
                    error, failed = str(exc), True
            for row in targets:
                row.update(status="failed" if failed else "unrun", error=error)
            continue
        try:
            plan = clm_official.load_plan(root)
            revision, sha256 = _validate_job(shared, plan, size, seed, shared_root, root)
            # Recompute metadata validation; do not trust a stale results.json.
            validated = {_key(row): row for row in clm_official.summary(root)}
            for row in targets:
                source = validated[_key(row)]
                row.update(
                    termination=source["termination"],
                    ops=source["ops"],
                    edits=source["edits"],
                    model_revision=revision,
                    model_sha256=sha256,
                )
                errors = [source["error"]] if source["error"] else []
                if source["status"] == "completed":
                    item = next(item for item in plan["episodes"] if _key(item) == _key(row))
                    episode = _object(Path(item["out"]) / "episode.json")
                    for key in ("team_source_commit", "team_source_sha256"):
                        if episode.get(key) != shared[key]:
                            errors.append(f"Episode source provenance differs from suite: {key}")
                if errors:
                    row.update(status="failed", error=" | ".join(errors))
                elif source["status"] == "completed":
                    row.update(status="graded", reward=source["reward"])
                elif source["status"] == "unrun":
                    row["error"] = "Episode has no runtime provenance"
                else:
                    row.update(status="failed", error="Episode started without a valid final grade")
        except (ValueError, KeyError, TypeError, OSError) as exc:
            # A job-level mismatch invalidates every score from that model/seed.
            for row in targets:
                row.update(status="failed", reward=None, error=str(exc))

    groups = defaultdict(list)
    comparisons = defaultdict(dict)
    for row in rows:
        groups[row["model"], row["task"], row["condition"]].append(row)
        comparisons[row["model"], row["task"]][row["seed"], row["condition"]] = row
    summary_rows = []
    for (model, task, condition), group in groups.items():
        graded = [row for row in group if row["status"] == "graded"]
        summary_rows.append(
            {
                "model": model,
                "task": task,
                "condition": condition,
                "expected": len(group),
                "graded": len(graded),
                "failed": sum(row["status"] == "failed" for row in group),
                "unrun": sum(row["status"] == "unrun" for row in group),
                "mean_reward_graded": mean(row["reward"] for row in graded) if graded else None,
                "stream_exhausted": sum(row["termination"] == "stream_exhausted" for row in graded),
                "budget_wall": sum(row["termination"] == "budget_wall" for row in graded),
                "submit": sum(row["termination"] == "submit" for row in graded),
            }
        )
    pair_rows = []
    both = {"none", "clm_self"}.issubset(shared["conditions"])
    for (model, task), group in comparisons.items():
        paired = []
        if both:
            for seed in shared["config"]["seeds"]:
                pair = [group[seed, condition] for condition in ("none", "clm_self")]
                if all(row["status"] == "graded" for row in pair):
                    paired.append((seed, pair[0]["reward"], pair[1]["reward"]))
        expected_pairs = len(shared["config"]["seeds"]) if both else 0
        pair_rows.append(
            {
                "model": model,
                "task": task,
                "expected_pairs": expected_pairs,
                "graded_pairs": len(paired),
                "excluded_pairs": expected_pairs - len(paired),
                "paired_task_seeds": ",".join(str(pair[0]) for pair in paired),
                "none_mean_paired": mean(pair[1] for pair in paired) if paired else None,
                "clm_mean_paired": mean(pair[2] for pair in paired) if paired else None,
                "clm_minus_none_paired": mean(pair[2] - pair[1] for pair in paired)
                if paired
                else None,
                "coverage": "complete"
                if both and len(paired) == expected_pairs
                else "partial"
                if paired
                else "unavailable",
            }
        )
    _tsv(suite / "results.tsv", list(rows[0]), rows)
    _tsv(suite / "summary.tsv", list(summary_rows[0]), summary_rows)
    _tsv(suite / "comparison.tsv", list(pair_rows[0]), pair_rows)
    write_json(suite / "results.json", rows)
    graded_count = sum(row["status"] == "graded" for row in rows)
    lines = [
        f"Validated grades: {graded_count}/{len(rows)} planned episodes.",
        f"Sampling profile: {shared['profile']}; {json.dumps(shared['sampling'], sort_keys=True)}",
        "Graded is not task completion: inspect stream_exhausted / budget_wall / submit.",
        "Missing, failed and provenance-mismatched scores are blank, never zero.",
        "summary.tsv means use graded episodes; comparison.tsv uses only matching graded seed pairs.",
        "Partial pairs are not the full-grid comparison. Coverage and denominators are explicit.",
        "Seeds vary task generation, not inference seeds. Shared task structure can correlate seeds; no independent-replicate confidence interval is inferred.",
        "Official CLM prompts, Bash/file tools and budget-final-turn rules differ from original team chat none.",
        "This validates saved metadata/task identities; it does not regrade snapshots or prove equality with historical teammate model bytes.",
        "model\ttask\tpaired/expected\tnone\tclm_self\tdelta",
    ]
    for row in pair_rows:

        def number(value):
            return "-" if value is None else f"{value:.4f}"

        lines.append(
            "\t".join(
                [
                    row["model"],
                    row["task"],
                    f"{row['graded_pairs']}/{row['expected_pairs']}",
                    number(row["none_mean_paired"]),
                    number(row["clm_mean_paired"]),
                    number(row["clm_minus_none_paired"]),
                ]
            )
        )
    (suite / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        suite / "report-status.json",
        {
            "status": "complete" if graded_count == len(rows) else "incomplete",
            "planned": len(rows),
            "graded": graded_count,
            "failed": sum(row["status"] == "failed" for row in rows),
            "unrun": sum(row["status"] == "unrun" for row in rows),
        },
    )
    return rows, summary_rows, pair_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows, _, _ = aggregate(args.suite)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        args.suite.mkdir(parents=True, exist_ok=True)
        # Never leave an earlier successful table beside a failed fresh validation.
        for name in ("results.tsv", "summary.tsv", "comparison.tsv", "results.json"):
            (args.suite / name).unlink(missing_ok=True)
        write_json(args.suite / "report-status.json", {"status": "failed", "error": str(exc)})
        (args.suite / "summary.txt").write_text(
            f"Report unavailable: {exc}\nNo score or denominator was inferred.\n", encoding="utf-8"
        )
        print(f"Report unavailable: {exc}")
        return 1
    print((args.suite / "summary.txt").read_text(encoding="utf-8"), end="")
    return int(any(row["status"] != "graded" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
