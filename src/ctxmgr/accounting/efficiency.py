"""Aggregate measured call latency/tokens; parameter-token figures are proxies, not FLOPs."""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def parameter_count(model):
    match = re.search(r"Qwen3\.5-(\d+(?:\.\d+)?)B", model or "")
    return float(match[1]) * 1e9 if match else None


def summarize_calls(calls):
    out = {}
    for group in ("actor", "manager", "total"):
        rows = [
            c for c in calls if group == "total" or (c["role"] == "actor") == (group == "actor")
        ]
        prompt = sum(c.get("prompt_tokens", 0) for c in rows)
        completion = sum(c.get("completion_tokens", 0) for c in rows)
        known = all(c.get("cached_prompt_tokens") is not None for c in rows)
        cached = sum(c.get("cached_prompt_tokens") or 0 for c in rows)
        params_known = all(parameter_count(c.get("model")) is not None for c in rows)
        raw = sum(
            (parameter_count(c.get("model")) or 0)
            * (c.get("prompt_tokens", 0) + c.get("completion_tokens", 0))
            for c in rows
        )
        reused = sum(
            (parameter_count(c.get("model")) or 0)
            * (
                c.get("prompt_tokens", 0)
                - (c.get("cached_prompt_tokens") or 0)
                + c.get("completion_tokens", 0)
            )
            for c in rows
        )
        out.update(
            {
                f"{group}_calls": len(rows),
                f"{group}_prompt_tokens": prompt,
                f"{group}_completion_tokens": completion,
                f"{group}_cached_tokens": cached if known else None,
                f"{group}_api_latency_s": sum(c.get("latency_s", 0) for c in rows),
                f"{group}_param_tokens_no_cache": raw if params_known else None,
                f"{group}_param_tokens_cache_adjusted": reused if known and params_known else None,
            }
        )
    return out


def episode_row(path, lenient=False):
    e = json.loads(path.read_text())
    run = path.parent
    calls = [json.loads(line) for line in (run / "calls.jsonl").read_text().splitlines()]
    score_path = run / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    row = {
        k: e.get(k)
        for k in (
            "family",
            "level",
            "seed",
            "condition",
            "actor_model",
            "manager_model",
            "precision",
            "terminated_by",
            "repo_commit",
            "contextbench_commit",
            "edit_prompt_version",
            "runner_elapsed_s",
            "evaluation_elapsed_s",
        )
    }
    row.update(
        run=str(run),
        instance_sha256=json.dumps(e.get("instance_sha256"), sort_keys=True),
        temperature=e.get("sampling", {}).get("temperature"),
        official_reward=score.get("official", {}).get("reward"),
        **summarize_calls(calls),
    )
    row["lenient_kv"] = None
    if lenient and e.get("family") == "kv_store":
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location(
            "lenient", Path(__file__).parents[3] / "scripts/lenient_kv.py"
        )
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        roots = [p for p in run.parents if (p / "instances").is_dir()]
        if roots:
            inst = roots[0] / "instances" / f"kv_store-{e['level']}-s{e['seed']}"
            if inst.exists():
                row["lenient_kv"] = module.lenient_score(run, inst)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--lenient", action="store_true")
    args = parser.parse_args()
    rows, errors = [], []
    paths = sorted({p for root in args.roots for p in root.rglob("episode.json")})
    for p in paths:
        try:
            rows.append(episode_row(p, args.lenient))
        except (OSError, ValueError, KeyError) as exc:
            errors.append({"path": str(p), "error": str(exc)})
    args.out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.out / "episodes.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(
            row[k]
            for k in (
                "family",
                "level",
                "actor_model",
                "condition",
                "precision",
                "temperature",
                "edit_prompt_version",
                "repo_commit",
            )
        )
        grouped[key].append(row)
    summaries = []
    for key, members in grouped.items():
        item = dict(
            zip(
                (
                    "family",
                    "level",
                    "actor_model",
                    "condition",
                    "precision",
                    "temperature",
                    "edit_prompt_version",
                    "repo_commit",
                ),
                key,
            )
        )
        item["n"] = len(members)
        item["terminations"] = {
            t: sum(r["terminated_by"] == t for r in members)
            for t in sorted({r["terminated_by"] for r in members})
        }
        for field in (
            "official_reward",
            "lenient_kv",
            "runner_elapsed_s",
            "total_api_latency_s",
            "total_param_tokens_no_cache",
            "total_param_tokens_cache_adjusted",
        ):
            values = [r[field] for r in members if r[field] is not None]
            item[field + "_mean"] = statistics.mean(values) if values else None
            item[field + "_n"] = len(values)
        summaries.append(item)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (args.out / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    (args.out / "README.md").write_text(
        "# Efficiency extraction\n\nAll failed/partial scored episodes are retained. "
        "API latency is the sum of observed request durations, not total job GPU time. "
        "Runner time excludes startup and evaluation and is available only for instrumented runs. "
        "Parameter × token measures use nominal model sizes and are approximate compute proxies, "
        "not architecture-aware FLOPs or dollar costs. Cache-adjusted values are null when "
        "cache reporting is missing. Do not pool differing hardware/protocols/revisions. "
        "Compare quality before cost; a budget-wall failure is not a cheap successful task.\n"
    )
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "groups": len(summaries),
                "errors": len(errors),
                "out": str(args.out),
            }
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
