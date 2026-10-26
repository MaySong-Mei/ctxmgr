"""Per-episode results table for all vLLM runs: one row per scored episode.

Usage: python scripts/episodes_table.py OUT.csv [PREVIOUS.csv]
Reads runs/<run>/<actor>/<condition>/<family>-<level>-s<seed>/ (episode.json, score.json,
calls.jsonl, edits.jsonl). Runs from before the harness logged `edit_prompt_version` (the draft
protocols) take their `edit_version` from PREVIOUS.csv when given.
"""

import csv
import json
import sys
from pathlib import Path

COLUMNS = [
    "job", "run", "edit_version", "actor", "manager", "condition", "family", "level", "seed",
    "served_max_len", "manager_max_tokens", "max_turns", "trigger", "temperature", "top_p", "top_k",
    "presence_penalty", "terminated_by", "actor_turns", "official_reward", "required_info_loss",
    "strict_exact_accuracy", "observed_peak_o200k", "mgmt_events", "events_applied",
    "manager_capped", "actor_tokens", "manager_tokens", "repo_commit",
]
MANAGER_ROLES = {"manager", "self_manager"}


def short(model):
    return model.rsplit("/", 1)[-1] if model else ""


def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".") if x != int(x) else f"{x:.1f}"
    return str(x)


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def row(run, ep_dir, previous):
    ep = json.loads((ep_dir / "episode.json").read_text(encoding="utf-8"))
    score_path = ep_dir / "score.json"
    if not score_path.exists():
        return None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    calls, edits = jsonl(ep_dir / "calls.jsonl"), jsonl(ep_dir / "edits.jsonl")
    sampling = ep.get("sampling") or {}
    diag = score.get("diagnostics") or {}
    key = (run, ep["condition"], ep["family"], ep["level"], str(ep["seed"]))
    edit_version = ep.get("edit_prompt_version") or ""
    if not edit_version and ep["condition"] != "none":
        edit_version = previous.get(key, "")
    commit = ep.get("repo_commit") or ""
    return {
        "job": run.split("-", 1)[1],
        "run": run,
        "edit_version": edit_version if ep["condition"] != "none" else "",
        "actor": short(ep.get("actor_model")),
        "manager": short(ep.get("manager_model")) if ep["condition"] != "none" else "",
        "condition": ep["condition"],
        "family": ep["family"],
        "level": ep["level"],
        "seed": ep["seed"],
        "served_max_len": ep.get("served_max_len"),
        "manager_max_tokens": ep.get("manager_max_tokens"),
        "max_turns": ep.get("max_turns"),
        "trigger": ep.get("manager_trigger_o200k"),
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "top_k": sampling.get("top_k"),
        "presence_penalty": sampling.get("presence_penalty"),
        "terminated_by": ep.get("terminated_by"),
        "actor_turns": len({c["turn"] for c in calls if c["role"] == "actor"}),
        "official_reward": round(score["official"]["reward"], 4),
        "required_info_loss": diag.get("required_info_loss"),
        "strict_exact_accuracy": diag.get("strict_exact_accuracy"),
        "observed_peak_o200k": diag.get("observed_peak_tokens"),
        "mgmt_events": len(edits),
        "events_applied": sum(bool(e.get("applied")) for e in edits),
        "manager_capped": sum(
            c["role"] in MANAGER_ROLES and c.get("finish_reason") == "length" for c in calls
        ),
        "actor_tokens": sum(
            c["prompt_tokens"] + c["completion_tokens"] for c in calls if c["role"] == "actor"
        ),
        "manager_tokens": sum(
            c["prompt_tokens"] + c["completion_tokens"] for c in calls if c["role"] in MANAGER_ROLES
        ),
        "repo_commit": commit[:7],
    }


def main():
    previous = {}
    if len(sys.argv) > 2:
        with open(sys.argv[2], newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = (r["run"], r["condition"], r["family"], r["level"], r["seed"])
                previous[key] = r["edit_version"]
    rows = []
    for run_dir in sorted(p for p in Path("runs").glob("*-*") if p.is_dir()):
        for ep_dir in sorted(run_dir.glob("*/*/*-s*")):
            if (ep_dir / "episode.json").exists():
                r = row(run_dir.name, ep_dir, previous)
                if r:
                    rows.append(r)
    with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r[k]) if isinstance(r[k], float) else ("" if r[k] is None else r[k]) for k in COLUMNS})
    print(f"wrote {len(rows)} rows to {sys.argv[1]}")


if __name__ == "__main__":
    main()
