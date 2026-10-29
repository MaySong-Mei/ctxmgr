"""Compute per episode and role on Needle: prefix-reuse FLOPs and GPU-seconds.

Usage: python scripts/cost_table.py OUT.md
Reads runs/*/<actor>/<condition>/needle_retention-*/ (episode.json, score.json, calls.jsonl) for
greedy bf16 runs at trigger 18,432 and max_len 81,920 whose calls all logged
`cached_prompt_tokens` (vLLM prompt_tokens_details; rounds 25-28). FLOPs: `ctxmgr.accounting.flops`
with the measured cache hits; the no-cache bound sets them to 0. GPU-seconds: summed call latency
(episodes run one after another in a job, actor and manager share one H200).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from ctxmgr.accounting.flops import CONSTANTS, call_flops, size_of

ROLES = ("actor", "manager")


def episode(d: Path):
    e = json.loads((d / "episode.json").read_text())
    if e.get("precision") != "bf16" or e.get("served_max_len") != 81920:
        return None
    if (e.get("sampling") or {}).get("temperature") not in (0, 0.0, None):
        return None
    if e.get("manager_trigger_o200k") not in (18432, None):
        return None
    calls = [json.loads(line) for line in (d / "calls.jsonl").read_text().splitlines() if line]
    if not calls or any(c.get("cached_prompt_tokens") is None for c in calls):
        return None
    acc = defaultdict(float)
    for c in calls:
        role = "actor" if c["role"] == "actor" else "manager"
        k = CONSTANTS[size_of(c["model"])]
        p, g, r = c["prompt_tokens"] or 0, c["completion_tokens"] or 0, c["cached_prompt_tokens"]
        acc[role] += call_flops(k, p, r, g)
        acc[role + "_nocache"] += call_flops(k, p, 0, g)
        acc[role + "_prompt"] += p
        acc[role + "_cached"] += r
        acc[role + "_s"] += c.get("latency_s") or 0
    cond = e["condition"]
    manager = size_of(e["manager_model"]) if cond.startswith("peer_") else ""
    acc["reward"] = json.loads((d / "score.json").read_text())["official"]["reward"]
    key = (
        e["level"],
        size_of(e["actor_model"]),
        cond,
        manager,
        "" if cond == "none" else e.get("edit_prompt_version") or "",
    )
    return key, (e["seed"], d.parts[-4]), acc


def main(out):
    cells = defaultdict(dict)
    for d in sorted(Path("runs").glob("*/Qwen3.5-*/*/needle_retention-*")):
        if (d / "episode.json").exists() and (got := episode(d)):
            key, seed_job, acc = got
            cells[key][seed_job] = acc
    lines = [
        "# Compute per Needle episode (prefix-reuse FLOPs, GPU-seconds)",
        "",
        "Built by `scripts/cost_table.py`. Means per episode over the listed jobs. PF = 10^15 FLOPs",
        "(`ctxmgr.accounting.flops`, measured prefix-cache hits); no-cache = same formula without cache",
        "hits; hit = cached / prompt tokens; GPU-s = summed call latency (actor and manager share one",
        "H200, one call at a time). Greedy, trigger 18,432, max_len 81,920; only runs that logged",
        "cached tokens (rounds 25-28). `none` ends at the budget wall after ~7 of 24-48 operations.",
        "",
        (
            "| level | actor | condition | manager | version | n | reward | actor PF | manager PF "
            "| total PF | manager share | no-cache PF | actor hit | manager hit | GPU-s actor "
            "| GPU-s manager | jobs |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = {s: i for i, s in enumerate(CONSTANTS)}
    for key in sorted(cells, key=lambda k: (k[0], order[k[1]], k[2] != "none", k[3], k[4])):
        level, actor, cond, manager, version = key
        eps = list(cells[key].values())
        n = len(eps)
        mean = defaultdict(float)
        for x in eps:
            for f, v in x.items():
                mean[f] += v / n

        def hit(role, mean=mean):
            p = mean[role + "_prompt"]
            return f"{mean[role + '_cached'] / p:.0%}" if p else "-"

        a, m = mean["actor"] / 1e15, mean["manager"] / 1e15
        nocache = (mean["actor_nocache"] + mean["manager_nocache"]) / 1e15
        jobs = ", ".join(sorted({j.split("-")[-1] for _, j in cells[key]}))
        lines.append(
            f"| {level} | {actor} | {cond} | {manager or '-'} | {version or '-'} | {n} "
            f"| {mean['reward']:.2f} | {a:.2f} | {m:.2f} | {a + m:.2f} | {m / (a + m):.0%} "
            f"| {nocache:.2f} | {hit('actor')} | {hit('manager')} | {mean['actor_s']:.0f} "
            f"| {mean['manager_s']:.0f} | {jobs} |"
        )
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(cells)} cells to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
