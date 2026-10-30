"""Per managed Needle cell: how often the manager protects needles when it edits, and the gain over `none`.

Usage: python scripts/mechanism_table.py OUT.md
Reads docs/tables/episodes_vllm.csv (H200 bf16, max_len 81,920, trigger 18,432) and each episode's
edits.jsonl. Per manager event: applied (at least one edit applied), protective (applied with a
`note` or `replace`, the two ways to keep needles while removing their block), unprotected delete
(applied with a `delete` and neither), empty (nothing applied: rejected reply, output cap, no edits).
Gain = cell mean reward - mean `none` reward for the same level, actor and decoding (pooled over jobs;
greedy `none` differs between jobs for 2 of 196 seed groups, by 0.04). Spearman rank correlations
are over cells with n >= 4.
"""

import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ("K4-C24", "K8-C48")
SKIP = {"none", "none_r", "rule_needle", "mask_arc"}


def episodes(path="docs/tables/episodes_vllm.csv"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if (
            r["family"] == "needle_retention"
            and r["level"] in LEVELS
            and r["served_max_len"] == "81920"
            and r["trigger"] in ("18432", "")
            and not r["condition"].endswith("_arc")
        ):
            r["decoding"] = "greedy" if r["temperature"] in ("0.0", "0") else "sampled"
            yield r


def events(r):
    d = Path("runs") / r["run"] / r["actor"] / r["condition"]
    d = d / f"needle_retention-{r['level']}-s{r['seed']}"
    lines = (d / "edits.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def main(out):
    rows = list(episodes())
    none = defaultdict(list)
    for r in rows:
        if r["condition"] == "none":
            none[r["level"], r["actor"], r["decoding"]].append(float(r["official_reward"]))
    cells = defaultdict(Counter)
    for r in rows:
        if r["condition"] in SKIP:
            continue
        key = (r["level"], r["actor"], r["condition"], r["manager"] or "-", r["edit_version"],
               r["decoding"])  # fmt: skip
        c = cells[key]
        c["n"] += 1
        c["reward"] += float(r["official_reward"])
        for e in events(r):
            ops = Counter(a["op"] for a in e["applied"])
            c["events"] += 1
            if not ops:
                c["empty"] += 1
                continue
            c["applied"] += 1
            if ops["note"] or ops["replace"]:
                c["protective"] += 1
            elif ops["delete"]:
                c["unprotected"] += 1
    lines = [
        "# Needle manager behaviour per cell (protective edits vs gain over `none`)",
        "",
        "Built by `scripts/mechanism_table.py` from `episodes_vllm.csv` and each episode's",
        "`edits.jsonl` (H200 bf16, max_len 81,920, trigger 18,432; archive conditions excluded).",
        "protective = applied events with a `note` or `replace`; unprotected = applied events that",
        "delete with neither; shares of applied events. empty = events with nothing applied, share of",
        "all events. gain = reward - `none` (same level, actor, decoding).",
        "",
    ]
    summary, table = [], []
    for level in LEVELS:
        xs, ys = [], []
        for key in sorted(k for k in cells if k[0] == level):
            c = cells[key]
            base = none.get((level, key[1], key[5]))
            if not base:
                continue
            reward = c["reward"] / c["n"]
            gain = reward - st.mean(base)
            app = c["applied"]
            prot = c["protective"] / app if app else None
            unprot = c["unprotected"] / app if app else None
            if prot is not None and c["n"] >= 4:
                xs.append(prot)
                ys.append(gain)
            table.append(
                f"| {' | '.join(key)} | {c['n']} | {reward:.2f} | {gain:+.2f} | "
                f"{'-' if prot is None else f'{prot:.2f}'} | "
                f"{'-' if unprot is None else f'{unprot:.2f}'} | "
                f"{c['empty'] / c['events']:.2f} | {c['events']} |"
                if c["events"]
                else f"| {' | '.join(key)} | {c['n']} | {reward:.2f} | {gain:+.2f} | - | - | - | 0 |"
            )
        summary.append(
            f"- {level}: Spearman(protective share, gain) = {spearman(xs, ys):.2f} over {len(xs)} "
            "cells with n >= 4 and at least one applied event."
        )
    lines += summary + [
        "",
        (
            "| level | actor | condition | manager | version | decoding | n | reward | gain | "
            "protective | unprotected | empty | events |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *table,
    ]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main(sys.argv[1])
