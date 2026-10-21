"""Pooled H200 bf16 grid: per actor x level x condition, mean score and paired test vs none.

Needle uses the official reward; KV uses the lenient score (scripts/lenient_kv.py).
"""
import json
import random
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, "scripts")
from lenient_kv import lenient_score

RUNS = {
    "Qwen3.5-0.8B": ["peer27b-30127434"],
    "Qwen3.5-2B": ["peer27b-30088179", "peer27b-30127435"],
    "Qwen3.5-4B": ["peer27b-30088182", "peer27b-30127436"],
    "Qwen3.5-9B": ["peer27b-30114789", "peer27b-30127437"],
    # Later runs overwrite earlier ones per (condition, seed): 27B `self` s42-45 comes from the
    # edit-v8 rerun 30118842, not the v7 run 30088184 (bare-edit format slip).
    "Qwen3.5-27B": ["baseline-30088184", "baseline-30118842", "baseline-30136264"],
}
LEVELS = ["needle_retention-K4-C24", "needle_retention-K8-C48", "kv_store-R520"]
CONDS = ["none", "self", "peer_same", "peer_27b"]


def score(run, actor, cond, level, seed):
    d = Path("runs") / run / actor / cond / f"{level}-s{seed}"
    if not (d / "score.json").exists():
        return None
    if level.startswith("kv_store"):
        return lenient_score(d, Path("runs") / run / "instances" / d.name)
    return json.loads((d / "score.json").read_text())["official"]["reward"]


rows = []
for actor, runs in RUNS.items():
    for level in LEVELS:
        data = {c: {} for c in CONDS}
        for run in runs:
            for cond in CONDS:
                for seed in range(42, 50):
                    v = score(run, actor, cond, level, seed)
                    if v is not None:
                        data[cond][seed] = v
        for cond in CONDS:
            if not data[cond]:
                continue
            mean = st.mean(data[cond].values())
            if cond == "none":
                rows.append((actor, level, cond, len(data[cond]), mean, None, None, None))
                continue
            seeds = sorted(set(data[cond]) & set(data["none"]))
            diffs = [data[cond][s] - data["none"][s] for s in seeds]
            pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            n = pos + neg
            k = max(pos, neg)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n) if n else 1.0
            random.seed(0)
            boots = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(10000))
            rows.append(
                (actor, level, cond, len(seeds), mean, st.mean(diffs), (boots[250], boots[9750]), (pos, neg, p))
            )

print("| actor | level | condition | n | mean | diff vs none | 95% CI | +/- | sign p (2-sided) |")
print("|---|---|---|---|---|---|---|---|---|")
for actor, level, cond, n, mean, diff, ci, test in rows:
    a = actor.replace("Qwen3.5-", "")
    lv = level.replace("needle_retention-", "Needle ").replace("kv_store-", "KV ")
    if diff is None:
        print(f"| {a} | {lv} | none | {n} | {mean:.2f} | | | | |")
    else:
        print(
            f"| {a} | {lv} | {cond} | {n} | {mean:.2f} | {diff:+.2f} | "
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}] | {test[0]}/{test[1]} | {test[2]:.3f} |"
        )
