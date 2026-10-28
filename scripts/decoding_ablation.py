"""Decoding ablation: greedy vs Qwen3.5 recommended sampling, paired over the same instances.

Usage: python scripts/decoding_ablation.py > docs/tables/decoding_ablation.md
Greedy = the pooled core grid (scripts/pooled_grid.py RUNS). Sampled = temperature 0.7, top_p 0.8,
top_k 20, presence_penalty 1.5, request seed = instance seed. Needle: official reward; KV R520:
lenient score. CI = bootstrap 95%; p = two-sided sign test over non-zero differences.
"""

import random
import statistics as st
import sys
from math import comb

sys.path.insert(0, "scripts")
from pooled_grid import RUNS as GREEDY
from pooled_grid import score

SAMPLED = {
    "Qwen3.5-0.8B": ["peer27b-30271270"],
    "Qwen3.5-2B": ["peer27b-30271271"],
    "Qwen3.5-4B": ["peer27b-30265858"],
    "Qwen3.5-9B": ["peer27b-30265859"],
    "Qwen3.5-27B": ["baseline-30335778", "baseline-30691916"],
}
SMALL_LEVELS = ["needle_retention-K4-C24", "kv_store-R520"]
BIG_LEVELS = ["needle_retention-K4-C24", "needle_retention-K8-C48"]
SEEDS = range(42, 50)


def load(runs, actor, cond, level):
    out = {}
    for run in runs:  # later runs overwrite earlier ones per seed
        for seed in SEEDS:
            v = score(run, actor, cond, level, seed)
            if v is not None:
                out[seed] = v
    return out


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    diffs = [a[s] - b[s] for s in seeds]
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n, k = pos + neg, max(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n) if n else 1.0
    random.seed(0)
    boots = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(10000))
    return f"{st.mean(diffs):+.2f} [{boots[250]:+.2f}, {boots[9750]:+.2f}]", f"{pos}/{neg}", p, len(seeds)


def label(level):
    return level.replace("needle_retention-", "Needle ").replace("kv_store-", "KV ")


def main():
    print("# Decoding ablation: greedy vs recommended sampling\n")
    print(__doc__.split("\n", 3)[3].strip().replace("\n", " "), "\n")
    print("## Condition vs `none`\n")
    print("| actor | level | condition | greedy mean | sampled mean | diff vs none, greedy | +/- | "
          "diff vs none, sampled | +/- | sign p (sampled) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for actor in SAMPLED:
        big = actor.endswith("27B")
        conds = ["none", "self", "peer_same"] + ([] if big else ["peer_27b"])
        for level in BIG_LEVELS if big else SMALL_LEVELS:
            g = {c: load(GREEDY[actor], actor, c, level) for c in conds}
            s = {c: load(SAMPLED[actor], actor, c, level) for c in conds}
            if not s["none"]:
                continue
            size = actor.replace("Qwen3.5-", "")
            for c in conds:
                gm = f"{st.mean(g[c].values()):.2f}" if g[c] else ""
                sm = f"{st.mean(s[c].values()):.2f}" if s[c] else ""
                if c == "none":
                    print(f"| {size} | {label(level)} | none | {gm} | {sm} | | | | | |")
                    continue
                pg, ps = paired(g[c], g["none"]), paired(s[c], s["none"])
                print(f"| {size} | {label(level)} | {c} | {gm} | {sm} | {pg[0]} | {pg[1]} | "
                      f"{ps[0]} | {ps[1]} | {ps[2]:.3f} |")
    print("\n## 27B actor: role separation (`peer_same` - `self`)\n")
    print("| level | greedy | +/- | sampled | +/- | sign p (sampled) | n |")
    print("|---|---|---|---|---|---|---|")
    actor = "Qwen3.5-27B"
    for level in BIG_LEVELS:
        g = {c: load(GREEDY[actor], actor, c, level) for c in ("self", "peer_same")}
        s = {c: load(SAMPLED[actor], actor, c, level) for c in ("self", "peer_same")}
        pg, ps = paired(g["peer_same"], g["self"]), paired(s["peer_same"], s["self"])
        print(f"| {label(level)} | {pg[0]} | {pg[1]} | {ps[0]} | {ps[1]} | {ps[2]:.3f} | {ps[3]} |")


if __name__ == "__main__":
    main()
