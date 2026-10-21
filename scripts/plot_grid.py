"""Figure: paired difference vs `none` for each managed condition, actor size and task.

Usage: python scripts/plot_grid.py OUT.png
Reads the same runs as scripts/pooled_grid.py. Small multiples instead of one colored chart:
rows = condition, columns = task, x = actor size, y = mean paired difference vs `none` with a
bootstrap 95% CI. One hue; identity is carried by panel position, not color.
"""

import random
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from pooled_grid import LEVELS, RUNS, score

BLUE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CONDS = ["self", "peer_same", "peer_27b"]
TITLES = {"needle_retention-K4-C24": "Needle K4-C24", "needle_retention-K8-C48": "Needle K8-C48",
          "kv_store-R520": "KV R520 (lenient)"}
ACTORS = list(RUNS)


def paired(actor, cond, level):
    data = {c: {} for c in ("none", cond)}
    for run in RUNS[actor]:
        for c in data:
            for seed in range(42, 50):
                v = score(run, actor, c, level, seed)
                if v is not None:
                    data[c][seed] = v
    seeds = sorted(set(data["none"]) & set(data[cond]))
    if not seeds:
        return None
    diffs = [data[cond][s] - data["none"][s] for s in seeds]
    random.seed(0)
    boots = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(5000))
    return st.mean(diffs), boots[125], boots[4875], len(seeds)


fig, axes = plt.subplots(3, 3, figsize=(10.5, 8), sharey=True, facecolor=SURFACE)
for i, cond in enumerate(CONDS):
    for j, level in enumerate(LEVELS):
        ax = axes[i][j]
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=MUTED, lw=1)
        for k, actor in enumerate(ACTORS):
            r = paired(actor, cond, level)
            if r is None:
                continue
            mean, lo, hi, n = r
            ax.plot([k, k], [lo, hi], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(k, mean, "o", color=BLUE, ms=8, mec=SURFACE, mew=2)
            ax.annotate(f"{mean:+.2f}", (k, hi), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(ACTORS)), [a.replace("Qwen3.5-", "") for a in ACTORS], color=INK)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        if i == 0:
            ax.set_title(TITLES[level], fontsize=11, color=INK)
        if j == 0:
            ax.set_ylabel(f"{cond}\nminus none", fontsize=10, color=INK)
fig.suptitle("Paired difference vs no management (mean, bootstrap 95% CI; 8 seeds per point)",
             fontsize=11, color=INK)
fig.text(0.5, 0.01, "actor size  ·  peer_27b = fixed 27B manager (none for a 27B actor)",
         ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(sys.argv[1], dpi=160, facecolor=SURFACE)
print("wrote", sys.argv[1])
