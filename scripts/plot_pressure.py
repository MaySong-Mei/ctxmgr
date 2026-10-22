"""Figure: Needle reward vs environment pressure, none vs peer_27b, one panel per actor size.

Usage: python scripts/plot_pressure.py OUT.png
Pressure = the instance's environment input tokens / 32,768 (manifest `environment_pressure`).
Points are means over seeds with min-max whiskers; K4-C24 / K8-C48 use 8 seeds, the rest 4.
"""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRID, SURFACE = (
    "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb",
)
CURVE_RUNS = {  # pressure-curve levels, plus the core-grid runs for K4-C24 / K8-C48
    "Qwen3.5-0.8B": ["peer27b-30174977", "peer27b-30127434"],
    "Qwen3.5-2B": ["peer27b-30171619", "peer27b-30088179", "peer27b-30127435"],
    "Qwen3.5-9B": ["peer27b-30174978", "peer27b-30114789", "peer27b-30127437"],
}
LEVELS = ["K4-C4", "K4-C8", "K4-C24", "K8-C48", "K8-C60", "K4-C96"]
SERIES = [("none", "no management", ORANGE), ("peer_27b", "27B manager", BLUE)]


def collect(actor, cond, level):
    vals, pressure = {}, None
    for run in CURVE_RUNS[actor]:
        for d in (Path("runs") / run / actor / cond).glob(f"needle_retention-{level}-s*"):
            if not (d / "score.json").exists():
                continue
            seed = int(d.name.rsplit("-s", 1)[1])
            vals[seed] = json.loads((d / "score.json").read_text())["official"]["reward"]
            manifest = Path("runs") / run / "instances" / d.name / "manifest.json"
            pressure = json.loads(manifest.read_text())["environment_pressure"]
    return pressure, list(vals.values())


fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, facecolor=SURFACE)
for ax, actor in zip(axes, CURVE_RUNS):
    ax.set_facecolor(SURFACE)
    for cond, label, color in SERIES:
        xs, ys, lo, hi = [], [], [], []
        for level in LEVELS:
            p, v = collect(actor, cond, level)
            if v:
                xs.append(p)
                ys.append(st.mean(v))
                lo.append(st.mean(v) - min(v))
                hi.append(max(v) - st.mean(v))
        for x, y, a, b in zip(xs, ys, lo, hi):  # min-max whisker, thinner than the line
            ax.plot([x, x], [y - a, y + b], color=color, lw=1, alpha=0.45, solid_capstyle="round")
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=MUTED)
    ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_xlim(0.4, 25)
    ticks = [0.5, 1, 3, 6, 12]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(actor.replace("Qwen3.5-", "") + " actor", fontsize=11, color=INK)
    ax.set_xlabel("environment pressure (input / 32K budget, log)", fontsize=9, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
axes[0].set_ylabel("Needle reward (official)", fontsize=10, color=INK)
axes[0].legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK)
fig.suptitle("Needle retention vs pressure: no management vs fixed 27B manager "
             "(mean, light whisker = min-max over seeds; dashed line = budget)", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(sys.argv[1], dpi=160, facecolor=SURFACE)
print("wrote", sys.argv[1])
