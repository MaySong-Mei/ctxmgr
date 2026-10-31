"""Figure: reward vs compute per Needle episode, one panel per level.

Usage: python scripts/plot_cost.py OUT.png
Reads docs/tables/cost_needle.md (built by scripts/cost_table.py). x = total prefix-reuse PFLOPs per
episode (actor + manager, log scale), y = mean reward. Color = who manages: a 27B or 9B peer manager
for a small actor, or the 27B model alone. Circles = one prompt wording each (no thinking),
triangles = manager thinking (v8). `none` and the task-aware rule are neutral. Every point is
labeled; the table has the values.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
LEVELS = ("K4-C24", "K8-C48")
TABLE = "docs/tables/cost_needle.md"

# (actor, condition) -> (group, color); groups with several wordings get one label.
GROUPS = {
    ("2B", "none"): ("2B, none", None),
    ("2B", "rule_needle"): ("2B + rule", INK),
    ("2B", "peer_9b"): ("2B + 9B manager", ORANGE),
    ("2B", "peer_27b"): ("2B + 27B manager", BLUE),
    ("4B", "peer_27b"): ("4B + 27B manager", BLUE),
    ("27B", "none"): ("27B, none", AQUA),
    ("27B", "self"): ("27B, self", AQUA),
    ("27B", "peer_same"): ("27B, peer_same", AQUA),
    ("27B", "rule_needle"): ("27B + rule", AQUA),
}
# Label offsets in points (dx, dy, ha) per (level, label).
OFFSETS = {
    ("K4-C24", "2B, none"): (-7, 8, "left"),
    ("K4-C24", "2B + rule"): (8, 0, "left"),
    ("K4-C24", "2B + 9B manager"): (0, -16, "center"),
    ("K4-C24", "2B + 9B manager, thinking"): (-8, 0, "right"),
    ("K4-C24", "2B + 27B manager"): (-9, -4, "right"),
    ("K4-C24", "2B + 27B manager, thinking"): (8, 5, "left"),
    ("K4-C24", "4B + 27B manager, thinking"): (8, 5, "left"),
    ("K4-C24", "27B, none"): (-8, -3, "right"),
    ("K4-C24", "27B, self"): (8, 0, "left"),
    ("K4-C24", "27B, peer_same"): (-8, 0, "right"),
    ("K4-C24", "27B + rule"): (0, -12, "center"),
    ("K8-C48", "2B, none"): (-7, 8, "left"),
    ("K8-C48", "2B + rule"): (8, 0, "left"),
    ("K8-C48", "2B + 9B manager"): (-9, -10, "right"),
    ("K8-C48", "2B + 9B manager, thinking"): (-8, 0, "right"),
    ("K8-C48", "2B + 27B manager"): (-9, 0, "right"),
    ("K8-C48", "2B + 27B manager, thinking"): (-8, 0, "right"),
    ("K8-C48", "4B + 27B manager, thinking"): (-8, 10, "right"),
    ("K8-C48", "27B, none"): (-8, 0, "right"),
    ("K8-C48", "27B + rule"): (0, -14, "center"),
}
# Which dot of a multi-wording cluster carries its label.
ANCHORS = {
    ("K4-C24", "2B + 9B manager"): "low",
    ("K4-C24", "2B + 27B manager"): "left",
    ("K8-C48", "2B + 9B manager"): "left",
    ("K8-C48", "2B + 27B manager"): "high",
}
PICK = {
    "low": lambda xy: min(xy, key=lambda p: p[1]),
    "high": lambda xy: max(xy, key=lambda p: p[1]),
    "left": min,
}


def cells(path=TABLE):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("| K"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        out.append(
            {
                "level": c[0],
                "actor": c[1],
                "cond": c[2],
                "version": c[4],
                "n": int(c[5]),
                "reward": float(c[6]),
                "pf": float(c[9]),
            }
        )
    return out


def points(level, rows):
    """label -> (color, marker, [(pf, reward), ...])."""
    pts = defaultdict(list)
    style = {}
    for r in rows:
        if r["level"] != level or (r["actor"], r["cond"]) not in GROUPS:
            continue
        group, color = GROUPS[r["actor"], r["cond"]]
        think = r["version"].endswith("+think")
        if r["actor"] == "4B" and not think:
            continue
        label = f"{group}, thinking" if think else group
        marker = "^" if think else "D" if r["cond"] == "rule_needle" else "o"
        pts[label].append((r["pf"], r["reward"]))
        style[label] = (color, marker)
    return {k: (*style[k], v) for k, v in pts.items()}


rows = cells()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, facecolor=SURFACE)
for ax, level in zip(axes, LEVELS, strict=True):
    ax.set_facecolor(SURFACE)
    for label, (color, marker, xy) in points(level, rows).items():
        xs, ys = zip(*xy, strict=True)
        if color is None:  # `none`: hollow ring
            ax.plot(xs, ys, marker, ms=8, mfc=SURFACE, mec=MUTED, mew=2, ls="none", zorder=3)
        else:
            ax.plot(
                xs,
                ys,
                marker,
                ms=9 if marker != "o" else 8,
                color=color,
                mec=SURFACE,
                mew=1.5,
                ls="none",
                zorder=3,
            )
        text = f"{label} ({len(xy)} wordings)" if len(xy) > 1 else label
        anchor = PICK[ANCHORS[level, label]](xy) if len(xy) > 1 else xy[0]
        dx, dy, ha = OFFSETS.get((level, label), (8, 0, "left"))
        ax.annotate(
            text,
            anchor,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            color=SECONDARY,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.08, 70)
    ax.set_ylim(-0.04, 1.08)
    ticks = [0.1, 0.3, 1, 3, 10, 30]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(f"Needle {level}", fontsize=11, color=INK)
    ax.set_xlabel("PFLOPs per episode (actor + manager, log scale)", fontsize=9.5, color=INK)
axes[0].set_ylabel("reward (mean over episodes)", fontsize=9.5, color=INK)

legend = [
    Line2D([], [], marker="o", ls="none", ms=8, color=BLUE, label="27B peer manager"),
    Line2D([], [], marker="o", ls="none", ms=8, color=ORANGE, label="9B peer manager"),
    Line2D([], [], marker="o", ls="none", ms=8, color=AQUA, label="27B model alone"),
    Line2D([], [], marker="^", ls="none", ms=9, color=SECONDARY, label="manager thinking"),
    Line2D(
        [], [], marker="o", ls="none", ms=8, mfc=SURFACE, mec=MUTED, mew=2, label="no management"
    ),
    Line2D([], [], marker="D", ls="none", ms=7, color=INK, label="task-aware rule"),
]
fig.legend(
    handles=legend,
    loc="lower center",
    ncol=6,
    frameon=False,
    fontsize=9,
    labelcolor=SECONDARY,
    bbox_to_anchor=(0.5, 0.0),
)
fig.suptitle("Reward vs compute per Needle episode", fontsize=12, color=INK, x=0.06, ha="left")
fig.text(
    0.06,
    0.905,
    "Prefix-reuse FLOPs with the prefix-cache hits vLLM measured; greedy, "
    "rounds 25-29 (docs/tables/cost_needle.md).\nEach peer-manager circle is one prompt "
    "wording without thinking; triangles are edit-v8 with manager thinking. 27B self and "
    "peer_same at K8-C48 predate cache logging.",
    fontsize=8.5,
    color=MUTED,
    ha="left",
    va="top",
)
fig.tight_layout(rect=(0, 0.06, 1, 0.86))
fig.savefig(sys.argv[1], dpi=160, facecolor=SURFACE)
print("wrote", sys.argv[1])
