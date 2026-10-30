"""Figure: share of protective manager edits vs gain over `none`, one point per managed Needle cell.

Usage: python scripts/plot_mechanism.py OUT.png
Reads docs/tables/mechanism_needle.md (built by scripts/mechanism_table.py). x = share of applied
manager events that `note` or `replace` (keep needles while removing their block), y = mean reward -
`none`. Color = size of the model that edits (the actor itself for `self`), an ordinal blue ramp;
shape = decoding (greedy, sampled, manager thinking). Cells with n < 4 or no applied edit are left out.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mechanism_table import LEVELS

INK, SECONDARY, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e6e5e0", "#fcfcfb"
RAMP = {"0.8B": "#86b6ef", "2B": "#5598e7", "4B": "#2a78d6", "9B": "#1c5cab", "27B": "#0d366b"}
SHAPES = {"greedy": "o", "sampled": "s", "thinking": "^"}
TABLE = "docs/tables/mechanism_needle.md"


def cells(path=TABLE):
    out, rho = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("- K"):
            rho[line[2:8]] = line.split("= ")[1].split(" ")[0]
        if not line.startswith("| K"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        level, actor, cond, manager, version, decoding, n = c[:7]
        if c[9] == "-" or int(n) < 4:
            continue
        editor = actor if cond == "self" else manager
        out.append(
            {
                "level": level,
                "size": editor.removeprefix("Qwen3.5-").removesuffix("-FP8"),
                "style": "thinking" if version.endswith("+think") else decoding,
                "gain": float(c[8]),
                "protective": float(c[9]),
            }
        )
    return out, rho


rows, rho = cells()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, facecolor=SURFACE)
for ax, level in zip(axes, LEVELS, strict=True):
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=MUTED, lw=1)
    for r in (r for r in rows if r["level"] == level):
        ax.plot(
            r["protective"],
            r["gain"],
            SHAPES[r["style"]],
            ms=8,
            color=RAMP[r["size"]],
            mec=SURFACE,
            mew=1.5,
            ls="none",
        )
    ax.text(0.02, 0.97, f"Spearman {rho[level]}", transform=ax.transAxes, va="top",
            fontsize=9, color=SECONDARY)  # fmt: skip
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.4, 0.8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(f"Needle {level}", fontsize=11, color=INK)
    ax.set_xlabel("share of applied manager edits that note or replace", fontsize=9.5, color=INK)
axes[0].set_ylabel("reward minus none", fontsize=9.5, color=INK)

legend = [
    Line2D([], [], marker="o", ls="none", ms=8, color=c, label=f"{s} editor")
    for s, c in RAMP.items()
] + [Line2D([], [], marker=m, ls="none", ms=8, color=SECONDARY, label=s) for s, m in SHAPES.items()]
fig.legend(handles=legend, loc="lower center", ncol=8, frameon=False, fontsize=9,
           labelcolor=SECONDARY, bbox_to_anchor=(0.5, 0.0))  # fmt: skip
fig.suptitle("Gain over no management rises with the share of protective edits", fontsize=12,
             color=INK, x=0.06, ha="left")  # fmt: skip
fig.text(0.06, 0.905, "One point per managed cell (condition x manager x prompt wording x decoding; "
         "n >= 4 episodes). Editor = the model that edits\n(the actor itself for self). "
         "docs/tables/mechanism_needle.md", fontsize=8.5, color=MUTED, ha="left", va="top")  # fmt: skip
fig.tight_layout(rect=(0, 0.06, 1, 0.86))
fig.savefig(sys.argv[1], dpi=160, facecolor=SURFACE)
print("wrote", sys.argv[1])
