"""Per managed Needle cell: paired difference vs `none` with a seed-level bootstrap CI and sign test.

Usage: python scripts/paired_table.py OUT.md
Same cells as scripts/mechanism_table.py. Each cell's rewards are first averaged per seed (a seed
run in two jobs counts once), then paired with that seed's mean `none` reward for the same level,
actor and decoding. 95% CI: percentile bootstrap over seeds (10,000 resamples, fixed RNG seed).
Sign test: two-sided exact binomial over seeds with a nonzero difference. Seeds 42-49 share each
level's layout (only the words differ), so the CI covers instance noise, not layout variation.
"""

import random
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

from mechanism_table import LEVELS, SKIP, episodes


def sign_p(pos, neg):
    n, k = pos + neg, max(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2**n) if n else 1.0


def main(out):
    rows = list(episodes())
    none, cells = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in rows:
        reward = float(r["official_reward"])
        if r["condition"] == "none":
            none[r["level"], r["actor"], r["decoding"], r["seed"]].append(reward)
        elif r["condition"] not in SKIP:
            key = (r["level"], r["actor"], r["condition"], r["manager"] or "-", r["edit_version"],
                   r["decoding"])  # fmt: skip
            cells[key][r["seed"]].append(reward)
    lines = [
        "# Needle: paired difference vs `none` per managed cell, with 95% CIs",
        "",
        "Built by `scripts/paired_table.py`. Rewards averaged per seed, paired with the seed's",
        "mean `none` (same level, actor, decoding). CI = percentile bootstrap over seeds; p = two-sided",
        "sign test over seeds. Seeds share each level's layout, so CIs cover instance noise only.",
        "",
        (
            "| level | actor | condition | manager | version | decoding | seeds | episodes | reward "
            "| diff vs none | 95% CI | +/- | sign p |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(cells, key=lambda k: (LEVELS.index(k[0]), *k[1:])):
        level, actor, *_, decoding = key
        by_seed = cells[key]
        seeds = sorted(s for s in by_seed if (level, actor, decoding, s) in none)
        if not seeds:
            continue
        means = [st.mean(by_seed[s]) for s in seeds]
        diffs = [
            m - st.mean(none[level, actor, decoding, s]) for m, s in zip(means, seeds, strict=True)
        ]
        rng = random.Random(0)
        boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        episodes_n = sum(len(by_seed[s]) for s in seeds)
        lines.append(
            f"| {' | '.join(key)} | {len(seeds)} | {episodes_n} | {st.mean(means):.2f} "
            f"| {st.mean(diffs):+.2f} | [{boots[250]:+.2f}, {boots[9750]:+.2f}] | {pos}/{neg} "
            f"| {sign_p(pos, neg):.3f} |"
        )
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines) - 8} cells to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
