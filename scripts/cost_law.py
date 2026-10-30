"""How much a peer manager reads: measured vs the trigger law T / (T - R), per Needle cell.

Usage: python scripts/cost_law.py OUT.md
A peer manager gets no prefix-cache hits, so at every event it re-reads the whole context (about T
tokens) and each event frees about T - R tokens (R = context left after the edit). Over a stream of
S new tokens it then reads about S * T / (T - R) tokens. Measured: manager prompt tokens / actor
uncached prompt tokens (the actor's uncached tokens are the stream plus its own turns). Predicted:
mean context before / mean (before - after) over the episode's events (o200k, from edits.jsonl).
Greedy bf16 runs at max_len 81,920 that logged cached tokens (rounds 25-28); archive conditions
excluded (recalled lines re-enter the context).
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from cost_table import episode as cost_episode


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(out):
    cells = defaultdict(list)
    for d in sorted(Path("runs").glob("*/Qwen3.5-*/peer_*/needle_retention-*")):
        if d.parent.name.endswith("_arc") or not (d / "episode.json").exists():
            continue
        if not (got := cost_episode(d)):
            continue
        key, _, acc = got
        events = jsonl(d / "edits.jsonl")
        if not events:
            continue
        before = st.mean(e["context_before_o200k"] for e in events)
        after = st.mean(e["context_after_o200k"] for e in events)
        uncached = acc["actor_prompt"] - acc["actor_cached"]
        if before <= after or not uncached:
            continue
        cells[key].append(
            (acc["manager_prompt"] / uncached, before / (before - after), len(events))
        )
    lines = [
        "# Peer-manager reads vs the trigger law T / (T - R) (Needle)",
        "",
        "Built by `scripts/cost_law.py`. measured = manager prompt tokens / actor uncached prompt",
        "tokens; predicted = mean context before an event / mean tokens the event frees (o200k).",
        "ratio = measured / predicted. Means over episodes.",
        "",
        "| level | actor | condition | manager | version | n | events | measured | predicted | ratio |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ratios = []
    for key in sorted(cells):
        v = cells[key]
        m, p = st.mean(x[0] for x in v), st.mean(x[1] for x in v)
        ratios.append(m / p)
        lines.append(
            f"| {' | '.join(key)} | {len(v)} | {st.mean(x[2] for x in v):.1f} | {m:.2f} | {p:.2f} "
            f"| {m / p:.2f} |"
        )
    q = st.quantiles(ratios, n=4)
    summary = (
        f"Across {len(ratios)} cells, measured / predicted has median {st.median(ratios):.2f} "
        f"(IQR {q[0]:.2f}-{q[2]:.2f})."
    )
    lines[5:5] = [summary, ""]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1])
