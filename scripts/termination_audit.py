"""Per Needle cell: how episodes end, how often the actor's reply hits the output cap, and reward split
into the share of the stream delivered and the share of delivered needles kept.

Usage: python scripts/termination_audit.py OUT.md
Reads the rows of docs/tables/episodes_vllm.csv that `mechanism_table.episodes` keeps, plus each
episode's result.json (ops delivered), score.json (needles matched) and calls.jsonl (actor
finish_reason). reward = delivered x kept, where delivered = ops delivered / ops in the stream and
kept = needles matched / needles delivered (every op carries the same number of needles). capped =
actor replies cut at the 2,048-token output cap (finish_reason `length`). cap loop = an episode that
ends at `max_steps` with its last 5 actor replies all capped (the reply never reaches the
READY_FOR_NEXT_OP marker, so the stream stops).
"""

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from mechanism_table import LEVELS, episodes

DONE = {"submit", "stream_exhausted"}
LOOP_TAIL = 5


def episode_dir(r):
    return (
        Path("runs") / r["run"] / r["actor"] / r["condition"]
        / f"needle_retention-{r['level']}-s{r['seed']}"
    )  # fmt: skip


def audit(r):
    d = episode_dir(r)
    result = json.loads((d / "result.json").read_text(encoding="utf-8"))
    official = json.loads((d / "score.json").read_text(encoding="utf-8"))["official"]
    lines = (d / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    actor = [c for c in map(json.loads, filter(None, lines)) if c["role"] == "actor"]
    capped = [c["finish_reason"] == "length" for c in actor]
    delivered = result["n_ops_delivered"] / result["n_ops_total"]
    needles = official["total"] * delivered
    return {
        "delivered": delivered,
        "kept": official["matched"] / needles if needles else None,
        "turns": len(actor),
        "capped": sum(capped),
        "any_cap": any(capped),
        "loop": r["terminated_by"] == "max_steps" and len(capped) >= LOOP_TAIL
        and all(capped[-LOOP_TAIL:]),
    }  # fmt: skip


def main(out):
    cells = defaultdict(list)
    for r in episodes():
        d = episode_dir(r)
        if not (d / "result.json").exists() or not (d / "score.json").exists():
            continue
        key = (r["level"], r["actor"], r["condition"], r["manager"] or "-", r["edit_version"] or "-",
               r["decoding"])  # fmt: skip
        cells[key].append({"reward": float(r["official_reward"]), "end": r["terminated_by"],
                           **audit(r)})  # fmt: skip
    by_actor = defaultdict(Counter)
    table = []
    for key in sorted(cells, key=lambda k: (LEVELS.index(k[0]), *k[1:])):
        eps = cells[key]
        n = len(eps)
        ends = Counter("done" if e["end"] in DONE else e["end"] for e in eps)
        kept = [e["kept"] for e in eps if e["kept"] is not None]
        turns = sum(e["turns"] for e in eps)
        if key[2] != "none":
            a = by_actor[key[1]]
            a["n"] += n
            a["max_steps"] += ends["max_steps"]
            a["any_cap"] += sum(e["any_cap"] for e in eps)
            a["loop"] += sum(e["loop"] for e in eps)
        table.append(
            f"| {' | '.join(key)} | {n} | {st.mean(e['reward'] for e in eps):.2f} | "
            f"{st.mean(e['delivered'] for e in eps):.2f} | "
            f"{st.mean(kept) if kept else float('nan'):.2f} | {ends['done']} | "
            f"{ends['budget_wall']} | {ends['max_steps']} | "
            f"{n - ends['done'] - ends['budget_wall'] - ends['max_steps']} | "
            f"{sum(e['capped'] for e in eps) / turns if turns else 0:.2f} | "
            f"{sum(e['any_cap'] for e in eps)} | {sum(e['loop'] for e in eps)} |"
        )
    summary = [
        f"- {actor}: {a['max_steps']}/{a['n']} managed episodes end at `max_steps`, {a['loop']} of "
        f"them in a cap loop; {a['any_cap']}/{a['n']} have at least one capped actor reply."
        for actor, a in sorted(by_actor.items(), key=lambda kv: float(kv[0].split("-")[1][:-1]))
    ]
    lines = [
        "# Needle: how episodes end, actor output-cap hits, and reward = delivered x kept",
        "",
        "Built by `scripts/termination_audit.py` from `episodes_vllm.csv` and each episode's",
        "`result.json`, `score.json` and `calls.jsonl` (H200 bf16, max_len 81,920, trigger 18,432;",
        "archive conditions excluded). delivered = share of the stream's ops delivered; kept = share",
        "of delivered needles in the final context; done = `submit` or `stream_exhausted`; wall =",
        "`budget_wall`; capped = share of actor replies cut at the 2,048-token output cap; cap loop =",
        f"ends at `max_steps` with the last {LOOP_TAIL} actor replies all capped.",
        "",
        "Managed episodes (every condition except `none`), by actor:",
        "",
        *summary,
        "",
        (
            "| level | actor | condition | manager | version | decoding | n | reward | delivered | "
            "kept | done | wall | max_steps | other | capped | episodes with a cap | cap loops |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        *table,
    ]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main(sys.argv[1])
