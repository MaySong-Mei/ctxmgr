"""Confirmatory analysis for docs/preregistration.md. Run unchanged after the held-out jobs finish.

Usage:
  python scripts/prereg_analysis.py              # seeds 50-57 / 50-65, frozen harness commit enforced
  python scripts/prereg_analysis.py --dry-run    # seeds 42-49, no commit check (pipeline test only)

Reads every runs/<run>/<actor>/<condition>/<family>-<level>-s<seed>/ episode. Needle: official reward.
KV: lenient recall (scripts/lenient_kv.py). Tests: one-sided exact sign-flip permutation tests on
shifted paired differences; a multi-part prediction passes only if every component passes
(intersection-union, p = max component p); Holm across predictions at alpha = 0.05.
"""

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from lenient_kv import lenient_score

FROZEN_COMMIT = "23d9a6b"
ALPHA = 0.05
SEEDS_8 = list(range(50, 58))
SEEDS_16 = list(range(50, 66))
NEEDLE, KV = "needle_retention", "kv_store"

# Component: (label, (actor, condition), (actor, condition) or None, family, level, kind, threshold).
# kinds: "gt" A - B > t; "lt" A - B < t; "ni" A - B > -t; "tost" |A - B| < t.
PREDICTIONS = [
    (
        "P1",
        16,
        [
            (f"{a} peer_27b-none", (a, "peer_27b"), (a, "none"), NEEDLE, "K4-C24", "gt", 0.30)
            for a in ("2B", "4B", "9B")
        ],
    ),
    (
        "P2",
        8,
        [
            (f"{a} self-none", (a, "self"), (a, "none"), NEEDLE, "K4-C24", "lt", 0.05)
            for a in ("2B", "4B")
        ],
    ),
    ("P3", 8, [("9B mask_arc-none", ("9B", "mask_arc"), ("9B", "none"), KV, "R1000", "gt", 0.70)]),
    (
        "P4",
        16,
        [
            (
                "9B mask_arc vs peer_27b_arc",
                ("9B", "mask_arc"),
                ("9B", "peer_27b_arc"),
                KV,
                "R1000",
                "tost",
                0.10,
            )
        ],
    ),
    (
        "P5",
        16,
        [
            (f"4B {c}-none", ("4B", c), ("4B", "none"), KV, "R1000", "lt", 0.20)
            for c in ("mask_arc", "peer_27b_arc")
        ],
    ),
    (
        "P6",
        8,
        [("27B self_arc-none_r", ("27B", "self_arc"), ("27B", "none_r"), KV, "R1000", "gt", 0.70)],
    ),
    (
        "P7",
        8,
        [
            c
            for arm in ("self_arc", "peer_same_arc", "mask_arc")
            for c in (
                (
                    f"9B peer_27b-{arm}",
                    ("9B", "peer_27b"),
                    ("9B", arm),
                    NEEDLE,
                    "K4-C24",
                    "gt",
                    0.20,
                ),
                (f"9B {arm}-none", ("9B", arm), ("9B", "none"), NEEDLE, "K4-C24", "lt", 0.10),
            )
        ],
    ),
    (
        "P8",
        16,
        [
            (
                "2B peer_27b vs 27B peer_same",
                ("2B", "peer_27b"),
                ("27B", "peer_same"),
                NEEDLE,
                "K4-C24",
                "ni",
                0.10,
            )
        ],
    ),
    (
        "P9",
        8,
        [
            (f"9B {c}-none", ("9B", c), ("9B", "none"), KV, "R2000", "gt", 0.50)
            for c in ("mask_arc", "peer_27b_arc")
        ],
    ),
    (
        "P10",
        8,
        [
            (f"{a} peer_27b-none", (a, "peer_27b"), (a, "none"), NEEDLE, "K2-C192", "gt", 0.05)
            for a in ("2B", "9B")
        ],
    ),
]


def needed_cells():
    cells = set()
    for _, _, comps in PREDICTIONS:
        for _, a, b, family, level, _, _ in comps:
            cells |= {(*a, family, level), (*b, family, level)}
    return cells


def load(seeds, dry_run):
    """(actor, condition, family, level, seed) -> score for the predicted cells.

    Confirmatory mode fails on a duplicate episode or a harness commit other than the frozen one.
    Dry-run mode (seeds 42-49, which were run several times) keeps the episode of the latest job.
    """
    scores, source, cells = {}, {}, needed_cells()
    for ep in sorted(Path("runs").glob("*-*/Qwen3.5-*/*/*-s*/episode.json")):
        d = ep.parent  # .../<family>-<level>-s<seed>, e.g. needle_retention-K4-C24-s50
        name, seed = d.name.rsplit("-s", 1)
        family, level = name.split("-", 1)
        seed = int(seed)
        actor = d.parent.parent.name.removeprefix("Qwen3.5-")
        if (actor, d.parent.name, family, level) not in cells:
            continue
        if seed not in seeds or not (d / "score.json").exists():
            continue
        e = json.loads(ep.read_text(encoding="utf-8"))
        if (e.get("sampling") or {}).get("temperature", 0) != 0:
            continue  # the confirmatory family is greedy only
        if not dry_run and not (e.get("repo_commit") or "").startswith(FROZEN_COMMIT):
            raise SystemExit(f"{d}: harness commit {e.get('repo_commit')} != {FROZEN_COMMIT}")
        key = (actor, d.parent.name, family, level, seed)
        job = int(d.parents[2].name.rsplit("-", 1)[1])
        if key in scores:
            if not dry_run:
                raise SystemExit(f"duplicate episode for {key}: {source[key]} and {d}")
            if job < source[key][0]:
                continue
        source[key] = (job, d)
        if family == KV:
            run = d.parents[2]
            scores[key] = lenient_score(d, run / "instances" / d.name)
        else:
            scores[key] = json.loads((d / "score.json").read_text())["official"]["reward"]
    return scores


def p_greater(values):
    """One-sided exact sign-flip test of mean(values) > 0; ties count against the prediction."""
    n, observed = len(values), st.mean(values)
    hits = sum(
        st.mean(s * v for s, v in zip(signs, values)) >= observed - 1e-12
        for signs in itertools.product((1, -1), repeat=n)
    )
    return hits / 2**n if observed > 0 else 1.0


def component(c, seeds, scores):
    label, (aa, ac), (ba, bc), family, level, kind, t = c
    a = [scores.get((aa, ac, family, level, s)) for s in seeds]
    b = [scores.get((ba, bc, family, level, s)) for s in seeds]
    if None in a or None in b:
        missing = sum(x is None for x in a) + sum(x is None for x in b)
        return {
            "label": label,
            "status": "incomplete",
            "p": 1.0,
            "mean": None,
            "note": f"{missing} episodes missing",
        }
    d = [x - y for x, y in zip(a, b)]
    mean = st.mean(d)
    if kind == "gt":
        p, wrong, p_opp = p_greater([x - t for x in d]), mean < t, p_greater([t - x for x in d])
    elif kind == "lt":
        p, wrong, p_opp = p_greater([t - x for x in d]), mean > t, p_greater([x - t for x in d])
    elif kind == "ni":
        p, wrong, p_opp = p_greater([x + t for x in d]), mean < -t, p_greater([-t - x for x in d])
    else:  # tost
        p = max(p_greater([x + t for x in d]), p_greater([t - x for x in d]))
        wrong = abs(mean) > t
        p_opp = p_greater([x - t for x in d]) if mean > t else p_greater([-t - x for x in d])
    falsified = wrong and p_opp < ALPHA
    favour = sum(
        (x > t)
        if kind == "gt"
        else (x < t)
        if kind == "lt"
        else (x > -t)
        if kind == "ni"
        else (abs(x) < t)
        for x in d
    )
    return {
        "label": label,
        "status": "falsified" if falsified else "tested",
        "p": p,
        "mean": mean,
        "note": f"{kind} {t:+.2f}; {favour}/{len(d)} seeds favour",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="seeds 42-49, no commit check")
    args = ap.parse_args()
    seeds_for = {8: SEEDS_8, 16: SEEDS_16}
    if args.dry_run:
        seeds_for = {8: list(range(42, 50)), 16: list(range(42, 50))}
    scores = load(set(seeds_for[8]) | set(seeds_for[16]), args.dry_run)

    results = []
    for pid, n, comps in PREDICTIONS:
        rows = [component(c, seeds_for[n], scores) for c in comps]
        results.append(
            {
                "id": pid,
                "rows": rows,
                "p": max(r["p"] for r in rows),
                "falsified": any(r["status"] == "falsified" for r in rows),
                "incomplete": any(r["status"] == "incomplete" for r in rows),
            }
        )
    # Holm step-down across predictions.
    ordered = sorted(results, key=lambda r: r["p"])
    m, rejecting = len(ordered), True
    for i, r in enumerate(ordered):
        r["holm_threshold"] = ALPHA / (m - i)
        rejecting = rejecting and r["p"] <= r["holm_threshold"]
        r["confirmed"] = rejecting and not r["incomplete"]
    print("| prediction | component | mean diff | test | p (component) | outcome |")
    print("|---|---|---|---|---|---|")
    for r in results:
        outcome = (
            "confirmed"
            if r["confirmed"]
            else "falsified"
            if r["falsified"]
            else "incomplete"
            if r["incomplete"]
            else "inconclusive"
        )
        summary = f"{outcome} (p={r['p']:.4f}, Holm threshold {r['holm_threshold']:.4f})"
        for j, c in enumerate(r["rows"]):
            mean = "" if c["mean"] is None else f"{c['mean']:+.3f}"
            first = j == 0
            print(
                f"| {r['id'] if first else ''} | {c['label']} | {mean} | {c['note']} | "
                f"{c['p']:.4f} | {summary if first else ''} |"
            )
    confirmed = sum(r["confirmed"] for r in results)
    print(
        f"\nConfirmed {confirmed}/{m}; falsified {sum(r['falsified'] for r in results)}; "
        f"incomplete {sum(r['incomplete'] for r in results)}"
        + ("  [DRY RUN: seeds 42-49, not confirmatory]" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
