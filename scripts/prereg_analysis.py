"""Confirmatory analysis for docs/preregistration.md. Run unchanged after the pre-registered jobs.

Usage:
  python scripts/prereg_analysis.py            # seeds 50-57 / 50-65; frozen setup enforced
  python scripts/prereg_analysis.py --dry-run  # seeds 42-49, setup not enforced (pipeline test only)

Reads runs/<run>/Qwen3.5-<actor>/<condition>/<family>-<level>-s<seed>/. Needle: official reward.
KV: lenient recall (scripts/lenient_kv.py).

Each prediction component is a statement about paired differences d = A - B on the same instance,
tested with an exact one-sided binomial sign test: k = number of instances on the predicted side of
the threshold (ties count against), p = P(Binomial(n, 1/2) >= k). The claim tested is therefore "the
difference is beyond the threshold on a majority of instances"; the mean is reported descriptively.
A prediction passes only if every component passes (intersection-union, p = max component p). Holm
step-down across the predictions at one-sided alpha = 0.05.
"""

import argparse
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, "scripts")
from lenient_kv import lenient_score

ALPHA = 0.05
FROZEN = {
    "repo_commit": "33b3997",  # prefix of the prereg-v1-harness tag on may/track2-baseline
    "contextbench_commit": "fb533c18c78a82d3f1b2047a9c1700735cf829ae",
    "precision": "bf16",
    "served_max_len": 81920,
    "trigger": 18432,
    "manager_max_tokens": 4096,
    "sampling": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "thinking": False,
        "top_k": None,
        "presence_penalty": None,
        "seed": None,
    },
}
EDIT_VERSION = {
    "none": None,
    "none_r": None,
    "self": "edit-v8",
    "peer_same": "edit-v8",
    "peer_27b": "edit-v8",
    "self_arc": "edit-v8-arc",
    "peer_same_arc": "edit-v8-arc",
    "peer_27b_arc": "edit-v8-arc",
    "mask_arc": "rule-mask-v1",
}
SEEDS = {8: list(range(50, 58)), 16: list(range(50, 66))}
NEEDLE, KV = "needle_retention", "kv_store"

# component: (label, (actor, cond), (actor, cond), family, level, kind, threshold)
# kinds (per instance): "gt" d > t; "lt" d < t; "ni" d > -t; "eq" |d| < t
PREDICTIONS = [
    ("P1", 16, [
        (f"{a} peer_27b-none", (a, "peer_27b"), (a, "none"), NEEDLE, "K4-C24", "gt", 0.30)
        for a in ("2B", "4B", "9B")
    ]),
    ("P2", 8, [
        (f"{a} self-none", (a, "self"), (a, "none"), NEEDLE, "K4-C24", "lt", 0.05)
        for a in ("2B", "4B")
    ]),
    ("P3", 8, [("9B mask_arc-none", ("9B", "mask_arc"), ("9B", "none"), KV, "R1000", "gt", 0.70)]),
    ("P4", 16, [
        ("9B mask_arc vs peer_27b_arc", ("9B", "mask_arc"), ("9B", "peer_27b_arc"), KV, "R1000",
         "eq", 0.10),
    ]),
    ("P5", 16, [
        (f"4B {c}-none", ("4B", c), ("4B", "none"), KV, "R1000", "lt", 0.20)
        for c in ("mask_arc", "peer_27b_arc")
    ]),
    ("P6", 8, [
        ("27B self_arc-none_r", ("27B", "self_arc"), ("27B", "none_r"), KV, "R1000", "gt", 0.70),
    ]),
    ("P7", 8, [
        c
        for arm in ("self_arc", "peer_same_arc", "mask_arc")
        for c in (
            (f"9B peer_27b-{arm}", ("9B", "peer_27b"), ("9B", arm), NEEDLE, "K4-C24", "gt", 0.20),
            (f"9B {arm}-none", ("9B", arm), ("9B", "none"), NEEDLE, "K4-C24", "lt", 0.10),
        )
    ]),
    ("P8", 16, [
        ("2B peer_27b vs 27B peer_same", ("2B", "peer_27b"), ("27B", "peer_same"), NEEDLE,
         "K4-C24", "ni", 0.10),
    ]),
    ("P9", 8, [
        ("9B mask_arc-none", ("9B", "mask_arc"), ("9B", "none"), KV, "R2000", "gt", 0.50),
        ("9B peer_27b_arc-none", ("9B", "peer_27b_arc"), ("9B", "none"), KV, "R2000", "gt", 0.50),
        ("9B mask_arc vs peer_27b_arc", ("9B", "mask_arc"), ("9B", "peer_27b_arc"), KV, "R2000",
         "eq", 0.15),
    ]),
    ("P10", 8, [
        (f"{a} peer_27b-none", (a, "peer_27b"), (a, "none"), NEEDLE, "K2-C192", "gt", 0.05)
        for a in ("2B", "9B")
    ]),
]


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(max(k, 0), n + 1))


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion k/n."""
    a = (1 - level) / 2

    def crossing(f):  # f increases in p; return the p where f(p) = a
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < a else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else crossing(lambda p: binom_sf(k, n, p))
    upper = 1.0 if k == n else 1 - crossing(lambda q: binom_sf(n - k, n, q))
    return lower, upper


def needed_cells() -> set:
    cells = set()
    for _, _, comps in PREDICTIONS:
        for _, a, b, family, level, _, _ in comps:
            cells |= {(*a, family, level), (*b, family, level)}
    return cells


def check_setup(e: dict, actor: str, cond: str, n_ops: int, where: Path) -> None:
    problems = []
    if not (e.get("repo_commit") or "").startswith(FROZEN["repo_commit"]):
        problems.append(f"repo_commit {e.get('repo_commit')}")
    for key in ("contextbench_commit", "precision", "served_max_len"):
        if e.get(key) != FROZEN[key]:
            problems.append(f"{key} {e.get(key)}")
    if e.get("sampling") != FROZEN["sampling"]:
        problems.append(f"sampling {e.get('sampling')}")
    if e.get("actor_model") != f"Qwen/Qwen3.5-{actor}":
        problems.append(f"actor_model {e.get('actor_model')}")
    manager = {
        "peer_27b": "Qwen/Qwen3.5-27B",
        "peer_27b_arc": "Qwen/Qwen3.5-27B",
        "peer_same": f"Qwen/Qwen3.5-{actor}",
        "peer_same_arc": f"Qwen/Qwen3.5-{actor}",
    }
    if e.get("manager_model") != manager.get(cond):
        problems.append(f"manager_model {e.get('manager_model')}")
    if e.get("edit_prompt_version") != EDIT_VERSION[cond]:
        problems.append(f"edit_prompt_version {e.get('edit_prompt_version')}")
    if cond not in ("none", "none_r"):
        if e.get("manager_trigger_o200k") != FROZEN["trigger"]:
            problems.append(f"trigger {e.get('manager_trigger_o200k')}")
        if e.get("manager_max_tokens") != FROZEN["manager_max_tokens"]:
            problems.append(f"manager_max_tokens {e.get('manager_max_tokens')}")
    if e.get("max_turns") != 2 * n_ops + 10:
        problems.append(f"max_turns {e.get('max_turns')} (n_ops {n_ops})")
    if problems:
        raise SystemExit(f"{where}: setup differs from the pre-registration: {'; '.join(problems)}")


def load(seeds: set, dry_run: bool) -> dict:
    """(actor, cond, family, level, seed) -> (score, instance hash, source dir, job id).

    An episode with episode.json but no score.json (e.g. server_rejected on turn 1) scores 0. One
    without episode.json is missing (its job was stopped). Two complete episodes for one key abort
    the confirmatory analysis; the dry run (seeds 42-49 were run several times) keeps the latest job.
    """
    found, cells = {}, needed_cells()
    for ep in sorted(Path("runs").glob("*-*/Qwen3.5-*/*/*-s*/episode.json")):
        d = ep.parent
        name, seed = d.name.rsplit("-s", 1)
        family, level = name.split("-", 1)
        actor, cond, seed = d.parent.parent.name.removeprefix("Qwen3.5-"), d.parent.name, int(seed)
        if seed not in seeds or (actor, cond, family, level) not in cells:
            continue
        e = json.loads(ep.read_text(encoding="utf-8"))
        run = d.parents[2]
        if dry_run:
            if (e.get("sampling") or {}).get("temperature", 0) != 0:
                continue
        else:
            manifest = json.loads((run / "instances" / d.name / "manifest.json").read_text())
            check_setup(e, actor, cond, manifest["n_ops"], d)
        if not any((d / "context_snapshots").glob("turn-*.json")):
            score = 0.0  # ended before any snapshot (e.g. server_rejected on turn 1)
        elif family == KV:
            score = lenient_score(d, run / "instances" / d.name)
        elif (d / "score.json").exists():
            score = json.loads((d / "score.json").read_text())["official"]["reward"]
        else:
            raise SystemExit(f"{d}: snapshots but no score.json (scorer failed); rescore first")
        key = (actor, cond, family, level, seed)
        job = int(run.name.rsplit("-", 1)[1])
        if key in found:
            if not dry_run:
                raise SystemExit(f"two complete episodes for {key}: {found[key][2]} and {d}")
            if job < found[key][3]:
                continue
        found[key] = (score, json.dumps(e.get("instance_sha256"), sort_keys=True), d, job)
    return found


def component(c, seeds, data) -> dict:
    label, (aa, ac), (ba, bc), family, level, kind, t = c
    a = [data.get((aa, ac, family, level, s)) for s in seeds]
    b = [data.get((ba, bc, family, level, s)) for s in seeds]
    missing = [s for s, x, y in zip(seeds, a, b) if x is None or y is None]
    if missing:
        return {"label": label, "status": "incomplete", "p": 1.0, "p_opp": 1.0,
                "text": f"missing seeds {missing}"}
    for s, x, y in zip(seeds, a, b):
        if x[1] != y[1]:
            raise SystemExit(f"{label} seed {s}: paired episodes ran different instances")
    d = [x[0] - y[0] for x, y in zip(a, b)]
    if kind == "gt":
        fav, wrong, symbol = sum(v > t for v in d), sum(v < t for v in d), f"d > {t:+.2f}"
    elif kind == "lt":
        fav, wrong, symbol = sum(v < t for v in d), sum(v > t for v in d), f"d < {t:+.2f}"
    elif kind == "ni":
        fav, wrong, symbol = sum(v > -t for v in d), sum(v < -t for v in d), f"d > {-t:+.2f}"
    else:
        fav, wrong, symbol = sum(abs(v) < t for v in d), sum(abs(v) > t for v in d), f"|d| < {t:.2f}"
    n = len(d)
    jobs = sorted({x[3] for x in a} | {y[3] for y in b})
    p, p_opp = binom_sf(fav, n), binom_sf(wrong, n)
    lo, hi = clopper_pearson(fav, n)
    return {
        "label": label,
        "status": "falsified" if p_opp < ALPHA else "tested",
        "p": p,
        "p_opp": p_opp,
        "text": f"{symbol}: {fav}/{n} (95% CI {lo:.2f}-{hi:.2f}), {wrong} opposite; "
        f"mean d {st.mean(d):+.3f}; jobs {', '.join(map(str, jobs))}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="seeds 42-49; setup not enforced")
    args = ap.parse_args()
    seeds_for = {8: list(range(42, 50)), 16: list(range(42, 50))} if args.dry_run else SEEDS
    data = load(set(seeds_for[8]) | set(seeds_for[16]), args.dry_run)

    results = {}
    for pid, n, comps in PREDICTIONS:
        rows = [component(c, seeds_for[n], data) for c in comps]
        results[pid] = {
            "rows": rows,
            "p": max(r["p"] for r in rows),
            "falsified": any(r["status"] == "falsified" for r in rows),
            "incomplete": any(r["status"] == "incomplete" for r in rows),
        }
    rejecting = True
    ordered = sorted(results, key=lambda k: results[k]["p"])
    for i, pid in enumerate(ordered):
        r = results[pid]
        r["threshold"] = ALPHA / (len(ordered) - i)
        rejecting = rejecting and r["p"] <= r["threshold"]
        confirmed = rejecting and not r["incomplete"]
        r["outcome"] = (
            "confirmed" if confirmed
            else "falsified" if r["falsified"]
            else "incomplete" if r["incomplete"]
            else "inconclusive"
        )

    print("| prediction | component | result | p | p (opposite) | outcome |")
    print("|---|---|---|---|---|---|")
    for pid, r in results.items():
        for j, c in enumerate(r["rows"]):
            head = f"**{r['outcome']}** (p={r['p']:.4f}, Holm {r['threshold']:.4f})"
            print(
                f"| {pid if j == 0 else ''} | {c['label']} | {c['text']} | {c['p']:.4f} | "
                f"{c['p_opp']:.4f} | {head if j == 0 else ''} |"
            )

    fal = {pid: r["falsified"] for pid, r in results.items()}
    p2_both = all(c["status"] == "falsified" for c in results["P2"]["rows"])

    def verdict(x: bool) -> str:
        return "FALSIFIED" if x else "not falsified"

    print("\nRule-level falsification (component opposite-side p < 0.05, unadjusted):")
    print(f"- R1: {verdict(fal['P1'] or fal['P10'] or p2_both)}")
    print(f"- R2: {verdict(fal['P3'] or fal['P4'] or fal['P9'])}; actor-size boundary (P5): "
          f"{verdict(fal['P5'])}; 27B extension (P6): {results['P6']['outcome']}")
    print(f"- R3: {verdict(fal['P7'])}")
    if args.dry_run:
        print("\n[DRY RUN: seeds 42-49, the data the rule was derived from; not confirmatory]")


if __name__ == "__main__":
    main()
