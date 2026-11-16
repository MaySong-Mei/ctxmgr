import collections
import math

rows = [l.rstrip("\n").split("\t") for l in open("bcp.tsv") if l.startswith("Qwen3.5-")]
acc = collections.defaultdict(dict)
wall = collections.defaultdict(dict)
for actor, cond, q, correct, judged, w in rows:
    a = actor.replace("Qwen3.5-", "").replace("-BF16", "")
    acc[(a, cond)][q] = correct == "true"
    wall[(a, cond)][q] = w == "1"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar(x, y):
    qs = sorted(set(x) & set(y))
    b = sum(x[q] and not y[q] for q in qs)
    c = sum(y[q] and not x[q] for q in qs)
    n = b + c
    k = min(b, c)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return b, c, p, len(qs)


order = [
    "clm_self",
    "clm_self_nonudge",
    "clm_fixed",
    "mgr_fixed@{s}",
    "mgr_request@{s}",
    "mgr_watch@{s}",
    "mgr_fixed@27B",
    "mgr_request@27B",
    "mgr_watch@27B",
]
print("| actor | condition | accuracy | 95% CI | context-limit hits | vs clm_self (+/−, p) |")
print("|---|---|---|---|---|---|")
for a in ["9B", "27B"]:
    seen = set()
    for t in order:
        c = t.format(s=a)
        if c in seen or (a, c) not in acc:
            continue
        seen.add(c)
        d = acc[(a, c)]
        k, n = sum(d.values()), len(d)
        lo, hi = wilson(k, n)
        wl = sum(wall[(a, c)].values())
        if c == "clm_self":
            cmp = "—"
        else:
            b, cc, p, m = mcnemar(d, acc[(a, "clm_self")])
            cmp = f"+{b}/−{cc}, p={p:.3f}"
        print(f"| {a} | {c} | {k}/{n} = {k / n:.0%} | {lo:.0%}–{hi:.0%} | {wl}/{n} | {cmp} |")
print()
for a, x, y in [
    ("9B", "mgr_watch@27B", "mgr_watch@9B"),
    ("9B", "mgr_fixed@27B", "mgr_fixed@9B"),
    ("9B", "clm_fixed", "clm_self_nonudge"),
    ("27B", "clm_fixed", "clm_self_nonudge"),
    ("27B", "mgr_watch@27B", "clm_self_nonudge"),
    ("9B", "mgr_watch@27B", "clm_self_nonudge"),
    ("27B", "clm_self", "clm_self_nonudge"),
    ("9B", "clm_self", "clm_self_nonudge"),
]:
    b, c, p, m = mcnemar(acc[(a, x)], acc[(a, y)])
    print(f"{a}: {x} vs {y}: +{b}/−{c} of {m}, p={p:.3f}")
