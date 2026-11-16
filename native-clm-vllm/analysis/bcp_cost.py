import collections

P = {"9B": 9e9, "27B": 27e9}
acc = {}
for l in open("bcp.tsv"):
    if l.startswith("Qwen3.5-"):
        a, c, q, ok, *_ = l.rstrip("\n").split("\t")
        acc[(a, c, q)] = ok == "true"
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for l in open("bcp_usage.tsv"):
    f = l.rstrip("\n").split("\t")
    a, c, q = f[:3]
    num = [float(x) if x else 0.0 for x in f[3:]]
    ap, ac_, acached, mp, mc, mcached = num + [0.0] * (6 - len(num))
    actor = a.replace("Qwen3.5-", "").replace("-BF16", "")
    mgr = c.split("@")[1] if "@" in c else None
    g = agg[(actor, c)]
    g["n"] += 1
    g["ok"] += acc.get((a, c, q), False)
    g["actor_pf"] += 2 * P[actor] * (ap - acached + ac_) / 1e15
    g["mgr_pf"] += (2 * P[mgr] * (mp - mcached + mc) / 1e15) if mgr else 0
    g["actor_tok"] += ap + ac_
    g["mgr_tok"] += mp + mc
print(
    "| actor | condition | accuracy | actor PF/ep | manager PF/ep | total PF/ep | PF per correct |"
)
print("|---|---|---|---|---|---|---|")
for (actor, c), g in sorted(agg.items(), key=lambda kv: (kv[0][0] != "9B", kv[0][1])):
    n = g["n"]
    tot = (g["actor_pf"] + g["mgr_pf"]) / n
    per = (g["actor_pf"] + g["mgr_pf"]) / g["ok"] if g["ok"] else float("inf")
    print(
        f"| {actor} | {c} | {g['ok'] / n:.0%} | {g['actor_pf'] / n:.2f} | {g['mgr_pf'] / n:.2f} | {tot:.2f} | {per:.2f} |"
    )
