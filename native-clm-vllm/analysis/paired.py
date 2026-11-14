import collections, math, random, sys

path = sys.argv[1] if len(sys.argv) > 1 else "cb.tsv"
R = {}
for line in open(path):
    label, actor, tag, ep, reward, term = line.rstrip("\n").split("\t")
    actor = actor.replace("Qwen3.5-", "").replace("-BF16", "")
    lvl, seed = ep.rsplit("-s", 1)
    fam = "Needle" if lvl.startswith("needle") else "KV"
    if label == "native":
        tag = {"none": "none", "clm_self": "clm_self"}[tag]
    if lvl in ("needle_retention-K4-C4", "kv_store-R130"):
        continue
    R[(actor, tag, lvl, int(seed))] = float(reward)


def pairs(actor, a, b, fam):
    out = []
    for (ac, tg, lvl, seed), v in R.items():
        if (
            ac == actor
            and tg == a
            and (fam == "all" or lvl.startswith({"Needle": "needle", "KV": "kv"}[fam]))
        ):
            w = R.get((actor, b, lvl, seed))
            if w is not None:
                out.append(v - w)
    return out


def sign_p(d):
    pos, neg = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def boot(d, B=4000, seed=0):
    rng = random.Random(seed)
    if not d:
        return (float("nan"),) * 2
    m = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(B))
    return m[int(0.025 * B)], m[int(0.975 * B)]


def row(actor, a, b, fam):
    d = pairs(actor, a, b, fam)
    if not d:
        return None
    lo, hi = boot(d)
    pos, neg = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    return f"| {actor} | {a} − {b} | {fam} | {len(d)} | {sum(d) / len(d):+.2f} [{lo:+.2f}, {hi:+.2f}] | {pos}/{neg} | {sign_p(d):.3f} |"


print("| actor | comparison | tasks | pairs | mean diff [95% CI] | better/worse | sign p |")
print("|---|---|---|---|---|---|---|")
for actor in ["2B", "4B", "9B", "27B"]:
    s = actor
    comps = [
        ("clm_self", "clm_self_noskill"),
        ("clm_self_noskill", "none"),
        ("clm_fixed", "clm_self_noskill"),
        (f"mgr_watch@{s}", "clm_self_noskill"),
        (f"mgr_fixed@{s}", "clm_self_noskill"),
        (f"mgr_request@{s}", "clm_self_noskill"),
    ]
    if actor != "27B":
        comps += [
            ("mgr_watch@27B", "clm_self_noskill"),
            ("mgr_fixed@27B", "clm_self_noskill"),
            ("mgr_watch@27B", f"mgr_watch@{s}"),
            ("mgr_fixed@27B", f"mgr_fixed@{s}"),
        ]
    for a, b in comps:
        for fam in ("Needle", "KV"):
            r = row(actor, a, b, fam)
            if r:
                print(r)
