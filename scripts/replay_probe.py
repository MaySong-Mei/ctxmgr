"""Replay probe for the 27B actor's output-cap loop on Needle (round 29, job 31451066).

Pre-registered in `docs/preregistration-v3.md` (may/week1). Rebuilds exact actor inputs from an
episode's context snapshots and re-sends them as single calls, changing one thing per arm:

  python scripts/replay_probe.py plan RUNS_DIR                    # list probe points, no calls
  python scripts/replay_probe.py run RUNS_DIR OUT.jsonl --base-url URL --model MODEL
  python scripts/replay_probe.py summarize OUT.jsonl

RUNS_DIR is `runs/baseline-<job>/<actor>` with `rule_needle/` and `none/` episode directories.
The input of actor call t is snapshot turn-t without its last message: each turn the harness
compacts, appends the operation (or the runner's reminder), calls the actor, appends the reply, and
the runner snapshots. Points per `rule_needle` episode with a capped reply: `a` = the first capped
call, `b` = the next call (its input ends with the capped reply and the reminder), `deep` = the
sixth call after `a` when every call in between was capped too.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from contextbench.stream import is_ready_line

from ctxmgr.harness.agents import NEEDLE_LINE_RE, Sampling

GREEDY = Sampling()
SAMPLED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
N_SAMPLES = 8
ROLLOUT_TURNS = 4
LONG_CAP = 8192
NEEDLE_ID_RE = re.compile(r"\[n(\d+)i\d+#[0-9a-f]+\]")
NEEDLE_HEADER = "NEEDLES (keep these lines verbatim in your context):"
CAP_REMINDER = (
    "[Your reply was cut at the 2,048-token output limit before READY_FOR_NEXT_OP, so the stream "
    "did not advance. Keep this reply short and put READY_FOR_NEXT_OP on a line of its own to "
    "receive the next operation.]"
)
# Appended to the task instruction (block 1), as the archive conditions do.
DISCLOSURES = {
    "D0": "",
    "D1": (
        "\n\nNote: messages in this conversation stay in your context until something removes them."
    ),
    "D2": (
        "\n\nNote: a context manager may remove old messages. Before it does, it copies every "
        "needle line into a notes block, so no needle is lost."
    ),
    "D3": "\n\nDo not repeat needle lines from earlier operations in your replies.",
}


EPISODE_RE = re.compile(r"needle_retention-(K\d+-C\d+)-s(\d+)$")


def snapshot(ep, turn):
    path = ep / "context_snapshots" / f"turn-{turn:04d}.json"
    messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def call_input(ep, turn):
    """The messages actor call `turn` received, and the reply it logged."""
    messages = snapshot(ep, turn)
    assert messages[-1]["role"] == "assistant", (ep, turn)
    return messages[:-1], messages[-1]["content"]


def actor_calls(ep):
    lines = (ep / "calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [c for c in map(json.loads, filter(None, lines)) if c["role"] == "actor"]


def points(runs):
    """Probe points per capped rule_needle episode: {(level, seed, point): turn}."""
    out = {}
    for ep in sorted((runs / "rule_needle").glob("needle_retention-*")):
        if not (ep / "calls.jsonl").exists():
            continue
        level, seed = EPISODE_RE.search(ep.name).groups()
        calls = actor_calls(ep)
        capped = [c["finish_reason"] == "length" for c in calls]
        if True not in capped:
            continue
        i = capped.index(True)
        out[level, seed, "a"] = (ep, calls[i]["turn"])
        if i + 1 < len(calls):
            out[level, seed, "b"] = (ep, calls[i + 1]["turn"])
        if i + 6 < len(calls) and all(capped[i : i + 7]):
            out[level, seed, "deep"] = (ep, calls[i + 6]["turn"])
    return out


def short_previous(messages):
    """Point b with the capped reply replaced by one that restates only the current op's needles
    (the 2B's format) and, like the original, lacks the marker, so the reminder still fits."""
    op = next(
        m for m in reversed(messages) if m["role"] == "user" and NEEDLE_HEADER in m["content"]
    )
    lines = NEEDLE_LINE_RE.findall(op["content"])
    out = [dict(m) for m in messages]
    assert out[-2]["role"] == "assistant" and out[-1]["role"] == "user"
    out[-2]["content"] = NEEDLE_HEADER + "\n" + "\n".join(lines)
    return out


def single_jobs(runs):
    """(record, messages, request kwargs) for every single-call arm; R1 first (run serially)."""
    jobs = []
    pts = points(runs)

    def add(arm, key, messages, sample=None, logged=None, **kw):
        level, seed, point = key
        rec = {"arm": arm, "level": level, "seed": seed, "point": point, "sample": sample,
               "turn": pts[key][1], "logged": logged}  # fmt: skip
        jobs.append((rec, messages, {**GREEDY.request_kwargs(), **kw}))

    for key, (ep, turn) in pts.items():
        msgs, logged = call_input(ep, turn)
        add("R1_exact", key, msgs, logged=logged)
    for key, (ep, turn) in pts.items():
        msgs, _ = call_input(ep, turn)
        if key[2] in ("a", "b"):
            add("R2_cap8k", key, msgs, max_tokens=LONG_CAP)
        if key[2] == "b" and key[0] == "K4-C24":
            add("R3a_short_prev", key, short_previous(msgs))
            add("R3b_drop_prev", key, msgs[:-2])
            for s in range(N_SAMPLES):
                add("R5a_sampled", key, msgs, sample=s, seed=s, **SAMPLED)
                add("R5b_sampled_pp", key, msgs, sample=s, seed=s, presence_penalty=1.5, **SAMPLED)
        if key[2] in ("b", "deep") and key[0] == "K4-C24":
            add("R6_cap_reminder", key, [*msgs[:-1], {"role": "user", "content": CAP_REMINDER}])
    for rec, _, kw in jobs:
        if "top_k" in kw:  # vLLM takes top_k in the request body
            kw["extra_body"] = {**kw["extra_body"], "top_k": kw.pop("top_k")}
    return jobs


def rollout_sources(runs):
    """Per `none` K4-C24 episode: system, instruction, the first ROLLOUT_TURNS op messages and the
    replies logged for them (no editing happens under `none`, so one snapshot holds them all)."""
    out = {}
    for ep in sorted((runs / "none").glob("needle_retention-K4-C24-s*")):
        messages = snapshot(ep, ROLLOUT_TURNS)
        ops = [m for m in messages[2:] if m["role"] == "user"]
        logged = [m["content"] for m in messages[2:] if m["role"] == "assistant"]
        assert len(ops) == ROLLOUT_TURNS and all(NEEDLE_HEADER in m["content"] for m in ops), ep
        out[EPISODE_RE.search(ep.name).group(2)] = (messages[0], messages[1], ops, logged)
    return out


def describe(reply, finish, tokens):
    ops = sorted({int(m) for m in NEEDLE_ID_RE.findall(reply)})
    return {
        "finish_reason": finish,
        "completion_tokens": tokens,
        "ready": is_ready_line(reply),
        "needle_lines": len(NEEDLE_LINE_RE.findall(reply)),
        "needle_ops": ops,
        "reply": reply,
    }


def complete(client, model, messages, kwargs):
    r = client.chat.completions.create(model=model, messages=messages, **kwargs)
    choice = r.choices[0]
    return describe(choice.message.content or "", choice.finish_reason, r.usage.completion_tokens)


def rollout(client, model, seed, source, arm):
    system, instruction, ops, logged = source
    messages = [system, {**instruction, "content": instruction["content"] + DISCLOSURES[arm]}]
    records = []
    for turn, op in enumerate(ops, 1):
        messages.append(op)
        out = complete(client, model, messages, GREEDY.request_kwargs())
        messages.append({"role": "assistant", "content": out["reply"]})
        records.append({"arm": f"R4_{arm}", "level": "K4-C24", "seed": seed, "point": "rollout",
                        "sample": None, "turn": turn,
                        "logged": logged[turn - 1] if arm == "D0" else None, **out})  # fmt: skip
    return records


def run(runs, out_path, base_url, model, workers):
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=3600)
    jobs = single_jobs(runs)
    sources = rollout_sources(runs)
    with open(out_path, "w", encoding="utf-8") as out:

        def write(rec):
            logged = rec.pop("logged")
            rec["same_as_logged"] = None if logged is None else rec["reply"] == logged
            out.write(json.dumps(rec) + "\n")
            out.flush()

        # R1 one call at a time, as the episodes ran; the other arms concurrently.
        for rec, messages, kw in (j for j in jobs if j[0]["arm"] == "R1_exact"):
            write({**rec, **complete(client, model, messages, kw)})
        rest = [j for j in jobs if j[0]["arm"] != "R1_exact"]
        with ThreadPoolExecutor(workers) as pool:
            singles = pool.map(lambda j: {**j[0], **complete(client, model, j[1], j[2])}, rest)
            rolls = [
                pool.submit(rollout, client, model, seed, src, arm)
                for seed, src in sources.items()
                for arm in DISCLOSURES
            ]
            for rec in singles:
                write(rec)
            for fut in rolls:
                for rec in fut.result():
                    write(rec)


def summarize(path):
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line
    ]
    groups = {}
    for r in rows:
        if r["point"] == "rollout" and r["turn"] != ROLLOUT_TURNS:
            continue
        groups.setdefault((r["arm"], r["level"], r["point"]), []).append(r)
    print("| arm | level | point | n | capped | ready | ready <= 2,048 tokens | median tokens | "
          "earlier ops restated | same as logged |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        g = groups[key]
        toks = sorted(r["completion_tokens"] for r in g)
        capped = sum(r["finish_reason"] == "length" for r in g)
        ready = sum(r["ready"] for r in g)
        short = sum(r["ready"] and r["completion_tokens"] <= 2048 for r in g)
        # replies that restate needles of an op before the latest op they mention
        cumulative = sum(
            any(op < max(r["needle_ops"], default=0) for op in r["needle_ops"]) for r in g
        )
        same = [r["same_as_logged"] for r in g if r["same_as_logged"] is not None]
        same = f"{sum(same)}/{len(same)}" if same else "-"
        print(f"| {' | '.join(key)} | {len(g)} | {capped} | {ready} | {short} | "
              f"{toks[len(toks) // 2]} | {cumulative} | {same} |")  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        c = sub.add_parser(name)
        c.add_argument("runs", type=Path)
        if name == "run":
            c.add_argument("out", type=Path)
            c.add_argument("--base-url", required=True)
            c.add_argument("--model", required=True)
            c.add_argument("--workers", type=int, default=32)
    sub.add_parser("summarize").add_argument("out", type=Path)
    args = p.parse_args()
    if args.cmd == "plan":
        jobs = single_jobs(args.runs)
        for key, (_, turn) in sorted(points(args.runs).items()):
            print(*key, f"turn {turn}")
        arms = {}
        for rec, _, _ in jobs:
            arms[rec["arm"]] = arms.get(rec["arm"], 0) + 1
        n_roll = len(rollout_sources(args.runs)) * len(DISCLOSURES)
        print(arms, f"R4 rollouts: {n_roll} x {ROLLOUT_TURNS} calls")
    elif args.cmd == "run":
        run(args.runs, args.out, args.base_url, args.model, args.workers)
    else:
        summarize(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
