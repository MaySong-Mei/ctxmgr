"""Identical-input replay of Needle manager events across managers, wordings and decoding.

Pre-registered in `docs/preregistration-v5.md` (may/week1). Rebuilds the exact peer-manager requests
of a source run's first three events per episode, sends each to one served manager under every
wording, applies the reply to a copy of the context, and scores it:

  python scripts/manager_replay.py plan SRC                       # list contexts; check the rebuild
  python scripts/manager_replay.py run SRC OUT.jsonl --base-url URL --model MODEL [--sampled]
      [--wordings v8,para3,...] [--sampled-wordings v8,para3,...]
  python scripts/manager_replay.py texts [WORDING ...]            # print instruction texts
  python scripts/manager_replay.py summarize OUT.jsonl [OUT.jsonl ...]

Besides v8 and the harness paraphrases, three ablations of `para3` (`docs/preregistration-v6.md`) are
built from its own text, so no step is reworded: `p3flat` (the numbered steps as one paragraph),
`p3bullets` (the steps as an unnumbered list) and `v8steps` (v8 with `para3`'s numbered steps added).

SRC is `runs/peer27b-<job>/Qwen3.5-2B/<condition>` of a run made with plain edit-v8. The context a
manager edits at turn t is snapshot t-1 (the harness compacts before it appends turn t's operation).
`plan` checks every rebuilt context against the logged `context_before_o200k` and every rebuilt v8
request against the logged manager `prompt_tokens_o200k`; `run` refuses to start on any mismatch.
Success = every needle line of the context survives the edits and they free at least 30% of it.
"""

import argparse
import copy
import json
import re
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from contextbench.tokens import count_tokens

from ctxmgr.harness import management
from ctxmgr.harness.agents import Sampling

WORDINGS = (None, "para1", "para2", "para3", "para4")  # None = plain edit-v8
SAMPLED_WORDINGS = (None, "para3", "para4")
N_SAMPLES = 4
EVENTS = 3  # first three manager events of each source episode
MANAGER_MAX_TOKENS = 4096
FREED_MIN = 0.30
NEEDLE_ID_RE = re.compile(r"\[n\d+i\d+#[0-9a-f]+\]")
EPISODE_RE = re.compile(r"needle_retention-(K\d+-C\d+)-s(\d+)$")
# Gemma 4 model card: one recommended setting for all uses.
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def wording_name(w):
    return w or "v8"


def _para3_parts():
    """`para3` split into (head lines, step-block lines incl. its header, tail lines, step texts)."""
    lines = management.EDIT_PARAPHRASES["para3"].split("\n")
    start = lines.index("Work through these steps:")
    end = next(i for i in range(start + 1, len(lines)) if not re.match(r"^(\d\. |   )", lines[i]))
    steps = []
    for line in lines[start + 1 : end]:
        if re.match(r"^\d\. ", line):
            steps.append(line[3:])
        else:
            steps[-1] += " " + line.strip()
    return lines[:start], lines[start:end], lines[end:], steps


def _ablations():
    head, block, tail, steps = _para3_parts()
    # Step 5 opens with a procedural "Write the reply."; as prose only its rule remains.
    prose = " ".join([*steps[:4], steps[4].removeprefix("Write the reply. ")])
    # Only the markers change; the step lines keep para3's own wrapping.
    bullets = [re.sub(r"^\d\. ", "-  ", line) for line in block]
    v8_head, v8_last = management.EDIT_INSTRUCTIONS.rsplit("\n", 1)
    return {
        "p3flat": "\n".join([*head, textwrap.fill(prose, 100), *tail]),
        "p3bullets": "\n".join([*head, *bullets, *tail]),
        "v8steps": "\n".join([v8_head, *block, v8_last]),
    }


ABLATIONS = _ablations()


def parse_wordings(text):
    """'v8,para3,p3flat' -> (None, 'para3', 'p3flat'); rejects unknown names."""
    names = [w.strip() for w in text.split(",") if w.strip()]
    known = {"v8", *management.EDIT_PARAPHRASES, *management.EDIT_VARIANTS, *ABLATIONS}
    unknown = [w for w in names if w not in known]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown wordings: {unknown}")
    return tuple(None if w == "v8" else w for w in names)


def instructions(wording):
    if wording in ABLATIONS:
        return ABLATIONS[wording]
    return management.with_variant(management.EDIT_INSTRUCTIONS, wording)


def snapshot(ep, turn):
    path = ep / "context_snapshots" / f"turn-{turn:04d}.json"
    messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def request(context, wording, wall):
    """The peer-manager request exactly as `PeerManagedAgent.request_edits` builds it."""
    size = count_tokens(context)[0]
    text = (
        f"Context of the assistant ({size} tokens; limit {wall}):\n\n"
        f"{management.render(context)}\n\n{instructions(wording)}"
    )
    return [
        {"role": "system", "content": management.PEER_SYSTEM},
        {"role": "user", "content": text},
    ]


def contexts(src):
    """[(level, seed, event index, context, wall, rebuild problems)] for the first EVENTS events."""
    out = []
    for ep in sorted(src.glob("needle_retention-*")):
        level, seed = EPISODE_RE.search(ep.name).groups()
        episode = json.loads((ep / "episode.json").read_text(encoding="utf-8"))
        wall = episode["budget"] - episode["reserve"]
        lines = (ep / "edits.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines if line][:EVENTS]
        calls = [
            json.loads(x)
            for x in (ep / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            if x
        ]
        manager_calls = [c for c in calls if c["role"] == "manager"]
        for k, event in enumerate(events):
            ctx = snapshot(ep, event["turn"] - 1)
            problems = []
            if count_tokens(ctx)[0] != event["context_before_o200k"]:
                problems.append("context size differs from the log")
            logged = [c for c in manager_calls if c["turn"] == event["turn"]]
            if (
                not logged
                or count_tokens(request(ctx, None, wall))[0] != logged[0]["prompt_tokens_o200k"]
            ):
                problems.append("v8 request size differs from the logged manager call")
            out.append((level, seed, k + 1, ctx, wall, problems))
    return out


def score(ctx, reply):
    edits, error = management.parse_edits(reply)
    after = copy.deepcopy(ctx)
    result = management.apply_edits(after, edits) if edits else None
    before_ids = set(NEEDLE_ID_RE.findall(" ".join(m["content"] for m in ctx)))
    after_ids = set(NEEDLE_ID_RE.findall(" ".join(m["content"] for m in after)))
    kept = len(before_ids & after_ids) / len(before_ids) if before_ids else 1.0
    freed = 1 - count_tokens(after)[0] / count_tokens(ctx)[0]
    ops = [a["op"] for a in result.applied] if result else []
    return {
        "parse_error": error,
        "apply_error": result.error if result else None,
        "ops": ops,
        "wrapped": bool(result and any(a.get("wrapped_bare_edit") for a in result.applied)),
        "kept": round(kept, 4),
        "freed": round(freed, 4),
        "success": kept >= 0.99 and freed >= FREED_MIN,
    }


def jobs(src, sampled, wordings=WORDINGS, sampled_wordings=SAMPLED_WORDINGS):
    out = []
    for level, seed, k, ctx, wall, _ in contexts(src):
        base = {"level": level, "seed": seed, "event": k}
        for w in wordings:
            kw = Sampling().request_kwargs(max_tokens=MANAGER_MAX_TOKENS, thinking=False)
            out.append(({**base, "wording": wording_name(w), "decoding": "greedy", "sample": None},
                        ctx, request(ctx, w, wall), kw))  # fmt: skip
        if sampled:
            for w in sampled_wordings:
                for s in range(N_SAMPLES):
                    kw = Sampling().request_kwargs(max_tokens=MANAGER_MAX_TOKENS, thinking=False)
                    kw.update(temperature=GEMMA_SAMPLING["temperature"],
                              top_p=GEMMA_SAMPLING["top_p"], seed=s)  # fmt: skip
                    kw["extra_body"] = {**kw["extra_body"], "top_k": GEMMA_SAMPLING["top_k"]}
                    out.append(({**base, "wording": wording_name(w), "decoding": "sampled",
                                 "sample": s}, ctx, request(ctx, w, wall), kw))  # fmt: skip
    return out


def run(src, out_path, base_url, model, sampled, workers, wordings, sampled_wordings):
    bad = [(lv, s, k, p) for lv, s, k, _, _, p in contexts(src) if p]
    if bad:
        print("rebuild check failed:", bad[:5])
        return 1
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=3600)

    def one(job):
        rec, ctx, messages, kw = job
        r = client.chat.completions.create(model=model, messages=messages, **kw)
        reply = r.choices[0].message.content or ""
        return {**rec, "model": model, "finish_reason": r.choices[0].finish_reason,
                "completion_tokens": r.usage.completion_tokens, **score(ctx, reply),
                "reply": reply}  # fmt: skip

    with open(out_path, "w", encoding="utf-8") as out, ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(one, jobs(src, sampled, wordings, sampled_wordings)):
            out.write(json.dumps(rec) + "\n")
            out.flush()
    return 0


def summarize(paths):
    rows = [
        json.loads(x) for p in paths for x in Path(p).read_text(encoding="utf-8").splitlines() if x
    ]
    groups = {}
    for r in rows:
        first = "first" if r["event"] == 1 else "later"
        groups.setdefault(
            (r["model"].rsplit("/", 1)[-1], r["decoding"], r["wording"], first), []
        ).append(r)
    print("| manager | decoding | wording | events | n | success | kept all | freed >= 30% | "
          "parse errors | capped |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        print(f"| {' | '.join(key)} | {n} | {sum(r['success'] for r in g) / n:.2f} | "
              f"{sum(r['kept'] >= 0.99 for r in g) / n:.2f} | "
              f"{sum(r['freed'] >= FREED_MIN for r in g) / n:.2f} | "
              f"{sum(bool(r['parse_error']) for r in g)} | "
              f"{sum(r['finish_reason'] == 'length' for r in g)} |")  # fmt: skip


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("plan")
    c.add_argument("src", type=Path)
    c = sub.add_parser("run")
    c.add_argument("src", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--base-url", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--sampled", action="store_true", help="add Gemma-recommended sampled repeats")
    c.add_argument("--workers", type=int, default=32)
    c.add_argument("--wordings", type=parse_wordings, default=WORDINGS, help="comma list")
    c.add_argument("--sampled-wordings", type=parse_wordings, default=SAMPLED_WORDINGS)
    sub.add_parser("texts").add_argument("wording", nargs="*")
    sub.add_parser("summarize").add_argument("out", type=Path, nargs="+")
    args = p.parse_args()
    if args.cmd == "texts":
        for w in args.wording or ["v8", *ABLATIONS]:
            print(f"===== {w}\n{instructions(None if w == 'v8' else w)}\n")
        return 0
    if args.cmd == "plan":
        ctxs = contexts(args.src)
        for level, seed, k, _, _, problems in ctxs:
            print(level, seed, f"event {k}", "OK" if not problems else "; ".join(problems))
        print(f"{len(ctxs)} contexts; {sum(bool(x[5]) for x in ctxs)} with rebuild problems")
        return 1 if any(x[5] for x in ctxs) else 0
    if args.cmd == "run":
        return run(args.src, args.out, args.base_url, args.model, args.sampled, args.workers,
                   args.wordings, args.sampled_wordings)  # fmt: skip
    summarize(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
