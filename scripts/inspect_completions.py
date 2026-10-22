"""Inspect model replies in run directories: which hit the length cap, and do they loop?

Usage: python scripts/inspect_completions.py RUN_DIR [RUN_DIR ...]
Reads `calls.jsonl` (finish_reason, token counts) and the last context snapshot (reply text).
Pure stdlib, so it runs anywhere the run directories are.
"""

import json
import sys
from pathlib import Path


def repetition(text: str, n: int = 8) -> float:
    """Share of word n-grams that are repeats: ~0 for normal prose, near 1 for a loop."""
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def inspect(run: Path) -> None:
    calls = [json.loads(line) for line in (run / "calls.jsonl").read_text().splitlines()]
    last = sorted((run / "context_snapshots").glob("turn-*.json"))[-1]
    messages = json.loads(last.read_text(encoding="utf-8"))["messages"]
    replies = [m["content"] for m in messages if m["role"] == "assistant"]
    episode = json.loads((run / "episode.json").read_text())
    capped = sum(c["finish_reason"] == "length" for c in calls)
    print(f"\n=== {run}  ({episode['actor_model']}, {episode['terminated_by']})")
    print(f"calls={len(calls)} capped={capped} replies_in_final_context={len(replies)}")
    for call, reply in zip(calls, replies):
        flag = "CAP" if call["finish_reason"] == "length" else "   "
        print(
            f"turn {call['turn']:>3} {flag} out={call['completion_tokens']:>5} "
            f"rep8={repetition(reply):.2f} chars={len(reply)}"
        )
        if call["finish_reason"] == "length":
            head, tail = reply[:300].replace("\n", " | "), reply[-300:].replace("\n", " | ")
            print(f"    head: {head}\n    tail: {tail}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        inspect(Path(arg))
