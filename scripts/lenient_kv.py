"""Lenient KV score next to the official one (issue #3): format-insensitive memory check.

Usage: python scripts/lenient_kv.py RUN_DIR INSTANCE_DIR [RUN_DIR INSTANCE_DIR ...]
A GET counts as answered if any ANSWER block for that key, in any snapshot, contains the gold value
after dropping the `#hash` suffix, punctuation and case. The official ContextBench reward requires
the value with its `#hash`; 4B/9B often drop the hash while recalling the words exactly.
Pure stdlib; reads only snapshots and gold.json.
"""

import json
import re
import sys
from pathlib import Path

ANSWER_RE = re.compile(
    r"<<<\s*ANSWER\s+key\s*=\s*([^\s>]+)\s*>+(.*?)<<<\s*ANSWER\s+END\s*>>>", re.S | re.I
)


def normalize(value: str) -> str:
    value = re.sub(r"#[0-9a-f]{6,}", "", value)
    return " ".join(re.sub(r"[^a-z ]", " ", value.lower()).split())


def lenient_score(run_dir: Path, instance_dir: Path) -> float:
    gold = json.loads((instance_dir / "gold.json").read_text(encoding="utf-8"))["queries"]
    answers: dict[str, set[str]] = {}
    for snap in sorted((run_dir / "context_snapshots").glob("turn-*.json")):
        for message in json.loads(snap.read_text(encoding="utf-8"))["messages"]:
            if message["role"] != "assistant":
                continue
            for key, value in ANSWER_RE.findall(message["content"]):
                answers.setdefault(key.strip(), set()).add(normalize(value))
    hits = sum(
        1
        for q in gold
        if normalize(q["value"]) and any(normalize(q["value"]) in a for a in answers.get(q["key"], ()))
    )
    return hits / len(gold)


if __name__ == "__main__":
    args = sys.argv[1:]
    for run, inst in zip(args[::2], args[1::2]):
        official = json.loads((Path(run) / "score.json").read_text())["official"]["reward"]
        print(f"{run}\tofficial={official:.3f}\tlenient={lenient_score(Path(run), Path(inst)):.3f}")
