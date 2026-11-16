"""Official primary score, plus separately named stricter project diagnostics."""

import json
from pathlib import Path

from contextbench import grade_run, load_instance
from contextbench.graders.base import messages_text, normalize
from contextbench.graders.kv import ANSWER_RE
from contextbench.tokens import count_tokens, snapshot_paths

from .core import BUDGET, RESERVE, require_exact_tokens, validate_instance


def score_run(instance_dir: Path | str, run_dir: Path | str) -> dict:
    require_exact_tokens()
    validate_instance(instance_dir)
    paths = snapshot_paths(run_dir)
    if not paths:
        raise ValueError(
            "Per-turn context snapshots required; report/trajectory fallback forbidden"
        )
    frames = []
    token_sizes = []
    for path in paths:
        snapshot = json.loads(path.read_text())
        messages = snapshot.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Missing/empty context in snapshot: {path}")
        tokens, method = count_tokens(messages)
        if method != "tiktoken:o200k_base" or snapshot.get("token_method") != method:
            raise ValueError(f"Wrong/missing token method: {path}")
        if snapshot.get("tokens") != tokens:
            raise ValueError(f"Snapshot token count does not match its messages: {path}")
        frames.append(messages)
        token_sizes.append(tokens)
    instance = load_instance(instance_dir)
    primary = grade_run(run_dir, instance.gold)
    if primary["graded_from"] != "snapshots" or not primary["grader_valid"]:
        raise ValueError("Official grader did not read valid snapshots")
    diagnostic = {
        "observed_peak_tokens": max(token_sizes),
        "snapshot_budget_exceeded": max(token_sizes) > BUDGET - RESERVE,
    }
    if instance.family == "needle_retention":
        expected = {normalize(line) for line in instance.gold["answer_lines"]}
        lines = set(messages_text(frames[-1]).splitlines())
        exact = set(instance.gold["answer_lines"]) & lines
        normalized = {normalize(line) for line in lines}
        diagnostic.update(
            {
                "required_info_loss": len(expected - normalized) / len(expected),
                "strict_verbatim_retention": len(exact) / len(expected),
            }
        )
    else:
        answers = {}
        for frame in frames:
            for key, value in ANSWER_RE.findall(messages_text(frame)):
                answers.setdefault(normalize(key), set()).add(normalize(value))
        queries = instance.gold["queries"]
        matched = sum(normalize(q["value"]) in answers.get(q["key"], set()) for q in queries)
        diagnostic.update(
            {
                "strict_exact_accuracy": matched / len(queries),
                "conflicting_answer_keys": sorted(
                    key for key, values in answers.items() if len(values) > 1
                ),
            }
        )
    return {"family": instance.family, "official": primary, "diagnostics": diagnostic}
