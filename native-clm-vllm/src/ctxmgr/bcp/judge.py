"""Judge BrowseComp-Plus answers with a local model (standard library HTTP only).

    python -m ctxmgr.bcp.judge run --root ROOT --data DIR --base-url URL --model M

The grading template and the yes/no parsing follow BrowseComp-Plus's
``scripts_evaluation/evaluate_run.py`` (texttron/BrowseComp-Plus, MIT), which in turn
follows BrowseComp. The CLM paper judged with Qwen3.5-27B at temperature 0; here the
judge runs with thinking off. An episode without an answer file counts as incorrect.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from .agent import load_question

GRADER_TEMPLATE = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


_CORRECT = (
    re.compile(r"\*\*correct:\*\*\s*(yes|no)", re.IGNORECASE),
    re.compile(r"\*\*correct\*\*:\s*(yes|no)", re.IGNORECASE),
    re.compile(r"correct:\s*(yes|no)", re.IGNORECASE),
)


def parse_correct(text: str) -> bool | None:
    for pattern in _CORRECT:
        match = pattern.search(text or "")
        if match:
            return match.group(1).lower() == "yes"
    return None


def ask(base_url: str, model: str, prompt: str, api_key: str) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def judge_root(root: Path, data: Path, base_url: str, model: str, api_key: str) -> list[dict]:
    rows = []
    for episode in sorted(root.glob("*/*/*/episode.json")):
        out = episode.parent
        verdict_path = out / "judge.json"
        meta = json.loads(episode.read_text())
        if meta.get("terminated_by") == "initializing":
            continue  # still running or never started
        if verdict_path.exists():
            rows.append(json.loads(verdict_path.read_text()))
            continue
        question = load_question(data, meta["query_id"])
        answer = (out / "answer.txt").read_text() if (out / "answer.txt").exists() else ""
        verdict = {"query_id": meta["query_id"], "path": str(out.relative_to(root))}
        if not answer.strip():
            verdict.update(correct=False, judged=False, reason="no answer file")
        else:
            reply = ask(
                base_url,
                model,
                GRADER_TEMPLATE.format(
                    question=question["query"], response=answer, correct_answer=question["answer"]
                ),
                api_key,
            )
            correct = parse_correct(reply)
            verdict.update(correct=bool(correct), judged=correct is not None, judge_reply=reply)
        verdict_path.write_text(json.dumps(verdict, indent=2) + "\n")
        rows.append(verdict)
    return rows


def main(argv=None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    args = parser.parse_args(argv)
    rows = judge_root(
        args.root, args.data, args.base_url, args.model, os.environ.get("OPENAI_API_KEY", "")
    )
    print(json.dumps({"judged": len(rows), "correct": sum(r["correct"] for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
