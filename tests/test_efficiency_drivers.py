"""Control-flow tests for the efficiency drivers with the model servers and subprocesses stubbed.

These catch driver-level bugs (argument wiring, condition lists, agent interfaces) without a GPU.
They do not test model behaviour.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "actor,smoke,expected",
    [
        ("2B", True, 2),
        ("2B", False, 12),
        ("4B", False, 12),
        ("9B", False, 28),
        ("27B", False, 24),
    ],
)
def test_grid_episode_counts_and_wiring(tmp_path, monkeypatch, actor, smoke, expected):
    grid = load("efficiency_grid")
    runs = []

    def fake_run(command, check):
        assert check
        if "ctxmgr.harness" in command:
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True)
            runs.append(command)

    monkeypatch.setattr(grid.subprocess, "run", fake_run)
    monkeypatch.setattr(grid, "reset", lambda url: None)
    argv = ["efficiency_grid.py", "--actor", actor, "--base-url", "http://a/v1"]
    argv += ["--out", str(tmp_path)]
    if actor != "27B":
        argv += ["--manager-url", "http://m/v1"]
    if smoke:
        argv.append("--smoke")
    monkeypatch.setattr(sys, "argv", argv)
    grid.main()
    assert len(runs) == expected
    for command in runs:
        condition = command[command.index("--condition") + 1]
        if condition.startswith("peer_27b"):
            assert command[command.index("--manager-base-url") + 1] == "http://m/v1"
            assert actor != "27B"
        out = Path(command[command.index("--out") + 1])
        assert json.loads((out / "timing.json").read_text())["started_unix_s"] > 0


def synthetic_dataset(n=60):
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    data = []
    for i in range(n):
        context = [
            [f"Title {i}-{j}", [" ".join(words[(i + j + k) % 6] for k in range(25))] * 5]
            for j in range(10)
        ]
        data.append(
            {
                "_id": f"id{i:03d}",
                "question": f"Question {i}?",
                "answer": "alpha",
                "supporting_facts": [[f"Title {i}-0", 0], [f"Title {i}-1", 1]],
                "context": context,
            }
        )
    return data


class FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, model, messages, **kwargs):
        last = messages[-1]["content"]
        content = '{"answer": "alpha", "supporting_facts": [["x", 0]]}' if "FINAL" in last else "OK"
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=10),
            completion_tokens_details=None,
        )
        choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
        return SimpleNamespace(choices=[choice], usage=usage)


@pytest.mark.parametrize(
    "conditions,manager",
    [(["none", "self", "peer_27b"], True), (["none", "self", "peer_same"], False)],
)
def test_multidoc_runs_every_condition_and_length(tmp_path, monkeypatch, conditions, manager):
    pilot = load("multidoc_pilot")
    data = synthetic_dataset()
    monkeypatch.setattr(pilot, "load_data", lambda path: data)
    monkeypatch.setattr(pilot.openai, "OpenAI", FakeClient)
    resets = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        resets.append(request.full_url)
        return Response()

    monkeypatch.setattr(pilot.urllib.request, "urlopen", fake_urlopen)
    argv = ["multidoc_pilot.py", "--data", str(tmp_path / "d.json"), "--out", str(tmp_path / "o")]
    argv += ["--model", "Qwen/Qwen3.5-2B", "--base-url", "http://a/v1", "--cases", "1"]
    argv += ["--conditions", *conditions]
    if manager:
        argv += ["--manager-url", "http://m/v1", "--manager-model", "Qwen/Qwen3.5-27B"]
    monkeypatch.setattr(sys, "argv", argv)
    pilot.main()
    results = [json.loads(p.read_text()) for p in (tmp_path / "o").rglob("result.json")]
    assert len(results) == len(conditions) * 2
    assert {r["condition"] for r in results} == set(conditions)
    long = [r for r in results if r["long_context"]]
    assert all(r["input_documents_o200k"] > 40000 for r in long)
    assert all(r["terminated_by"] != "request_failed" for r in results), results
    managed = [r for r in long if r["condition"] != "none"]
    assert all(r["management_events"] > 0 for r in managed)
    assert len(resets) == len(results) * (2 if manager else 1)
