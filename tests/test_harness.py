"""Harness tests with a fake OpenAI client: runner protocol, logs and scoring, no GPU needed."""

import json
import re
from types import SimpleNamespace

import httpx2
import openai

from ctxmgr.harness.agents import NoManagementAgent, Sampling
from ctxmgr.harness.episode import run_episode
from ctxmgr.tasks import generate_instance

GET_RE = re.compile(r"^GET\s+(\S+)", re.MULTILINE)


class FakeCompletions:
    """Answers KV GETs from a lookup (empty = know nothing) and always advances."""

    def __init__(self, values=None, reject_after=None):
        self.values = values or {}
        self.reject_after = reject_after
        self.requests = []

    def create(self, *, model, messages, **kwargs):
        self.requests.append({"model": model, "messages": list(messages), **kwargs})
        if self.reject_after is not None and len(self.requests) > self.reject_after:
            response = httpx2.Response(400, request=httpx2.Request("POST", "http://fake/v1"))
            raise openai.BadRequestError("prompt too long", response=response, body=None)
        blocks = [
            f"<<<ANSWER key={key}>>>\n{self.values[key]}\n<<<ANSWER END>>>"
            for key in GET_RE.findall(messages[-1]["content"])
            if key in self.values
        ]
        content = "\n".join([*blocks, "READY_FOR_NEXT_OP"])
        usage = SimpleNamespace(
            prompt_tokens=100 * len(messages),
            completion_tokens=10,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage)


def fake_agent(completions):
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return NoManagementAgent(client=client, model="fake/model", sampling=Sampling())


def kv_instance(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("kv_store", "R130", 42, instance)
    gold = json.loads((instance / "gold.json").read_text())
    return instance, {q["key"]: q["value"] for q in gold["queries"]}


def test_no_management_episode_logs_and_scores(tmp_path):
    instance, values = kv_instance(tmp_path)
    completions = FakeCompletions(values)
    out = tmp_path / "run"
    outcome = run_episode(fake_agent(completions), instance, out, condition="none")

    assert outcome["episode"]["terminated_by"] == "stream_exhausted"
    assert outcome["score"]["official"]["reward"] == 1
    for name in ("episode.json", "calls.jsonl", "score.json", "result.json"):
        assert (out / name).exists()
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == len(completions.requests)
    assert {c["role"] for c in calls} == {"actor"}
    assert all(c["prompt_tokens_o200k"] > 0 for c in calls)
    # No management: every request carries the full history, and thinking is off by default.
    first, last = completions.requests[0], completions.requests[-1]
    assert last["messages"][: len(first["messages"])] == first["messages"]
    assert first["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_model_that_answers_nothing_scores_zero(tmp_path):
    instance, _ = kv_instance(tmp_path)
    outcome = run_episode(
        fake_agent(FakeCompletions()), instance, tmp_path / "run", condition="none"
    )
    assert outcome["score"]["official"]["reward"] == 0


def test_server_rejection_is_recorded_not_raised(tmp_path):
    instance, values = kv_instance(tmp_path)
    out = tmp_path / "run"
    outcome = run_episode(
        fake_agent(FakeCompletions(values, reject_after=1)), instance, out, condition="none"
    )
    episode = json.loads((out / "episode.json").read_text())
    assert episode["terminated_by"] == "server_rejected"
    assert "prompt too long" in episode["error"]
    assert outcome["score"] is not None  # the one completed turn is still scored


def test_turn_past_the_wall_is_not_sent_to_the_server(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("needle_retention", "K4-C24", 42, instance)
    completions = FakeCompletions()
    out = tmp_path / "run"
    outcome = run_episode(fake_agent(completions), instance, out, condition="none")

    assert outcome["episode"]["terminated_by"] == "budget_wall"
    result = json.loads((out / "result.json").read_text())
    # Every turn but the wall-crossing one reached the server, and none was over the wall.
    assert len(completions.requests) == result["n_turns"] - 1
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert max(c["prompt_tokens_o200k"] for c in calls) <= 32768 - 2048


class NeverReady(FakeCompletions):
    """Answers but never writes the READY marker, like 2B in the GET phase of KV R1000."""

    def create(self, *, model, messages, **kwargs):
        reply = super().create(model=model, messages=messages, **kwargs)
        reply.choices[0].message.content = "ok"
        return reply


def test_actor_that_never_advances_stops_at_max_turns(tmp_path):
    instance, _ = kv_instance(tmp_path)
    out = tmp_path / "run"
    outcome = run_episode(fake_agent(NeverReady()), instance, out, condition="none")
    episode = outcome["episode"]
    assert episode["terminated_by"] == "max_steps"
    n_ops = json.loads((instance / "manifest.json").read_text())["n_ops"]
    assert episode["max_turns"] == 2 * n_ops + 10
    assert len(list((out / "context_snapshots").glob("turn-*.json"))) == episode["max_turns"]


def test_sampling_defaults_unchanged_and_ablation_flags_sent():
    from ctxmgr.harness.agents import Sampling

    greedy = Sampling().request_kwargs()
    assert greedy["temperature"] == 0.0 and greedy["top_p"] == 1.0
    assert "presence_penalty" not in greedy and "seed" not in greedy
    assert "top_k" not in greedy["extra_body"]
    sampled = Sampling(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, seed=42)
    kwargs = sampled.request_kwargs(max_tokens=4096)
    assert kwargs["extra_body"]["top_k"] == 20 and kwargs["presence_penalty"] == 1.5
    assert kwargs["seed"] == 42 and kwargs["max_tokens"] == 4096


def test_max_turns_per_op_raises_the_turn_cap(tmp_path):
    instance, _ = kv_instance(tmp_path)
    out = tmp_path / "run"
    outcome = run_episode(
        fake_agent(NeverReady()), instance, out, condition="none", max_turns_per_op=3
    )
    n_ops = json.loads((instance / "manifest.json").read_text())["n_ops"]
    assert outcome["episode"]["max_turns"] == 3 * n_ops + 10
