"""Self/peer management: edit protocol and episode wiring, with fake clients (no GPU)."""

import json
import re
from types import SimpleNamespace

from test_harness import FakeCompletions

from ctxmgr.harness import management
from ctxmgr.harness.agents import PeerManagedAgent, Sampling, SelfManagedAgent
from ctxmgr.harness.episode import run_episode
from ctxmgr.tasks import generate_instance

BLOCK_RE = re.compile(
    r"^\[block (\d+) \| (\w+)\]\n(.*?)(?=\n\n\[block \d+ \||\Z)", re.DOTALL | re.MULTILINE
)
NEEDLE_RE = re.compile(r"^\[n\d+i\d+#[0-9a-f]+\].*$", re.MULTILINE)


def reply(content):
    usage = SimpleNamespace(
        prompt_tokens=1,
        completion_tokens=1,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage)


class NeedleKeeper:
    """Peer manager: keep only needle lines; delete every other editable block."""

    def __init__(self):
        self.requests = []

    def create(self, *, model, messages, **kwargs):
        self.requests.append({"model": model, "messages": messages})
        edits = []
        for i, _role, text in BLOCK_RE.findall(messages[-1]["content"]):
            i = int(i)
            if i < management.PROTECTED:
                continue
            needles = NEEDLE_RE.findall(text)
            edits.append(
                {"op": "replace", "id": i, "text": "\n".join(needles)}
                if needles
                else {"op": "delete", "ids": [i]}
            )
        return reply(json.dumps({"edits": edits}))


class SelfDeleter(FakeCompletions):
    """Actor that, when asked to manage, deletes all its own old replies."""

    def create(self, *, model, messages, **kwargs):
        last = messages[-1]["content"]
        if last.startswith("[CONTEXT MANAGEMENT]"):
            self.requests.append({"model": model, "messages": list(messages), **kwargs})
            ids = [
                i
                for i, m in enumerate(messages[:-1])
                if i >= management.PROTECTED and m["role"] == "assistant"
            ]
            return reply("Sure:\n" + json.dumps({"edits": [{"op": "delete", "ids": ids}]}))
        return super().create(model=model, messages=messages, **kwargs)


def client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def blocks(n):
    roles = ["system", "user"] + ["user", "assistant"] * n
    return [{"role": r, "content": str(i)} for i, r in enumerate(roles[: n + 2])]


def contents(messages):
    return [m["content"] for m in messages]


def test_apply_edits_applies_valid_edits_and_reports_unknown_ones():
    messages = blocks(2)
    result = management.apply_edits(
        messages,
        [{"op": "delete", "ids": [3]}, {"op": "replace", "id": 2, "text": "kept"}, {"op": "x"}],
    )
    assert contents(messages) == ["0", "1", "kept"]
    assert result.applied == [{"op": "delete", "ids": [3]}, {"op": "replace", "id": 2, "chars": 4}]
    assert "unknown edit" in result.error


def test_delete_naming_protected_ids_is_rejected_whole():
    # 2B's pilot reply: "delete every block from 0 up". Before, ids 0/1 were dropped and the rest
    # applied, wiping the context.
    messages = blocks(10)
    for edit in (
        {"op": "delete", "ids": list(range(12))},
        {"op": "delete", "from": 0, "to": 11},
        {"op": "delete", "ids": [1, 5]},
    ):
        result = management.apply_edits(messages, [edit])
        assert result.applied == [] and "protected" in result.error
    assert len(messages) == 12


def test_delete_with_missing_or_malformed_ids_is_rejected_whole():
    messages = blocks(4)
    for edit in (
        {"op": "delete", "ids": [3, 99]},
        {"op": "delete", "ids": [3, "4"]},
        {"op": "delete", "ids": [3, True]},
        {"op": "delete", "ids": []},
        {"op": "delete", "from": 4, "to": 3},
        {"op": "delete", "from": 2, "to": 9},
        {"op": "delete", "ids": [3], "from": 2, "to": 3},
        {"op": "delete"},
    ):
        result = management.apply_edits(messages, [edit])
        assert result.applied == [] and result.error, edit
    assert len(messages) == 6


def test_rejected_delete_does_not_block_other_edits():
    messages = blocks(4)
    result = management.apply_edits(
        messages, [{"op": "delete", "ids": [0, 2]}, {"op": "delete", "ids": [4]}]
    )
    assert contents(messages) == ["0", "1", "2", "3", "5"]
    assert result.applied == [{"op": "delete", "ids": [4]}] and "protected" in result.error


def test_delete_range_is_inclusive():
    messages = blocks(10)
    result = management.apply_edits(messages, [{"op": "delete", "from": 2, "to": 9}])
    assert contents(messages) == ["0", "1", "10", "11"]
    assert result.applied == [{"op": "delete", "ids": list(range(2, 10))}]
    assert result.error is None


def test_edit_with_id_and_text_but_no_op_is_a_replace():
    messages = blocks(2)
    result = management.apply_edits(messages, [{"id": 3, "text": "summary"}])
    assert contents(messages) == ["0", "1", "2", "summary"]
    assert result.applied == [{"op": "replace", "id": 3, "chars": 7}] and result.error is None
    # Still subject to the same checks as an explicit replace.
    result = management.apply_edits(messages, [{"id": 0, "text": "x"}, {"ids": [2]}])
    assert contents(messages) == ["0", "1", "2", "summary"]
    assert "bad replace id 0" in result.error and "unknown edit" in result.error


def test_instructions_offer_ranges():
    assert '"from"' in management.EDIT_INSTRUCTIONS and '"to"' in management.EDIT_INSTRUCTIONS


def test_parse_edits_tolerates_prose_and_rejects_garbage():
    assert management.parse_edits('ok {"edits": [{"op": "delete", "ids": [2]}]}')[1] is None
    assert management.parse_edits("no json here") == ([], "no JSON object in reply")
    assert management.parse_edits('{"edits": 3}')[1] == "missing 'edits' list"


def test_peer_manager_keeps_needles_under_pressure(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("needle_retention", "K4-C24", 42, instance)
    actor, manager = FakeCompletions(), NeedleKeeper()
    agent = PeerManagedAgent(
        client=client(actor),
        model="fake/actor",
        sampling=Sampling(),
        manager_client=client(manager),
        manager_model="fake/manager",
    )
    out = tmp_path / "run"
    outcome = run_episode(agent, instance, out, condition="peer_27b", manager_model="fake/manager")

    # Without management this level hits the wall in ~8 turns; the manager lets it finish.
    assert outcome["episode"]["terminated_by"] == "stream_exhausted"
    assert outcome["score"]["official"]["reward"] == 1
    edits = [json.loads(line) for line in (out / "edits.jsonl").read_text().splitlines()]
    assert edits and all(e["editor"] == "manager" and e["error"] is None for e in edits)
    assert all(e["context_after_o200k"] < e["context_before_o200k"] for e in edits)
    roles = {json.loads(line)["role"] for line in (out / "calls.jsonl").read_text().splitlines()}
    assert roles == {"actor", "manager"}
    # The manager never sees gold or future operations: only the rendered live context.
    assert all("gold" not in r["messages"][-1]["content"].lower() for r in manager.requests)


def test_self_management_exchange_stays_out_of_the_context(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("needle_retention", "K4-C24", 42, instance)
    actor = SelfDeleter()
    agent = SelfManagedAgent(
        client=client(actor), model="fake/actor", sampling=Sampling(), trigger=5000
    )
    out = tmp_path / "run"
    run_episode(agent, instance, out, condition="self")

    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert any(c["role"] == "self_manager" for c in calls)
    edits = [json.loads(line) for line in (out / "edits.jsonl").read_text().splitlines()]
    assert edits and all(e["editor"] == "actor" for e in edits)
    # The raw manager reply is logged next to the parsed edits.
    assert all(e["reply"].startswith("Sure:\n{") for e in edits)
    for path in sorted((out / "context_snapshots").glob("turn-*.json")):
        messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
        assert not any(m["content"].startswith("[CONTEXT MANAGEMENT]") for m in messages)
