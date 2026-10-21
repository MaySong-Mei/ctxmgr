"""Self/peer management: edit protocol and episode wiring, with fake clients (no GPU)."""

import json
import re
from types import SimpleNamespace

import httpx2
import openai
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


def test_valid_reply_applies_all_its_edits():
    messages = blocks(2)
    result = management.apply_edits(
        messages, [{"op": "delete", "ids": [3]}, {"op": "replace", "id": 2, "text": "kept"}]
    )
    assert contents(messages) == ["0", "1", "kept"]
    assert result.applied == [{"op": "delete", "ids": [3]}, {"op": "replace", "id": 2, "chars": 4}]
    assert result.error is None


def test_one_invalid_edit_rejects_the_whole_reply():
    messages = blocks(2)
    result = management.apply_edits(
        messages,
        [{"op": "delete", "ids": [3]}, {"op": "replace", "id": 2, "text": "kept"}, {"op": "x"}],
    )
    assert contents(messages) == ["0", "1", "2", "3"]
    assert result.applied == []
    assert result.error.startswith("rejected whole reply") and "unknown edit" in result.error


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
        {"op": "delete", "from": 2, "to": 10**30},  # must not expand the range first
        {"op": "delete", "ids": [3], "from": 2, "to": 3},
        {"op": "delete"},
    ):
        result = management.apply_edits(messages, [edit])
        assert result.applied == [] and result.error, edit
    assert len(messages) == 6


def test_27b_consolidation_into_block_1_does_not_lose_the_deleted_blocks():
    # 27B (edit-v2 pilot): "copy the needles into block 1" + "delete 2..11". The copy is invalid, so
    # the delete must not run either; independent validation lost every needle here.
    messages = blocks(10)
    result = management.apply_edits(
        messages,
        [
            {"op": "delete", "ids": list(range(2, 12))},
            {"op": "replace", "id": 1, "text": "needles"},
        ],
    )
    assert len(messages) == 12 and result.applied == []
    assert "bad replace id 1" in result.error


def test_note_creates_then_appends_to_the_notes_block_after_the_instruction():
    messages = blocks(4)
    result = management.apply_edits(
        messages, [{"op": "note", "text": "n1"}, {"op": "delete", "from": 2, "to": 5}]
    )
    header = management.NOTES_HEADER
    assert contents(messages) == ["0", "1", f"{header}\nn1"]
    assert result.applied == [{"op": "note", "chars": 2}, {"op": "delete", "ids": [2, 3, 4, 5]}]
    messages += [{"role": "user", "content": "6"}]
    # v4: a later note only carries the new content and is appended.
    result = management.apply_edits(
        messages, [{"op": "note", "text": "n2"}, {"op": "delete", "ids": [3]}]
    )
    assert contents(messages) == ["0", "1", f"{header}\nn1\nn2"]
    assert result.error is None


def test_several_notes_in_one_reply_are_appended_in_order():
    messages = blocks(2)
    edits = [{"op": "note", "text": "a"}, {"op": "note", "text": "b"}]
    result = management.apply_edits(messages, edits)
    assert contents(messages)[2] == f"{management.NOTES_HEADER}\na\nb"
    assert len(result.applied) == 2 and result.error is None


def test_v5_replace_of_the_notes_block_is_read_as_a_note():
    messages = blocks(2)
    management.apply_edits(messages, [{"op": "note", "text": "old\nkeep"}])
    result = management.apply_edits(
        messages, [{"op": "replace", "id": 2, "text": "keep"}, {"op": "note", "text": "new"}]
    )
    assert contents(messages)[2] == f"{management.NOTES_HEADER}\nold\nkeep\nnew"
    assert len(messages) == 5
    assert {"op": "note", "chars": 4, "from_replace": True} in result.applied
    assert "read as a note" in result.error and "dropped 1 lines" in result.error


def test_v5_deletes_skip_the_notes_block():
    messages = blocks(2)
    management.apply_edits(messages, [{"op": "note", "text": "old"}])
    management.apply_edits(messages, [{"op": "delete", "ids": [2]}, {"op": "note", "text": "new"}])
    assert contents(messages)[2] == f"{management.NOTES_HEADER}\nold\nnew"


def test_v5_27b_rebuild_pattern_keeps_notes_and_drops_duplicates():
    # 27B (edit-v4, K8-C48): "delete from 2 to N" over the notes block plus a note re-copying every
    # needle. v5 keeps the notes block, adds only the new needles and deletes the chunks.
    messages = blocks(2)
    management.apply_edits(
        messages, [{"op": "note", "text": "n1\nn2"}, {"op": "delete", "ids": [3]}]
    )
    messages += [{"role": "user", "content": "chunk with n3"}]
    result = management.apply_edits(
        messages, [{"op": "delete", "from": 2, "to": 4}, {"op": "note", "text": "n1\nn2\nn3"}]
    )
    assert contents(messages) == ["0", "1", f"{management.NOTES_HEADER}\nn1\nn2\nn3"]
    assert "never deleted" in result.error and "dropped 2 lines" in result.error


def test_v5_context_view_labels_the_notes_block():
    messages = blocks(2)
    management.apply_edits(messages, [{"op": "note", "text": "n1"}])
    assert "[block 2 | notes]" in management.render(messages)
    assert "[block 2 | notes]" in management.index(messages)


def test_instructions_offer_notes_and_say_replies_are_atomic():
    text = management.EDIT_INSTRUCTIONS
    assert '"op": "note"' in text and "applied as a whole" in text


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
    assert result.applied == [{"op": "replace", "id": 3, "chars": 7, "inferred_op": True}]
    assert result.error is None
    # Still subject to the same checks as an explicit replace.
    result = management.apply_edits(messages, [{"id": 0, "text": "x"}, {"foo": [2]}])
    assert contents(messages) == ["0", "1", "2", "summary"]
    assert "bad replace id 0" in result.error and "unknown edit" in result.error


def test_edit_with_ids_or_range_but_no_op_is_a_delete_and_marked():
    messages = blocks(6)
    result = management.apply_edits(messages, [{"ids": [2]}, {"from": 4, "to": 5}])
    assert contents(messages) == ["0", "1", "3", "6", "7"]
    assert result.applied == [
        {"op": "delete", "ids": [2], "inferred_op": True},
        {"op": "delete", "ids": [4, 5], "inferred_op": True},
    ]
    # Explicit ops are not marked.
    result = management.apply_edits(messages, [{"op": "delete", "ids": [2]}])
    assert result.applied == [{"op": "delete", "ids": [2]}]


def test_replace_wins_over_delete_of_the_same_block():
    # e.g. "replace 5 with the needle lines, then delete 2..7": the needles must survive.
    messages = blocks(8)
    result = management.apply_edits(
        messages,
        [{"op": "replace", "id": 5, "text": "needles"}, {"op": "delete", "from": 2, "to": 7}],
    )
    assert contents(messages) == ["0", "1", "needles", "8", "9"]
    assert result.applied == [
        {"op": "replace", "id": 5, "chars": 7},
        {"op": "delete", "ids": [2, 3, 4, 6, 7]},
    ]
    assert "kept replaced blocks [5]" in result.error


def test_replacing_a_block_twice_keeps_the_last_and_logs_once():
    messages = blocks(2)
    result = management.apply_edits(
        messages,
        [{"op": "replace", "id": 2, "text": "a"}, {"op": "replace", "id": 2, "text": "bb"}],
    )
    assert contents(messages) == ["0", "1", "bb", "3"]
    assert result.applied == [{"op": "replace", "id": 2, "chars": 2}]
    assert "replaced twice" in result.error


def test_negative_ids_are_invalid_not_protected():
    messages = blocks(4)
    for edit in ({"op": "delete", "ids": [-1]}, {"op": "delete", "from": -1, "to": 3}):
        result = management.apply_edits(messages, [edit])
        assert result.applied == [] and "bad delete" in result.error


def test_instructions_offer_ranges_and_say_what_to_keep():
    text = management.EDIT_INSTRUCTIONS
    assert '"from"' in text and '"to"' in text
    assert "block 1" in text and "Never delete the only copy" in text


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


def test_episode_logs_edit_prompt_version(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("needle_retention", "K4-C4", 42, instance)
    agent = SelfManagedAgent(client=client(SelfDeleter()), model="fake/actor", sampling=Sampling())
    outcome = run_episode(agent, instance, tmp_path / "run", condition="self")
    assert outcome["episode"]["edit_prompt_version"] == management.EDIT_PROMPT_VERSION


class Rejecting:
    """Manager server that refuses every request (e.g. prompt + max_tokens over max_len)."""

    def __init__(self):
        self.kwargs = []

    def create(self, *, model, messages, **kwargs):
        self.kwargs.append(kwargs)
        response = httpx2.Response(400, request=httpx2.Request("POST", "http://fake/v1"))
        raise openai.BadRequestError("max_model_len exceeded", response=response, body=None)


def test_manager_has_its_own_output_budget_and_rejections_do_not_end_the_episode(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("needle_retention", "K4-C24", 42, instance)
    actor, manager = FakeCompletions(), Rejecting()
    agent = PeerManagedAgent(
        client=client(actor),
        model="fake/actor",
        sampling=Sampling(),
        manager_client=client(manager),
        manager_model="fake/manager",
    )
    out = tmp_path / "run"
    outcome = run_episode(agent, instance, out, condition="peer_27b", manager_model="fake/manager")

    assert {r["max_tokens"] for r in actor.requests} == {2048}
    assert {k["max_tokens"] for k in manager.kwargs} == {4096}
    assert outcome["episode"]["manager_max_tokens"] == 4096
    assert outcome["episode"]["terminated_by"] == "budget_wall"
    edits = [json.loads(line) for line in (out / "edits.jsonl").read_text().splitlines()]
    assert edits and all(e["error"].startswith("server rejected the edit request") for e in edits)


def test_v8_bare_edit_object_is_read_as_one_edit_and_marked():
    # 27B as `self` wrote replies like this in 26 of 31 v7 events ("missing 'edits' list").
    edits, error = management.parse_edits('{"op": "delete", "ids": [3]}')
    assert error is None and len(edits) == 1
    messages = blocks(2)
    result = management.apply_edits(messages, edits)
    assert contents(messages) == ["0", "1", "2"]
    assert result.applied == [{"op": "delete", "ids": [3], "wrapped_bare_edit": True}]
    # A bare object without an op or inferable fields is still an error.
    assert management.parse_edits('{"foo": 1}')[1] == "missing 'edits' list"
