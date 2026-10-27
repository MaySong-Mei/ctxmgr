"""Archive-with-recall arms: placeholders, RECALL round trips and episode logs, no GPU."""

import json
import re

from test_harness import GET_RE
from test_management import client, reply

from ctxmgr.harness import management
from ctxmgr.harness.agents import (
    ARCHIVE_INSTRUCTIONS,
    MaskAgent,
    NoManagementAgent,
    PeerManagedAgent,
    Sampling,
)
from ctxmgr.harness.episode import run_episode
from ctxmgr.tasks import generate_instance


def test_archive_keeps_deleted_and_replaced_text_and_leaves_placeholders():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "=== set batch (keys 0-9) ===\nSET K1 = a b"},
        {"role": "assistant", "content": "Stored.\nREADY_FOR_NEXT_OP"},
        {"role": "user", "content": "keep me"},
    ]
    archive = []
    result = management.apply_edits(
        messages,
        [{"op": "delete", "ids": [2, 3]}, {"op": "replace", "id": 4, "text": "short"}],
        archive,
    )
    assert result.error is None
    assert archive == [
        "=== set batch (keys 0-9) ===\nSET K1 = a b",
        "Stored.\nREADY_FOR_NEXT_OP",
        "keep me",
    ]
    assert [m["content"] for m in messages[2:]] == [
        "[ARCHIVED block: === set batch (keys 0-9) ===]",
        "[ARCHIVED block: Stored.]",
        "short",
    ]
    assert [m["role"] for m in messages] == ["system", "user", "user", "assistant", "user"]
    # Deleting a placeholder again removes it without archiving it.
    management.apply_edits(messages, [{"op": "delete", "ids": [2]}], archive)
    assert len(archive) == 3 and len(messages) == 4


def test_placeholder_of_an_answer_reply_leaves_no_sentinel():
    # Regression: "[ARCHIVED block: <<<ANSWER key=K1>>>]" made the KV grader swallow the next
    # real ANSWER block (found by the peer_27b_arc episode test below).
    stub = management.placeholder({"role": "assistant", "content": "<<<ANSWER key=K1>>>\nv"})
    assert "<" not in stub and ">" not in stub and "K1" in stub


def test_without_an_archive_delete_still_removes_blocks():
    messages = [{"role": "user", "content": str(i)} for i in range(4)]
    management.apply_edits(messages, [{"op": "delete", "ids": [2]}])
    assert [m["content"] for m in messages] == ["0", "1", "3"]


def test_archive_instructions_replace_gone_for_good():
    assert "gone for good" in management.EDIT_INSTRUCTIONS
    assert "gone for good" not in management.ARCHIVE_EDIT_INSTRUCTIONS
    assert "RECALL <key>" in management.ARCHIVE_EDIT_INSTRUCTIONS


SET_RE = re.compile(r"^SET (\S+) = (.+)$", re.MULTILINE)


class ContextReader:
    """Actor that answers a GET from what is in its context, and RECALLs the key when the value is
    not there but something was archived."""

    def __init__(self):
        self.requests = []

    def create(self, *, model, messages, **kwargs):
        self.requests.append(list(messages))
        last = messages[-1]["content"]
        seen = {k: v for m in messages for k, v in SET_RE.findall(m["content"])}
        if last.startswith("[RECALL "):
            keys = re.findall(r"^\[RECALL (\S+)\]", last, re.MULTILINE)
        else:
            keys = GET_RE.findall(last)
        archived = any(m["content"].startswith("[ARCHIVED block:") for m in messages)
        missing = [k for k in keys if k not in seen]
        if missing and archived and not last.startswith("[RECALL "):
            return reply("\n".join(f"RECALL {k}" for k in missing))
        answers = [f"<<<ANSWER key={k}>>>\n{seen[k]}\n<<<ANSWER END>>>" for k in keys if k in seen]
        return reply("\n".join([*answers, "READY_FOR_NEXT_OP"]))


def kv_r520(tmp_path):
    instance = tmp_path / "instance"
    generate_instance("kv_store", "R520", 42, instance)
    return instance


def test_mask_arc_archives_old_batches_and_recall_restores_every_answer(tmp_path):
    instance = kv_r520(tmp_path)
    reader = ContextReader()
    agent = MaskAgent(client=client(reader), model="fake/actor", sampling=Sampling(), trigger=8000)
    outcome = run_episode(agent, instance, tmp_path / "run", condition="mask_arc")
    episode, score = outcome["episode"], outcome["score"]
    assert episode["edit_prompt_version"] == "rule-mask-v1"
    assert episode["archive"] is True and episode["archived_blocks"] > 0
    assert episode["terminated_by"] in ("stream_exhausted", "submit")
    recalls = [
        json.loads(line) for line in (tmp_path / "run" / "recalls.jsonl").read_text().splitlines()
    ]
    assert recalls and all(r["hits"] >= 1 for r in recalls)
    assert score["official"]["reward"] == 1.0
    # The instruction tells the actor about RECALL; the protected blocks were never archived.
    assert ARCHIVE_INSTRUCTIONS.strip() in reader.requests[0][1]["content"]
    assert all(not m["content"].startswith("[ARCHIVED") for m in reader.requests[-1][:2])


def test_recall_of_an_unknown_key_says_so_and_none_r_never_archives(tmp_path):
    asked = []

    class AlwaysRecall:
        def create(self, *, model, messages, **kwargs):
            asked.append(messages[-1]["content"])
            if messages[-1]["content"].startswith("[RECALL "):
                return reply("READY_FOR_NEXT_OP")
            return reply("RECALL K99999")

    agent = NoManagementAgent(
        client=client(AlwaysRecall()), model="fake/actor", sampling=Sampling(), use_archive=True
    )
    agent.reset("sys", "task")
    assert agent.on_user_turn("=== op ===") == "READY_FOR_NEXT_OP"
    assert asked[-1] == "[RECALL K99999]\n(no archived line contains this key)"
    assert agent.archive == [] and agent.recalls[0]["hits"] == 0


def test_recall_rounds_are_capped(tmp_path):
    class Stubborn:
        def __init__(self):
            self.calls = 0

        def create(self, *, model, messages, **kwargs):
            self.calls += 1
            return reply("RECALL K1")

    stubborn = Stubborn()
    agent = NoManagementAgent(
        client=client(stubborn), model="fake/actor", sampling=Sampling(), use_archive=True
    )
    agent.reset("sys", "task")
    agent.on_user_turn("=== op ===")
    assert stubborn.calls == 1 + 3  # first reply + MAX_RECALL_ROUNDS


def test_default_conditions_do_not_recall_or_mention_the_archive(tmp_path):
    class RecallOnce:
        def __init__(self):
            self.calls = []

        def create(self, *, model, messages, **kwargs):
            self.calls.append(messages)
            return reply("RECALL K1\nREADY_FOR_NEXT_OP")

    actor = RecallOnce()
    agent = NoManagementAgent(client=client(actor), model="fake/actor", sampling=Sampling())
    agent.reset("sys", "task")
    agent.on_user_turn("=== op ===")
    assert len(actor.calls) == 1
    assert "Archive" not in actor.calls[0][1]["content"]


def test_peer_arc_manager_gets_archive_instructions_and_deletes_become_placeholders(tmp_path):
    instance = kv_r520(tmp_path)
    seen = []

    class ArchiveAll:
        def create(self, *, model, messages, **kwargs):
            seen.append(messages[-1]["content"])
            ids = [
                int(i)
                for i in re.findall(r"^\[block (\d+) \|", messages[-1]["content"], re.MULTILINE)
            ]
            return reply(json.dumps({"edits": [{"op": "delete", "from": 2, "to": max(ids)}]}))

    agent = PeerManagedAgent(
        client=client(ContextReader()),
        model="fake/actor",
        sampling=Sampling(),
        trigger=8000,
        use_archive=True,
        manager_client=client(ArchiveAll()),
        manager_model="fake/manager",
    )
    outcome = run_episode(agent, instance, tmp_path / "run", condition="peer_27b_arc")
    assert outcome["episode"]["edit_prompt_version"] == management.ARCHIVE_PROMPT_VERSION
    assert seen and "RECALL <key>" in seen[0] and "gone for good" not in seen[0]
    assert outcome["score"]["official"]["reward"] == 1.0
