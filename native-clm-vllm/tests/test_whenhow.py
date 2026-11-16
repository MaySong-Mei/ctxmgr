"""When-versus-how conditions against the real pinned CLM classes, with a fake sandbox.

Needs the CLM runtime environment:
    PYTHONPATH=src uv run --project envs/clm python -m unittest tests.test_whenhow
Skipped in the root environment, where the official CLM packages are absent.
"""

import asyncio
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HAVE_CLM = importlib.util.find_spec("clm_contextbench") is not None
FILLER = "<<<FILLER " + "lorem ipsum dolor " * 7000 + "FILLER>>>"
CTX_FILE = "/tmp/.live_ctx/LIVE_CTX_MAIN.txt"


class FakeEnvironment:
    """Holds sandbox files; a command containing DELETE_FILLER edits the mirror."""

    def __init__(self):
        self.files = {}
        self.commands = []

    async def upload_file(self, source, target):
        self.files[target] = Path(source).read_text(encoding="utf-8")

    async def download_file(self, source, target):
        Path(target).write_text(self.files[source], encoding="utf-8")

    async def exec(self, command, timeout_sec=None):
        self.commands.append(command)
        if "DELETE_FILLER" in command:
            self.files[CTX_FILE] = self.files[CTX_FILE].replace(FILLER, "[filler removed]")
        return types.SimpleNamespace(stdout="", stderr="", return_code=0)


def response(content="", command=None):
    tool_calls = None
    if command is not None:
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": f'{{"command": "{command}"}}'},
            }
        ]
    message = types.SimpleNamespace(
        model_dump=lambda: {"role": "assistant", "content": content, "tool_calls": tool_calls}
    )
    usage = types.SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=0),
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)


def conversation(filler=True):
    turns = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task instruction"},
        {"role": "user", "content": "=== chunk 1 ===\nNEEDLE n1 keep me"},
        {"role": "assistant", "content": "ok"},
    ]
    if filler:
        turns.append({"role": "user", "content": FILLER})
    return turns


@unittest.skipUnless(HAVE_CLM, "needs the CLM runtime environment (envs/clm)")
class WhenHowTests(unittest.TestCase):
    def setUp(self):
        from ctxmgr.harness import clm_whenhow

        self.w = clm_whenhow
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logs = Path(self.temp.name)

    def make(self, cls, **kwargs):
        agent = cls(
            logs_dir=self.logs,
            model_name="openai/actor",
            api_base="http://127.0.0.1:1/v1",
            skill="none",
            **kwargs,
        )
        agent._live = conversation()
        return agent

    def run_async(self, coroutine):
        return asyncio.run(coroutine)

    def test_conversation_starts_above_trigger(self):
        agent = self.make(self.w.FixedTriggerStream)
        self.assertEqual(agent.trigger_tokens, self.w.TRIGGER_TOKENS)
        self.assertGreater(agent._tokens(), self.w.TRIGGER_TOKENS)

    def test_fixed_trigger_opens_a_two_turn_window_and_blocks_edits_after(self):
        from clm_harness.context_utils.context_string import render_editable

        agent = self.make(self.w.FixedTriggerStream)
        env = FakeEnvironment()
        self.run_async(agent.before_iteration(env))
        self.assertTrue(agent._ctx.open)
        self.assertIn("[HARNESS]", agent._live[-1]["content"])
        self.assertEqual(agent.events[0]["kind"], "harness_trigger")

        messages = agent._live
        rendered = render_editable(messages, protect=2)
        record = {"edit": "none", "before": 0, "after": 0, "prefix": 0}
        note, changed = agent._ctx._apply_edit(
            messages, rendered, rendered.replace(FILLER, "[x]"), 0, 30720, record
        )
        self.assertTrue(changed, note)
        self.assertNotIn(FILLER, "".join(str(m.get("content")) for m in messages))

        # The next iteration keeps the window open; the one after closes it.
        self.run_async(agent.before_iteration(env))
        self.assertTrue(agent._ctx.open)
        self.run_async(agent.before_iteration(env))
        self.assertFalse(agent._ctx.open)
        messages.append({"role": "user", "content": "more text " * 50})
        rendered = render_editable(messages, protect=2)
        note, changed = agent._ctx._apply_edit(
            messages, rendered, rendered.replace("more text", "x"), 0, 30720, record
        )
        self.assertFalse(changed)
        self.assertIn("NOT applied", note)
        self.assertEqual(agent._ctx.n_blocked, 1)

    def test_fixed_trigger_waits_below_threshold(self):
        agent = self.make(self.w.FixedTriggerStream)
        agent._live = conversation(filler=False)
        self.run_async(agent.before_iteration(FakeEnvironment()))
        self.assertFalse(agent._ctx.open)
        self.assertEqual(agent.events, [])

    def test_manager_edits_actor_context_and_logs_its_calls(self):
        agent = self.make(self.w.ManagedStream, manager_mode="fixed", manager_model="openai/mgr")
        env = FakeEnvironment()
        replies = [response("Removing filler.", "python3 edit.py DELETE_FILLER"), response("DONE")]
        with patch.object(agent, "_manager_completion", AsyncMock(side_effect=replies)) as call:
            self.run_async(agent.before_iteration(env))
        text = "".join(str(m.get("content")) for m in agent._live)
        self.assertNotIn(FILLER, text)
        self.assertIn("NEEDLE n1 keep me", text)
        event = agent.events[0]
        self.assertEqual(
            (event["kind"], event["commands"], event["applied_edits"]),
            ("harness_trigger", 1, 1),
        )
        self.assertLess(event["tokens_after"], event["tokens_before"])
        self.assertEqual(call.await_count, 2)
        # The manager saw the actor's context in its own fresh session.
        first_session = call.await_args_list[0].args[0]
        self.assertIn("NEEDLE n1 keep me", first_session[1]["content"])
        summary = agent.summary()
        self.assertEqual((summary["n_manager_calls"], summary["manager_prompt_tokens"]), (2, 2000))
        # Below the trigger again, so the manager is not called on the next iteration.
        with patch.object(agent, "_manager_completion", AsyncMock()) as again:
            self.run_async(agent.before_iteration(env))
        again.assert_not_awaited()

    def test_manager_hears_when_its_edit_changed_nothing(self):
        agent = self.make(self.w.ManagedStream, manager_mode="fixed")
        env = FakeEnvironment()
        replies = [response("", "ls /tmp"), response("", "DELETE_FILLER"), response("DONE")]
        with patch.object(agent, "_manager_completion", AsyncMock(side_effect=replies)) as call:
            self.run_async(agent.before_iteration(env))
        log = agent.events[0]["command_log"]
        self.assertEqual([entry["command"] for entry in log], ["ls /tmp", "DELETE_FILLER"])
        self.assertIn("NO change", log[0]["note"])
        self.assertIn("edit applied", log[1]["note"])
        # The manager session (one list, extended in place) carries the feedback as a tool result.
        session = call.await_args_list[-1].args[0]
        tool_results = [m["content"] for m in session if m["role"] == "tool"]
        self.assertIn("NO change", tool_results[0])
        self.assertIn("edit applied", tool_results[1])

    def test_actor_edits_are_never_applied_under_a_manager(self):
        from clm_harness.context_utils.context_string import render_editable

        agent = self.make(self.w.ManagedStream, manager_mode="fixed")
        messages = agent._live
        rendered = render_editable(messages, protect=2)
        record = {"edit": "none", "before": 0, "after": 0, "prefix": 0}
        note, changed = agent._ctx._apply_edit(
            messages, rendered, rendered.replace(FILLER, "[x]"), 0, 30720, record
        )
        self.assertFalse(changed)
        self.assertIn("separate manager", note)

    def test_watching_manager_can_decline(self):
        agent = self.make(self.w.ManagedStream, manager_mode="watch")
        agent._live = conversation(filler=False)
        env = FakeEnvironment()
        with patch.object(agent, "_manager_completion", AsyncMock()) as idle:
            self.run_async(agent.before_iteration(env))  # records the starting size only
        idle.assert_not_awaited()
        agent._live.append({"role": "user", "content": FILLER})
        with patch.object(
            agent, "_manager_completion", AsyncMock(return_value=response("NOT_NOW"))
        ):
            self.run_async(agent.before_iteration(env))
        event = agent.events[0]
        self.assertEqual(
            (event["kind"], event["declined"], event["commands"]), ("manager_watch", True, 0)
        )
        self.assertIn(FILLER, "".join(str(m.get("content")) for m in agent._live))

    def test_actor_request_marker_triggers_manager_once(self):
        from clm_contextbench import ClmStream

        agent = self.make(self.w.ManagedStream, manager_mode="request")
        env = FakeEnvironment()
        with patch.object(ClmStream, "_is_successful_submit", return_value=False):
            agent._is_successful_submit(
                "echo REQUEST_COMPACTION && echo READY_FOR_NEXT_OP", None, ""
            )
        replies = [response("", "DELETE_FILLER"), response("DONE")]
        with patch.object(agent, "_manager_completion", AsyncMock(side_effect=replies)):
            self.run_async(agent.before_iteration(env))
        self.assertEqual(agent.events[0]["kind"], "actor_request")
        with patch.object(agent, "_manager_completion", AsyncMock()) as again:
            self.run_async(agent.before_iteration(env))
        again.assert_not_awaited()

    def test_system_prompts_state_each_timing_rule(self):
        from clm_harness.clm_agent.harness import _SYSTEM_TEMPLATE
        from contextbench.stream import MARKER_PARAGRAPH

        official = _SYSTEM_TEMPLATE.replace("{{finish_instructions}}", MARKER_PARAGRAPH)
        fixed = self.make(self.w.FixedTriggerStream).system_prompt(official)
        self.assertIn("Free up context by editing that file", fixed)
        self.assertIn("harness decides when you compact", fixed)
        self.assertTrue(fixed.endswith(official.split(MARKER_PARAGRAPH, 1)[1]))
        managed = self.make(self.w.ManagedStream, manager_mode="fixed").system_prompt(official)
        self.assertNotIn("Free up context by editing that file", managed)
        self.assertIn("separate manager maintains your context", managed)
        self.assertNotIn("REQUEST_COMPACTION", managed)
        request = self.make(self.w.ManagedStream, manager_mode="request").system_prompt(official)
        self.assertIn("echo REQUEST_COMPACTION", request)

    def test_trial_config_passes_manager_settings(self):
        from ctxmgr.harness import clm_upstream

        task = self.logs / "task"
        task.mkdir()
        config = clm_upstream.trial_config(
            task,
            self.logs / "out",
            model="Qwen3.5-4B-BF16",
            base_url="http://127.0.0.1:9/v1",
            condition="mgr_watch",
            manager_model="Qwen3.5-27B-BF16",
            manager_base_url="http://127.0.0.1:10/v1",
        )
        kwargs = config.agent.kwargs
        self.assertEqual(config.agent.import_path, "ctxmgr.harness.clm_whenhow:ManagedStream")
        self.assertEqual(
            (kwargs["manager_mode"], kwargs["manager_model"], kwargs["manager_api_base"]),
            ("watch", "openai/Qwen3.5-27B-BF16", "http://127.0.0.1:10/v1"),
        )
        self.assertEqual(kwargs["skill"], "none")
        same = clm_upstream.trial_config(
            task, self.logs / "out2", model="m", base_url="http://b/v1", condition="mgr_fixed"
        )
        self.assertEqual(
            (same.agent.kwargs["manager_model"], same.agent.kwargs["manager_api_base"]),
            ("openai/m", "http://b/v1"),
        )
        plain = clm_upstream.trial_config(
            task, self.logs / "out3", model="m", base_url="http://b/v1", condition="clm_fixed"
        )
        self.assertNotIn("manager_mode", plain.agent.kwargs)


if __name__ == "__main__":
    unittest.main()
