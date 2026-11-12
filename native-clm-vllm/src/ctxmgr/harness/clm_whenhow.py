"""When-versus-how conditions on top of the unchanged official CLM loop.

Every condition here inherits ``ClmStream`` (the paper's ContextBench agent): the same
system prompt, bash tool, stream protocol, budget gate and final-turn rule. Only two
things vary, and they vary independently:

* WHEN context is edited: the actor decides (native CLM), the harness decides (a fixed
  token trigger), the actor asks a manager, or a manager decides while watching.
* WHO edits: the actor itself, or a separate manager session with the same bash tool on
  the same mirrored context file.

Conditions (see ``CONDITIONS`` in ``clm_upstream``):

``clm_self_noskill``  native CLM without the task recipe (actor: when + how).
``clm_fixed``         actor edits, but only inside a window the harness opens at the
                      trigger; edits at other times are not applied (harness: when).
``mgr_fixed``         harness trigger; a manager session edits (harness: when, manager: how).
``mgr_request``       the actor runs ``echo REQUEST_COMPACTION``; a manager edits
                      (actor: when, manager: how).
``mgr_watch``         a manager looks after every 4,096 tokens of growth and decides
                      whether to edit (manager: when + how).

A same-size manager is a separate session of the actor's own server; a larger manager
is another server. Manager calls never reuse the actor's conversation, so they cannot
hit the actor's prefix cache; their tokens and latency are logged per call.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import litellm
from clm_contextbench import ClmStream
from clm_harness.clm_agent import harness as clm_harness_module
from clm_harness.context_env.env import ContextEnv
from clm_harness.context_utils.context_string import render_editable
from clm_harness.utils import tokens as tk
from clm_harness.utils.lm_compat import token_cap_key
from clm_harness.utils.tool_schemas import BASH_TOOL

TRIGGER_FRACTION = 0.6  # Of the enforced budget: 18,432 of 30,720 on ContextBench.
TRIGGER_TOKENS = 18432
WINDOW_TURNS = 2  # The trigger turn and the one after it.
REFIRE_GROWTH = 2048  # Re-trigger only after this much growth past the last attempt.
WATCH_EVERY = 4096  # A watching manager looks after this much growth.
MANAGER_MAX_COMMANDS = 6
REQUEST_MARKER = "REQUEST_COMPACTION"

_CLOSED_NOTE = (
    "\n[LIVE_CTX_MAIN.txt: edit NOT applied — in this run context edits are applied "
    "only when the harness asks you to compact.]"
)
_MANAGED_NOTE = (
    "\n[LIVE_CTX_MAIN.txt: edit NOT applied — in this run a separate manager maintains "
    "your context.]"
)


def _editing_guidance() -> str:
    """The official prompt's how-to-edit paragraphs, shared verbatim with managers."""
    template = clm_harness_module._SYSTEM_TEMPLATE
    start = template.index("**Locate text with code")
    end = template.index("{{finish_instructions}}")
    return template[start:end].strip()


MANAGER_SYSTEM = """You maintain the working memory of another AI agent.

The agent is doing the task shown to you and cannot see you. Its conversation is
mirrored to `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`. Whatever that file contains after your
edits is exactly what the agent will hold on its next turn. The agent is graded only on
what remains in its context, so keep everything it still needs and remove what it does
not.

Edit the file with the `bash` tool, one command per turn. Each result tells you whether
the file changed and the agent's new context size. When you are finished, reply with the
single word DONE and no tool call.

{guidance}
"""

MANAGER_REQUEST = """The agent's task:

<task>
{task}
</task>

The agent's context limit is {limit} tokens and it currently holds about {tokens}. Its
current context (the editable part of `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`) is:

<context>
{context}
</context>

{ask}"""

ASK_COMPACT = "Free up space now, keeping everything the agent still needs."
ASK_WATCH = (
    "First decide whether compacting now is worthwhile for the agent. If it is not, reply "
    "with the single word NOT_NOW and no tool call. Otherwise free up space, keeping "
    "everything the agent still needs."
)


_MANAGEMENT_HEADING = "## Managing your context\n"
_MIRROR_INTRO = "Your conversation is mirrored to "


def _split_system(system: str, finish: str) -> tuple[str, str, str]:
    """(before the section, management section, the finish paragraph and after)."""
    if system.count(_MANAGEMENT_HEADING) != 1 or not finish or system.count(finish) != 1:
        raise ValueError("Unexpected official CLM system prompt; cannot locate management text")
    before, section = system.split(_MANAGEMENT_HEADING, 1)
    management, after = section.split(finish, 1)
    return before + _MANAGEMENT_HEADING, management, finish + after


def _insert_before_marker(system: str, policy: str, finish: str) -> str:
    """Keep the official management section and append a condition policy to it."""
    head, management, tail = _split_system(system, finish)
    return head + management + policy + tail


def _replace_management(system: str, policy: str, finish: str) -> str:
    """Keep the shared goal/budget sentences; replace the self-editing instructions."""
    head, management, tail = _split_system(system, finish)
    if management.count(_MIRROR_INTRO) != 1:
        raise ValueError("Unexpected official CLM management section")
    common_goal = management.split(_MIRROR_INTRO, 1)[0]
    return head + common_goal + policy + tail


class GatedContextEnv(ContextEnv):
    """``ContextEnv`` whose actor-side edits are applied only while ``open``."""

    def __init__(self, *args: Any, closed_note: str = _CLOSED_NOTE, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.open = False
        self.closed_note = closed_note
        self.n_blocked = 0

    def apply_edit_unchecked(self, *args: Any) -> tuple[str, bool]:
        return super()._apply_edit(*args)

    def _apply_edit(self, messages, rendered, read_back, before, limit, edit_rec):
        if self.open:
            return super()._apply_edit(messages, rendered, read_back, before, limit, edit_rec)
        self.n_blocked += 1
        edit_rec.update(edit="blocked", before=before, after=before)
        return self.closed_note, False


def _gated_copy(old: ContextEnv, closed_note: str) -> GatedContextEnv:
    new = GatedContextEnv(
        paths=old.paths,
        budget=old.budget,
        protect=old.protect,
        persistent_bash=old.persistent_bash,
        command_timeout=old.command_timeout,
        observation_max_chars=old.observation_max_chars,
        obs_cfg=old.obs_cfg,
        allow_edit_growth=old.allow_edit_growth,
        context_budget_tokens=old.context_budget_tokens,
        closed_note=closed_note,
    )
    old.host_mirror.unlink(missing_ok=True)
    return new


class WhenHowMixin:
    """Adds one async hook before each loop iteration's budget check.

    Mix in before ``ClmStream`` (ContextBench) or ``ClmAgent`` (general tasks).
    """

    def __init__(self, *args: Any, closed_note: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = _gated_copy(self._ctx, closed_note)
        self._host_mirror = self._ctx.host_mirror
        self.events: list[dict[str, Any]] = []
        self._live: list[dict[str, Any]] | None = None

    def _tokens(self) -> int:
        return self._budget.count(self._live or [])

    async def before_iteration(self, environment: Any) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def system_prompt(self, system: str) -> str:
        """Rewrite the official system prompt for this condition (default: unchanged)."""
        return system

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        maybe = self._ckpt.maybe
        replay = self._resume.replay

        async def hooked(env: Any, messages: list[dict[str, Any]]) -> Any:
            self._live = messages
            if not getattr(self, "cb_done", False):
                await self.before_iteration(env)
            return await maybe(env, messages)

        def rewritten(messages: list[dict[str, Any]], *a: Any, **kw: Any) -> Any:
            out = replay(messages, *a, **kw)
            if not out or out[0].get("role") != "system":
                raise ValueError("Expected the official system prompt as the first message")
            out[0] = {**out[0], "content": self.system_prompt(out[0]["content"])}
            return out

        self._ckpt.maybe = hooked
        self._resume.replay = rewritten
        try:
            await super().run(instruction, environment, context)
        finally:
            self._ckpt.maybe = maybe
            self._resume.replay = replay
            if hasattr(self, "_cb_record"):
                self._cb_record["whenhow"] = self.summary()
            (self.logs_dir / "whenhow.json").write_text(
                json.dumps({"summary": self.summary(), "events": self.events}, indent=2) + "\n"
            )

    def summary(self) -> dict[str, Any]:
        return {"n_events": len(self.events), "n_blocked_actor_edits": self._ctx.n_blocked}


class FixedTriggerMixin(WhenHowMixin):
    """The actor edits, but only in a window the harness opens at a token trigger."""

    def __init__(self, *args: Any, trigger_tokens: int | None = None, **kwargs: Any):
        super().__init__(*args, closed_note=_CLOSED_NOTE, **kwargs)
        self.trigger_tokens = (
            int(trigger_tokens)
            if trigger_tokens
            else round(TRIGGER_FRACTION * self._budget.strict_target)
        )
        self._window = 0
        self._next_fire = self.trigger_tokens

    @staticmethod
    def name() -> str:
        return "clm-contextbench-fixed-trigger"

    def system_prompt(self, system: str) -> str:
        policy = (
            "**In this run the harness decides when you compact.** When your context "
            f"passes {self.trigger_tokens} tokens it asks you to compact; edits to "
            "the file are applied only in that turn and the next. Edits at other times "
            "are not applied.\n\n"
        )
        return _insert_before_marker(system, policy, self.finish_policy.prompt_fragment())

    async def before_iteration(self, environment: Any) -> None:
        tokens = self._tokens()
        if self._window > 0:
            self._window -= 1
            if self._window == 0:
                self._ctx.open = False
                self._next_fire = max(self.trigger_tokens, tokens + REFIRE_GROWTH)
                self.events[-1].update(tokens_after=tokens)
        if self._window == 0 and tokens < self.trigger_tokens:
            self._next_fire = self.trigger_tokens
        if self._window == 0 and tokens >= self._next_fire:
            self._window = WINDOW_TURNS
            self._ctx.open = True
            self.events.append({"kind": "harness_trigger", "tokens_before": tokens})
            self._live.append(
                {
                    "role": "user",
                    "content": (
                        f"[HARNESS] Your context is about {tokens} tokens (limit "
                        f"{self._budget.strict_target}). Compact it now by editing "
                        "/tmp/.live_ctx/LIVE_CTX_MAIN.txt. Edits are applied only during "
                        f"this compaction window: this turn and the next {WINDOW_TURNS - 1}."
                    ),
                }
            )


class ManagedMixin(WhenHowMixin):
    """A separate manager session edits the actor's mirrored context."""

    MODES = ("fixed", "request", "watch")

    def __init__(
        self,
        *args: Any,
        manager_mode: str = "fixed",
        manager_model: str | None = None,
        manager_api_base: str | None = None,
        manager_api_key: str | None = None,
        trigger_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, closed_note=_MANAGED_NOTE, **kwargs)
        if manager_mode not in self.MODES:
            raise ValueError(f"manager_mode must be one of {self.MODES}")
        self.manager_mode = manager_mode
        self.manager_model = manager_model or self.model_name
        self.manager_api_base = manager_api_base or self.api_base
        self.manager_api_key = manager_api_key or os.environ.get("CTXMGR_MANAGER_API_KEY")
        self.trigger_tokens = (
            int(trigger_tokens)
            if trigger_tokens
            else round(TRIGGER_FRACTION * self._budget.strict_target)
        )
        self._next_fire = self.trigger_tokens
        self._requested = False
        self._last_watch: int | None = None
        self.manager_calls: list[dict[str, Any]] = []

    @staticmethod
    def name() -> str:
        return "clm-contextbench-managed"

    def system_prompt(self, system: str) -> str:
        policy = (
            "Your conversation is mirrored to `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`, but **in "
            "this run a separate manager maintains your context**: between your turns it "
            "may delete or shorten earlier parts of this conversation. Do not edit that file "
            "yourself; your edits are not applied. Use the `bash` tool to do the task and "
            "follow its turn protocol.\n\n"
        )
        if self.manager_mode == "request":
            policy += (
                "When you want the manager to free up room, include "
                f"`echo {REQUEST_MARKER}` in a command (it can share a command with other "
                "steps via `&&`). The manager runs before your next turn.\n\n"
            )
        return _replace_management(system, policy, self.finish_policy.prompt_fragment())

    def summary(self) -> dict[str, Any]:
        calls = self.manager_calls
        return {
            **super().summary(),
            "manager_mode": self.manager_mode,
            "manager_model": self.manager_model,
            "n_manager_calls": len(calls),
            "manager_prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
            "manager_completion_tokens": sum(c.get("completion_tokens", 0) for c in calls),
            "manager_cached_tokens": sum(c.get("cached_tokens", 0) for c in calls),
            "manager_seconds": round(sum(c.get("seconds", 0.0) for c in calls), 2),
        }

    def _is_successful_submit(self, command: str, result: Any, tool_content: str) -> bool:
        if self.manager_mode == "request" and REQUEST_MARKER in (command or ""):
            self._requested = True
        return super()._is_successful_submit(command, result, tool_content)

    async def before_iteration(self, environment: Any) -> None:
        tokens = self._tokens()
        if self.manager_mode == "fixed":
            if tokens < self.trigger_tokens:
                self._next_fire = self.trigger_tokens
            elif tokens >= self._next_fire:
                after = await self.manage(environment, watch=False, reason="harness_trigger")
                self._next_fire = max(self.trigger_tokens, after + REFIRE_GROWTH)
        elif self.manager_mode == "request":
            if self._requested:
                self._requested = False
                await self.manage(environment, watch=False, reason="actor_request")
        else:
            if self._last_watch is None:
                self._last_watch = tokens
            elif tokens >= self._last_watch + WATCH_EVERY:
                self._last_watch = await self.manage(
                    environment, watch=True, reason="manager_watch"
                )

    async def _manager_completion(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.manager_model,
            "messages": messages,
            "tools": [BASH_TOOL],
            token_cap_key(self.manager_model): self.max_tokens,
            "api_base": self.manager_api_base,
            "timeout": clm_harness_module._LLM_TIMEOUT_SECONDS,
            "request_timeout": clm_harness_module._LLM_TIMEOUT_SECONDS,
            "num_retries": 2,
            "drop_params": True,
        }
        if self.manager_api_key:
            kwargs["api_key"] = self.manager_api_key
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.send_chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        return await asyncio.to_thread(litellm.completion, **kwargs)

    async def manage(self, environment: Any, *, watch: bool, reason: str) -> int:
        """Run one manager session on the actor's live messages; return tokens after."""
        actor = self._live
        ctx = self._ctx
        before = self._tokens()
        rendered = await ctx.write_mirror(environment, actor)
        task = str(actor[1].get("content", "")) if len(actor) > 1 else ""
        session: list[dict[str, Any]] = [
            {"role": "system", "content": MANAGER_SYSTEM.format(guidance=_editing_guidance())},
            {
                "role": "user",
                "content": MANAGER_REQUEST.format(
                    task=task,
                    limit=self._budget.strict_target,
                    tokens=before,
                    context=rendered,
                    ask=ASK_WATCH if watch else ASK_COMPACT,
                ),
            },
        ]
        event: dict[str, Any] = {
            "kind": reason,
            "tokens_before": before,
            "commands": 0,
            "applied_edits": 0,
            "declined": False,
            "calls": [],
        }
        for _ in range(MANAGER_MAX_COMMANDS + 1):
            started = time.monotonic()
            try:
                response = await self._manager_completion(session)
            except Exception as exc:  # noqa: BLE001 - a failed manager leaves the actor as is
                event["error"] = f"{type(exc).__name__}: {exc}"[:500]
                break
            call = _usage(response)
            call["seconds"] = round(time.monotonic() - started, 3)
            call["sent_tokens"] = tk.count_tokens(session)[0]
            self.manager_calls.append(call)
            event["calls"].append(call)
            message = response.choices[0].message.model_dump()
            message["content"] = message.get("content") or ""
            action = clm_harness_module._extract_tool_call(message)
            command = action.get("command") or ""
            if action["type"] not in ("command", "done") or not command:
                message.pop("tool_calls", None)
                event["declined"] = "NOT_NOW" in message["content"]
                break
            if event["commands"] >= MANAGER_MAX_COMMANDS:
                break
            session.append(message)
            event["commands"] += 1
            result = await ctx.execute(command, environment)
            read_back = await ctx.read_mirror(environment)
            note = ""
            if read_back is not None and read_back.strip() != rendered.strip():
                count = tk.count_tokens(actor)[0]
                limit = self._budget.strict_target or self.context_budget_tokens or 0
                prefix = tk.count_tokens(actor[: ctx.protect])[0]
                record = {"edit": "manager", "before": count, "after": count, "prefix": prefix}
                note, changed = ctx.apply_edit_unchecked(
                    actor, rendered, read_back, count, limit, record
                )
                if changed:
                    event["applied_edits"] += 1
                rendered = await ctx.write_mirror(environment, actor)
            output = ctx.format_tool_result(result)
            session.append(
                {
                    "role": "tool",
                    "tool_call_id": action.get("tool_call_id") or "",
                    "content": f"{output}{note}\n[agent context: ~{self._tokens()} tokens]",
                }
            )
        event["tokens_after"] = self._tokens()
        self.events.append(event)
        return event["tokens_after"]


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    out = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    if usage is None:
        return out
    out["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
    out["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        out["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0
    return out


def render_for_tests(messages: list[dict[str, Any]], protect: int = 2) -> str:
    """Expose the mirror rendering used for managers, for unit tests."""
    return render_editable(messages, protect=protect)


class FixedTriggerStream(FixedTriggerMixin, ClmStream):
    """Harness-timed self-management on ContextBench."""


class ManagedStream(ManagedMixin, ClmStream):
    """Manager-edited context on ContextBench."""
