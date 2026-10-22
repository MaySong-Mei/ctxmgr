"""Agents implementing ContextBench's runner protocol on an OpenAI-compatible (vLLM) server.

Conditions: `none` (append only), `self` (the actor edits its own context from inside its
conversation) and `peer_same` / `peer_27b` (a separate manager session edits it). Both managed
conditions share the edit protocol in `management.py` and trigger at the same context size.
"""

import time
from dataclasses import dataclass, field
from typing import ClassVar

import openai
from contextbench.tokens import count_tokens

from . import management


class ServerRejected(RuntimeError):
    """The server refused a request (e.g. prompt longer than the served `max_len`)."""


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048  # matches the benchmark's 2,048-token reserve
    thinking: bool = False  # Qwen3.5 thinks by default; off keeps reasoning text out of the context
    # Unset by default (not sent): greedy decoding, as in all runs before the sampling ablation.
    top_k: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    def request_kwargs(self, max_tokens: int | None = None) -> dict:
        extra = {"chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.top_k is not None:
            extra["top_k"] = self.top_k
        kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "extra_body": extra,
        }
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs


@dataclass
class ChatAgent:
    """One model, one live message list. Subclasses add context management."""

    client: openai.OpenAI
    model: str
    sampling: Sampling = field(default_factory=Sampling)
    role: str = "actor"
    messages: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    turn: int = 0
    # o200k wall (budget - reserve). A turn whose input already crosses it is not sent: the runner
    # ends the episode on `budget_wall` anyway, and the server might reject the oversized prompt.
    wall: int | None = None

    # --- ContextBench runner protocol -------------------------------------------------------------
    def reset(self, system: str, instruction: str) -> None:
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        self.calls = []
        self.turn = 0

    def on_user_turn(self, text: str) -> str:
        self.turn += 1
        self.before_turn()
        # The runner's invariant: the operation enters the context intact before the model acts.
        self.messages.append({"role": "user", "content": text})
        if self.wall is not None and count_tokens(self.messages)[0] > self.wall:
            return ""
        reply = self.complete(self.messages, role=self.role)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def context(self) -> list[dict]:
        return self.messages

    # --- hooks / helpers --------------------------------------------------------------------------
    def before_turn(self) -> None:
        """Context management runs here, after the previous turn was snapshotted."""

    def complete(
        self, messages: list[dict], *, role: str, client=None, model=None, max_tokens=None
    ) -> str:
        client, model = client or self.client, model or self.model
        start = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **self.sampling.request_kwargs(max_tokens)
            )
        except openai.BadRequestError as error:
            raise ServerRejected(str(error)) from error
        choice = response.choices[0]
        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None)
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
        self.calls.append(
            {
                "turn": self.turn,
                "role": role,
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "prompt_tokens_o200k": count_tokens(messages)[0],
                "cached_prompt_tokens": cached,
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                "finish_reason": choice.finish_reason,
                "latency_s": round(time.monotonic() - start, 3),
            }
        )
        return choice.message.content or ""


class NoManagementAgent(ChatAgent):
    """Condition `none`: append everything, never edit. Runs until the budget wall or the end."""


@dataclass
class ManagedAgent(ChatAgent):
    """Runs the shared edit protocol before a turn once the context exceeds `trigger` o200k tokens.

    The management exchange itself is never kept in the actor's context, so only the edits differ
    between conditions.
    """

    editor: ClassVar[str] = "actor"
    trigger: int = 18432  # 60% of the 30,720-token wall
    # Edit replies have their own output budget: the actor's 2,048 matches the benchmark reserve, but a
    # manager output never enters the actor's context, and 2,048 cut off 27B notes of ~54 needles.
    manager_max_tokens: int = 4096
    edits: list[dict] = field(default_factory=list)

    def reset(self, system: str, instruction: str) -> None:
        super().reset(system, instruction)
        self.edits = []

    def before_turn(self) -> None:
        before = count_tokens(self.messages)[0]
        if before <= self.trigger:
            return
        try:
            reply = self.request_edits(before)
        except ServerRejected as rejected:
            # e.g. prompt + manager_max_tokens over the served max_len: a failed event, not the end.
            reply, rejected_error = "", f"server rejected the edit request: {str(rejected)[:300]}"
        else:
            rejected_error = None
        proposed, error = management.parse_edits(reply)
        error = rejected_error or error
        result = management.apply_edits(self.messages, proposed)
        self.edits.append(
            {
                "turn": self.turn,
                "editor": self.editor,
                "trigger": "threshold",
                "context_before_o200k": before,
                "context_after_o200k": count_tokens(self.messages)[0],
                "applied": result.applied,
                "error": error or result.error,
                "reply": reply,
            }
        )

    def request_edits(self, size: int) -> str:
        raise NotImplementedError


@dataclass
class SelfManagedAgent(ManagedAgent):
    """Condition `self`: the actor is asked, inside its own conversation, to edit its context."""

    editor = "actor"

    def request_edits(self, size: int) -> str:
        request = (
            f"[CONTEXT MANAGEMENT] Your context is {size} tokens; the limit is {self.wall}.\n"
            f"{management.EDIT_INSTRUCTIONS}\n\nBlocks:\n{management.index(self.messages)}"
        )
        return self.complete(
            [*self.messages, {"role": "user", "content": request}],
            role="self_manager",
            max_tokens=self.manager_max_tokens,
        )


@dataclass
class PeerManagedAgent(ManagedAgent):
    """Conditions `peer_same` / `peer_27b`: a separate manager session reads and edits the context."""

    manager_client: openai.OpenAI | None = None
    manager_model: str | None = None
    editor = "manager"

    def request_edits(self, size: int) -> str:
        request = (
            f"Context of the assistant ({size} tokens; limit {self.wall}):\n\n"
            f"{management.render(self.messages)}\n\n{management.EDIT_INSTRUCTIONS}"
        )
        return self.complete(
            [
                {"role": "system", "content": management.PEER_SYSTEM},
                {"role": "user", "content": request},
            ],
            role="manager",
            client=self.manager_client,
            model=self.manager_model,
            max_tokens=self.manager_max_tokens,
        )
