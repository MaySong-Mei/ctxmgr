"""Agents implementing ContextBench's runner protocol on an OpenAI-compatible (vLLM) server.

Only the no-management condition lives here so far: it is the baseline every other condition is
compared against, and it needs no edit tools. CLM self-management and peer managers (Track 2) should
subclass `ChatAgent` and change only what happens to `self.messages` between turns.
"""

import time
from dataclasses import dataclass, field

import openai
from contextbench.tokens import count_tokens


class ServerRejected(RuntimeError):
    """The server refused a request (e.g. prompt longer than the served `max_len`)."""


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048  # matches the benchmark's 2,048-token reserve
    thinking: bool = False  # Qwen3.5 thinks by default; off keeps reasoning text out of the context

    def request_kwargs(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.thinking}},
        }


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

    def complete(self, messages: list[dict], *, role: str) -> str:
        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, **self.sampling.request_kwargs()
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
                "model": self.model,
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
