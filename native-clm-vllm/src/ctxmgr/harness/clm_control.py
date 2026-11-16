"""Append-only control for the official CLM loop, outside the upstream package.

The CLM treatment continues to instantiate the official ``ClmStream`` directly.
This control inherits that same loop, stream, tool execution and budget policy.
Only management instructions and acceptance of context-file edits differ.
It is a new matched control, not Ruthvik's historical plain-chat baseline.
"""

from clm_contextbench import ClmStream
from clm_harness.context_env.env import ContextEnv
from contextbench.stream import MARKER_PARAGRAPH

_MANAGEMENT_HEADING = "## Managing your context\n"
_MIRROR_INTRO = "Your conversation is mirrored to "
_DISABLED_NOTE = (
    "\n[LIVE_CTX_MAIN.txt: edit REJECTED — context management is disabled; "
    "the conversation remains unchanged and append-only.]"
)
_DISABLED_POLICY = (
    "Your conversation is append-only. Earlier messages cannot be deleted, replaced, "
    "or compacted. `/tmp/.live_ctx/LIVE_CTX_MAIN.txt` is a refreshed view of the "
    "conversation; changes to that file are rejected and never change the messages "
    "you receive. Do not attempt context edits. Use the same `bash` tool to solve "
    "the task and follow its turn protocol.\n\n"
)


def without_management_instructions(system):
    """Replace only the pinned system management section, keeping common text."""
    if system.count(_MANAGEMENT_HEADING) != 1 or system.count(MARKER_PARAGRAPH) != 1:
        raise ValueError("Unexpected official CLM system prompt; cannot isolate management text")
    before, section = system.split(_MANAGEMENT_HEADING, 1)
    management, after = section.split(MARKER_PARAGRAPH, 1)
    if management.count(_MIRROR_INTRO) != 1:
        raise ValueError("Unexpected official CLM management section")
    # Keep the shared task-success goal, budget value and per-result size explanation.
    common_goal = management.split(_MIRROR_INTRO, 1)[0]
    return before + _MANAGEMENT_HEADING + common_goal + _DISABLED_POLICY + MARKER_PARAGRAPH + after


class AppendOnlyContextEnv(ContextEnv):
    """Run normal tools but never accept their changes to the live conversation."""

    def _apply_edit(self, messages, rendered, read_back, before, limit, edit_rec):
        # This check is at the synchronization boundary, not a shell-command filter:
        # aliases, subprocesses and arbitrary file overwrites cannot bypass it.
        self.n_ctx_rejected += 1
        edit_rec.update(edit="rejected_management_disabled", before=before, after=before)
        return _DISABLED_NOTE, False

    async def step(self, command, messages, *, environment, pending=None):
        obs = await super().step(command, messages, environment=environment, pending=pending)
        # Preserve token counting/output capping; remove only advice to perform a
        # disabled action. The common budget controller and its final turn are unchanged.
        obs.readout = obs.readout.replace(
            f" — OVER; compact {self.paths.ctx_file} now",
            " — OVER; context management is disabled",
        )
        if obs.touched_ctx and obs.notes and obs.notes != _DISABLED_NOTE:
            obs.notes = (
                "\n[LIVE_CTX_MAIN.txt: conversation unchanged; context management is disabled.]"
            )
        return obs


class NoManagementStream(ClmStream):
    """Official CLM loop with management instructions and context rewrites disabled."""

    def __init__(self, *args, skill="none", **kwargs):
        if skill is not None and str(skill).strip().lower() != "none":
            raise ValueError("The append-only control requires skill='none'")
        if int(kwargs.get("max_num_retry_on_limit", 0)) != 0:
            raise ValueError("The append-only control forbids history rollback retries")
        if kwargs.get("resume_from"):
            raise ValueError("The append-only control requires a fresh episode, without resume")
        super().__init__(*args, skill="none", **kwargs)
        old = self._ctx
        self._ctx = AppendOnlyContextEnv(
            paths=old.paths,
            budget=old.budget,
            protect=old.protect,
            persistent_bash=old.persistent_bash,
            command_timeout=old.command_timeout,
            observation_max_chars=old.observation_max_chars,
            obs_cfg=old.obs_cfg,
            allow_edit_growth=old.allow_edit_growth,
            context_budget_tokens=old.context_budget_tokens,
        )
        old.host_mirror.unlink(missing_ok=True)
        self._host_mirror = self._ctx.host_mirror

    @staticmethod
    def name():
        return "clm-contextbench-append-only-control"

    async def run(self, instruction, environment, context):
        replay = self._resume.replay

        def append_only_prompt(messages, *args, **kwargs):
            out = replay(messages, *args, **kwargs)
            if not out or out[0].get("role") != "system":
                raise ValueError("Expected the official system prompt as the first message")
            out[0] = {
                **out[0],
                "content": without_management_instructions(out[0]["content"]),
            }
            return out

        self._resume.replay = append_only_prompt
        try:
            await super().run(instruction, environment, context)
        finally:
            self._resume.replay = replay
