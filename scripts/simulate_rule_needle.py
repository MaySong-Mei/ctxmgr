"""Offline simulation: what the task-aware `rule_needle` manager scores on Needle Retention.

Usage: python scripts/simulate_rule_needle.py OUT_DIR [levels...]
The actor is a fake that only advances (no GPU): this isolates what management alone can achieve.
Prints mean official reward, terminations and peak o200k context per level for `none` and
`rule_needle` over seeds 42-49.
"""

import statistics as st
import sys
from pathlib import Path
from types import SimpleNamespace

from ctxmgr.harness.agents import NoManagementAgent, RuleNeedleAgent, Sampling
from ctxmgr.harness.episode import run_episode
from ctxmgr.tasks import generate_instance

LEVELS = ["K4-C4", "K4-C8", "K4-C24", "K8-C48", "K8-C60", "K4-C96", "K2-C192"]


class Advance:
    """Minimal actor: acknowledges every operation and advances."""

    def create(self, *, model, messages, **kwargs):
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=1,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
        message = SimpleNamespace(content="READY_FOR_NEXT_OP")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage)


def main():
    out = Path(sys.argv[1])
    levels = sys.argv[2:] or LEVELS
    client = SimpleNamespace(chat=SimpleNamespace(completions=Advance()))
    print("| level | condition | mean reward | terminated_by | mean peak o200k |")
    print("|---|---|---|---|---|")
    for level in levels:
        for name, cls in (("none", NoManagementAgent), ("rule_needle", RuleNeedleAgent)):
            rewards, ends, peaks = [], {}, []
            for seed in range(42, 50):
                instance = out / "instances" / f"{level}-s{seed}"
                if not instance.exists():
                    generate_instance("needle_retention", level, seed, instance)
                agent = cls(client=client, model="fake/advance", sampling=Sampling())
                result = run_episode(agent, instance, out / name / f"{level}-s{seed}", condition=name)
                score = result["score"]["official"]
                rewards.append(score["reward"])
                peaks.append(score["agent_peak_tokens"])
                end = result["episode"]["terminated_by"]
                ends[end] = ends.get(end, 0) + 1
            print(f"| {level} | {name} | {st.mean(rewards):.3f} | {ends} | {st.mean(peaks):,.0f} |")


if __name__ == "__main__":
    main()
