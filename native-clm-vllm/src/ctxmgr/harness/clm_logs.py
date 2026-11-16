"""Translate saved official CLM artifacts; never mutate the agent's live context."""

import json
import re
import shutil
from pathlib import Path

from contextbench.tokens import count_tokens, snapshot_paths

from ctxmgr.tasks import score_run, validate_instance
from ctxmgr.tasks.core import UPSTREAM_COMMIT

from .clm_upstream import CLM_COMMIT, write_json


def _read(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def export_calls(native, model):
    """Native export aligns successful replies with requests; do not guess missing usage."""
    frames = [(p, _read(p, {})) for p in snapshot_paths(native)]
    sent = [(p, f) for p, f in frames if f.get("kind") == "agent"]
    trajectory = _read(native / "trajectory.ctx.json", {})
    check = trajectory.get("extra", {}).get("export_check", {})
    steps = [
        s
        for segment in trajectory.get("segments", [])
        for s in segment.get("steps", [])
        if s.get("source") == "agent"
    ]
    aligned = check.get("consistent") is True and len(steps) == len(sent)
    timings = [t for t in _read(native / "timing.json", []) if "llm_s" in t]
    calls, edits = [], []
    for index, (path, frame) in enumerate(sent):
        metrics = (steps[index].get("metrics") or {}) if aligned else {}
        timing = timings[index] if len(timings) == len(sent) else {}
        calls.append(
            {
                "turn": frame["step"],
                "role": "actor",
                "model": model,
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "prompt_tokens_o200k": count_tokens(frame["messages"])[0],
                "cached_prompt_tokens": metrics.get("cached_tokens"),
                "reasoning_tokens": None,
                "latency_s": timing.get("llm_s"),
                "snapshot": path.name,
                "metrics_available": bool(metrics),
            }
        )
        if timing.get("ctx_touched") or timing.get("ctx_synced"):
            # Only the native receipt reports size immediately after the edit.
            # Next prompt size also includes the new command/result/op, so isn't a substitute.
            observation = (steps[index].get("observation") or {}) if aligned else {}
            text = "\n".join(str(r.get("content", "")) for r in observation.get("results", []))
            receipt = re.search(r"\[LIVE_CTX_MAIN\.txt: edit applied[^\]]*?~(\d+)->(\d+)", text)
            edits.append(
                {
                    "turn": frame["step"],
                    "editor": "actor",
                    "trigger": "self_initiated",
                    "op": "bash_live_context",
                    "applied": bool(timing.get("ctx_synced")),
                    "context_before_o200k": frame["tokens"],
                    "context_after_o200k": int(receipt[2]) if receipt else None,
                    "command_excerpt": timing.get("cmd"),
                    "return_code": timing.get("return_code"),
                }
            )
    complete = (
        aligned
        and bool(calls)
        and all(
            c["prompt_tokens"] is not None and c["completion_tokens"] is not None for c in calls
        )
    )
    return calls, edits, {"per_call_metrics_complete": complete, "native_export_check": check}


def collect_run(instance, out, *, metadata=None, failure=None):
    instance, out = Path(instance), Path(out)
    validate_instance(instance)
    metadata = dict(metadata or _read(out / "episode.json", {}))
    manifest = _read(instance / "manifest.json", {})
    if metadata.get("instance_sha256") != manifest["sha256"]:
        raise ValueError("This run belongs to a different task instance")
    native = out / "harbor" / "trial" / "agent"
    # Byte-preserve native snapshots. Track 1 checks their exact counts before scoring.
    target = out / "context_snapshots"
    paths = snapshot_paths(native)
    if paths:
        target.mkdir(exist_ok=True)
        for path in paths:
            dest = target / path.name
            if dest.exists() and dest.read_bytes() != path.read_bytes():
                raise ValueError(f"Refusing to overwrite a different collected snapshot: {dest}")
            shutil.copyfile(path, dest)
        if {p.name for p in snapshot_paths(out)} != {p.name for p in paths}:
            raise ValueError("Collected snapshots include stale/unrecognized turns")
    read_errors = []

    def read_record(path):
        try:
            return _read(path, {})
        except (ValueError, OSError) as exc:
            read_errors.append(
                {
                    "file": str(path),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            return {}

    stream = read_record(native / "interactive_stream.json")
    usage = read_record(native / "usage.json")
    harbor_result = read_record(out / "harbor" / "trial" / "result.json")
    failure = failure or harbor_result.get("exception_info") or metadata.get("error")
    native_reason = stream.get("terminated_by")
    reason = native_reason or ("error" if failure else "interrupted")
    overflow = metadata.get("native_server_context_overflow")
    overflow_evidence = "not_observed"
    if isinstance(overflow, bool):
        overflow_evidence = "official_agent_flag"
    else:
        overflow = None
        # At these pinned revisions ClmStream sets budget_wall only for a
        # finalized budget OR a server overflow. Recover only the unambiguous
        # old-run case; do not guess when finalization or usage is missing.
        if (
            metadata.get("clm_commit") == CLM_COMMIT
            and metadata.get("contextbench_commit") == UPSTREAM_COMMIT
            and native_reason == "budget_wall"
            and usage.get("budget_finalized") is False
        ):
            overflow = True
            overflow_evidence = "pinned_native_budget_wall_without_finalization"
        elif native_reason == "budget_wall" and usage.get("budget_finalized") is True:
            overflow_evidence = "budget_finalized_server_overflow_unknown"
    if overflow is True:
        reason = "server_rejected"
        failure = failure or {
            "exception_type": "ServerContextOverflow",
            "exception_message": "The model server rejected a context; native snapshots and partial score are retained.",
        }
    if reason == "interrupted" and failure is None:
        failure = {
            "exception_type": "IncompleteTrial",
            "exception_message": "Native stream has no terminal reason; partial run only",
        }
    score, score_error = None, None
    if paths:
        try:
            score = score_run(instance, out)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            score_error = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
    else:
        score_error = {
            "exception_type": "MissingSnapshots",
            "exception_message": "No model snapshots; no valid score",
        }
    try:
        calls, edits, accounting = export_calls(native, metadata.get("actor_model"))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        calls, edits = [], []
        accounting = {"per_call_metrics_complete": False, "export_error": str(exc)}
        read_errors.append({"exception_type": type(exc).__name__, "exception_message": str(exc)})
    metadata.update(
        terminated_by=reason,
        native_terminated_by=native_reason,
        server_context_overflow=overflow,
        server_context_overflow_evidence=overflow_evidence,
        error=failure or score_error or (read_errors[0] if read_errors else None),
        score_error=score_error,
        native_read_errors=read_errors,
        accounting=accounting,
        native_logs=str(native),
        native_usage=usage,
    )
    write_json(out / "episode.json", metadata)
    write_json(
        out / "result.json",
        {
            "terminated_by": reason,
            "native_terminated_by": native_reason,
            "server_context_overflow": overflow,
            "error": metadata["error"],
            "n_ops_delivered": stream.get("n_ops_delivered"),
            "n_ops_total": stream.get("n_ops_total"),
            "source": "official ClmStream / Harbor",
        },
    )
    write_json(out / "score.json", score)
    _jsonl(out / "calls.jsonl", calls)
    _jsonl(out / "edits.jsonl", edits)
    if stream:
        write_json(out / "interactive_stream.json", stream)
    return {"episode": metadata, "score": score}
