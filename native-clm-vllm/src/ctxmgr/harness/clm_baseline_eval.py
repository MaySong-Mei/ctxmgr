"""Run the original team chat baseline with configurable shared sampling settings.

This is Ruthvik/May's NoManagementAgent and ContextBench episode runner, not
the separate NoManagementStream control. Only constructor sampling and host
provenance are added here; task delivery, prompts, stopping and scoring remain
those of the original baseline.
"""

import json
import os
import sys
import traceback
from contextlib import suppress
from pathlib import Path

import openai

from . import episode
from .agents import NoManagementAgent, Sampling
from .clm_settings import resolve_settings, settings_from_overrides


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run_baseline(
    instance: Path,
    out: Path,
    *,
    model: dict,
    base_url: str,
    settings=None,
    served_max_len=81920,
) -> int:
    """Return 0 for a scored run, 1 for an error; never reuse an existing run."""
    instance, out = Path(instance), Path(out)
    # The upstream runner also checks this. Do it before exception handling so
    # a retry cannot add failure metadata to, or overwrite, a previous run.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Baseline output is not empty: {out}; use a new directory")
    from .clm_official import CLIENT_TRANSPORT, validate_model_manifest, verify_team_sources

    team_sources = verify_team_sources()
    validate_model_manifest(model)
    settings = (
        resolve_settings("official")
        if settings is None
        else settings_from_overrides(settings["profile"], settings["profile_overrides"])
    )
    out.mkdir(parents=True, exist_ok=True)
    sampling = Sampling(
        temperature=settings["sampling"]["temperature"],
        top_p=settings["sampling"]["top_p"],
        max_tokens=settings["sampling"]["max_tokens"],
        thinking=settings["sampling"]["enable_thinking"],
    )
    provenance = {
        "condition": "none",
        "implementation": "ctxmgr.harness.agents:NoManagementAgent",
        "runner": "ctxmgr.harness.episode:run_episode",
        "team_source_commit": team_sources["commit"],
        "team_source_sha256": team_sources["sha256"],
        "profile": settings["profile"],
        "profile_overrides": settings["profile_overrides"],
        "backend": model["backend"],
        "server_version": model["server_version"],
        "precision": model["precision"],
        "model_sha256_kind": model["model_sha256_kind"],
        "client_transport": CLIENT_TRANSPORT,
        "actor_model": model.get("model_alias"),
        "model_revision": model.get("revision", model.get("model_revision")),
        "model_sha256": model.get("sha256", model.get("model_sha256")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "served_max_len": served_max_len,
        "budget_ruler": "o200k",
        "budget_final_turn_policy": "original_chat_runner:no_extra_final_turn",
        "historical_sampling_reproduced": vars(sampling)
        == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048, "thinking": False},
        "sampling": vars(sampling).copy(),
    }
    client, agent, error = None, None, None
    status = "failed"
    try:
        client = openai.OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "local"))
        agent = NoManagementAgent(client=client, model=model["model_alias"], sampling=sampling)
        outcome = episode.run_episode(
            agent,
            instance,
            out,
            condition="none",
            served_max_len=served_max_len,
            precision=provenance["precision"],
        )
        metadata = outcome["episode"]
        error = metadata.get("error")
        if metadata.get("terminated_by") == "server_rejected" and not error:
            error = "Original baseline runner reported server_rejected"
        if not outcome.get("score") and not error:
            error = "Original baseline runner returned no score"
        # Only add host provenance after the original runner finishes. Sampling
        # retains the original runner's field names, including 'thinking'.
        metadata.update(provenance)
        metadata["error"] = error
        metadata["status"] = "failed" if error else "completed"
        _write_json(out / "episode.json", metadata)
        status = metadata["status"]
    except Exception as exc:  # noqa: BLE001 - retain diagnostics and fail the job
        error = f"{type(exc).__name__}: {exc}"
        (out / "baseline-failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        metadata_path = out / "episode.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata.update(provenance)
        metadata.update(
            status="failed",
            error=error,
            actor_model=model.get("model_alias"),
        )
        metadata.setdefault("terminated_by", "runtime_error")
        _write_json(metadata_path, metadata)
        # The original episode runner only handles ServerRejected. Other errors
        # may bypass its call-log write; preserve every completed call we have.
        if agent is not None and not (out / "calls.jsonl").exists():
            (out / "calls.jsonl").write_text(
                "".join(json.dumps(call) + "\n" for call in agent.calls), encoding="utf-8"
            )
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()
        # Must be after contextbench.run: it rejects nonempty output directories.
        _write_json(out / "baseline-config.json", provenance)
        _write_json(out / "baseline-status.json", {"status": status, "error": error})
    if error:
        if not (out / "result.json").exists():
            _write_json(
                out / "result.json",
                {
                    "terminated_by": metadata.get("terminated_by", "runtime_error"),
                    "error": error,
                    "status": "failed",
                },
            )
        print(f"Baseline failed: {error}; see {out}", file=sys.stderr)
        return 1
    return 0
