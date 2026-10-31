"""Check the native client's tool transport; never execute a generated command.

This separate, fresh conversation is not a benchmark episode. A model declining
to call a tool is a warning, not grounds to exclude that model from evaluation.
"""

import argparse
import json
import os
from pathlib import Path

from .clm_settings import (
    add_sampling_arguments,
    resolve_settings,
    settings_from_args,
    settings_from_overrides,
)
from .clm_upstream import verify_upstream, write_json


def probe(base_url, model, out, profile="team-baseline", *, settings=None, served_max_len=None):
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm
    from clm_harness.utils.tool_schemas import BASH_TOOL

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    verify_upstream()
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    resolved = (
        resolve_settings(profile)
        if settings is None
        else settings_from_overrides(settings["profile"], settings["profile_overrides"])
    )
    settings = resolved["sampling"]
    record = {
        "model": model,
        "base_url": base_url,
        "profile": resolved["profile"],
        "profile_overrides": resolved["profile_overrides"],
        "served_max_len": served_max_len,
        "settings": settings,
        "benchmark_episode": False,
        "commands_executed": False,
        "requests": [],
        "responses": [],
        "warnings": [],
        "error": None,
    }
    messages = [
        {
            "role": "system",
            "content": "This is an API transport check, separate from any benchmark task.",
        },
        {
            "role": "user",
            "content": "Use the bash tool once with command `printf CLM_PROBE`. "
            "After receiving the tool result, answer OK.",
        },
    ]

    def query():
        request = {
            "model": f"openai/{model}",
            "messages": messages,
            "tools": [BASH_TOOL],
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_tokens": settings["max_tokens"],
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": settings["enable_thinking"]}
            },
        }
        record["requests"].append(json.loads(json.dumps(request)))
        response = litellm.completion(
            **request,
            api_base=base_url,
            timeout=600,
            request_timeout=600,
            num_retries=0,
            drop_params=True,
        )
        record["responses"].append(response.model_dump(mode="json"))
        if not response.choices or response.choices[0].message is None:
            raise ValueError("Server returned no choice/message")
        if response.usage is None:
            record["warnings"].append("Server did not report token usage")
        message = response.choices[0].message
        record["reasoning_observed"] = record.get("reasoning_observed", False) or bool(
            getattr(message, "reasoning_content", None)
        )
        if getattr(message, "reasoning_content", None) and not settings["enable_thinking"]:
            record["warnings"].append("Reasoning content returned despite thinking=false")
        return message

    try:
        message = query()
        calls = message.tool_calls or []
        record["tool_call_observed"] = bool(calls)
        record["roundtrip_observed"] = False
        valid = bool(calls)
        for call in calls:
            try:
                arguments = json.loads(call.function.arguments)
                if (
                    call.function.name != "bash"
                    or not call.id
                    or not isinstance(arguments, dict)
                    or not isinstance(arguments.get("command"), str)
                ):
                    valid = False
            except (TypeError, ValueError):
                valid = False
        if valid:
            # Preserve reasoning fields just as the official harness does.
            replay = message.model_dump()
            replay["content"] = replay.get("content") or ""
            messages.append(replay)
            for call in calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": "CLM_PROBE (simulated transport response; command not executed)",
                    }
                )
            query()
            record["roundtrip_observed"] = True
        else:
            record["warnings"].append(
                "Model did not emit a valid Bash call; tool roundtrip unverified. "
                "Keep this model in the benchmark."
            )
        if settings["enable_thinking"] and not record.get("reasoning_observed"):
            record["warnings"].append(
                "Thinking was requested, but no separate reasoning field was observed; "
                "this probe alone does not verify the server's reasoning mode."
            )
        record["outcome"] = "passed_with_warning" if record["warnings"] else "passed"
    except Exception as exc:  # noqa: BLE001 - retain transport failures, not fabricated scores
        record["outcome"] = "transport_error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    write_json(out, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    add_sampling_arguments(parser, default_profile="team-baseline")
    parser.add_argument("--served-max-len", type=int)
    args = parser.parse_args(argv)
    result = probe(
        args.base_url,
        args.model,
        args.out,
        settings=settings_from_args(args),
        served_max_len=args.served_max_len,
    )
    print(json.dumps({k: result[k] for k in ("model", "outcome", "warnings", "error")}))
    return 1 if result["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
