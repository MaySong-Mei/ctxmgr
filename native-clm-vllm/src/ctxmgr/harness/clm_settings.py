"""Resolve supported sampling overrides without changing either agent implementation."""

import math
import os

PROFILE_OVERRIDES = {
    "official": {},
    "team-baseline": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "enable_thinking": False,
    },
}
SAMPLING_KEYS = ("temperature", "top_p", "max_tokens", "enable_thinking")


def profile_overrides(profile):
    """Return the original named profile overrides, without importing an agent."""
    try:
        return dict(PROFILE_OVERRIDES[profile])
    except KeyError as exc:
        raise ValueError(f"Unknown CLM evaluation profile: {profile}") from exc


def resolve_settings(
    profile="official", *, temperature=None, top_p=None, max_tokens=None, thinking=None
):
    """Apply explicit sampling flags over a named profile and official constants."""
    overrides = profile_overrides(profile)
    supplied = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "enable_thinking": thinking,
    }
    for key in ("temperature", "top_p"):
        value = supplied[key]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (key == "temperature" and value < 0)
            or (key == "top_p" and not 0 < value <= 1)
        ):
            raise ValueError(f"Invalid {key}: {value!r}")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    if thinking is not None and not isinstance(thinking, bool):
        raise ValueError("thinking must be a boolean")
    overrides.update({key: value for key, value in supplied.items() if value is not None})
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from clm_contextbench import PAPER_KWARGS

    effective = {**PAPER_KWARGS, **overrides}
    return {
        "profile": profile,
        "profile_overrides": overrides,
        "agent_kwargs": effective,
        "sampling": {key: effective[key] for key in SAMPLING_KEYS},
    }


def add_sampling_arguments(parser, default_profile="official"):
    """Add profile and optional overrides; absent booleans remain None."""
    profile_overrides(default_profile)
    parser.add_argument("--profile", choices=tuple(PROFILE_OVERRIDES), default=default_profile)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--thinking", dest="thinking", action="store_true")
    mode.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)
    return parser


def settings_from_args(args):
    return resolve_settings(
        getattr(args, "profile", "official"),
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
        max_tokens=getattr(args, "max_tokens", None),
        thinking=getattr(args, "thinking", None),
    )


def settings_from_overrides(profile="official", sampling_overrides=None):
    """Resolve the native supported-kwargs spelling, rejecting arbitrary agent kwargs."""
    values = dict(sampling_overrides or {})
    unexpected = set(values) - set(SAMPLING_KEYS)
    if unexpected:
        raise ValueError(f"Unsupported sampling override keys: {sorted(unexpected)}")
    return resolve_settings(
        profile,
        temperature=values.get("temperature"),
        top_p=values.get("top_p"),
        max_tokens=values.get("max_tokens"),
        thinking=values.get("enable_thinking"),
    )
