from ctxmgr.accounting.efficiency import summarize_calls


def test_both_roles_count_and_unknown_cache_stays_unknown():
    calls = [
        {
            "role": "actor",
            "model": "Qwen/Qwen3.5-2B",
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cached_prompt_tokens": 80,
            "latency_s": 2,
        },
        {
            "role": "manager",
            "model": "Qwen/Qwen3.5-27B",
            "prompt_tokens": 50,
            "completion_tokens": 5,
            "cached_prompt_tokens": None,
            "latency_s": 3,
        },
    ]
    out = summarize_calls(calls)
    assert out["total_api_latency_s"] == 5
    assert out["total_param_tokens_no_cache"] == 2e9 * 110 + 27e9 * 55
    assert out["actor_param_tokens_cache_adjusted"] == 2e9 * 30
    assert out["total_param_tokens_cache_adjusted"] is None


def test_hotpot_scoring_and_format_are_separate():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    spec = spec_from_file_location(
        "multidoc", Path(__file__).parents[1] / "scripts/multidoc_pilot.py"
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.answer_metrics("The Eiffel Tower", "Eiffel Tower") == (1, 1)
    assert module.answer_metrics("yes Paris", "yes") == (0, 0)
    assert module.parse_prediction('```json\n{"answer":"Paris"}\n```')["answer"] == "Paris"
    assert module.parse_prediction("Paris") is None
