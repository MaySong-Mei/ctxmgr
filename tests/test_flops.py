from ctxmgr.accounting.flops import CONSTANTS, call_flops, size_of


def test_27b_constants_match_clm():
    # CLM (arXiv 2609.37725) reports these for the 27B hybrid Qwen.
    k = CONSTANTS["27B"]
    assert round(k.c_token / 1e9, 2) == 48.70
    assert round(k.c_attn / 1e5, 2) == 3.93


def test_clm_worked_example():
    # CLM: P = 20,000, G = 500 costs about 1.41e14 FLOPs with 18,000 tokens reused and 10.81e14
    # with none (without the output head, which CLM omits).
    k = CONSTANTS["27B"]
    head = k.c_head * 500
    assert round((call_flops(k, 20_000, 18_000, 500) - head) / 1e14, 2) == 1.41
    assert round((call_flops(k, 20_000, 0, 500) - head) / 1e14, 2) == 10.81


def test_cache_only_lowers_cost_and_sizes_parse():
    k = CONSTANTS["2B"]
    assert call_flops(k, 1000, 900, 10) < call_flops(k, 1000, 0, 10)
    assert size_of("Qwen/Qwen3.5-27B-FP8") == "27B"
    assert size_of("Qwen/Qwen3.5-0.8B") == "0.8B"
