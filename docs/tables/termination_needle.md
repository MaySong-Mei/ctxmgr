# Needle: how episodes end, actor output-cap hits, and reward = delivered x kept

Built by `scripts/termination_audit.py` from `episodes_vllm.csv` and each episode's
`result.json`, `score.json` and `calls.jsonl` (H200 bf16, max_len 81,920, trigger 18,432;
archive conditions excluded). delivered = share of the stream's ops delivered; kept = share
of delivered needles in the final context; done = `submit` or `stream_exhausted`; wall =
`budget_wall`; capped = share of actor replies cut at the 2,048-token output cap; cap loop =
ends at `max_steps` with the last 5 actor replies all capped.

Managed episodes (every condition except `none`), by actor:

- Qwen3.5-0.8B: 3/144 managed episodes end at `max_steps`, 0 of them in a cap loop; 13/144 have at least one capped actor reply.
- Qwen3.5-2B: 1/448 managed episodes end at `max_steps`, 0 of them in a cap loop; 59/448 have at least one capped actor reply.
- Qwen3.5-4B: 6/216 managed episodes end at `max_steps`, 5 of them in a cap loop; 10/216 have at least one capped actor reply.
- Qwen3.5-9B: 0/113 managed episodes end at `max_steps`, 0 of them in a cap loop; 1/113 have at least one capped actor reply.
- Qwen3.5-27B: 38/120 managed episodes end at `max_steps`, 22 of them in a cap loop; 100/120 have at least one capped actor reply.
- Qwen3.5-27B-FP8: 0/8 managed episodes end at `max_steps`, 0 of them in a cap loop; 4/8 have at least one capped actor reply.

| level | actor | condition | manager | version | decoding | n | reward | delivered | kept | done | wall | max_steps | other | capped | episodes with a cap | cap loops |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K4-C24 | Qwen3.5-0.8B | none | - | - | greedy | 20 | 0.33 | 0.33 | 1.00 | 0 | 20 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | none | - | - | sampled | 8 | 0.22 | 0.22 | 1.00 | 0 | 8 | 0 | 0 | 0.31 | 3 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 8 | 0.97 | 1.00 | 0.97 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_27b | Qwen3.5-27B | edit-v8 | sampled | 8 | 0.45 | 0.68 | 0.76 | 5 | 0 | 3 | 0 | 0.13 | 6 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_27b | Qwen3.5-27B-FP8 | edit-v7 | greedy | 4 | 0.90 | 1.00 | 0.90 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_4b | Qwen3.5-4B | edit-v8 | greedy | 8 | 0.24 | 0.85 | 0.31 | 4 | 4 | 0 | 0 | 0.10 | 1 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_9b | Qwen3.5-9B | edit-v8 | greedy | 8 | 0.16 | 1.00 | 0.16 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_same | Qwen3.5-0.8B | edit-v7 | greedy | 4 | 0.33 | 0.33 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_same | Qwen3.5-0.8B | edit-v8 | greedy | 16 | 0.33 | 0.33 | 1.00 | 0 | 16 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | peer_same | Qwen3.5-0.8B | edit-v8 | sampled | 8 | 0.22 | 0.22 | 1.00 | 0 | 8 | 0 | 0 | 0.31 | 3 | 0 |
| K4-C24 | Qwen3.5-0.8B | self | - | edit-v7 | greedy | 4 | 0.33 | 0.33 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | self | - | edit-v8 | greedy | 8 | 0.33 | 0.33 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-0.8B | self | - | edit-v8 | sampled | 8 | 0.22 | 0.22 | 1.00 | 0 | 8 | 0 | 0 | 0.31 | 3 | 0 |
| K4-C24 | Qwen3.5-27B | none | - | - | greedy | 25 | 0.29 | 0.29 | 1.00 | 0 | 25 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-27B | none | - | - | sampled | 8 | 0.29 | 0.29 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-27B | peer_same | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.89 | 0.99 | 0.90 | 3 | 0 | 1 | 0 | 0.45 | 4 | 1 |
| K4-C24 | Qwen3.5-27B | peer_same | Qwen3.5-27B | edit-v8 | greedy | 20 | 0.78 | 0.97 | 0.80 | 14 | 0 | 6 | 0 | 0.52 | 20 | 3 |
| K4-C24 | Qwen3.5-27B | peer_same | Qwen3.5-27B | edit-v8 | sampled | 8 | 0.67 | 0.97 | 0.70 | 6 | 0 | 2 | 0 | 0.45 | 7 | 2 |
| K4-C24 | Qwen3.5-27B | self | - | edit-v7 | greedy | 4 | 0.29 | 0.29 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-27B | self | - | edit-v8 | greedy | 24 | 0.39 | 0.99 | 0.39 | 23 | 0 | 1 | 0 | 0.16 | 13 | 1 |
| K4-C24 | Qwen3.5-27B | self | - | edit-v8 | sampled | 8 | 0.60 | 0.91 | 0.69 | 6 | 1 | 1 | 0 | 0.47 | 7 | 0 |
| K4-C24 | Qwen3.5-27B-FP8 | none | - | - | greedy | 4 | 0.29 | 0.29 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-27B-FP8 | peer_same | Qwen3.5-27B-FP8 | edit-v7 | greedy | 4 | 0.90 | 1.00 | 0.90 | 4 | 0 | 0 | 0 | 0.46 | 4 | 0 |
| K4-C24 | Qwen3.5-27B-FP8 | self | - | edit-v7 | greedy | 4 | 0.29 | 0.29 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | none | - | - | greedy | 128 | 0.33 | 0.33 | 1.00 | 0 | 128 | 0 | 0 | 0.01 | 1 | 0 |
| K4-C24 | Qwen3.5-2B | none | - | - | sampled | 16 | 0.33 | 0.33 | 1.00 | 0 | 16 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v7 | greedy | 4 | 1.00 | 1.00 | 1.00 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 20 | 0.99 | 1.00 | 0.99 | 20 | 0 | 0 | 0 | 0.01 | 1 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8 | sampled | 8 | 0.76 | 1.00 | 0.76 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+example | greedy | 8 | 0.38 | 1.00 | 0.38 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+mustnote | greedy | 8 | 0.95 | 1.00 | 0.95 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para1 | greedy | 8 | 0.68 | 1.00 | 0.68 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para2 | greedy | 8 | 0.95 | 1.00 | 0.95 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para3 | greedy | 8 | 1.00 | 1.00 | 1.00 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para4 | greedy | 8 | 0.51 | 1.00 | 0.51 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+think | greedy | 8 | 0.93 | 1.00 | 0.93 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B-FP8 | edit-v5 | greedy | 4 | 0.79 | 1.00 | 0.79 | 4 | 0 | 0 | 0 | 0.11 | 1 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B-FP8 | edit-v6 | greedy | 4 | 0.38 | 1.00 | 0.38 | 4 | 0 | 0 | 0 | 0.11 | 1 | 0 |
| K4-C24 | Qwen3.5-2B | peer_27b | Qwen3.5-27B-FP8 | edit-v7 | greedy | 8 | 0.97 | 1.00 | 0.97 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_4b | Qwen3.5-4B | edit-v8 | greedy | 8 | 0.34 | 0.66 | 0.53 | 1 | 7 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8 | greedy | 8 | 0.16 | 0.92 | 0.24 | 7 | 1 | 0 | 0 | 0.14 | 5 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8 | sampled | 8 | 0.14 | 1.00 | 0.14 | 8 | 0 | 0 | 0 | 0.03 | 2 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+example | greedy | 8 | 0.36 | 0.97 | 0.38 | 7 | 1 | 0 | 0 | 0.09 | 3 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+mustnote | greedy | 8 | 0.14 | 1.00 | 0.14 | 8 | 0 | 0 | 0 | 0.09 | 3 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para1 | greedy | 8 | 0.33 | 0.33 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para2 | greedy | 8 | 0.23 | 0.67 | 0.57 | 4 | 4 | 0 | 0 | 0.18 | 4 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para3 | greedy | 8 | 0.18 | 0.83 | 0.34 | 6 | 2 | 0 | 0 | 0.10 | 3 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para4 | greedy | 8 | 0.16 | 0.83 | 0.33 | 6 | 2 | 0 | 0 | 0.10 | 3 | 0 |
| K4-C24 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+think | greedy | 8 | 0.66 | 0.81 | 0.84 | 4 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_same | Qwen3.5-2B | edit-v7 | greedy | 8 | 0.37 | 0.38 | 0.99 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_same | Qwen3.5-2B | edit-v8 | greedy | 12 | 0.36 | 0.37 | 0.98 | 0 | 12 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | peer_same | Qwen3.5-2B | edit-v8 | sampled | 8 | 0.34 | 0.34 | 0.99 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | rule_needle | - | rule-needle-v1 | greedy | 8 | 1.00 | 1.00 | 1.00 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | self | - | edit-v7 | greedy | 8 | 0.33 | 0.33 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-2B | self | - | edit-v8 | greedy | 12 | 0.33 | 0.33 | 1.00 | 0 | 12 | 0 | 0 | 0.07 | 1 | 0 |
| K4-C24 | Qwen3.5-2B | self | - | edit-v8 | sampled | 8 | 0.34 | 0.44 | 0.84 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | none | - | - | greedy | 60 | 0.33 | 0.33 | 1.00 | 0 | 60 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | none | - | - | sampled | 16 | 0.33 | 0.33 | 1.00 | 0 | 16 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.81 | 1.00 | 0.81 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 28 | 0.80 | 1.00 | 0.80 | 28 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v8 | sampled | 16 | 0.77 | 1.00 | 0.77 | 16 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v8+think | greedy | 8 | 0.83 | 0.88 | 0.95 | 5 | 0 | 3 | 0 | 0.43 | 4 | 3 |
| K4-C24 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8 | greedy | 8 | 0.16 | 1.00 | 0.16 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8+example | greedy | 8 | 0.29 | 1.00 | 0.29 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8+mustnote | greedy | 8 | 0.17 | 1.00 | 0.17 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8+think | greedy | 8 | 0.43 | 0.67 | 0.76 | 3 | 5 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_same | Qwen3.5-4B | edit-v7 | greedy | 8 | 0.31 | 0.62 | 0.64 | 2 | 6 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_same | Qwen3.5-4B | edit-v8 | greedy | 12 | 0.33 | 0.56 | 0.76 | 1 | 11 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | peer_same | Qwen3.5-4B | edit-v8 | sampled | 8 | 0.33 | 0.42 | 0.82 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | self | - | edit-v7 | greedy | 8 | 0.33 | 0.33 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | self | - | edit-v8 | greedy | 12 | 0.33 | 0.33 | 1.00 | 0 | 12 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-4B | self | - | edit-v8 | sampled | 8 | 0.33 | 0.35 | 0.95 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | none | - | - | greedy | 36 | 0.32 | 0.32 | 1.00 | 0 | 36 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | none | - | - | sampled | 16 | 0.33 | 0.33 | 1.00 | 0 | 16 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | peer_27b | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.75 | 1.00 | 0.75 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 29 | 0.82 | 1.00 | 0.82 | 29 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | peer_27b | Qwen3.5-27B | edit-v8 | sampled | 16 | 0.67 | 1.00 | 0.67 | 16 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | peer_same | Qwen3.5-9B | edit-v7 | greedy | 8 | 0.24 | 0.59 | 0.63 | 3 | 5 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | peer_same | Qwen3.5-9B | edit-v8 | greedy | 4 | 0.24 | 0.66 | 0.58 | 2 | 2 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | peer_same | Qwen3.5-9B | edit-v8 | sampled | 8 | 0.16 | 1.00 | 0.16 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | self | - | edit-v7 | greedy | 8 | 0.17 | 1.00 | 0.17 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | self | - | edit-v8 | greedy | 4 | 0.17 | 1.00 | 0.17 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K4-C24 | Qwen3.5-9B | self | - | edit-v8 | sampled | 8 | 0.17 | 1.00 | 0.17 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | none | - | - | greedy | 20 | 0.17 | 0.17 | 1.00 | 0 | 20 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 8 | 0.30 | 0.99 | 0.31 | 7 | 1 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | peer_27b | Qwen3.5-27B-FP8 | edit-v7 | greedy | 4 | 0.28 | 1.00 | 0.28 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | peer_4b | Qwen3.5-4B | edit-v8 | greedy | 8 | 0.17 | 0.36 | 0.52 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | peer_9b | Qwen3.5-9B | edit-v8 | greedy | 8 | 0.05 | 1.00 | 0.05 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | peer_same | Qwen3.5-0.8B | edit-v7 | greedy | 4 | 0.17 | 0.17 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | peer_same | Qwen3.5-0.8B | edit-v8 | greedy | 16 | 0.16 | 0.16 | 1.00 | 0 | 16 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | self | - | edit-v7 | greedy | 4 | 0.17 | 0.17 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-0.8B | self | - | edit-v8 | greedy | 8 | 0.17 | 0.17 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-27B | none | - | - | greedy | 8 | 0.15 | 0.15 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-27B | none | - | - | sampled | 8 | 0.15 | 0.15 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-27B | peer_same | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.34 | 0.45 | 0.75 | 0 | 3 | 1 | 0 | 0.71 | 4 | 0 |
| K8-C48 | Qwen3.5-27B | peer_same | Qwen3.5-27B | edit-v8 | greedy | 4 | 0.34 | 0.65 | 0.53 | 0 | 1 | 3 | 0 | 0.69 | 4 | 0 |
| K8-C48 | Qwen3.5-27B | peer_same | Qwen3.5-27B | edit-v8 | sampled | 16 | 0.30 | 0.59 | 0.67 | 5 | 0 | 11 | 0 | 0.79 | 16 | 8 |
| K8-C48 | Qwen3.5-27B | self | - | edit-v7 | greedy | 4 | 0.15 | 0.15 | 1.00 | 0 | 4 | 0 | 0 | 0.35 | 1 | 0 |
| K8-C48 | Qwen3.5-27B | self | - | edit-v8 | greedy | 8 | 0.14 | 1.00 | 0.14 | 7 | 0 | 1 | 0 | 0.31 | 8 | 0 |
| K8-C48 | Qwen3.5-27B | self | - | edit-v8 | sampled | 16 | 0.19 | 0.80 | 0.32 | 5 | 0 | 11 | 0 | 0.71 | 16 | 7 |
| K8-C48 | Qwen3.5-2B | none | - | - | greedy | 112 | 0.15 | 0.15 | 1.00 | 0 | 112 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | none | - | - | sampled | 8 | 0.15 | 0.15 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.33 | 1.00 | 0.33 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 4 | 0.29 | 1.00 | 0.29 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+example | greedy | 8 | 0.16 | 1.00 | 0.16 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+mustnote | greedy | 8 | 0.40 | 0.96 | 0.43 | 7 | 1 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para1 | greedy | 8 | 0.34 | 1.00 | 0.34 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para2 | greedy | 8 | 0.53 | 0.80 | 0.69 | 3 | 5 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para3 | greedy | 8 | 0.35 | 0.35 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+para4 | greedy | 8 | 0.19 | 1.00 | 0.19 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B | edit-v8+think | greedy | 8 | 0.63 | 0.66 | 0.95 | 4 | 4 | 0 | 0 | 0.19 | 3 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B-FP8 | edit-v5 | greedy | 4 | 0.26 | 1.00 | 0.26 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B-FP8 | edit-v6 | greedy | 4 | 0.17 | 1.00 | 0.17 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_27b | Qwen3.5-27B-FP8 | edit-v7 | greedy | 8 | 0.32 | 1.00 | 0.32 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_4b | Qwen3.5-4B | edit-v8 | greedy | 8 | 0.17 | 0.23 | 0.77 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8 | greedy | 8 | 0.06 | 1.00 | 0.06 | 8 | 0 | 0 | 0 | 0.04 | 3 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8 | sampled | 8 | 0.09 | 0.97 | 0.09 | 6 | 1 | 1 | 0 | 0.02 | 2 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+example | greedy | 8 | 0.22 | 0.53 | 0.47 | 1 | 7 | 0 | 0 | 0.27 | 5 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+mustnote | greedy | 8 | 0.10 | 0.93 | 0.14 | 7 | 1 | 0 | 0 | 0.17 | 4 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para1 | greedy | 8 | 0.13 | 0.39 | 0.70 | 2 | 6 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para2 | greedy | 8 | 0.05 | 1.00 | 0.05 | 8 | 0 | 0 | 0 | 0.09 | 4 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para3 | greedy | 8 | 0.07 | 1.00 | 0.07 | 8 | 0 | 0 | 0 | 0.07 | 5 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+para4 | greedy | 8 | 0.07 | 1.00 | 0.07 | 8 | 0 | 0 | 0 | 0.08 | 5 | 0 |
| K8-C48 | Qwen3.5-2B | peer_9b | Qwen3.5-9B | edit-v8+think | greedy | 8 | 0.34 | 0.45 | 0.87 | 0 | 8 | 0 | 0 | 0.09 | 1 | 0 |
| K8-C48 | Qwen3.5-2B | peer_same | Qwen3.5-2B | edit-v7 | greedy | 8 | 0.15 | 0.15 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | peer_same | Qwen3.5-2B | edit-v8 | greedy | 12 | 0.15 | 0.15 | 1.00 | 0 | 12 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | rule_needle | - | rule-needle-v1 | greedy | 8 | 1.00 | 1.00 | 1.00 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | self | - | edit-v7 | greedy | 8 | 0.17 | 0.18 | 0.95 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-2B | self | - | edit-v8 | greedy | 4 | 0.16 | 0.16 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | none | - | - | greedy | 32 | 0.16 | 0.16 | 1.00 | 0 | 32 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.31 | 1.00 | 0.31 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 4 | 0.24 | 1.00 | 0.24 | 4 | 0 | 0 | 0 | 0.04 | 1 | 0 |
| K8-C48 | Qwen3.5-4B | peer_27b | Qwen3.5-27B | edit-v8+think | greedy | 8 | 0.82 | 0.83 | 0.98 | 5 | 1 | 2 | 0 | 0.36 | 3 | 1 |
| K8-C48 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8 | greedy | 8 | 0.06 | 1.00 | 0.06 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8+example | greedy | 8 | 0.14 | 1.00 | 0.14 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8+mustnote | greedy | 8 | 0.06 | 1.00 | 0.06 | 8 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | peer_9b | Qwen3.5-9B | edit-v8+think | greedy | 8 | 0.23 | 0.46 | 0.67 | 2 | 5 | 1 | 0 | 0.37 | 2 | 1 |
| K8-C48 | Qwen3.5-4B | peer_same | Qwen3.5-4B | edit-v7 | greedy | 4 | 0.16 | 0.28 | 0.62 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | peer_same | Qwen3.5-4B | edit-v8 | greedy | 12 | 0.16 | 0.32 | 0.56 | 0 | 12 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | self | - | edit-v7 | greedy | 4 | 0.16 | 0.16 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-4B | self | - | edit-v8 | greedy | 4 | 0.15 | 0.15 | 1.00 | 0 | 4 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-9B | none | - | - | greedy | 8 | 0.16 | 0.16 | 1.00 | 0 | 8 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-9B | peer_27b | Qwen3.5-27B | edit-v7 | greedy | 4 | 0.33 | 0.82 | 0.43 | 2 | 2 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-9B | peer_27b | Qwen3.5-27B | edit-v8 | greedy | 4 | 0.31 | 0.90 | 0.37 | 3 | 1 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-9B | peer_same | Qwen3.5-9B | edit-v7 | greedy | 4 | 0.07 | 1.00 | 0.07 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-9B | peer_same | Qwen3.5-9B | edit-v8 | greedy | 4 | 0.09 | 0.98 | 0.09 | 3 | 1 | 0 | 0 | 0.01 | 1 | 0 |
| K8-C48 | Qwen3.5-9B | self | - | edit-v7 | greedy | 4 | 0.07 | 1.00 | 0.07 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
| K8-C48 | Qwen3.5-9B | self | - | edit-v8 | greedy | 4 | 0.08 | 1.00 | 0.08 | 4 | 0 | 0 | 0 | 0.00 | 0 | 0 |
