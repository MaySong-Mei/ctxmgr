# Identical-input replay of manager events (round 32, jobs 31558632-31558635)

Built by `scripts/manager_replay.py summarize` (harness `may/manager-replay` b8252b3) from
`runs/mreplay-<job>/replay.jsonl`; design in `docs/preregistration-v5.md`. Inputs: the first three
manager events of job 31545119's 16 episodes (2B actor + Gemma 4 31B, edit-v8); first = event 1
(shared by every condition), later = events 2-3. success = every needle id of the context survives
the edits (registered measure, which also counts ids in actor replies; see deviation D2) and at
least 30% is freed. capped = reply cut at 4,096 tokens. Sampled = Gemma's recommended setting,
4 request seeds per context.

| manager | decoding | wording | events | n | success | kept all | freed >= 30% | parse errors | capped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-27B | greedy | para1 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | para1 | later | 32 | 0.25 | 0.31 | 0.94 | 2 | 2 |
| Qwen3.5-27B | greedy | para2 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | para2 | later | 32 | 0.84 | 0.94 | 0.91 | 3 | 3 |
| Qwen3.5-27B | greedy | para3 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | para3 | later | 32 | 0.75 | 0.97 | 0.78 | 7 | 7 |
| Qwen3.5-27B | greedy | para4 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | para4 | later | 32 | 0.06 | 0.12 | 0.94 | 2 | 2 |
| Qwen3.5-27B | greedy | v8 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | v8 | later | 32 | 0.50 | 0.53 | 0.97 | 1 | 1 |
| Qwen3.5-9B | greedy | para1 | first | 16 | 0.00 | 1.00 | 0.00 | 16 | 16 |
| Qwen3.5-9B | greedy | para1 | later | 32 | 0.00 | 0.28 | 0.72 | 7 | 7 |
| Qwen3.5-9B | greedy | para2 | first | 16 | 0.00 | 0.69 | 0.31 | 11 | 11 |
| Qwen3.5-9B | greedy | para2 | later | 32 | 0.00 | 0.06 | 0.94 | 1 | 1 |
| Qwen3.5-9B | greedy | para3 | first | 16 | 0.00 | 0.38 | 0.62 | 6 | 6 |
| Qwen3.5-9B | greedy | para3 | later | 32 | 0.00 | 0.00 | 1.00 | 0 | 0 |
| Qwen3.5-9B | greedy | para4 | first | 16 | 0.00 | 0.44 | 0.56 | 7 | 7 |
| Qwen3.5-9B | greedy | para4 | later | 32 | 0.00 | 0.12 | 0.88 | 2 | 2 |
| Qwen3.5-9B | greedy | v8 | first | 16 | 0.00 | 0.62 | 0.38 | 10 | 10 |
| Qwen3.5-9B | greedy | v8 | later | 32 | 0.00 | 0.06 | 0.94 | 2 | 2 |
| gemma-4-12B-it | greedy | para1 | first | 16 | 0.38 | 0.62 | 0.75 | 1 | 0 |
| gemma-4-12B-it | greedy | para1 | later | 32 | 0.28 | 0.50 | 0.78 | 7 | 0 |
| gemma-4-12B-it | greedy | para2 | first | 16 | 0.00 | 0.88 | 0.12 | 4 | 0 |
| gemma-4-12B-it | greedy | para2 | later | 32 | 0.00 | 0.78 | 0.22 | 12 | 1 |
| gemma-4-12B-it | greedy | para3 | first | 16 | 0.31 | 0.50 | 0.81 | 0 | 0 |
| gemma-4-12B-it | greedy | para3 | later | 32 | 0.66 | 0.94 | 0.72 | 8 | 4 |
| gemma-4-12B-it | greedy | para4 | first | 16 | 0.00 | 0.00 | 1.00 | 0 | 0 |
| gemma-4-12B-it | greedy | para4 | later | 32 | 0.03 | 0.25 | 0.75 | 1 | 0 |
| gemma-4-12B-it | greedy | v8 | first | 16 | 0.25 | 0.69 | 0.56 | 4 | 0 |
| gemma-4-12B-it | greedy | v8 | later | 32 | 0.06 | 0.47 | 0.56 | 7 | 4 |
| gemma-4-12B-it | sampled | para3 | first | 64 | 0.34 | 0.62 | 0.72 | 9 | 2 |
| gemma-4-12B-it | sampled | para3 | later | 128 | 0.34 | 0.88 | 0.47 | 59 | 7 |
| gemma-4-12B-it | sampled | para4 | first | 64 | 0.00 | 0.12 | 0.88 | 1 | 0 |
| gemma-4-12B-it | sampled | para4 | later | 128 | 0.05 | 0.39 | 0.64 | 18 | 2 |
| gemma-4-12B-it | sampled | v8 | first | 64 | 0.17 | 0.67 | 0.50 | 15 | 0 |
| gemma-4-12B-it | sampled | v8 | later | 128 | 0.09 | 0.48 | 0.60 | 36 | 11 |
| gemma-4-31B-it | greedy | para1 | first | 16 | 0.69 | 0.69 | 1.00 | 0 | 0 |
| gemma-4-31B-it | greedy | para1 | later | 32 | 0.72 | 0.75 | 0.97 | 1 | 1 |
| gemma-4-31B-it | greedy | para2 | first | 16 | 0.88 | 0.88 | 1.00 | 0 | 0 |
| gemma-4-31B-it | greedy | para2 | later | 32 | 0.84 | 0.91 | 0.94 | 2 | 2 |
| gemma-4-31B-it | greedy | para3 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| gemma-4-31B-it | greedy | para3 | later | 32 | 0.88 | 0.94 | 0.94 | 2 | 1 |
| gemma-4-31B-it | greedy | para4 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| gemma-4-31B-it | greedy | para4 | later | 32 | 0.69 | 0.72 | 0.97 | 1 | 1 |
| gemma-4-31B-it | greedy | v8 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| gemma-4-31B-it | greedy | v8 | later | 32 | 0.84 | 0.91 | 0.94 | 2 | 2 |
