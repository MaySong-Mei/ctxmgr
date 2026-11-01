# para3 ablation on the manager replay (round 33, jobs 31571897 and 31571899)

Built by `scripts/manager_replay.py summarize` (harness `may/manager-replay` da7578b) from
`runs/mreplay-<job>/replay.jsonl`; design in `docs/preregistration-v6.md`. Same 48 fixed contexts
and success measure as `manager_replay.md`. Wordings: v8 and `para3` (in-job controls), `p3flat`
(`para3`'s steps as prose), `p3bullets` (numbers replaced by bullets), `v8steps` (v8 plus
`para3`'s step block). Sampled = Gemma's recommended setting, 4 request seeds per context.

| manager | decoding | wording | events | n | success | kept all | freed >= 30% | parse errors | capped |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-27B | greedy | p3bullets | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | p3bullets | later | 32 | 0.69 | 0.97 | 0.72 | 9 | 9 |
| Qwen3.5-27B | greedy | p3flat | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | p3flat | later | 32 | 0.75 | 0.97 | 0.78 | 7 | 7 |
| Qwen3.5-27B | greedy | para3 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | para3 | later | 32 | 0.75 | 0.97 | 0.78 | 7 | 7 |
| Qwen3.5-27B | greedy | v8 | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | v8 | later | 32 | 0.56 | 0.56 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | v8steps | first | 16 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| Qwen3.5-27B | greedy | v8steps | later | 32 | 0.88 | 0.91 | 0.97 | 1 | 1 |
| gemma-4-12B-it | greedy | p3bullets | first | 16 | 0.50 | 0.81 | 0.69 | 3 | 0 |
| gemma-4-12B-it | greedy | p3bullets | later | 32 | 0.53 | 0.91 | 0.62 | 11 | 2 |
| gemma-4-12B-it | greedy | p3flat | first | 16 | 0.00 | 0.94 | 0.06 | 3 | 0 |
| gemma-4-12B-it | greedy | p3flat | later | 32 | 0.44 | 0.88 | 0.56 | 13 | 2 |
| gemma-4-12B-it | greedy | para3 | first | 16 | 0.31 | 0.56 | 0.75 | 1 | 0 |
| gemma-4-12B-it | greedy | para3 | later | 32 | 0.66 | 0.94 | 0.72 | 8 | 3 |
| gemma-4-12B-it | greedy | v8 | first | 16 | 0.25 | 0.81 | 0.44 | 3 | 0 |
| gemma-4-12B-it | greedy | v8 | later | 32 | 0.03 | 0.44 | 0.56 | 7 | 4 |
| gemma-4-12B-it | greedy | v8steps | first | 16 | 0.38 | 0.50 | 0.88 | 2 | 0 |
| gemma-4-12B-it | greedy | v8steps | later | 32 | 0.09 | 0.81 | 0.28 | 20 | 7 |
| gemma-4-12B-it | sampled | p3bullets | first | 64 | 0.42 | 0.72 | 0.70 | 13 | 1 |
| gemma-4-12B-it | sampled | p3bullets | later | 128 | 0.37 | 0.88 | 0.48 | 56 | 10 |
| gemma-4-12B-it | sampled | p3flat | first | 64 | 0.06 | 0.77 | 0.30 | 7 | 0 |
| gemma-4-12B-it | sampled | p3flat | later | 128 | 0.29 | 0.84 | 0.45 | 61 | 6 |
| gemma-4-12B-it | sampled | para3 | first | 64 | 0.34 | 0.62 | 0.72 | 8 | 1 |
| gemma-4-12B-it | sampled | para3 | later | 128 | 0.35 | 0.88 | 0.48 | 59 | 9 |
| gemma-4-12B-it | sampled | v8 | first | 64 | 0.17 | 0.67 | 0.50 | 15 | 0 |
| gemma-4-12B-it | sampled | v8 | later | 128 | 0.09 | 0.50 | 0.58 | 38 | 13 |
| gemma-4-12B-it | sampled | v8steps | first | 64 | 0.20 | 0.67 | 0.53 | 19 | 1 |
| gemma-4-12B-it | sampled | v8steps | later | 128 | 0.05 | 0.77 | 0.27 | 69 | 30 |
