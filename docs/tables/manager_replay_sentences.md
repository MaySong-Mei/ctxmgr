# Sentence-level para3 ablation on the manager replay (round 34, job 31576318)

Built by `scripts/manager_replay.py summarize` (harness `may/manager-replay` 14b1a23) from
`runs/mreplay-31576318/replay.jsonl`; design in `docs/preregistration-v7.md`. Gemma 4 12B on the
same 48 fixed contexts and success measure as `manager_replay.md`. Wordings: v8 and `para3`
(in-job controls); `p3noframe` (no framing sentence); `p3nostep1`-`p3nostep5` (one step
removed, later steps renumbered); `v8p3frame` (v8 with `para3`'s opening). Sampled = Gemma's
recommended setting, 8 request seeds per context.

| manager | decoding | wording | events | n | success | kept all | freed >= 30% | parse errors | capped |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12B-it | greedy | p3noframe | first | 16 | 0.44 | 0.88 | 0.56 | 5 | 0 |
| gemma-4-12B-it | greedy | p3noframe | later | 32 | 0.66 | 0.81 | 0.84 | 3 | 1 |
| gemma-4-12B-it | greedy | p3nostep1 | first | 16 | 0.19 | 0.81 | 0.38 | 3 | 0 |
| gemma-4-12B-it | greedy | p3nostep1 | later | 32 | 0.34 | 0.78 | 0.56 | 13 | 2 |
| gemma-4-12B-it | greedy | p3nostep2 | first | 16 | 0.88 | 0.88 | 1.00 | 0 | 0 |
| gemma-4-12B-it | greedy | p3nostep2 | later | 32 | 0.22 | 0.91 | 0.31 | 16 | 0 |
| gemma-4-12B-it | greedy | p3nostep3 | first | 16 | 0.81 | 1.00 | 0.81 | 2 | 0 |
| gemma-4-12B-it | greedy | p3nostep3 | later | 32 | 0.62 | 0.91 | 0.72 | 7 | 1 |
| gemma-4-12B-it | greedy | p3nostep4 | first | 16 | 0.00 | 0.19 | 0.81 | 1 | 0 |
| gemma-4-12B-it | greedy | p3nostep4 | later | 32 | 0.09 | 0.47 | 0.62 | 9 | 4 |
| gemma-4-12B-it | greedy | p3nostep5 | first | 16 | 0.81 | 1.00 | 0.81 | 0 | 0 |
| gemma-4-12B-it | greedy | p3nostep5 | later | 32 | 0.25 | 0.97 | 0.28 | 21 | 2 |
| gemma-4-12B-it | greedy | para3 | first | 16 | 0.31 | 0.50 | 0.81 | 1 | 0 |
| gemma-4-12B-it | greedy | para3 | later | 32 | 0.62 | 0.91 | 0.72 | 8 | 2 |
| gemma-4-12B-it | greedy | v8 | first | 16 | 0.25 | 0.75 | 0.50 | 4 | 0 |
| gemma-4-12B-it | greedy | v8 | later | 32 | 0.06 | 0.50 | 0.53 | 9 | 7 |
| gemma-4-12B-it | greedy | v8p3frame | first | 16 | 0.19 | 0.69 | 0.50 | 2 | 0 |
| gemma-4-12B-it | greedy | v8p3frame | later | 32 | 0.16 | 0.91 | 0.25 | 21 | 10 |
| gemma-4-12B-it | sampled | p3noframe | first | 128 | 0.59 | 0.87 | 0.71 | 17 | 0 |
| gemma-4-12B-it | sampled | p3noframe | later | 256 | 0.28 | 0.84 | 0.43 | 117 | 22 |
| gemma-4-12B-it | sampled | p3nostep1 | first | 128 | 0.16 | 0.64 | 0.52 | 7 | 0 |
| gemma-4-12B-it | sampled | p3nostep1 | later | 256 | 0.28 | 0.71 | 0.57 | 101 | 13 |
| gemma-4-12B-it | sampled | p3nostep2 | first | 128 | 0.55 | 0.88 | 0.68 | 20 | 0 |
| gemma-4-12B-it | sampled | p3nostep2 | later | 256 | 0.21 | 0.95 | 0.25 | 139 | 13 |
| gemma-4-12B-it | sampled | p3nostep3 | first | 128 | 0.68 | 0.93 | 0.75 | 13 | 0 |
| gemma-4-12B-it | sampled | p3nostep3 | later | 256 | 0.40 | 0.94 | 0.46 | 121 | 23 |
| gemma-4-12B-it | sampled | p3nostep4 | first | 128 | 0.03 | 0.09 | 0.94 | 0 | 0 |
| gemma-4-12B-it | sampled | p3nostep4 | later | 256 | 0.18 | 0.59 | 0.58 | 95 | 23 |
| gemma-4-12B-it | sampled | p3nostep5 | first | 128 | 0.45 | 0.77 | 0.67 | 14 | 0 |
| gemma-4-12B-it | sampled | p3nostep5 | later | 256 | 0.27 | 0.95 | 0.32 | 153 | 21 |
| gemma-4-12B-it | sampled | para3 | first | 128 | 0.34 | 0.64 | 0.70 | 19 | 0 |
| gemma-4-12B-it | sampled | para3 | later | 256 | 0.38 | 0.88 | 0.49 | 114 | 15 |
| gemma-4-12B-it | sampled | v8 | first | 128 | 0.20 | 0.72 | 0.48 | 41 | 0 |
| gemma-4-12B-it | sampled | v8 | later | 256 | 0.09 | 0.49 | 0.58 | 78 | 19 |
| gemma-4-12B-it | sampled | v8p3frame | first | 128 | 0.12 | 0.70 | 0.42 | 28 | 0 |
| gemma-4-12B-it | sampled | v8p3frame | later | 256 | 0.17 | 0.74 | 0.42 | 126 | 43 |
