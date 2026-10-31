# Replay probe (round 30, job 31513325): 27B actor output-cap loop

Built by `scripts/replay_probe.py summarize` (harness `may/track2-baseline` 51b6825) from
`runs/replay-31513325/replay.jsonl`: single calls rebuilt from job 31451066's snapshots;
pre-registered in `docs/preregistration-v3.md`. capped = cut at `max_tokens`; ready = marker on
a line of its own; earlier ops restated = the reply restates needles of an op before the latest
op it mentions; same as logged = identical text to the episode's reply (R1, R4 D0). R4 rows are the
4th turn of each rollout.

| arm | level | point | n | capped | ready | ready <= 2,048 tokens | median tokens | earlier ops restated | same as logged |
|---|---|---|---|---|---|---|---|---|---|
| R1_exact | K4-C24 | a | 8 | 8 | 0 | 0 | 2048 | 8 | 8/8 |
| R1_exact | K4-C24 | b | 8 | 8 | 0 | 0 | 2048 | 8 | 7/8 |
| R1_exact | K4-C24 | deep | 8 | 8 | 0 | 0 | 2048 | 8 | 8/8 |
| R1_exact | K8-C48 | a | 8 | 8 | 0 | 0 | 2048 | 8 | 8/8 |
| R1_exact | K8-C48 | b | 8 | 2 | 6 | 6 | 274 | 2 | 8/8 |
| R1_exact | K8-C48 | deep | 2 | 2 | 0 | 0 | 2048 | 2 | 2/2 |
| R2_cap8k | K4-C24 | a | 8 | 0 | 8 | 0 | 2114 | 8 | - |
| R2_cap8k | K4-C24 | b | 8 | 0 | 8 | 0 | 2114 | 8 | - |
| R2_cap8k | K8-C48 | a | 8 | 0 | 8 | 0 | 2116 | 8 | - |
| R2_cap8k | K8-C48 | b | 8 | 0 | 8 | 6 | 271 | 2 | - |
| R3a_short_prev | K4-C24 | b | 8 | 7 | 1 | 1 | 2048 | 7 | - |
| R3b_drop_prev | K4-C24 | b | 8 | 8 | 0 | 0 | 2048 | 8 | - |
| R4_D0 | K4-C24 | rollout | 8 | 0 | 8 | 8 | 616 | 8 | 8/8 |
| R4_D1 | K4-C24 | rollout | 8 | 0 | 8 | 8 | 638 | 8 | - |
| R4_D2 | K4-C24 | rollout | 8 | 0 | 8 | 8 | 634 | 8 | - |
| R4_D3 | K4-C24 | rollout | 8 | 0 | 8 | 8 | 645 | 8 | - |
| R5a_sampled | K4-C24 | b | 64 | 59 | 5 | 5 | 2048 | 59 | - |
| R5b_sampled_pp | K4-C24 | b | 64 | 58 | 6 | 6 | 2048 | 58 | - |
| R6_cap_reminder | K4-C24 | b | 8 | 4 | 4 | 4 | 2048 | 4 | - |
| R6_cap_reminder | K4-C24 | deep | 8 | 3 | 5 | 5 | 431 | 3 | - |
