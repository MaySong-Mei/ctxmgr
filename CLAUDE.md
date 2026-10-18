# CLAUDE.md

Course project (Yale, cluster course id `cpsc4880`): who should manage an LLM agent's context?
This repo is public — never commit credentials, cluster usernames, model weights or results.
Agent operating instructions and current status: @AGENTS.md

## Experiment

Core grid = actor size × 4 conditions on the official ContextBench (released; pinned at
`fb533c18c78a82d3f1b2047a9c1700735cf829ae`, Track 1 adapter and contract in `docs/tasks-and-evaluation.md`
on `ruthvik/track1-tasks`, PR #1). Families: KV Store and Needle Retention at official named levels.

- Actors: Qwen3.5 dense instruct models 0.8B, 2B, 4B, 9B, 27B (not the `-Base` variants).
- Conditions: no management · CLM self-management · same-size peer manager · fixed 27B peer manager
  (a 27B actor gets none / self / same-size peer only). Self and peer managers use the same edit tools
  and instructions; the manager triggers at a context threshold.
- Budget: 32,768 o200k tokens with 2,048 reserved, same for every actor size. Exact o200k is required;
  served-model (Qwen) token counts are logged separately.
- Metrics: official ContextBench reward graded from per-turn context snapshots (KV: ANSWER blocks across
  snapshots; Needle: lines in the final snapshot), plus Track 1's strict diagnostics, plus compute
  including manager calls (adapt CLM `flops_metrics/`). The gold oracle is an implementation check,
  never a baseline.

## Tracks / ownership

| Track | Owner | Code |
|---|---|---|
| 1 Tasks & eval | Ruthvik | `src/ctxmgr/tasks/` |
| 2 Harness & managers (vLLM agent loop, CLM edit tools, manager hook, per-step logs) | Hancheng | `src/ctxmgr/harness/` |
| 3 Infra, runs, analysis, report skeleton | May | `src/ctxmgr/{serving,accounting,analysis}/`, `slurm/`, `report/` |

Tracks 1 and 2 must agree on the interface (task format in → context/answers out); document it in `docs/`.
Branches: `<name>/<track>` (e.g. `may/track3-infra`), merged to `main` via PR.
Timeline: weeks 1–2 tasks + harness + pilot on 0.8B; weeks 3–5 core grid; weeks 6–8 failure analysis + 4-page NeurIPS write-up.

## Compute (Yale Bouchet, `education_gpu`)

- Enforced limit: **1 GPU per user, 1 job at a time, 24 h wall** (docs say 2 GPUs; QoS enforces 1).
- GPUs: A5000 24 GB (32), RTX 5000 Ada 32 GB (12), A40 48 GB (4, one node).
- Rules of Behavior: all computation through Slurm, never on login nodes; end idle interactive sessions;
  no floods of <5-min jobs; don't split work across accounts to dodge fair-use limits; course accounts are coursework-only.
- Storage: weights/results in project dir `/nfs/roberts/project/<course>/<user>` (backed up);
  scratch is purged after 30 days (caches only). Paths are set in `slurm/env.sh`.
- Connection setup: `docs/cluster_setup.md`. Login needs Duo, which only a human can approve.

## Models & precision

- Weights (bf16): 0.8B 1.7 GB, 2B 4.5 GB, 4B 9.3 GB, 9B 19.3 GB, 27B 55.6 GB. Official quants exist only for 27B
  (`Qwen3.5-27B-FP8` 30.9 GB, `-GPTQ-Int4` 30.2 GB).
- 27B fits no single GPU in bf16 → use `Qwen3.5-27B-FP8` on an A40 (Ampere: weight-only FP8, slower).
- 9B actor + 27B-FP8 manager does not fit one A40 in bf16 → candidate: 9B in FP8 for all four of its conditions.
- Keep precision identical across the four conditions of a given actor size. Plan a bf16-vs-FP8 control on one small size.
- Qwen3.5 is multimodal (`Qwen3_5ForConditionalGeneration`) with hybrid linear/full attention (1 in 4 layers full),
  so KV cache is small (~64 KB/token for 27B). Serve text-only (`--limit-mm-per-prompt` image/video = 0).
  Thinking is on by default; control it via `chat_template_kwargs.enable_thinking`.
- vLLM pinned to 0.30.0 (`uv sync --group serve`, Linux only); it pulls torch 2.13 built for CUDA 13.
  GPU nodes run driver 580 (CUDA 13 OK) but have no `nvcc`, so FlashInfer JIT kernels fail →
  `VLLM_USE_FLASHINFER_SAMPLER=0` (set in `slurm/env.sh`). Attention backend is FLASH_ATTN (no JIT).
- Pilot (2026-10-12, 0.8B on A5000): smoke test passed; 27K-token KV recall correct; ~313 tok/s decode;
  KV cache 1.36M tokens; job ~7 min, of which vLLM startup ~3 min (fixed per-job overhead).
- Pilot (2026-10-14, A40, max_len 32K, gpu_util 0.90; single-request decode is memory-bandwidth bound):
  2B: 3.6 GiB weights, KV 2.72M tok, 128 tok/s · 4B: 8.0 GiB, KV 0.88M tok, 59 tok/s ·
  9B: 16.8 GiB, KV 0.60M tok, 33 tok/s. All recalled the 27K-token KV probe. 9B on A5000 still untested.
- Pilot (2026-10-15, A40, `slurm/a40_27b.sbatch` first run, max_len 32K): every phase FAILED, all on
  configuration, none on capacity. 27B-FP8 download 5 min (setup job OK). 27B-FP8 loads: 28.0 GiB
  weights, 25 s (8 s cached); torch.compile 48 s first time (later cached). Alone at util 0.90: 8.32 GiB
  left for KV; co-location share 0.75: 3.79 GiB = 53,930 tokens, i.e. only 1.65 concurrent 32K requests
  for the manager. Both then refused to start: default `max_num_seqs` 256 exceeds the Mamba state blocks
  (173 at 0.90, 79 at 0.75); fix `--max-num-seqs 64`. 9B with online FP8 crashed in torch.compile
  (Inductor "auto_functionalized was not removed", vLLM 0.30 / torch 2.13); fix `--enforce-eager`
  (eager decode is a lower bound) or an offline FP8 checkpoint. Actors never started, so co-location
  fit is still unknown. Both fixes are in the script.
- Pilot (2026-10-16, A40 re-run with `--max-num-seqs 64`, max_len 32K): job exit 1 (3 of 5 phases fail).
  27B-FP8 alone (util 0.90) PASS: Marlin weight-only FP8 kernel, KV 9.03 GiB = 128K tokens (3.9× 32K),
  27K-token probe correct (prefill 22 s), decode 20.3 tok/s. Co-located 27B-FP8 (0.75) + 2B (0.21) PASS,
  both smoke tests concurrently: 27B KV 63K tokens (1.9× 32K), decode still 20.3 tok/s, prefill 28 s;
  2B KV 342K tokens, 58 tok/s (vs 128 alone); 43.1 of 46.1 GB used. 27B + 4B FAIL on capacity: 4B weights
  8.0 GiB leave −0.19 GiB KV in the 0.21 share. 9B online FP8 FAIL (alone and co-located): vLLM picks the
  CUTLASS w8a8 kernel, which errors on Ampere (`cutlass_scaled_mm_sm80_epilogue`); the 27B checkpoint
  works because it gets Marlin. 9B-FP8 weights are 10.4 GiB, above the 0.21 share (~9.4 GiB) anyway.
  So on one A40 only 2B (and presumably 0.8B) can sit beside the 27B manager at 32K.
- First end-to-end baseline (2026-10-16, `slurm/baseline_smoke.sbatch` on `may/track2-baseline` d94f74a,
  2B bf16, condition none, seed 42, max_len 40,960, thinking off, temp 0; landed on an RTX 5000 Ada since
  `--gpus=1` does not pin a type): job COMPLETED in 10 min (vLLM startup ~2.5 min). KV R130: ran all
  50 turns, official reward 1.0 (24/24), peak 10.5K o200k. Strict exact accuracy 0.0 only because 2B
  writes a stray `>` after the ANSWER tag, so every value starts with "> " (official grader matches by
  substring). Needle K4-C4: cut off at turn 9 by vLLM (`server_rejected`: 38,913 Qwen prompt tokens +
  2,048 max_tokens > 40,960) while still under the o200k budget (peak 24.6K), so its 0.75 (12/16) is
  not a full-episode score. Qwen/o200k ratio: 1.25 on KV, 1.40 on Needle. 2B often hit the 2,048-token
  completion cap on Needle. Runs are under `runs/baseline-<job>` in the project dir (not in git).
- Needle K4-C4 re-run (2026-10-16, same instance hash, 2B, `MAX_LEN=49152`, RTX 5000 Ada): job COMPLETED,
  no vLLM errors. Turns 1–9 identical to the 40,960 run (temp 0), then the episode ended at turn 11 on
  ContextBench's o200k `budget_wall` (peak 31.4K o200k, above budget − reserve = 30,720, so
  `snapshot_budget_exceeded` is true). Official reward 1.0 (16/16), required-info loss 0, strict
  verbatim retention 1.0. Max Qwen prompt 42,076 + 2,048 completion fits 49,152; ratio 1.37–1.40.
  8 of 11 completions hit the 2,048 cap (`finish_reason=length`). KV cache at 49,152: 1.90M tokens.
- No-management scaling baseline (2026-10-17, job on `may/track2-baseline` baddbbf, one A5000 job for
  0.8B/2B/4B, bf16, temp 0, thinking off, max_tokens 2,048, seed 42, `MAX_LEN=49152`; 19.6 min total).
  All three servers started (4B: 8.0 GiB weights, KV 356K tokens = 7.2× 49K). Ratio = max Qwen/o200k.

  | model | level | end | turns | reward | diagnostic | peak o200k | capped | ratio |
  |---|---|---|---|---|---|---|---|---|
  | 0.8B | KV R130 | budget_wall | 286 | 0.00 (0/24) | strict 0.00 | 30,722 | 0 | 1.22 |
  | 0.8B | Needle K4-C4 | stream_exhausted | 4 | 1.00 (16/16) | loss 0.00 | 16,446 | 0 | 1.40 |
  | 0.8B | KV R1000 | budget_wall | 7 | 0.00 (0/24) | strict 0.00 | 34,912 | 0 | 1.24 |
  | 0.8B | Needle K4-C24 | budget_wall | 8 | 0.33 (32/96) | loss 0.67 | 32,409 | 0 | 1.40 |
  | 2B | KV R130 | stream_exhausted | 50 | 1.00 (24/24) | strict 0.00 | 10,512 | 0 | 1.25 |
  | 2B | Needle K4-C4 | budget_wall | 14 | 1.00 (16/16) | loss 0.00 | 31,546 | 8 | 1.40 |
  | 2B | KV R1000 | budget_wall | 7 | 0.00 (0/24) | strict 0.00 | 34,912 | 0 | 1.24 |
  | 2B | Needle K4-C24 | budget_wall | 8 | 0.33 (32/96) | loss 0.67 | 33,367 | 0 | 1.40 |
  | 4B | KV R130 | submit | 26 | 0.00 (0/24) | strict 0.00 | 9,533 | 0 | 1.24 |
  | 4B | Needle K4-C4 | submit | 4 | 1.00 (16/16) | loss 0.00 | 17,230 | 0 | 1.39 |
  | 4B | KV R1000 | budget_wall | 7 | 0.00 (0/24) | strict 0.00 | 35,220 | 0 | 1.24 |
  | 4B | Needle K4-C24 | server_rejected | 7 | 0.29 (28/96) | loss 0.71 | 29,728 | 0 | 1.40 |

  Reading (from `scripts/inspect_completions.py` and the final snapshots):
  - Scores mostly reflect output format, not memory. 0.8B KV R130 wrote a malformed ANSWER
    (`key=00122>>>>`, no END tag, no `#hash`) and never `READY_FOR_NEXT_OP`, so the stream never
    advanced; it repeated the same reply ~280 times until the wall. 4B KV R130 wrote well-formed
    blocks but dropped the `#hash` suffix of every value (official needs it as a substring), then
    submitted after 26 of 50 turns. 2B keeps the hash but adds a stray `>`.
  - Echoing: 2B copies the NEEDLES block verbatim into its replies (and on K4-C4 echoes until the
    2,048 cap); 4B restates the needles as a numbered list and claims to delete filler it cannot
    delete; 0.8B replies only `READY_FOR_NEXT_OP` (no echo). Since Needle is graded from the final
    context, echoes in replies count as retention for no-management.
  - KV R1000 and Needle K4-C24 overflow the budget within 7–8 turns for every size, as expected
    without management (KV 0, Needle ~0.3).
  - 4B Needle K4-C24 hit `server_rejected` (47,105 Qwen prompt + 2,048 > 49,152): the harness sends
    the turn that crosses the o200k wall to the server (peaks up to 35.2K o200k ≈ 49K Qwen), so either
    check the wall before the call (Track 2) or serve ~53K.
  - 2B KV R130 reproduces the RTX 5000 Ada run exactly; 2B Needle K4-C4 does not (14 turns vs 11):
    temp 0 is not bitwise deterministic across GPU types, so keep one GPU type per comparison.
  - The vLLM ERROR lines in the 0.8B log are shutdown noise when the server is stopped between models.
- First managed pilot (2026-10-17, `may/track2-baseline` aa5b2f8 = draft management c717c61 + manager
  reply logged in `edits.jsonl`; pytest 33 passed on a CPU node first). One A5000 job, 2B and 4B ×
  none / self / peer_same × KV R1000 / Needle K4-C24, same fixed settings as the scaling baseline,
  trigger 18,432 o200k, `MAX_LEN=49152`. Turns = actor calls. Events = management calls; "applied" /
  "error" count events with any applied edit / any error (one event can be both). Tokens = prompt +
  completion summed per role.

  | model | cond | level | end | turns | reward | diag | peak | events | applied/err | tokens actor · manager |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 2B | none | KV R1000 | budget_wall | 6 | 0 (0/24) | strict 0 | 34,908 | 0 | – | 132K · – |
  | 2B | self | KV R1000 | budget_wall | 6 | 0 (0/24) | strict 0 | 34,908 | 3 | 0/0 | 132K · 95K |
  | 2B | peer_same | KV R1000 | budget_wall | 6 | 0 (0/24) | strict 0 | 34,908 | 3 | 0/1 | 132K · 96K |
  | 2B | none | Needle K4-C24 | budget_wall | 7 | 0.33 (32/96) | loss 0.67 | 33,245 | 0 | – | 166K · – |
  | 2B | self | Needle K4-C24 | stream_exhausted | 24 | 0.17 (16/96) | loss 0.83 | 22,828 | 4 | 4/4 | 425K · 123K |
  | 2B | peer_same | Needle K4-C24 | budget_wall | 17 | 0.17 (16/96) | loss 0.83 | 31,482 | 6 | 1/6 | 404K · 208K |
  | 4B | none | KV R1000 | budget_wall | 6 | 0 (0/24) | strict 0 | 35,058 | 0 | – | 133K · – |
  | 4B | self | KV R1000 | budget_wall | 6 | 0 (0/24) | strict 0 | 35,058 | 3 | 0/3 | 133K · 98K |
  | 4B | peer_same | KV R1000 | budget_wall | 6 | 0 (0/24) | strict 0 | 35,058 | 3 | 0/3 | 133K · 96K |
  | 4B | none | Needle K4-C24 | budget_wall | 7 | 0.33 (32/96) | loss 0.67 | 34,022 | 0 | – | 169K · – |
  | 4B | self | Needle K4-C24 | budget_wall | 7 | 0.33 (32/96) | loss 0.67 | 34,536 | 3 | 3/0 | 169K · 111K |
  | 4B | peer_same | Needle K4-C24 | budget_wall | 19 | 0.33 (32/96) | loss 0.67 | 33,506 | 11 | 8/4 | 481K · 369K |

  Manager replies (`edits.jsonl`): the JSON is usually parseable, but the edits are almost never useful.
  - 2B: either `{"edits": []}` (all KV events, both conditions) or "delete every block from 0 up".
    `apply_edits` drops the protected ids and applies the rest, so each such event wiped the context to
    ~600 tokens (needles included): 2B self Needle went 21.5K → 0.6K four times and ended with half
    the reward of no management. Long id lists also hit the 2,048 cap, leaving JSON unclosed ("no JSON
    object").
  - 4B KV: wrote replace edits without `"op"` (rejected as unknown), tried to replace protected block 0,
    or echoed a whole SET-BATCH block until the cap. Had they applied, they would have overwritten its
    replies with "stored all key-value pairs K00000–K00099" summaries (values lost). No KV edit applied.
  - 4B peer Needle deleted filler blocks correctly (`delete [10]`, `[10, 11]`, ~4K o200k each), which
    stretched the episode from 7 to 19 turns, but too little was freed and the reward stayed 0.33.
    4B self Needle only replaced its own reply with a needle list (context grew), so it ended like none.
  - Managers cost a lot: 50–80% of the actor's tokens (4B peer Needle 369K vs 481K).
  - For Track 2 before freezing the protocol: reject a delete that names protected ids instead of
    applying the rest; decide whether a missing `"op"` with `id`+`text` means replace; allow id ranges
    or give the manager more than 2,048 output tokens; KV R1000 needs a way to keep values while
    dropping SET-BATCH blocks (the actor never printed ANSWER blocks before the wall).
  - 2B log has the usual 8 shutdown ERROR lines; 4B none. KV caches: 2B 1.39M, 4B 387K tokens.
- peer_27b pilot (2026-10-17, `slurm/peer27b_pilot.sbatch` on `may/track2-baseline` 8b4cd8a = hardened
  protocol from PR #4; A40, 27B-FP8 manager + 2B actor co-located, Needle K4-C24, seed 42, trigger
  18,432, `MAX_LEN=49152`, 7.9 min). First attempt failed: at util 0.75 the 27B has 2.37 GiB KV, below
  the 3.3 GiB one 49,152-token request needs; `BIG_UTIL=0.79 ACTOR_UTIL=0.17` works.

  | cond | end | turns | reward | loss | peak o200k | events → applied | tokens actor · manager |
  |---|---|---|---|---|---|---|---|
  | none | budget_wall | 7 | 0.33 | 0.67 | 33,245 | – | 166K · – |
  | self | budget_wall | 7 | 0.33 | 0.67 | 33,245 | 3 → 0 | 166K · 108K |
  | peer_same | budget_wall | 7 | 0.33 | 0.67 | 33,245 | 3 → 0 | 166K · 107K |
  | peer_27b | stream_exhausted | 24 | 0.17 | 0.83 | 21,121 | 4 → 4 | 416K · 120K |

  - Hardening worked as intended for 2B: its "delete 0..N" edits are now rejected whole (protected ids),
    so self / peer_same no longer wipe the context; they change nothing and end exactly like none.
  - 27B writes well-formed edits but uses them to wipe: all four events were `delete from 2 to 11`
    (every editable block, needles included), 21K → 0.6K o200k each time. The episode now finishes the
    stream (24 turns, never hits the wall) but only the needles delivered after the last wipe survive,
    so the reward halves. Once it also tried to rewrite protected block 1 (the task instruction).
  - So a 27B manager is not enough under this protocol: it frees space but does not keep what the task
    needs. Candidate causes to test: the manager prompt never states the task goal beyond block 1;
    "free space" is the only explicit objective; no feedback on what was lost.
- peer_27b re-run with edit instructions v2 (2026-10-17, job 29919206, `may/track2-baseline` 6db5ded:
  "judged by task completion", "read block 1 for what to keep", "replace with the needed lines",
  "never delete the only copy"; utils 0.79/0.17; 17.8 min). none / self / peer_same unchanged
  (0.33, all 2B edits rejected or empty). peer_27b: stream_exhausted after 44 turns, **reward 0.00**
  (loss 1.00), 6 events, actor 807K · manager 192K tokens.
  - 27B now sometimes does the right thing: at turns 21 and 28 it deleted filler and replaced chunk
    blocks with exactly their NEEDLES lines (21K → 1.1K, 23K → 1.6K).
  - At turns 6 and 16 it tried to consolidate the needles into **block 1** (a replace of the task
    instruction with 22 needle lines appended) together with a delete of blocks 2–11. Block 1 is
    protected, so the replace was rejected, but edits are validated independently, so the delete
    went through and the needles were lost. The same happened once in the v1 run (turn 11).
  - Its last event (turn 44) was `delete from 2 to 38` with no replace, wiping everything before the
    final grade, hence 0.00.
  - Protocol issues this exposes: (1) independent validation breaks a manager's plan when one edit
    of it is rejected (all-or-nothing per reply would have kept the needles); (2) there is no legal
    place to consolidate kept content except rewriting an existing block, so 27B reached for the
    instruction block; (3) the instructions alone do not stop a wholesale delete near the end.

## Dev

- Python 3.12, uv. `uv sync`, `uv run pytest`, `uv run ruff check .`.
- Pilot: `sbatch slurm/setup_env.sbatch [models]` (CPU node) then `sbatch slurm/serve_smoke.sbatch [model ...]` (1 GPU; env MAX_LEN, GPU_UTIL).

## Open decisions

- Precision for 9B; how to run the 9B × fixed-27B-manager cell.
- Where the fixed 27B manager runs: co-located at util 0.75 it has ~63K KV tokens (~1.9 concurrent 32K
  requests), so manager calls would serialize, and only 2B/0.8B actors fit beside it (4B and 9B do not).
  Alternatives: 27B alone on the A40 (128K KV tokens) serving manager calls in a separate phase, or
  batching all peer_27b cells per actor.
- Instances / pressure levels / seeds per cell → total GPU-hours under the 1-GPU limit.
- Whether to ask course staff for a higher GPU limit.
- Prior-art check: does the CLM paper's multi-agent section already run an equivalent peer-manager
  comparison? Settle before the core grid; it decides the novelty claim.
- Deferred per the instructor's core grid (Track 1 AGENTS.md): scheduled summarization, next-size-up
  peers, MoE. Suggest dropping the proposal's oracle-retention baseline: ContextBench's gold oracle reads the answer
  key and is only an implementation check.
- Manager scope: the peer manager may only edit context (same edit tools as self-management), never
  answer the task or add reasoning, or task skill leaks into the peer conditions.
- Served context length: the 32K budget is in o200k tokens, but vLLM `max_len` counts Qwen tokens, which
  can be more for the same text. Measured ratio 1.25 (KV) to 1.40 (Needle) on the 2B baseline, and
  40,960 was too small for Needle. Covering the full 30,720-token o200k context at 1.40 plus a 2,048
  completion needs ~45K, so `max_len` 49,152 is the candidate; it let the 2B Needle episode reach the
  o200k budget wall (peak prompt 42K Qwen). It shrinks KV memory and co-location fit (27B at util 0.75
  would hold ~1.3 such requests).
- Strict KV diagnostic (Track 1): exact match fails on formatting like a stray `>` after the ANSWER tag
  (0/24 strict vs 24/24 official on the 2B baseline). Decide whether strict should strip such tag debris.
- Context-management logging from the first run (draft in `docs/log_schema.md`): per-call usage by role,
  edits, triggers, termination, on top of ContextBench snapshots.
