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

## Compute (Yale Bouchet, account `cpsc4880`)

- The course account can submit to `education_gpu` **and** the general GPU partitions (verified
  2026-10-20: a 2× A40 single-node job on `gpu` ran under `cpsc4880`). The 1-GPU cap is a property of
  the `education_gpu` partition, not of the account.
- `education_gpu`: **1 GPU per user, 1 job at a time, 24 h wall** (docs say 2 GPUs; QoS enforces 1).
  GPUs: A5000 24 GB (32), RTX 5000 Ada 32 GB (12), A40 48 GB (4, one node).
- General partitions (per-user QoS limits): `gpu` (A40 / A5000 / L40S / RTX 5000 Ada; ≤32 GPUs,
  12 jobs, 2 d wall), `gpu_h100`, `gpu_h200` (≤16 GPUs, 6 jobs, 2 d), `gpu_b200`, `gpu_rtx6000`,
  `gpu_devel` (≤2 GPUs, 1 job, 6 h). No `priority*` tiers.
- Multi-GPU submission gotchas: request GPUs on one node (`-N1 --gpus-per-node=a40:2`); plain
  `--gpus=2` may split them across two nodes, which breaks vLLM tensor parallelism. Use `sbatch` or
  pass `--qos=normal`: a login-node `srun` defaults to the `interactive` QoS, which `gpu*` partitions
  reject (`Invalid qos specification`).
- Usage on any partition is charged to the shared `cpsc4880` fairshare, so heavy general-partition use
  lowers classmates' priority; keep jobs sized to what the grid needs.
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
- peer_27b re-run with edit protocol v3 (2026-10-17, job 29924731, `may/track2-baseline` 80bc314:
  all-or-nothing replies + `note` op for a notes block after the instruction; 16.8 min).

  | cond | end | turns | reward | loss | peak o200k | events → applied | tokens actor · manager |
  |---|---|---|---|---|---|---|---|
  | none | budget_wall | 7 | 0.33 | 0.67 | 33,245 | – | 166K · – |
  | self | budget_wall | 7 | 0.33 | 0.67 | 32,868 | 3 → 3 | 165K · 108K |
  | peer_same | budget_wall | 10 | 0.33 | 0.67 | 33,209 | 6 → 6 | 272K · 215K |
  | peer_27b | stream_exhausted | 24 | **0.71** (68/96) | 0.29 | 29,532 | 7 → 3 | 525K · 246K |

  - 27B uses the notes block as intended: "copy the needles into a note, delete blocks 2–11" (turn 6,
    20 needles; turn 11, 40 needles; each 21K → 1.1–1.7K o200k). First time any managed condition beats
    none.
  - The four failed events (turns 16, 17, 23, 24) all hit the 2,048-token manager output cap
    (`finish_reason=length`): each note rewrites the whole notes block, which by then holds 53–54
    needles, so the JSON is cut off and the reply rejected. The lost needles (28/96) come from that.
  - 2B self deleted single small blocks (−125 o200k each); 2B peer_same deleted block 2 every time
    (−4K when it was a chunk), so both still hit the wall with the same reward as none.
  - Next protocol fix candidates: an append-only note (cost grows with new content, not total notes)
    or a larger manager output budget; record whichever in the edit contract and bump the version.
- peer_27b re-run with edit protocol v4 (2026-10-18, job 29927590, `may/track2-baseline` 8573b2e:
  append-only notes; 13.5 min). none / self / peer_same unchanged at 0.33 (32/96). peer_27b:
  budget_wall after 17 turns, **reward 0.75 (72/96)**, loss 0.25, actor 352K · manager 177K tokens.
  - Turn 6 as in v3 (note with 20 needles, delete 2–11). Turn 11: instead of appending, 27B replaced
    the notes block with all 40 needles. Turns 16–18: "delete 2–12" (including the notes block) plus
    a note re-copying all 54 needles; all three hit the 2,048 output cap and were rejected, so the
    context grew to the wall.
  - So append-only does not change 27B's behavior: it rebuilds the whole notes block every time
    (by replace, or by delete + note). The binding limit is the manager's output budget, which is
    separate from the actor's 2,048-token benchmark reserve: ~96 needles need roughly 3.5K Qwen
    output tokens.
- peer_27b re-run with `manager_max_tokens` = 4096 (2026-10-19, job 30037280, `may/track2-baseline`
  d15e82a, edit-v4; actor stays 2,048; 13.1 min). none / self / peer_same unchanged at 0.33 (32/96).
  peer_27b: **stream_exhausted after 24 turns, reward 0.79 (76/96)**, loss 0.21, peak 22.1K o200k,
  actor 441K · manager 127K tokens, no capped manager reply (longest 2,302 tokens).
  - Turns 6, 11, 16: notes with 20 / 40 / 60 needles plus deletes (21K → 1.1K / 1.7K / 2.2K). The
    turn-16 note (60 needles, 2,302 tokens) is the one 2,048 used to cut off.
  - Turn 21: `delete 3–12` with no note, dropping the 20 needles that arrived after turn 16; they
    are the whole 0.21 loss. No protocol error remains: the last failure is the manager's choice.
  - Progression on this instance (2B actor, Needle K4-C24, seed 42): none 0.33 · 27B manager v1 0.17
    · v2 0.00 · v3 0.71 · v4 0.75 · v4 + 4096 budget 0.79. Same-size 2B managers stay at 0.33 in all
    versions (their edits are rejected, empty, or delete one block). Single seed: not yet a result.
- Multi-seed peer_27b pilot (2026-10-19, job 30043145, `may/track2-baseline` 863dce2, edit-v4,
  manager_max_tokens 4096, 2B actor, seeds 42–45, 56.6 min, all 32 episodes completed).

  | level | none | self | peer_same | peer_27b (per seed 42/43/44/45) |
  |---|---|---|---|---|
  | Needle K4-C24 | 0.33 | 0.33 | 0.33 | **0.69** (0.79 / 0.58 / 0.79 / 0.58) |
  | KV R1000 | 0.00 | 0.00 | 0.00 | 0.07 (0.00 / 0.00 / 0.29 / 0.00) |

  - Needle: the 27B gain holds on every seed (min 0.58 vs 0.33), and peer_27b always finishes the
    stream (4 events, all applied, none capped). Seed 42 reproduces the previous run exactly (0.79).
    none / self / peer_same are 0.33 on every seed. Seeds change the words, not the structure, so
    results pair up (42 = 44, 43 = 45, down to token counts): weak replicates, not independent ones.
  - KV R1000: 27B frees space by `delete from 2 to 9` (all SET batches, 20K → 0.7K) without keeping
    any values in a note, so the stream finishes but GETs find nothing. Only seed 44 scored (0.29).
    The edit protocol lets a manager keep values (note/replace); 27B does not choose to.
  - Cost: Needle peer_27b actor 435–441K + manager 123–126K tokens vs none 165K (3.4×). KV
    peer_27b actor 473–869K + manager 50–73K vs none 132K.
  - 2B self / peer_same managers: 2–6 events per episode, mostly applied (single-block deletes), no
    score change. Same-size management is a no-op for 2B on both tasks.
- peer_27b extension (2026-10-19, job 30046489, 863dce2, same settings, `none` + `peer_27b` only,
  seeds 46–49 on Needle K4-C24 and K8-C48, 2 h 8 min, all 16 episodes completed). Pooled with
  job 30043145 for K4-C24 seeds 42–45:

  | level, seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
  |---|---|---|---|---|---|---|---|---|
  | K4-C24 none | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.04* |
  | K4-C24 peer_27b | 0.79 | 0.58 | 0.79 | 0.58 | 0.79 | 1.00 | 0.58 | 1.00 |
  | K8-C48 none | | | | | 0.15 | 0.15 | 0.15 | 0.15 |
  | K8-C48 peer_27b | | | | | 0.35 | 0.35 | 0.35 | 0.35 |

  \* `server_rejected` at turn 7: 47,105 Qwen prompt tokens + 2,048 > `max_len` 49,152 while still
  under the o200k wall (the Qwen/o200k ratio problem Ruthvik reported on #3; serve ≥ 81,920).
  - K4-C24: peer_27b > none on 8/8 seeds, mean paired difference +0.47 (bootstrap 95% CI
    0.33–0.64), one-sided sign test p = 0.004; excluding the flawed s49 none, 7/7, p = 0.008, mean
    +0.40. Caveats: temperature 0, seeds share the level's structure (only 4 distinct difference
    values), one actor size, vLLM/FP8 stack (not comparable with the llama.cpp/Q8_0 baseline).
  - K8-C48 (384 needles): peer_27b 0.35 vs 0.15 on all 4 seeds (identical outcomes), but every
    episode still hits the budget wall: 3 manager replies per episode hit the 4,096 output cap
    because 27B rewrites the whole notes block. 4/4 positive, p = 0.06.
- Round 1 of the auto loop: edit-v5 (2026-10-20, job 30068993, `may/track2-baseline` 60c4848; notes
  block kept by the protocol: deletes skip it, replace of it = note, notes deduplicated; instructions
  said "never copy the existing notes again"; actor `max_len` 81,920, manager 49,152; 56.6 min).

  | level | none (s46–49) | peer_27b v5 (s46 / 47 / 48 / 49) | v4 for reference |
  |---|---|---|---|
  | K4-C24 | 0.33 ×4 | 1.00 / 0.79 / 0.79 / 0.58 | 0.79 / 1.00 / 0.58 / 1.00 |
  | K8-C48 | 0.15 ×4 | 0.27 / 0.25 / 0.23 / 0.29 | 0.35 ×4 |

  - The actor `max_len` fix worked: every `none` episode now ends on the o200k wall (s49 was
    `server_rejected` at 49,152 before).
  - v5 removed the output-cap failures (0 capped replies; K8-C48 now finishes the stream, 48 turns,
    10–11 events, all applied) but K8-C48 reward fell. In s46 only 2 of 10 events wrote a note
    (turn 6: chunks 0–4; turn 26: chunks 20–24); the other 8 deleted chunk blocks without noting
    their needles. "Never copy the existing notes again" seems to have suppressed noting altogether.
  - K4-C24 is unchanged in distribution (mean 0.79 both versions).
  - Next (v6): instruct "before deleting a block with needed lines, note those lines from that
    block in the same reply; repeats are dropped automatically", and rerun the same cells.
- Round 2: edit-v6 (2026-10-20, job 30069061, af240e3; instruction "note needed lines from every
  block before deleting it, every time; lines already in notes are dropped automatically"; peer_27b
  only, same instances; 43.4 min). **Worse:** K4-C24 0.375 on all 4 seeds, K8-C48 0.167 on all 4.
  27B wrote one note at the first event (chunks 0-4) and afterwards only deleted (`delete 3-12`, no
  note) in every event. No caps, all events applied. Hypothesis: the label "notes, always kept" and
  "always kept" in the instructions read as "needed lines are saved for me".
- Round 3: edit-v7 (job 30069142, 139c19e): label just `notes`; instructions say the notes block
  holds only what the editor writes and a deleted block is gone for good. Same cells.
  Result (78.1 min): **best version so far.**

  | level | none | v4 (s46-49) | v6 | v7 (s46 / 47 / 48 / 49) |
  |---|---|---|---|---|
  | K4-C24 | 0.33 | 0.79 | 0.375 | 1.00 / 1.00 / 0.79 / 1.00 (mean 0.95) |
  | K8-C48 | 0.15 | 0.35 | 0.167 | 0.42 / 0.33 / 0.25 / 0.42 (mean 0.35) |

  - 27B notes again: K4-C24 s46 noted 20/20/60/20 needles over its 4 events (the 60 includes repeats,
    dropped by dedup). K8-C48 s46: 12 events, notes in 5 (40/80/32/108/40 needles), none in 7, so
    needles are still lost there; 3 replies over all K8-C48 episodes hit the 4,096 cap.
  - All episodes finish the stream. Manager cost K8-C48 313-365K vs actor 864-951K.
  - Caution: v5-v7 were tuned on seeds 46-49. **v7 is now frozen**; next rounds evaluate it on other
    seeds and conditions instead of tuning further.
- Round 4: frozen edit-v7 on held-out seeds 42-45 (2026-10-21, job 30069159, 139c19e; 2B actor,
  none / self / peer_same / peer_27b, K4-C24 + K8-C48, 87.4 min, 32 episodes completed).

  | level | none | self | peer_same | peer_27b (s42 / 43 / 44 / 45) |
  |---|---|---|---|---|
  | K4-C24 | 0.33 | 0.33 | 0.375 / 0.375 / 0.33 / 0.375 | 1.00 / 1.00 / 1.00 / 1.00 |
  | K8-C48 | 0.15 | 0.17 | 0.17 / 0.15 / 0.15 / 0.15 | 0.25 / 0.31 / 0.33 / 0.23 |

  Paired vs `none`, v7 (peer_27b pooled over held-out s42-45 and tuning s46-49, n = 8):
  - K4-C24 peer_27b: +0.64 (bootstrap 95% CI +0.59 to +0.67), 8/8 positive, sign test p = 0.004.
    Held-out seeds alone: 1.00 on all four, so no sign that v7 overfit s46-49.
  - K8-C48 peer_27b: +0.17 (CI +0.13 to +0.22), 8/8 positive, p = 0.004. Still loses most needles
    (some events delete without noting; occasional 4,096 caps).
  - 2B self: +0.00 (K4-C24) / +0.02 (K8-C48); 2B peer_same: +0.03 / +0.01 (n = 4 each). Same-size
    management does not help 2B. 2B peer_same now writes long notes and hits the 4,096 cap in 2-3
    of 4 events; 2B self edits mostly apply but free little.
  - Next: round 5 = same grid with the 0.8B actor (fits beside 27B at util 0.17).
- Round 5: frozen edit-v7 with the **0.8B actor** (2026-10-21, job 30069161, seeds 42-45,
  K4-C24 + K8-C48, 66.1 min, 32 episodes completed).

  | level | none | self | peer_same | peer_27b (s42 / 43 / 44 / 45) | 2B peer_27b (round 4) |
  |---|---|---|---|---|---|
  | K4-C24 | 0.33 | 0.33 | 0.33 | 0.79 / 1.00 / 1.00 / 0.79 (0.90) | 1.00 ×4 |
  | K8-C48 | 0.17 | 0.17 | 0.17 | 0.25 / 0.31 / 0.33 / 0.23 (0.28) | identical values |

  - 0.8B self / peer_same: almost no valid edit replies ("no JSON object", "missing 'edits' list"),
    so they equal none.
  - peer_27b with 0.8B is nearly identical to 2B; on K8-C48 the four rewards match 2B exactly. On
    Needle under peer management the actor only replies READY and the manager decides what survives,
    so **Needle does not test actor capability in peer conditions**; actor size barely matters.
    KV (the actor must answer GETs from what was kept) is the task where actor and manager interact.
  - Next: round 6 = KV with v7 (2B actor, none vs peer_27b, R1000 + R2000, seeds 42-45). Under v4,
    27B deleted all SET batches without noting values (0.07).
- Round 6 first attempt (job 30069167) cancelled after 9.7 h: in KV R1000 s42 under peer_27b the 2B
  actor answered GETs but never wrote READY_FOR_NEXT_OP; the manager kept the context small, so the
  budget wall never ended it (53,920 turns, 3.4 GB of snapshots in `runs/peer27b-30069167`, left
  for May to delete). Fix `may/track2-baseline` 1cc9124: `max_turns` = 2 x n_ops + 10 (`max_steps`).
- Round 6: KV with frozen v7 (2026-10-21, job 30070167, 2B, none vs peer_27b, R1000 + R2000, seeds
  42-45, 25.5 min). none 0.00 on all 8 (budget wall, turn 6). peer_27b 0.00 except R1000 s44 (0.29);
  R1000 s42 hit `max_steps` (78 turns); one R2000 edit request was rejected by the server.
  - 27B never notes a value: every event is `delete 2-9` (all SET batches, 20K -> 0.7K o200k), so
    GETs find nothing. v7's "note what the task needs" did not change this on KV.
  - Structural limit: the KV instruction says no files/tools, everything must stay in context, and
    any key may be queried. R1000 values alone are ~35K o200k (> 30,720 usable), R2000 ~70K, so
    full retention is impossible; a manager can at best keep a share of the SET lines. 27B keeps none.
  - Next: round 7 = KV R520 (environment pressure 0.85: management is triggered by the actor's
    replies, but all values fit), 2B, none vs peer_27b, seeds 42-45.
- Round 7: KV R520 (2026-10-21, job 30070468, 7.3 min). none 0.00 on all 4 (3 finish the stream
  with every value still in context, peak ~28.6K; s45 hits the wall). peer_27b 0.00 / 0.00 / 0.00 /
  0.25; one event per episode, again `delete 2-9` with no note.
  - The actor is the floor: 2B's GET answers are malformed and invented
    (`<<<ANSWER K00009=aa978ddb>>>` instead of the 24-word value + `#hash`), even with all values in
    context. 2B can answer KV R130 (1.00) but not R520. KV cannot measure management with a 2B actor.
  - Larger actors cannot share the A40 with the vLLM 27B-FP8 manager, so peer_27b with 4B/9B needs
    the llama.cpp stack (Ruthvik, #2) or a second GPU. What vLLM can still do on one A40: 4B/9B with
    none / self / peer_same.
  - Next: round 8 = 4B and 9B (bf16, alone on an A40, `max_len` 81,920), none / self / peer_same,
    KV R520 + Needle K4-C24, seeds 42-45 (job 30070484, `baseline_smoke.sbatch` dd57739).
  Result (1 h 49 min, 48 episodes; official / lenient KV per `scripts/lenient_kv.py`, which ignores
  the `#hash` suffix, punctuation and case; means over s42-45):

  | | KV R520 none | KV self | KV peer_same | Needle K4-C24 none | Needle self | Needle peer_same |
  |---|---|---|---|---|---|---|
  | 4B | 0.25 / **1.00** | 0.06 / 0.14 | 0.15 / 0.27 | 0.33 | 0.33 | 0.31 |
  | 9B | 0.00 / **0.97** | 0.14 / 0.27 | 0.03 / 0.03 | 0.31 | 0.17 | 0.23 |

  - Without management 4B and 9B recall nearly every KV R520 value; the official reward is ~0 only
    because they drop the `#hash` suffix of the value (issue #3 format confound, now quantified).
  - Same-size management **hurts** on KV R520 (lenient 1.00 -> 0.03-0.27): the manager deletes SET
    batches that still hold queried values.
  - Needle K4-C24: 4B/9B self and peer_same finish more of the stream for 9B (submit at turn 24) but
    wipe needles (0.17); no condition beats none.
  - Next: round 9 = 4B/9B, KV R1000 (pressure > 1, management needed), none / self / peer_same.
- Round 9: 4B/9B KV R1000 (2026-10-21, job 30083525, 27 min). Official / lenient means over s42-45:
  4B none 0 / 0, self 0 / 0 (no valid JSON), peer_same 0.06 / 0.25; 9B none 0 / 0, self 0.14 / 0.25,
  peer_same 0.03 / 0.07. none always hits the wall at turn 6, before any GET. The managed 0.17-0.29
  come from the last SET batch surviving the deletes; no manager notes values (same pattern as the
  27B manager's 0.25-0.29). Copying runs with per-file scp stalled (1.7 of 26 MB in 46 min); a
  single `tar czf - | tar xzf -` stream takes 12 s.
- Next: round 10 = **27B-FP8 as actor** (alone on the A40, util 0.90, `max_len` 81,920), none / self
  / peer_same (the core grid's 27B row), Needle K4-C24 + KV R520 + KV R1000, seeds 42-45.
- Round 10 (job 30084447, A40, 27B-FP8 actor, 8.2 h): KV R520 none **1.00** official (27B keeps
  the `#hash`), self 0.23, peer_same 0.14; KV R1000 none 0.00, self 0.25, peer_same 0.24; Needle
  K4-C24 none 0.29, self 0.29, peer_same **0.90**.
- **Move to H200** (2026-10-20): the course account can use the general `gpu*` partitions (class
  caps per Aakash's guide: 6 GPUs on gpu/gpu_rtx6000, 32 on gpu_h200; we use 1-3). Fixes needed:
  `VLLM_USE_DEEP_GEMM=0` (DeepGEMM FP8 JIT needs nvcc; job 30085999), 1 h startup wait (cold 27B
  bf16 compile + CUDA graphs ~17 min; job 30087088), free ports at run time (30088183 and 30088184
  shared a node and port 28184; the 9B job talked to the 27B server, 404s, cancelled by YCRC staff
  after 2.6 h), and a `check_model` guard. 27B now runs in **bf16** (manager 0.55 / actor 0.38 of
  one H200; 27B KV 355K tokens).
- H200 bf16 grid, edit-v7, seeds 42-45, Needle official, KV official / lenient (means, n = 4):

  | actor | level | none | self | peer_same | peer_27b |
  |---|---|---|---|---|---|
  | 2B | Needle K4-C24 | 0.33 | 0.33 | 0.38 | **1.00** |
  | 2B | Needle K8-C48 | 0.15 | 0.17 | 0.15 | **0.33** |
  | 2B | KV R520 | 0.00 / 0.25 | 0.08 / 0.25 | 0.03 / 0.27 | 0.06 / 0.16 |
  | 4B | Needle K4-C24 | 0.33 | 0.33 | 0.31 | **0.81** |
  | 4B | Needle K8-C48 | 0.16 | 0.16 | 0.16 | **0.31** |
  | 4B | KV R520 | 0.25 / **1.00** | 0.01 / 0.08 | 0.15 / 0.27 | 0.15 / 0.27 |
  | 9B | Needle K4-C24 | 0.32 | 0.17 | 0.26 | **0.75** |
  | 9B | Needle K8-C48 | 0.16 | 0.07 | 0.07 | **0.33** |
  | 9B | KV R520 | 0.00 / **0.97** | 0.14 / 0.27 | 0.03 / 0.03 | 0.14 / 0.27 |
  | 27B | Needle K4-C24 | 0.29 | 0.29 | **0.89** | – |
  | 27B | Needle K8-C48 | 0.15 | 0.15 | **0.34** | – |
  | 27B | KV R520 | **1.00** / 1.00 | 0.27 / 0.27 | 0.27 / 0.27 | – |

  Jobs 30088179 (2B), 30088182 (4B), 30114789 (9B, rerun after the port collision), 30088184 (27B
  actor). v8 rerun of 27B self: job 30118842.
  - bf16 reproduces the FP8 picture: a 27B manager lifts small actors on Needle; same-size
    management does nothing for 2B/4B.
  - **On KV R520 every management condition hurts actors that can do the task** (4B lenient 1.00 ->
    0.08-0.27; 27B 1.00 -> 0.27): managers delete SET batches whose values are queried later.
  - 27B actor: peer_same 0.89 vs self 0.29 on Needle K4-C24 looks like a role-separation effect,
    but 27B as `self` wrote a bare edit object (no `{"edits": [...]}`) in 26 of 31 events (16 of 28
    under FP8), so nothing applied; as `peer_same` 0 of 104. The gap is at least partly a
    self-mode format slip. edit-v8 (`may/track2-baseline` 014daa7) reads a bare object as one edit
    (logged `wrapped_bare_edit`); next: rerun 27B self under v8 to separate role from format.
  - v8 rerun of 27B self (job 30118842, H200 bf16, 2 h 19 min): all 86 events now apply (71 via
    the bare-object path), yet Needle K4-C24 0.36 (v7 0.29; peer_same 0.89), K8-C48 0.15 (peer_same
    0.34), KV R520 0.27 (= peer_same). The format fix does not close the gap. Mechanism: as `self`
    27B emits **one edit per reply** (K4-C24: 21 delete-only, 4 note-only events; one note, then
    deletes that drop later needles), as `peer_same` it emits a plan (17 delete+note, 5
    delete+replace, 6 delete-only). So role separation matters for 27B, through how complete each
    management reply is, not through edit validity.
- Round 12: trigger 27,000 instead of 18,432 (jobs 30125285 4B, 30125287 9B; H200 bf16, edit-v8,
  seeds 42-45; `none` unaffected). KV R520 lenient (18,432 in brackets): 4B self 0.44 (0.08),
  peer_same **1.00** (0.27), peer_27b 0.30 (0.27); 9B self 0.28 (0.27), peer_same 0.28 (0.03),
  peer_27b 0.28 (0.27). Needle K4-C24 peer_27b: 4B 0.73 (0.81), 9B 0.66 (0.75); self/peer_same
  unchanged. A later trigger removes the KV harm only for 4B same-size management; the 27B manager
  still deletes value-holding SET batches whenever it runs (~0.28), and Needle gains shrink a little.
  The trigger matters but does not change the main conclusions; keep 18,432 as the default.
- Round 13: seeds 46-49 for 2B/4B/9B and 0.8B on seeds 42-49 (jobs 30127434-37, H200 bf16,
  edit-v8, 27-48 min each). Pooled 8-seed grid with paired tests: `docs/tables/h200_bf16_pooled.md`
  (`scripts/pooled_grid.py`). Summary (diff vs none; all 8/8 same sign, p = 0.008 unless noted):
  - **peer_27b on Needle helps every actor size**: K4-C24 +0.64 (0.8B), +0.67 (2B), +0.46 (4B),
    +0.48 (9B); K8-C48 +0.14, +0.17, +0.11, +0.16.
  - **Same-size management never helps small/mid actors** (0.8B-4B within +-0.03); for 9B it hurts
    (Needle self -0.16 / -0.08, peer_same -0.07 (p 0.13) / -0.08).
  - **On KV R520 every management condition hurts actors that can do the task**: 4B -0.72 to -0.88,
    9B -0.70 to -0.95 (lenient). 0.8B can't do KV at all (0.00); 2B is noisy (0.46, mixed signs).
  - 27B actor: see round 14.
- Round 14: 27B actor seeds 46-49 (job 30136264, H200 bf16, edit-v8, 6.5 h). Pooled n = 8 (`self`
  from edit-v8 runs 30118842 + 30136264 only): Needle K4-C24 none 0.29, self 0.38 (+0.08, 5/1,
  p = 0.22), peer_same 0.85 (+0.56, 8/0, p = 0.008); K8-C48 none 0.15, self 0.14 (+0.00), peer_same
  0.34 (+0.19, 8/0); KV R520 none 1.00, self 0.28 (-0.72), peer_same 0.23 (-0.77), both 0/8.
  **Role effect at 27B, paired peer_same - self: +0.48 (K4-C24) and +0.20 (K8-C48), 8/8 seeds,
  p = 0.008**, with edit validity equal (v8). Same weights, same edit protocol; only who issues the
  edit request differs.
- Round 15: Needle pressure curve, none vs peer_27b (H200 bf16, edit-v8, seeds 42-45; jobs
  30171619 2B, 30174977 0.8B, 30174978 9B; K4-C24 / K8-C48 from the 8-seed grid). Means:

  | level (env. pressure) | 0.8B none / 27B | 2B none / 27B | 9B none / 27B |
  |---|---|---|---|
  | K4-C4 (0.5) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
  | K4-C8 (0.98) | 0.78 / 0.94 | 1.00 / 1.00 | 1.00 / 1.00 |
  | K4-C24 (2.9) | 0.33 / 0.97 | 0.33 / 1.00 | 0.32 / 0.80 |
  | K8-C48 (6.0) | 0.17 / 0.30 | 0.15 / 0.31 | 0.16 / 0.32 |
  | K8-C60 (7.5) | 0.13 / 0.21 | 0.12 / 0.27 | 0.12 / 0.25 |
  | K4-C96 (11.7) | 0.05 / 0.25 | 0.08 / 0.35 | 0.08 / 0.26 |

  Management never hurts below the budget and the 27B manager stays above none at every pressure;
  needle density matters (K8 levels below K4-C96). Figures: `docs/figures/needle_pressure_curve.png`
  (`scripts/plot_pressure.py`), `docs/figures/grid_diff_vs_none.png` (`scripts/plot_grid.py`).
- Round 16: KV R130 / R260, 4B and 9B, none vs peer_27b (jobs 30179678/9, H200 bf16, seeds 42-45,
  6 min each). Pressure 0.24 / 0.44: the trigger is never reached (0 events), so peer_27b = none
  (lenient 4B 1.00 / 1.00, 9B 1.00 / 0.99). KV curve: management changes nothing until it fires,
  and as soon as it fires (R520, R1000) lenient recall drops to ~0.25-0.28 for every manager.
- Round 17: 27B actor at higher Needle pressure, none / self / peer_same (job 30180821, H200 bf16,
  edit-v8, seeds 42-45, 13.5 h). K8-C60: 0.12 / 0.10 / 0.28; K4-C96: 0.07 / 0.13 / 0.30.
  peer_same - self +0.13..+0.23 and +0.15..+0.18 (4/4 seeds each); peer_same - none +0.12..+0.24
  (4/4). `none` hits the budget wall at the same point on every seed (identical scores). Managed
  episodes often end at max_steps (K4-C96: self 1/4, peer_same 2/4; K8-C60: 1/4 each), so their
  scores are a lower bound. The role-separation gain seen at K4-C24 / K8-C48 holds at higher pressure.
- Round 18: decoding ablation, Qwen3.5 recommended non-thinking sampling (temperature 0.7, top_p 0.8,
  top_k 20, presence_penalty 1.5, request seed = instance seed) instead of greedy, for actor and
  manager. 4B and 9B, Needle K4-C24, none vs peer_27b, seeds 42-49 (jobs 30263988/9, ~15 min each).
  peer_27b - none: 4B greedy +0.46 -> sampled +0.44 [+0.35, +0.52]; 9B +0.48 -> +0.33 [+0.23, +0.43];
  8/8 seeds positive in both (p=0.008). `none` unchanged (0.33, budget wall); peer_27b means
  4B 0.80 -> 0.77, 9B 0.80 -> 0.67 (paired sampled - greedy n.s.). The effect does not depend on greedy.
  Jobs 30260229/30 failed (submit args shifted); 30260909/10 ran greedy because the sbatch defined
  SAMPLING without passing it (fixed in e9a3b8d). That accidental rerun reproduced the greedy grid
  (4B +0.47, 9B +0.48), so greedy vLLM runs are effectively deterministic.
- Round 19: full core grid under the same sampling, 4B and 9B, Needle K4-C24 + KV R520 (lenient),
  seeds 42-49 (jobs 30265858/9, 28 / 37 min). Diff vs none, greedy -> sampled:
  Needle 4B self 0.00 -> 0.00, peer_same -0.01 -> 0.00, peer_27b +0.46 -> +0.44;
  Needle 9B self -0.16 -> -0.17 (0/8), peer_same -0.07 -> -0.17 (0/7, p=0.016), peer_27b +0.48 -> +0.33;
  KV 4B self -0.88 -> -0.89, peer_same -0.72 -> -0.76, peer_27b -0.72 -> -0.74;
  KV 9B self -0.70 -> -0.70, peer_same -0.95 -> -0.95, peer_27b -0.70 -> -0.79 (all 0/8).
  Every grid conclusion holds under sampling. Seeded sampling is reproducible: the Needle 4B
  none / peer_27b cells repeat round 18 exactly.
- Round 20: same sampled grid for 0.8B and 2B (jobs 30271270/1, 40 min). Diff vs none, greedy -> sampled:
  Needle 2B self 0.00 -> +0.01, peer_same +0.03 -> +0.01, peer_27b +0.67 -> +0.43 (8/8);
  Needle 0.8B self / peer_same 0.00 -> 0.00, peer_27b +0.64 -> +0.23 [+0.05, +0.42] (5/1, p=0.22);
  KV 2B self -0.01 -> -0.12, peer_same -0.06 -> -0.12, peer_27b -0.33 -> -0.42 (none 0.46 -> 0.49,
  bimodal per seed); KV 0.8B none 0.00 -> 0.08 (one seed 0.62), all managed 0.00-0.08.
  0.8B degenerates under sampling: 130 of 600 sampled actor calls on Needle hit max_tokens
  (finish_reason length), 0 under greedy; 2B/4B/9B have 0 in both. Needle 0.8B `none` falls to 0.04 on
  3/8 seeds, and peer_27b episodes run into max_steps. With sampling the peer_27b gain stays
  significant for 2B/4B/9B (+0.43 / +0.44 / +0.33) but not for 0.8B.
- Round 21: 27B actor under the same sampling, Needle K4-C24, none / self / peer_same, seeds 42-49
  (job 30335778, 4.6 h, edit-v8; baseline_smoke.sbatch takes SAMPLING since 62e0b38). Greedy reference =
  pooled runs 30088184 / 30118842 / 30136264 (self all v8; peer_same s42-45 v7, s46-49 v8).
  Means greedy -> sampled: none 0.29 -> 0.29, self 0.38 -> 0.60, peer_same 0.85 -> 0.67.
  peer_same - self: greedy +0.48 [+0.41, +0.55] (8/0) -> sampled +0.06 [-0.13, +0.26] (4/4, p=1.0).
  self - none: +0.08 (5/1) -> +0.31 (7/0, p=0.016); peer_same - none: +0.56 -> +0.37 (8/0 both).
  Paired sampled - greedy: self +0.23 (6/2), peer_same -0.19 (1/6). Self edit events applied:
  greedy v8 49/49 (runs 30118842 + 30136264), sampled 54/57, so the self gain is not fewer failed
  edits. The 27B role-separation gain at K4-C24 does not hold under recommended
  sampling; under sampling both managed conditions beat none.
- Round 22: same as round 21 at Needle K8-C48 (job 30691916, edit-v8, seeds 42-49). Means greedy ->
  sampled: none 0.15 -> 0.15, self 0.14 -> 0.18, peer_same 0.34 -> 0.30.
  peer_same - self: greedy +0.20 [+0.15, +0.23] (8/0) -> sampled +0.12 [+0.02, +0.21] (6/1, p=0.125).
  peer_same - none: +0.19 (8/0) -> +0.15 [+0.07, +0.24] (6/0, p=0.031); self - none: 0.00 -> +0.03 (5/2).
  Sampled self ends at max_steps on 8/8 seeds and peer_same on 7/8 (scores are lower bounds).
  At K8-C48 the role-separation gain shrinks under sampling but keeps its sign (CI above 0);
  at K4-C24 (round 21) it vanished.
- Round 23: round 22's self / peer_same rerun with max_turns_per_op 4 (cap 202 instead of 106; job
  30791880, 1bffcd8, ~17 h). Means self 0.18 -> 0.20, peer_same 0.30 -> 0.31 (paired +0.02 each,
  n.s.). Still at the cap: self 3/8, peer_same 4/8. peer_same - self +0.11 [-0.00, +0.23] (5/2),
  peer_same - none +0.17 (6/0, p=0.031), self - none +0.05 (6/1). The 106-turn cap did not
  materially lower round 22's scores; seeded sampling reproduced peer_same s44-49 exactly.
- Round 24: archive-with-recall on KV (46756f1; plan in `reports/Context manager experiment plan.md`,
  experiment 1). Deleted/replaced blocks move to an archive, leaving `[ARCHIVED block: ...]`; the
  actor may write `RECALL <key>` and the archived lines with that key re-enter the context. Arms:
  none_r (archive instructions, nothing archived), self_arc, peer_same_arc, peer_27b_arc, mask_arc
  (rule: archive oldest blocks to half the trigger). Smoke 30937295 (9B R520 s42), grid
  30939991/4/7 (2B/4B/9B, R520 + R1000, 9 conditions incl. the delete arms, seeds 42-49, greedy,
  45-75 min each). Lenient KV means:
  | actor | level | none | none_r | delete arms (self / peer_same / peer_27b) | arc arms (self / peer_same / peer_27b / mask) |
  |---|---|---|---|---|---|
  | 9B | R520 | 0.98 | 0.98 | 0.28 / 0.03 / 0.28 | 0.91 / 0.94 / 0.91 / 0.99 |
  | 9B | R1000 | 0.00 | 0.00 | 0.23 / 0.08 / 0.23 | 1.00 / 1.00 / 0.97 / 0.99 |
  | 4B | R520 | 1.00 | 1.00 | 0.09 / 0.28 / 0.28 | 0.48 / 0.54 / 0.51 / 0.45 |
  | 4B | R1000 | 0.00 | 0.00 | 0.05 / 0.23 / 0.22 | 0.11 / 0.17 / 0.09 / 0.06 |
  | 2B | R520 | 0.46 | 0.00 | 0.45 / 0.40 / 0.12 | 0.00 for all |
  - 9B: archiving removes the KV harm (arc - none at R520 -0.08..+0.01, n.s.) and at R1000 turns
    management from 0.08-0.23 into ~1.0 (arc - none +0.97..+1.00, 8/0; arc - delete +0.74..+0.92,
    8/0). ~23 RECALLs per episode. Manager identity no longer matters: mask_arc (no LLM) equals the
    LLM arcs (peer_27b_arc - mask_arc -0.02 at R1000, -0.08 at R520, n.s.). Official (hash-exact)
    scores also rise (9B R1000 0.72-1.00) because recalled lines carry the `#hash`.
  - 4B: partial at R520 (arc 0.45-0.54 vs delete 0.09-0.28, ~7-10 RECALLs per episode of 24 GETs);
    at R1000 it never RECALLs and answers with an invented hash, so arc <= delete. The bottleneck
    is the actor's retrieval initiative.
  - 2B: the added archive paragraph breaks its ANSWER format (`<<<ANSWER K00310>>>>`, no `key=`),
    so every archive-instruction arm scores 0 while values are often right; 2B is not interpretable.
  - 9B `peer_27b` (delete) equals `self` again on every seed (identical edits), as in the greedy grid.
- Round 25: pre-registered validation of the decision rule. Design review
  (`reports/Validation experiment design check.md`), two adversarial reviews, freeze with tags
  `prereg-v1` (track3) and `prereg-v1-harness` (track2, 33b3997). Canaries 30980396/7: two of three
  reproduced; 9B peer_27b s46 gave 0.67 vs 0.83 (identical prompt, different greedy reply at turn 6:
  server-state non-determinism; deviation D1). Blocks J1-J13 = jobs 30981378-30981390 (13 H200 jobs,
  12 done in < 1 h each, 27B J13 ~8 h). Frozen analysis: 7/10 confirmed (P1, P2, P3, P6, P7, P8,
  P10), 3 inconclusive (P4, P5, P9), 0 falsified; R1-R3 not falsified. Details:
  `docs/preregistration-results.md`.
- Round 26: manager-size ladder on Needle (harness 527811a: `peer_4b` / `peer_9b`, any `peer_<size>`
  uses the MANAGER server; CLI refuses a size mismatch). Jobs 31069247/8/9 (0.8B/2B/4B actor, 9B
  manager; fresh `none` and `peer_same` in the same job) and 31069250/1 (0.8B/2B actor, 4B manager);
  K4-C24 + K8-C48, seeds 42-49, greedy, 7-30 min each. `peer_27b` from the pooled grid.
  K4-C24 means (none 0.33): `peer_4b` 0.24 / 0.34 (0.8B / 2B), `peer_9b` 0.16 / 0.16 / 0.16
  (0.8B / 2B / 4B; -0.17..-0.18, 0/7-0/8), `peer_27b` 0.97 / 1.00 / 0.80. K8-C48: `peer_9b`
  -0.09..-0.11 (0/8 each), `peer_4b` 0.00 / +0.03, `peer_27b` +0.11..+0.17.
  Mechanism: the 9B manager's edits apply (4-10 events, nearly all applied) but it deletes every
  block from 2 to the end without noting needles first (e.g. 2B s46: 29,267 -> 578 tokens, 9 needle
  lines left, reward 0.04); the 27B manager notes the needles before each delete (97 lines kept,
  1.00). So 9B's harm is edit judgment / instruction adherence, not the actor role (as a manager it
  cannot submit) and not the JSON protocol. Zero-shot editing judgment appears only between 9B and
  27B in this family.
- Round 27: prompt variants and a live rule (harness bc94295 `--edit-variant`, logged as
  `edit-v8+<variant>`; 9b37541 offline rule simulation). Jobs 31255823/4 (2B actor, 9B manager,
  example / mustnote), 31255826/7 (4B actor, same), 31255829/30 (2B actor, 27B manager, example /
  mustnote), 31255831 (2B actor, `rule_needle` + `none`); Needle K4-C24 + K8-C48, seeds 42-49,
  greedy, 11-49 min each. Diff vs `none` (K4-C24 / K8-C48):
  9B +example 2B +0.03 (4/3) / +0.07 (7/1), 4B -0.04 / -0.02; 9B +mustnote 2B -0.20 (0/8) / -0.05,
  4B -0.17 (0/8) / -0.10 (0/8); 27B +example +0.04 / +0.02; 27B +mustnote +0.62 (8/0) / +0.26 (8/0).
  `rule_needle` live: 1.000 on all 16 episodes. Applied notes: 9B example 31/208 events, 9B mustnote
  4/231, 27B example 16/106, 27B mustnote 59/126. 27B under `example` notes at the first event and
  then only deletes (the v6 failure). No prompt makes 9B help; the example removes its harm. The
  27B manager is as prompt-sensitive (example collapses it). `results.md` finding 7 updated.
- Round 28: paraphrases and manager thinking (harness 0eec53e: `--edit-variant para1..para4`, four
  full rewrites of v8 with the same content and syntax, no example; `--manager-thinking` turns on
  thinking for the manager only, parses the edits after `</think>`, logged as `edit-v8+think`).
  Needle K4-C24 + K8-C48, seeds 42-49, greedy, `none` in each job. Jobs 31283581/3/5/8 (9B manager,
  para1-4, 2B actor), 31283589/90/93/96 (27B manager, para1-4), 31262552/3 (9B thinking, 2B/4B actor,
  16,384 manager tokens; 31262341/2 failed on a LEVELS format slip), 31283597/8 (27B thinking, 2B/4B).
  Diff vs `none` (K4-C24 / K8-C48), 2B actor: 9B para1 0.00 / -0.02 (all 24 K4 replies runaway delete
  lists capped at 4,096), para2 -0.10 / -0.10 (0/8), para3 -0.16 / -0.07 (0/8), para4 -0.17 / -0.08
  (0/8), no note in 445 events; 27B para1 +0.35 / +0.20, para2 +0.61 / +0.39, para3 +0.67 (1.00 every
  seed) / +0.21, para4 +0.17 / +0.05, all 8/0. Thinking means (2B K4 / 2B K8 / 4B K4 / 4B K8): none
  0.33 / 0.15 / 0.33 / 0.16; 9B 0.16 / 0.06 / 0.16 / 0.06 -> thinking 0.66 / 0.34 / 0.43 / 0.23; 27B
  1.00 / 0.31 / 0.80 / 0.27 -> thinking 0.93 / 0.63 / 0.83 / 0.82. 9B thinking - 9B: +0.51 / +0.28
  (8/0) / +0.27 / +0.17 (7/1); 27B thinking - 9B thinking: +0.27 (7/1) / +0.29 (6/2) / +0.40 (8/0) /
  +0.59 (8/0). Thinking: 5-8K output tokens per manager call, 8-42% capped at 16,384 (no edit), 27B
  jobs 4.2 / 6.2 h, 5 of 32 27B episodes at max_steps. 9B with thinking rarely notes (3/234 events;
  replaces chunks with their needles or deletes selectively); 27B with thinking notes often
  (4B K8: 81/128). Conclusion: size and reasoning both matter; the 9B/27B gap holds across 7 wordings
  and with thinking on both. `results.md` finding 7 updated.
- Cost analysis (2026-10-29): `src/ctxmgr/accounting/flops.py` (CLM prefix-reuse FLOPs per call,
  constants from each Qwen3.5 `config.json`; the 27B reproduces CLM's 48.70 GFLOP/token and
  3.93e5 per pair, tested in `tests/test_flops.py`) and `scripts/cost_table.py` ->
  `docs/tables/cost_needle.md` (76 Needle cells, runs with measured `cached_prompt_tokens`, rounds
  25-28). Peer managers get ~0% cache hits (each event follows the last edits; the request starts
  with the context size), the actor ~73%, self-managers 92-96%. Manager FLOPs ≈ C_token ratio x actor
  FLOPs (27B v8: 17x for 2B, 6.4x for 4B, 3.7x for 9B). K4-C24: 2B + 27B 0.99 at 7.5 PF / 90 GPU-s vs
  27B peer_same 0.77 at 28.5 PF / 1,312 s vs rule 1.00 at 0.56 PF / 10 s. Thinking: 1.3-1.5x FLOPs,
  4-5x GPU time. `results.md` finding 8.
- Next-task review (2026-10-30): two rounds of hostile reviews (task validity with offline MuSiQue /
  HotpotQA retention simulations, research strategy, revised-plan methods and claims). HotpotQA is
  rule-solvable (best cheap rule keeps all gold for 0.93-1.00 of questions); MuSiQue leaves headroom
  only at K >= 4 questions per episode; a live MuSiQue port does not fit the harness in a week.
  Plan: write-up first; 27B + rule and 9B-manager-sampled must-runs; second manager family (Gemma 4)
  with an identical-input replay probe; optional offline MuSiQue selector probe.
- Week-1 analysis (2026-10-30), no GPU:
  - `scripts/mechanism_table.py` -> `docs/tables/mechanism_needle.md`: the share of applied manager
    events that note or replace vs gain over `none`, Spearman 0.67 (65 cells, K4-C24) / 0.62
    (54, K8-C48). Figure `docs/figures/needle_protective_vs_gain.png` (`scripts/plot_mechanism.py`).
    The 4B self-manager and 0.8B `peer_same` never delete (wall). `results.md` finding 9.
  - `scripts/paired_table.py` -> `docs/tables/paired_needle.md`: paired diff vs `none` per cell with
    seed-level bootstrap CIs. Greedy `none` differs between jobs for 2 of 196 seed groups (9B K4-C24
    s43/s46, 0.29 vs 0.33).
  - `scripts/cost_law.py` -> `docs/tables/cost_law_needle.md`: peer-manager reads vs T / (T - R),
    median measured/predicted 0.94 (IQR 0.74-0.98, 55 cells); 27B manager 0.94-0.98, reads each
    streamed token 1.04-1.20 times. Finding 8 bullet.
- Round 29 (submitted 2026-10-30 17:07, after `docs/preregistration-v2.md` was pushed in 3b7c719):
  harness `may/track2-baseline` 0eec53e; Needle K4-C24 + K8-C48, seeds 42-49; settings as
  `slurm/prereg_submit.sh` (`MAX_NUM_SEQS=64 MAX_TURNS_PER_OP=2`, trigger 18,432, max_len 81,920,
  `PROMPT_TOKENS_DETAILS=1`, `gpu_h200`).
  - 31451066 `r29-27b-rule`: `baseline_smoke.sbatch`, Qwen3.5-27B, `none rule_needle`, greedy,
    util 0.90 (P11, P12).
  - 31451067 `r29-9b-sampled`: `peer27b_pilot.sbatch`, 2B actor + 9B manager (0.55/0.38),
    `none peer_9b`, `SAMPLING='--temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5'`
    (P13, P14).
  - 31451067 done (20 min). `peer_9b` − `none`, sampled, paired: K4-C24 −0.19 [−0.22, −0.17] (0/8,
    p=0.008), K8-C48 −0.06 [−0.09, −0.03] (1/7, p=0.07) → **P13 confirmed**. Protective share 0.03 /
    0.14 → **P14 inconclusive** (K8-C48 between 0.10 and 0.25). Exploratory: 9B replies cut at the
    output cap 0.38 / 0.08 greedy → 0 / 0 sampled; the extra applied edits are unprotected deletes.
  - 31451066 (27B + rule) partial at 4.2 h: K8-C48 1.00 on 5/5 seeds; K4-C24 0.58 on 5/6 seeds (1.00
    on s45), so P11 is falsified as registered (max attainable K4 mean 0.74). Cause, post hoc: the 27B
    restates every needle received in each reply (+~148 tokens/op at K4); at op 14 the reply is cut at
    the 2,048-token output cap before `READY_FOR_NEXT_OP`, the runner's reminder yields the same cut
    reply, and the episode loops to `max_steps` (14/24 ops delivered = 56/96 needles). The rule lost
    nothing. K8 s42 escaped after one cut reply by switching to id ranges; K4 s45 after 9.
  - 31451066 done (6.3 h, exit 0, a1124u19n01, no vLLM ERROR). 27B + rule: K4-C24 0.64 (7/8 loop,
    s45 1.00), K8-C48 0.79 (s47/s48 loop at 8/48 and 7/48 ops, 0.17/0.15; others 1.00) → **P11
    falsified** at both levels. **P12 falsified**: 14.2 PF (K4, +63% vs 8.7) and 25.5 PF (K8, +6% vs
    24.1). 27B `none` K8-C48 0.15. Termination table now 47/136 managed 27B-actor episodes at
    `max_steps`, 31 cap loops. Cost figure regenerated with the 27B + rule point.
  - Replay probe drafted (not submitted): `scripts/replay_probe.py` + `slurm/replay_probe.sbatch` on
    the harness branch (uncommitted), `docs/preregistration-v3.md` (draft). Single calls rebuilt from
    31451066 snapshots, arms R1-R6, ~320 calls, one H200.
  - Process slip: earlier on 2026-10-30 a few read-only `python3` one-liners ran on the Bouchet login
    node to read logs, against the "no Python on login nodes" rule. Pull run dirs and analyse
    locally instead.
- Round 30 (submitted 2026-10-31 03:4x EDT, after `docs/preregistration-v3.md` was frozen in
  d5342ac): job 31513325 `r30-replay`, `slurm/replay_probe.sbatch` at harness `may/track2-baseline`
  51b6825, `gpu_h200`, one H200, defaults (27B bf16, max_len 81,920, util 0.90, 64 seqs). Replays
  single calls rebuilt from 31451066's snapshots (arms R1-R6, 362 calls); output
  `runs/replay-31513325/replay.jsonl`, summary table at the end of the job log. Q1-Q5.
- Termination audit (2026-10-30, no GPU; `scripts/termination_audit.py` →
  `docs/tables/termination_needle.md`, after a literature scan and two adversarial reviews): 38/120
  managed 27B-actor Needle episodes end at `max_steps`, 22 in a cap loop; 100/120 have a capped actor
  reply. Smaller actors: 0–6 `max_steps` per size (4B under thinking managers: 5 cap loops). The 2B
  restates only the current op's needles (flat ~170 / ~320 tokens per reply). The finding-4 gap at
  K4-C24 greedy is retention (both arms deliver ≥ 0.97 of the stream); at K8-C48 it mixes in the
  loop. Corrected the results.md Limits line that read the +0.02 from doubling the turn cap as "the
  cap is not driving those results"; scoped the outline's one-sentence claim to small actors.

## Dev

- Python 3.12, uv. `uv sync`, `uv run pytest`, `uv run ruff check .`.
- Pilot: `sbatch slurm/setup_env.sbatch [models]` (CPU node) then `sbatch slurm/serve_smoke.sbatch [model ...]` (1 GPU; env MAX_LEN, GPU_UTIL).

## Open decisions

- Precision for 9B; how to run the 9B × fixed-27B-manager cell.
- Where the fixed 27B manager runs: co-located at util 0.75 it has ~63K KV tokens (~1.9 concurrent 32K
  requests), so manager calls would serialize, and only 2B/0.8B actors fit beside it (4B and 9B do not).
  Alternatives: 27B alone on the A40 (128K KV tokens) serving manager calls in a separate phase, or
  batching all peer_27b cells per actor.
- Instances / pressure levels / seeds per cell → total GPU-hours. The 1-GPU limit applies only to
  `education_gpu`; the general `gpu` partition allows a 2× A40 node (27B manager + 4B/9B actor in vLLM)
  and parallel jobs (see Compute).
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
