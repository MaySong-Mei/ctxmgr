# Track 1: tasks and evaluation

Owner: Ruthvik. Status: local implementation and oracle validation; model experiments have not run. Updated 2026-10-14.

The official [ContextBench](https://github.com/RulinShao/ContextBench) has been released. The proposal's earlier assumption that it needed rebuilding is superseded. We use its generators, level definitions, instructions, feasibility checks and primary graders, pinned at `fb533c18c78a82d3f1b2047a9c1700735cf829ae` in `pyproject.toml` and `uv.lock`. No upstream source was vendored. ContextBench is Apache-2.0; the separate CLM repo uses CC BY-NC 4.0. Check the relevant license if importing CLM tools later.

## Local quickstart

```bash
uv sync --locked
uv run python -m ctxmgr.tasks generate kv_store --level R130 --seed 42 --out runs/tasks/kv-R130-s42
uv run python -m ctxmgr.tasks validate runs/tasks/kv-R130-s42
uv run python -m ctxmgr.tasks oracle runs/tasks/kv-R130-s42 --run-dir runs/oracle/kv-R130-s42
uv run python -m ctxmgr.tasks score runs/tasks/kv-R130-s42 --run-dir runs/oracle/kv-R130-s42
uv run pytest -q
```

For needles, use `generate needle_retention --level K4-C4`. Every output directory must be new. Generated instances/runs live under ignored `runs/`, not Git. The oracle reads evaluator gold; its 1.0 score is an implementation check and **never an experimental baseline or evidence about Qwen**. On Bouchet, all Python commands, including generation and tests, must run inside a batch job or interactive compute allocation. These examples are for the local machine.

## Task contract for the harness owner

`generate_instance(family, level, seed, out_dir)` creates upstream `chunks.jsonl`, `gold.json`, `meta.json`, a feasibility receipt, our manifest, and `public/stream.json`. `validate_instance(out_dir)` verifies hashes and rechecks feasibility. The manifest records resolved generator knobs, token method, commit, measured pressure, safety checks and SHA-256 hashes. An exact `o200k_base` encoder is mandatory: approximate token or cl100k fallbacks fail explicitly.

The evaluator owns the complete instance directory. Give the actor/manager only the current operation and its live context; do not expose evaluator `gold.json`, future operations, manifests that identify future queried keys, or the full stream through model-accessible filesystem tools. `public/stream.json` is a gold-free **environment input artifact**, not a file the actor should browse. Its instruction and operations contain the original inputs; it must still be delivered one operation at a time. Gold separation here is logical, not an OS sandbox: the harness must provide that isolation.

The preferred integration is the upstream pure-Python runner:

```python
from contextbench import run

result = run(agent, instance_dir, budget=32768, reserve=2048, out_dir=run_dir)
```

The agent implements `reset(system, instruction)`, `on_user_turn(text) -> str`, and `context() -> list[dict]`. It must append each incoming operation intact as a user message before processing it, and return the actual live messages from `context()`. Do not trim/offload an incoming operation before the actor receives it. Management applies after receipt; same-size/larger peers receive only the information permitted by the experimental design. Keep instructions, seed, budget and task stream identical across conditions. Method-specific tool skills are separate from the common task instruction.

Advance with standalone `READY_FOR_NEXT_OP` (chat) or `echo READY_FOR_NEXT_OP` (shell). The runner rejects early submission and implements budget walls. `run()` snapshots after each turn, so compact at the beginning of the next turn, after the previous answer has been snapshotted. A custom tool loop must also snapshot tool results and answers before compaction. Manager calls, tool steps, edits, actor/manager token usage, trigger thresholds, actual served-model token counts and termination reasons belong to Track 2/3 logs; task code does not estimate compute.

`score_run(instance_dir, run_dir)` requires canonical `context_snapshots/turn-*.json` with real messages and exact, consistent token counts. It refuses report-only/trajectory fallback, corrupt snapshots, or mismatched instance hashes. Use the upstream runner's snapshot convention or `contextbench.tokens.record_snapshot`; check that writes succeed. Return values have `official` and `diagnostics` blocks. Retain `result.json` too, particularly `terminated_by`; a budget-wall run must not be reported as a completed episode.

## Scoring decisions: official score first

| Task | Official metric at pinned revision | Project diagnostic |
|---|---|---|
| KV Store | Fraction of gold GETs with the correct value in an ANSWER block in **any** context snapshot. Whitespace is normalized; the expected value may be a substring of a longer answer. Different attempts are collected. | `strict_exact_accuracy`: whole normalized answer value equals gold, across snapshots. Also list keys with conflicting answers. |
| Needle Retention | Fraction of gold needle lines in the **final** context snapshot, normalizing whitespace. Disk files do not count. | `required_info_loss = 1 - official reward`; `strict_verbatim_retention` also checks the exact original line. |

The earlier plan's “all answers must remain in final context” is too strict for official KV grading. Conversely, an input SET record by itself is not a KV answer: a matching ANSWER block is required. Do not silently replace the official metric with our stricter diagnostics. Report both if they diverge. KV missing/wrong answers do not prove irreversible information loss: the correct value may still exist in an offloaded file. A causal KV loss metric would require checking the agent's authorized external store; it is not inferred here.

For both tasks, `observed_peak_tokens` is recomputed from snapshots and `snapshot_budget_exceeded` flags a value above 30,720. It cannot detect unsnapshotted model calls; the harness must enforce and log those, including the served Qwen tokenizer separately from the benchmark's common o200k accounting.

## Official levels and pressure

KV uses 100-record SET batches, 24-word values and 24 GETs. Core levels: R130, R260, R520, R1000, R2000, R4000, R8000, R16000.

Needle core levels: K4-C4, K4-C8, K4-C24, K8-C48, K8-C60, K4-C96, K2-C192, K2-C224. K is needles per chunk; C is chunks. Do not hold K constant across the full grid: official high-pressure levels reduce K to keep total required retention feasible. Each chunk has 140 filler lines and 10 words per line.

The upstream gate counts streamed operations (using its fixed header placeholder) and checks max operation <=5,529 tokens, retained mass <=13,824, and at least four operations. Named sub-budget controls bypass the pressure>1 gate only. The gate's `retained_tokens` is a task-specific feasibility estimate, not a model's measured memory usage.

Our `environment_pressure` counts the common task instruction plus each actually rendered operation, divided by 32,768; it excludes method prompts, system prompts, actor/manager output and tool traffic. `gate_pressure` is the upstream gate's different definition. Log both, and use a clearly labeled axis. Do not relabel levels as exact 0.25x/0.5x/1x: measured values differ. The release includes a highest needle level around 27x, beyond the earlier proposal's approximate 24x; keep the named level and report its measured pressure.

Pilot configuration is in `configs/tasks-pilot.yaml`. Start with one below-budget control and two pressure levels per task, shared seeds 42-45. The instructor's core condition grid remains unchanged. Generate the identical instances once and reuse across actor/manager conditions. Keep additional seeds for a held-out follow-up if prompts are tuned. No summarize, MoE or combine-values condition is added by this work.

## Validation and reproducibility

Run `uv run python -m ctxmgr.tasks.audit --out docs/task-audit.csv` for all 16 main-figure KV/needle levels x four seeds. It rechecks generation feasibility, requires exact tokenization, runs the official gold oracle, grades snapshots and rejects non-perfect oracle or budget-overflow results. Generated streams and snapshots are temporary; only aggregate CSV rows are saved. The CSV is a task-validation artifact, not model results.

Tests cover official levels, seed reproducibility, public/evaluator separation, overwrite refusal, tamper detection, token fallback refusal, missing/corrupt snapshots, oracle scoring, final-only needle retention, process-level KV scoring, and the distinction between permissive official and strict diagnostic KV matching.

## Handoff and remaining work

Hancheng (Track 2) can start from the three-method agent interface above. May (Track 3) can consume manifests, oracle validation and nested score records. Before claiming project results, the team needs a model-connected harness, identical context-editing permissions across appropriate conditions, a no-management implementation, and one end-to-end smallest-model pilot. May's tested serving commands remain outside this checkout. Real run testing and the final report depend on those other tracks. Track 1's benchmark literature brief is in `docs/benchmark-literature.md`.

Shared coordination, first-model acceptance criteria and remaining Track 1 work are in [project status](project-status.md). Initial local implementation does not mean Track 1 is finished: integration checks, failure review and the final report contribution remain.
