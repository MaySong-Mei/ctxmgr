# Native CLM: fresh five-model vLLM/BF16 evaluation

This package runs a new, independent experiment. It never resumes, changes, or
merges grades from the earlier llama.cpp/Q8_0 suite or its recovery directory.
No llama.cpp executable, GGUF model, GitHub CLI, token, or email service is required.

## Fixed experiment

| Setting | Value |
| --- | --- |
| Actors | Qwen3.5-0.8B, 2B, 4B, 9B, 27B, official Hugging Face weights |
| Precision / serving | BF16 / vLLM 0.30.0 |
| GPU | One H200 per job; default one job running at a time |
| Conditions | Original team chat `none`; native released CLM `clm_self` |
| KV levels | R130, R1000, R2000 |
| Needle levels | K4-C4, K4-C24, K8-C48 |
| Task seeds | 42, 43, 44, 45 |
| Sampling | Official CLM: thinking on, temperature 0.7, top_p 0.95, max_tokens 8192 |
| Benchmark budget | 32768 raw o200k tokens; reserve 2048; `CLM_TOK_RATIO=0` |
| Served-model context | 81920, separately from the benchmark token ruler |
| Total | **240 episodes**, in **20 GPU array tasks**, 12 episodes per task |

The CLM agent uses its pinned `PAPER_KWARGS` without sampling overrides. The four
sampling values above are also passed to the original chat baseline. The official
CLM skill, tools, native turn limits, task generator, and graders are preserved.
This is the requested two-condition grid, not the course's full peer-manager grid.

`--generation-config vllm` disables model-card sampling defaults; Qwen's recommended
top_k=20/presence_penalty=1.5/top_p=0.8 settings are **not** substituted for CLM's.
No task seed is sent as an inference seed. The serving seed is recorded as 0.

## 1. Clone the dedicated GitHub branch on Bouchet

The code lives under `native-clm-vllm/` on branch
`hancheng/native-clm-vllm-bf16` of `MaySong-Mei/ctxmgr`. On a Bouchet login
node, first check that the destination is unused, then clone the branch:

```bash
[[ ! -e "$HOME/ctxmgr-native-vllm" ]] || { echo 'Directory already exists'; exit 1; }
git clone --single-branch --branch hancheng/native-clm-vllm-bf16 \
  https://github.com/MaySong-Mei/ctxmgr.git "$HOME/ctxmgr-native-vllm"
cd "$HOME/ctxmgr-native-vllm/native-clm-vllm"
sha256sum -c SHA256SUMS
```

For later code updates, use `git -C "$HOME/ctxmgr-native-vllm" pull --ff-only`.
Submission freezes the exact source in a new suite directory, so an update to
the checkout does not silently change an active or completed run. Do not edit
an existing suite's frozen `source/`.

Verify the two prerequisites inherited from the previous CLM experiment:

```bash
export PATH="$HOME/.local/bin:$PATH"
command -v uv
ls -lh /nfs/roberts/project/cpsc4880/cpsc4880_hx259/containers/contextbench.sif
```

The previous pipeline used `uv` and this container. If either is missing,
resolve its location before submission. For an alternate container path or
module, copy `configs/clm-vllm.env` to a resource configuration and change
`CLM_IMAGE` or `CONTAINER_MODULE`. This package reuses the existing task
container; it does not build or overwrite it.

## 2. H200 readiness and first-run checks

`check` below checks Slurm resource requests but does not load models. This
release has not yet run vLLM on the user's H200 allocation. The full array
starts automatically after setup succeeds. Each GPU cell initializes its model
and performs a client/server transport probe before its 12 benchmark episodes;
startup or protocol failures are recorded. Inspect the setup log and early cell
logs after submission. Indices 12–15 exercise 9B and 16–19 exercise 27B. A
successful CPU test or Slurm resource check alone does not establish GPU
correctness.

## 3. Check resources, then submit once

The check validates all three resource requests with Slurm; it submits no work:

```bash
cd "$HOME/ctxmgr-native-vllm/native-clm-vllm"
bash scripts/clm_vllm.sh check
```

If all requests are accepted, submit the full experiment:

```bash
cd "$HOME/ctxmgr-native-vllm/native-clm-vllm"
bash scripts/clm_vllm.sh submit
```

Save the printed **Suite** path. The command creates a new directory similar to:

```text
/nfs/roberts/project/cpsc4880/cpsc4880_hx259/runs/clm-vllm-20261029T...-......
```

The jobs are submitted together:

1. CPU setup: creates locked server/client environments in the new suite, runs
   tests and pinned-source checks, checks the container, downloads/verifies the
   five immutable HF models, and generates 24 shared task instances.
2. GPU array: 20 model/seed tasks wait for successful setup. Each starts one vLLM
   server, probes the actual client/tool API, and runs its 12 episodes. Array `%1`
   limits concurrent GPU tasks to one; no next-task submission is needed.
3. CPU report: waits until the GPU array finishes, including failed tasks, then
   writes the full 240-row report with explicit missing/invalid grades.

Every `submit` call starts a **separate full experiment**. Do not repeat it to check
progress. The launcher records IDs before releasing the held setup job. If a
submission response is uncertain, inspect `jobs.tsv`, `submission/`, and `squeue`
before retrying; do not release a held setup job without checking its dependencies.

Initial environment installation and model downloads can be substantial. The new
BF16 weights are roughly 85–95 GB before cache/environment overhead. Existing
matching HF cache objects are reused. Setup has a 12-hour limit to leave room
for downloads and full weight verification. The 27B server
gets up to one hour for initial compilation/readiness; a long startup log alone is
not evidence of a hung episode. Queue time and complete runtime depend on Bouchet.

Defaults: `cpsc4880` account for `cpsc4880_*` users, `normal` QoS, CPU `day`, GPU
`gpu_h200`, one H200, 8 CPUs, 128 GB host RAM, 24-hour limit per GPU task. Host RAM
is separate from GPU memory. All Python, installation, hashing of weights, tests,
and model work run on allocated compute nodes, never the login node.

Optional parallelism: copy the resource config, change `CLM_PARALLEL=2`, and use
`check --config "$HOME/my-clm-resources.env"` followed by `submit --config ...`.
Keep the H200 GPU type and experiment settings unchanged. Higher concurrency needs
available quota and starts additional servers. Default sequential scheduling is
appropriate for the course account.

## 4. Inspect status and current results

Replace the placeholder with the **actual Suite path printed by submit**:

```bash
suite='/nfs/roberts/project/cpsc4880/cpsc4880_hx259/runs/REPLACE_WITH_PRINTED_SUITE'
bash "$HOME/ctxmgr-native-vllm/native-clm-vllm/scripts/clm_vllm.sh" status "$suite"
squeue -u "$USER"
```

Read the setup and GPU task logs:

```bash
tail -n 60 "$suite"/logs/clm-setup-*.out
ls -lh "$suite/logs"
cat "$suite/cells.tsv"
```

GPU log names are `clm-cell-ARRAYID_INDEX.out`. Indices 0–3 are 0.8B,
4–7 are 2B, 8–11 are 4B, 12–15 are 9B, and 16–19 are 27B; each group
uses seeds 42–45.
For one task, use its exact file name with `tail -n 80`. Its `cells/` directory
also contains `cell-status.json`, `server.log`, `transport-probe.json`, individual
episode logs, snapshots, scores, and model/serving provenance.

To refresh a combined snapshot while the GPU jobs run:

```bash
bash "$HOME/ctxmgr-native-vllm/native-clm-vllm/scripts/clm_vllm.sh" report "$suite"
```

Wait for the printed CPU report job to finish, then read:

```bash
cat "$suite/report/report-status.json"
cat "$suite/report/report.md"
```

`report` never launches GPU work. A partial report exits 1 to make incomplete
coverage visible; it still writes the available results. A report requested
before setup finishes says `preparation_not_ready`. Do not interpret either as
loss of existing grades. Final success is **240/240 validated grades**, with
`status: complete`. Slurm `COMPLETED` alone does not establish benchmark coverage.

## Outputs and failure behavior

| File | Meaning |
| --- | --- |
| `jobs.tsv`, `cells.tsv` | Scheduler IDs and the fixed 20-task mapping |
| `source/`, `source.sha256`, `config.env` | Frozen source and resource configuration |
| `ready.json` | Setup receipt: versions, shared-plan, container and model manifest hashes |
| `models/*/model.json` | Immutable model revision, complete file inventory and hashes |
| `shared/instances/` | New shared task instances; evaluator-owned gold stays outside agent tools |
| `cells/` | All episode artifacts, including errors, zero scores and early terminations |
| `report/results.tsv` | All 240 planned rows, source job IDs and explicit status |
| `report/summary.tsv` | Means and graded/expected denominators by model/task/condition |
| `report/comparison.tsv` | Means/differences using matching valid task-seed pairs only |
| `report/report.md` | Complete English Markdown summary, per-seed and termination tables |
| `report/provenance.json` | Reproduction details; may contain local filesystem paths |
| `report/report-status.json` | Coverage and SHA-256 hashes of the report artifacts |

An episode error is retained while remaining episodes are attempted if the server
is alive. A server/transport startup error stops that task. SIGTERM/SIGINT trigger
cleanup of episode/server process groups; Slurm still controls the allocation.
Incomplete jobs are not silently retried or assigned zero. A task refuses to reuse
its existing output directory; this release is for a fresh full run, not in-place
recovery. Reports preserve any validated grades from failed jobs.

## Scientific interpretation

- Native `clm_self` and original chat `none` have deliberately different prompts,
  tools, reasoning-history behavior and budget-final-turn rules. This comparison
  is between systems, not an editing-only ablation.
- The unchanged baseline keeps content-only history. The unchanged native CLM
  keeps its own reasoning-aware history. The local gateway translates vLLM's
  `reasoning` field to the client's `reasoning_content` spelling, without moving
  reasoning into content or altering generation parameters.
- `budget_wall` and `max_steps` can yield valid grades without completing the task
  stream. Report them separately from `stream_exhausted`/`submit`.
- Shared task structure can correlate seeds. The generated report does not infer
  independent-replicate confidence intervals or significance from four seeds.
- This new BF16 series must remain separate from the completed 144-episode Q8_0
  series and May's JSON self/peer or recommended-sampling rounds.
- No GitHub report publication or email is automatic in this package. Review `report.md`
  before publishing it through the existing GitHub reporting workflow. The prior
  publisher's 144-episode recovery schema is not compatible with this 240-episode
  suite; do not point that publisher at this directory.

## Validation and implementation notes

See `VALIDATION.md` for the checks actually performed before delivery. Local tests
do not establish H200 runtime compatibility or GPU accuracy. The compute setup
repeats tests against the locked native client, and each GPU job runs an actual
transport probe before benchmark execution. A model that declines to emit a tool
call receives a recorded warning, not automatic exclusion; transport failures
stop the task.

See `docs/clm-vllm-interface.md` for the module and CLI contracts, immutable upstream
pins, model revisions, and primary serving references. This package includes
historical project modules for source continuity; the supported new-run entry
point is `scripts/clm_vllm.sh`. Historical scripts/CLIs do not define this experiment.

AI assistance implemented the new orchestration, transport compatibility,
provenance validation, tests and reporting. Humans remain responsible for
experimental interpretation and publication.
