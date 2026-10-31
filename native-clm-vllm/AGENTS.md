# AGENTS.md — ctxmgr

Shared instructions for humans and coding agents working on **Who Should Manage the Context?**, CPSC 4880/5880, Fall 2026. Synthetic tasks and open-weight models only; no clinical data.

## Team and ownership

| Owner | Track | Primary paths |
|---|---|---|
| Ruthvik Edara | 1: tasks, validation, scoring, benchmark literature | `src/ctxmgr/tasks/`, task tests, `docs/tasks-and-evaluation.md` |
| Hancheng Xiang | 2: agent harness, tools, self/peer management, method literature | `src/ctxmgr/harness/` |
| May Mei | 3: serving, Slurm, runs, compute accounting, analysis, manager/scaling literature | `src/ctxmgr/serving/`, `src/ctxmgr/accounting/`, `src/ctxmgr/analysis/`, `slurm/` |

Read `docs/project-status.md` and the applicable interface docs before editing. Work across tracks when necessary, but record interface changes so another person's agent does not implement against an outdated contract. Do not assume a teammate's work is present until you have inspected its branch/files.

## Scientific scope

Research questions: how self-management scales; whether a separate manager helps; whether any gain comes from role separation or manager capability.

Instructor's core grid comes first. For each smaller actor: no management, CLM self, same-size peer, and one **fixed 27B** peer. A 27B actor needs no management, self, and same-size peer only. Proposed dense actor sizes: 0.8B/2B/4B/9B/27B; verify deployed model IDs. Summarization, next-size-up peers, MoE, combine-values tasks and multi-actor amortization are deferred.

Specify the management trigger, what constitutes self vs same-size peer, the context supplied to each manager, edit permissions and output budget before collecting comparison results. Keep task streams and applicable tools/skills comparable; log any deliberate difference. Same-size peer is a separate role/session, not necessarily a second loaded copy of the weights. Record model revision, precision/quantization, generation settings, prompt versions and server versions: a change in precision can confound a scale comparison.

## Tasks and evaluation

ContextBench is released at https://github.com/RulinShao/ContextBench. Use the official dependency pinned at `fb533c18c78a82d3f1b2047a9c1700735cf829ae`; do not rebuild or silently upgrade it. `docs/tasks-and-evaluation.md` defines our adapter and the harness handoff. Budget: 32,768 o200k tokens, with 2,048 reserved. Exact o200k is required; count served-model tokens separately too.

- KV reward reads ANSWER blocks across per-turn snapshots. Official matching normalizes whitespace and tolerates a correct value within a longer answer. Whole-value exact matching is a separately named diagnostic.
- Needle reward reads lines from the final snapshot with whitespace normalization; exact verbatim matching is a separate diagnostic.
- No gold files or future operations may be accessible to the actor/manager. The gold-free full stream is an environment input, not an agent-browsable file. Deliver each operation intact, one at a time, before context management.
- Snapshot actual live messages and tool results before compaction removes answers. Use the official runner if practical; its agent interface is `reset(system, instruction)`, `on_user_turn(text)`, and `context()`.
- Keep oracle validation distinct from model results. A gold oracle reads the answer key and is never a baseline. Keep termination reasons and failures in results; do not drop failed episodes silently.
- Account for both actor and manager compute. Accuracy alone is insufficient for efficiency claims. Missing KV answers do not by themselves prove information destruction if offloaded data still exists.

## Development and coordination

GitHub is the shared code and handoff source. Use a branch per track/change, pull/fetch before integration, and merge reviewed changes through PRs. Do not force-push, overwrite another track's work or discard uncommitted changes. Record material interface changes, validation, blockers and next steps in `docs/project-status.md`. Do not mark unrun tests or planned capabilities complete.

Local setup: `uv sync --locked`. Tests: `uv run pytest -q`. For task changes, also use Ruff checks/formatting on touched Python files and the task audit documented in the handoff. Keep generated weights, caches, streams, logs and runs out of Git. Commit source, configs, small aggregate validation tables and reproduction instructions.

Keep this file portable: no dependence on an individual's personal vault, home directory, credentials, or agent harness. Keep it with the final artifact; revise stale instructions as the project evolves. Record material AI assistance for the required AI-use disclosure; humans review generated code and scientific claims.

## Bouchet

- **All Python runs on allocated compute resources**, including tests and short checks: batch jobs (`sbatch`) or an interactive `salloc` allocation. Never execute Python on login nodes. Verify the execution hostname; allocation alone does not guarantee the current shell is on a compute node.
- Confirm course-account partitions, limits, storage paths and May's tested serving environment before launching jobs. Do not assume research-account examples or general cluster notes match course permissions.
- Save reproducible allocation/serving commands in `slurm/` and setup notes in the repo. Do not commit credentials, tokens or private keys. Avoid competing jobs or duplicate model servers under the same course-account allocation.
- Release interactive steps and allocations when finished; disconnect SSH when no longer needed. For intentionally continuing batch jobs, record job IDs, log paths and status. Do not imply background monitoring continues after a conversation turn ends.
- Put large models/caches/results in approved course storage, not a home-directory cache. Preserve enough run provenance to reproduce every figure.

## Report

Four-page NeurIPS-style report plus reproducibility instructions/checklist, contribution statement and AI-use disclosure. Negative results are acceptable with careful analysis. Literature scopes follow the three tracks. Do not claim general long-horizon reasoning improvements from two synthetic retention/retrieval tasks alone.
