# Native CLM recovery automation

This standalone adapter continues an **already prepared native CLM recovery** on
Bouchet. It waits for recorded jobs, validates saved grades, submits only cells
that have no recorded attempt, and creates an English GitHub Discussion report.
It does not import or replace the current team harness.

The implementation targets the reviewed 144-episode Q8_0 experiment: Qwen3.5
0.8B/2B/4B, six task levels, task seeds 42–45, `none` and native `clm_self`.
It uses the recovery's existing Python environment, frozen source, tools,
model manifests, sampling settings, and resource requests. It is not a general
launcher for new grids or the team's vLLM `self`/`peer_same` conditions.

## What happens automatically

1. A small CPU controller runs in Slurm's `day` partition. Disconnecting SSH does
   not stop that allocation. Its default limit is 24 hours; it consumes one CPU
   while waiting. The separate partition avoids occupying the education GPU
   queue's single-job slot. Account/QOS limits and actual availability still apply.
2. The controller reads both existing job ledgers and waits for active jobs.
   GPU jobs must finish with `COMPLETED` and exit `0:0`.
3. It requests a fresh CPU report using the **frozen recovery helper**. That helper
   revalidates source, task identities, model metadata, selection, and saved grades
   against the original experiment. It does not regrade snapshots.
4. If coverage is incomplete, every recorded recovery cell must have all its
   selected grades before the next unsubmitted cell can start. Recorded failures
   are never retried automatically. Missing/invalid scores never become zeros.
5. Complete coverage produces a public report preview. With explicit `--publish`,
   the controller creates or updates its own marked comment in the configured
   GitHub Discussion. The same report is not posted repeatedly.

The original 88 grades retain precedence. The original suite, source and raw
results are read only. New state and frozen automation code copies live under
`RECOVERY_DIR/automation/`; validated combined reports remain under
`RECOVERY_DIR/report/`. Do not add files to the recovery's `source/` or `tools/`:
its provenance checker intentionally rejects changes there.

## Install in a separate checkout

Run these Bash commands on the Bouchet login node. They run no Python:

```bash
git clone --single-branch --branch hancheng/clm-recovery-automation \
    https://github.com/MaySong-Mei/ctxmgr.git "$HOME/ctxmgr-automation"
cd "$HOME/ctxmgr-automation"
git rev-parse HEAD
```

If the destination already exists, inspect it first; do not overwrite it. Keep the
existing project environment and recovery directory in place. No `uv sync` is
needed for this adapter's standard-library code.

Define the recovery directory in each new shell:

```bash
recovery="/nfs/roberts/project/cpsc4880/$USER/runs/clm-recovery-20261021T203955Z-RLNKpS"
```

Check the scheduler request without creating a job:

```bash
bash "$HOME/ctxmgr-automation/scripts/clm_automation.sh" check "$recovery"
```

Then start the controller with local report preview only:

```bash
bash "$HOME/ctxmgr-automation/scripts/clm_automation.sh" start "$recovery"
```

The command prints a controller job ID and log path. It submits held, records the
job, then releases it. A repeated start returns the existing active controller
instead of submitting another one. Each start freezes its automation code before
submission, so a later git update does not change an in-flight controller.

Inspect progress:

```bash
bash "$HOME/ctxmgr-automation/scripts/clm_automation.sh" status "$recovery"
squeue -u "$USER"
```

Read the log path printed at submission. `status` includes the saved recovery
report; its coverage can lag an active GPU job. Only a fresh successful report
establishes current validated coverage. A job being graded does not mean it
finished the full task stream: inspect `budget_wall`, `stream_exhausted`, and
`submit` separately.

Stop orchestration gracefully:

```bash
bash "$HOME/ctxmgr-automation/scripts/clm_automation.sh" stop "$recovery"
```

The stop request does not cancel experiment or report jobs. After the controller
ends, `start` resumes from the ledgers. If the controller times out or its node
fails, existing experiment jobs remain recorded and the next controller checks
them before taking further action. Do not run manual `recover.sh next/report`
commands concurrently with automation; use one submission owner.

## GitHub publication and email notifications

Publication uses the [GitHub Discussions GraphQL API](https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions)
through [GitHub CLI](https://cli.github.com/manual/gh_api). It needs `gh` on the
compute node, outbound HTTPS, and an authenticated GitHub identity allowed to
write comments in the target repository. Check `gh --version` and
`gh auth status --hostname github.com` on the cluster; complete `gh auth login`
yourself if required. Never put a token into a committed file, command example,
report, or chat. The adapter does not provision credentials.

After reviewing a generated preview, start a controller with publication enabled:

```bash
bash "$HOME/ctxmgr-automation/scripts/clm_automation.sh" start "$recovery" \
    --publish --repository MaySong-Mei/ctxmgr --discussion 6
```

If another controller is still active, this command leaves its configuration
unchanged. Wait for it to end, or request a graceful stop and then start again.
A completed recovery generates a fresh validation report and publishes without
submitting another GPU job.

The adapter checks all Discussion comment pages for its marker and only updates
a comment owned by the currently authenticated identity. A lost write response
stops publication; a subsequent invocation reads GitHub again before deciding
whether a write is needed. No CI, p-value, new experimental round number, or
teammate mention is invented.

GitHub delivers notifications according to each recipient's subscription and
[notification settings](https://docs.github.com/en/subscriptions-and-notifications/get-started/configuring-notifications).
This is the mechanism behind GitHub Discussion notification emails. Updating an
existing comment may not send a new email. This adapter does not send SMTP email,
choose recipients, or guarantee an email delivery. Use one publisher and receipt
per experiment: GitHub has no unique-key constraint for comment creation, so two
independent publishers on different hosts can race despite the local lock.

Only allowlisted scientific fields are rendered. Raw error messages, local paths,
environment variables, authentication data, and full `results.json` files are
excluded. Publication normally requires complete coverage. The standalone
publisher's `--allow-partial` is an explicit diagnostic option; partial reports
stay clearly labeled and the controller never uses this option.

## Recovery from ambiguous submissions

If `sbatch` fails or returns an unrecognizable response, a journal remains in
`automation/launch-intent.txt` or `automation/submission-intent.json`. This is an
intentional stop: the scheduler might have accepted a job before the response was
lost. Do not remove the journal and resubmit without checking `squeue`, `sacct`,
the frozen ledgers, and the corresponding logs.

For a held job already recorded in a ledger, verify its ID and release that same
job with `scontrol release JOB_ID`. If a job was accepted but not recorded, record
the correct job in the appropriate ledger only after matching its command,
recovery path and selected cell. If the scheduler confirms no job was accepted,
archive the intent journal and restart. Never invent a replacement job ID.
The controller can reconcile its submission journal when the ledger proves the
intended job was recorded. Failures and unknown accounting stop new work.

An incomplete report's exit `1:0` is distinct from a validator error or failed
GPU job. `budget_wall` with a valid grade is an experimental outcome, not a reason
to rerun. Review a failed GPU attempt manually; this adapter never chooses the
best score among retries or erases a recorded attempt.

## Interfaces and testing

The login interface is `scripts/clm_automation.sh check|start|status|stop`.
`tools/clm_automation/controller.py` runs only on an allocated compute node and
calls the existing recovery helper. `github_report.py` exposes report rendering
and GitHub publication as separate functions and CLI subcommands, so a report
can also be rendered or published on a local workstation.

Local development tests require Python 3.12+, Bash, `flock`, and GNU coreutils:

```bash
python -m unittest discover -s tests/clm_automation -p 'test_*.py' -v
bash -n scripts/clm_automation.sh tools/clm_automation/worker.sbatch
```

On Bouchet, run all Python (including tests and one-line checks) on allocated
compute resources. Mocks exercise scheduler failure/restart behavior and GitHub
API handling; they do not establish actual cluster scheduling, GPU execution,
GitHub credentials, or email delivery. The uploaded snapshot lacks raw episode
directories, so fresh provenance validation remains a cluster-side step.

Validation on 2026-10-26: all 49 automation tests passed, Bash syntax checks
passed, and the actual uploaded frozen plan and 132/144 report passed adapter
checks. The final recorded GPU job was recognized without resubmission. Ruff was
not installed in the local review environment; no Ruff result is claimed.

The native CLM report is a separate experiment from the team's JSON-edit `self`
and `peer_same` experiments. Its model precision, prompts, tool protocol and
sampling differ. The published comparison must retain those distinctions and
must not pool the two sets of rewards. See [the intake snapshot](clm-recovery-snapshot.md).

AI assistance was used for implementation, tests and documentation. Human review
of the code and scientific interpretation remains part of the project handoff.
