# AGENTS.md

Operating instructions for coding agents working on this repo for May (Track 3: infra, runs,
analysis). Project context lives in `CLAUDE.md`; this file covers how to work and where things stand.
The repo is public: keep usernames, NetIDs, emails, phone numbers and keys out of every commit.

## Working with May

- Reply in Chinese unless asked otherwise; reports and docs for the team are in English.
- Explain what a step does and why when running remote jobs; May wants to understand the mechanics.
- Confirm before anything outward-facing or hard to undo (new repos, invites, emails to the team,
  deleting cluster data). Routine pushes to May's own branch and job submissions are fine.
- Course accounts may only be used for their own course. This project uses the **cpsc4880** account.
  May also has an account for another course; never use it for this project.

## Cluster access (Yale Bouchet)

Each machine needs its own SSH key and config (see `docs/cluster_setup.md`; never copy private keys
between machines). The agent expects an SSH host alias `bouchet-cpsc4880` with
`ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 4h`.

Login needs Duo, which only May can approve on May's phone. Pattern:

1. `ssh -O check bouchet-cpsc4880` — if a master is running, use `ssh -o BatchMode=yes bouchet-cpsc4880 '<cmd>'`.
2. Otherwise tell May a Duo push is coming, then open the master with `expect`, answering option 1 (push):
   ```bash
   /usr/bin/expect <<'EOF'
   set timeout 120
   spawn ssh -o ControlMaster=yes -o ControlPersist=4h -fN bouchet-cpsc4880
   expect {
     -re "Passcode or option" { send "1\r"; exp_continue }
     -re "Success" { exp_continue }
     -re "denied" { exit 1 }
     timeout { exit 1 }
     eof
   }
   EOF
   ```
3. On timeout, do not resend pushes in a loop; ask May when they are ready (or for a Duo passcode).

Connectivity: off campus, SSH needs the Yale VPN. If `bouchet.ycrc.yale.edu` does not resolve,
Tailscale DNS may be shadowing it. Check with `nc -z -G 8 bouchet.ycrc.yale.edu 22` (macOS) or
`nc -z -w 8 bouchet.ycrc.yale.edu 22` (Linux/WSL).
On Windows, run all SSH steps inside WSL (Ubuntu), never Windows OpenSSH: it lacks `ControlMaster`.
WSL is a separate device with its own key; setup in `docs/cluster_setup.md`.
The master dies when the machine sleeps or changes network; jobs keep running regardless.

On the cluster, the repo clone is at `/nfs/roberts/project/cpsc4880/$USER/ctxmgr`, tracking
`may/track3-infra`. Login nodes are for git, `sbatch`, `squeue`, `sacct` and reading logs only.

## Job workflow

1. Edit locally, commit and push to `may/track3-infra`.
2. On the cluster: `cd .../ctxmgr && git pull && sbatch ...`. Chain dependent jobs with
   `--dependency=afterok:<id>` (e.g. a CPU download job, then a GPU job).
3. `education_gpu` allows one submitted job per user: queue work there sequentially. The `cpsc4880`
   account can also use the general GPU partitions (`gpu`, `gpu_h200`, ...) for multi-GPU or parallel
   jobs; request GPUs with `-N1 --gpus-per-node=<type>:<n>` and submit via `sbatch` (see `CLAUDE.md` Compute).
4. Monitor from a background shell loop that polls `sacct -j <id> -X -n -o State` every 60 s and
   tolerates transient SSH failures (give up after ~5 consecutive failures), then prints the job log,
   `logs/smoke-*.json` and key vLLM log lines.
5. Verify every run on four levels: job state + exit code; GPU model + driver (`nvidia-smi` in log);
   vLLM startup (`Model loading took`, `GPU KV cache size`, no `ERROR`); smoke-test checks and JSON.
6. Record results in `CLAUDE.md` (pilot section) and commit.

Known quirks:
- `du` on the HF cache shows ~10 MB per model; check real sizes with `ls -lL` in the snapshot dir.
- uv warns it cannot hardlink (cache on scratch, venv on project); harmless.
- vLLM startup takes 2–4 min per model (compile + CUDA graphs); avoid many short jobs.

## Status (2026-10-16)

- Done: environment on cluster (vLLM 0.30.0, torch 2.13 cu13); 0.8B/2B/4B/9B downloaded and
  smoke-tested (results in `CLAUDE.md`); English pilot report written for the team (Claude Docs).
  27B-FP8 downloaded and smoke-tested on the A40 (`a40_27b.sbatch` re-run): 27B alone and 27B + 2B
  co-located pass; 27B + 4B does not fit; 9B online FP8 fails on Ampere (details in `CLAUDE.md`).
  First end-to-end no-management baseline (2B, KV R130 + Needle K4-C4) ran from `may/track2-baseline`;
  Needle was cut off by vLLM `max_len` 40,960 (details in `CLAUDE.md`).
- The cluster has a second worktree, `.../ctxmgr-baseline` on `may/track2-baseline`, with its own venv;
  the main clone stays on `may/track3-infra`. Pin a GPU type (`--gpus=a5000:1` etc.) for comparable runs.
- vLLM gotchas for Qwen3.5 (hybrid attention): set `--max-num-seqs` at or below the Mamba state
  blocks the log reports. Online `--quantization fp8` is unusable on the A40 with vLLM 0.30
  (torch.compile crash; with `--enforce-eager`, CUTLASS sm80 kernel error). Offline FP8 checkpoints
  (Marlin kernel) work.
- Next: decide 9B precision (bf16 alone on A40/A5000, or an offline FP8 checkpoint) and where the
  27B manager runs (open decisions in `CLAUDE.md`); 9B on an A5000; concurrent-throughput benchmark per size for the GPU-hour budget;
  a mock end-to-end grid runner (load once, many concurrent instances, per-step logs, compute
  accounting from CLM `flops_metrics/`); agree log format with Tracks 1/2 (draft: `docs/log_schema.md`).
- Open team decisions are listed in `CLAUDE.md`.
