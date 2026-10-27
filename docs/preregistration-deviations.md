# Deviations from the pre-registration (`prereg-v1`)

## D1: canary C1 did not reproduce one greedy result (recorded before any block was submitted)

- **Observed.** Canary jobs 30980396 (C1) and 30980397 (C2), harness `33b3997`, seed 46:
  - 9B `mask_arc` KV R1000: lenient 1.0000, reference 1.0000 (job 30939997). Reproduced.
  - 27B `none` Needle K4-C24: official 0.2917, reference 0.2917 (job 30136264). Reproduced.
  - 9B `peer_27b` Needle K4-C24: official **0.6667**, reference 0.8333 (job 30127437). Not reproduced.
  - `cached_prompt_tokens` is logged (non-null on all calls), so `--enable-prompt-tokens-details` works.
- **Cause.** Context snapshots are identical through turn 5, and the turn-6 actor request has the same
  prompt (6,981 served tokens). Under greedy decoding the 9B actor still returned a different reply
  (28 vs 75 completion tokens: a shorter acknowledgement of a filler deletion), after which the
  contexts diverge. All four management events made the same edit requests. The settings recorded in
  `episode.json` match (bf16, max_len 81,920, trigger 18,432, manager max_tokens 4,096, greedy,
  edit-v8; harness 014daa7 vs 33b3997 differ only in code paths not used by `peer_27b`). The
  difference is the server state: in job 30127437 this episode was the fourth on the actor server
  (after `none`, `self`, `peer_same`), in the canary it was the first. vLLM greedy outputs are not
  batch- or cache-state invariant, as noted in `reports/Validation experiment design check.md`.
- **Consequence.** Greedy runs are not bitwise reproducible across jobs; a repeated instance is a
  near-repeat, not an identical draw. This noise is already present in the seed 42–49 data the rule
  came from (e.g. the accidental greedy rerun, jobs 30260909/10, gave 4B +0.47 vs +0.46 and 9B +0.48
  vs +0.48 with per-seed differences). It adds instance-level variance; it does not bias conditions
  against each other, because the two conditions of a pair run the same instance and each episode is
  scored once.
- **Decision.** The cause is identified and is not a harness or configuration fault, so the blocks
  are submitted unchanged, as the canary rule allows. Nothing in the design, predictions or analysis
  changes. The exact-reproduction requirement is downgraded to "settings and harness reproduce; greedy
  outputs may differ by server state", and any confirmatory result is interpreted with this
  non-determinism in mind.
