# Results tables

`episodes_vllm.csv`: one row per scored episode on the vLLM stack (A40: bf16 actors, Qwen3.5-27B-FP8
manager; H200: bf16 actors and bf16 Qwen3.5-27B manager; see `job`), built from the run directories (`episode.json`, `score.json`,
`calls.jsonl`, `edits.jsonl`). Raw runs live in the course project directory, not in git.

- `edit_version`: edit protocol for managed conditions. `edit-v0-draft` = first draft (c717c61),
  `edit-v1-hardened` = PR #4 before instructions were versioned; later versions are logged by the
  harness (`EDIT_PROMPT_VERSION`). Empty for `none`.
- `official_reward`: ContextBench official score. `required_info_loss` (Needle) and
  `strict_exact_accuracy` (KV) are Track 1 diagnostics.
- `mgmt_events` / `events_applied` / `manager_capped`: management calls, calls with at least one
  applied edit, manager replies cut off by the output cap.
- Token columns sum prompt + completion tokens of the served model per role.
- Not comparable with llama.cpp/Q8_0 runs (different serving stack and precision).
