# Validation: size-matched GPU scheduling

The scientific grid remains five Qwen3.5 BF16 sizes, six official task levels,
seeds 42–45, `none` / native `clm_self`, and 240 planned episodes. Sampling
and pinned source/model revisions are unchanged. Scheduling now uses five
model-specific arrays and records the requested and actual GPU type.

Local verification:

- `bash -n` passed for the login-node launcher and compute worker scripts.
- `sha256sum -c SHA256SUMS --quiet` passed after this revision.
- 52 focused grid/model/launcher/native/integration tests: 51 passed, one
  real-upstream integration test skipped because its dependency is unavailable
  in this local Python environment.
- 74 total tests: 71 passed, two transport tests could not run because the
  pinned LiteLLM 1.105.1 is absent locally, and one integration test skipped.
  The compute setup reruns the suite in its locked environment.
- Synthetic scheduler tests check the five partition/device/memory requests,
  four seed indices per model, held setup, all-array report dependency,
  immutable suite snapshot and failure handling. No local GPU inference was run.

Still required on the cluster: run the non-submitting resource check, verify
pinned model downloads and real model startup/tool protocol on each selected
GPU, then inspect report coverage. Slurm completion alone does not establish
240 validated grades. No GPU accuracy or speed result is claimed here.
