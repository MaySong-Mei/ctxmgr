# Validation: 27B cell bounds and guarded continuation

The fixed cell entry point accepts all 20 grid indices. The new continuation
launcher permits only the four absent 27B cells in an otherwise validated
192/240 suite. It preserves the actual frozen runtime (including experiment
variants), records its narrow runtime-only guard repair, and checks existing
grade artifacts before each worker and after the final report.

Current local verification: 16 suite tests, 11 existing launcher tests and
12 continuation tests passed (39 total). These cover the 27B boundary, held
preflight dependencies, duplicate/partial-cell rejection, scheduler failures,
uncertain submission responses, preserved grades, unknown-runtime rejection,
real frozen-runtime invocation through a fake client, and report hashes.
`bash -n` passed for the new launcher and worker. Release checksums were
regenerated and verified. No live Slurm submission or GPU inference was run
in this development environment; the original cluster environment and
source layout are checked by the CPU preflight before GPU work can start.

The dependent continuation report records the patch scope and recalculates
its hashes. Use `status` to inspect it. Running the original frozen `report`
command later regenerates the original report format; the separate
continuation receipt and per-cell provenance remain available.

## Earlier size-matched scheduling validation

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
