# Native CLM vLLM/BF16 interface

## Scope and fixed sources

This standalone release uses the user's uploaded `ctxmgr-automation-review-26N2hP.tar.gz`
(SHA-256 `0cafca6037461e40cf6c24e15b0a2c6613fdfe7b1d21fc1fc7f9916edfc8eeb3`) as its
source baseline. It adds orchestration and backend compatibility; it does not
replace the frozen agent/task/scoring files with a teammate's newer protocol.

| Component | Pin |
| --- | --- |
| Native CLM | `facebookresearch/context-language-models@483198229ab7b90c0eed4651306ab3b3505aaf50` |
| ContextBench | `RulinShao/ContextBench@fb533c18c78a82d3f1b2047a9c1700735cf829ae` |
| Original team source bundle | `863dce2953805747dfcf30653c6d710a590a0b2c`, verified file hashes |
| Serving | `vllm==0.30.0`, root `uv.lock` |
| Native client | `envs/clm/uv.lock`, including `litellm==1.105.1` |
| Transport adapter | `reasoning-alias-v1` |

`agents.py`, `management.py`, `episode.py`, and the hashed task/scorer sources are
unchanged. The upstream CLM package is installed from its exact commit and checked
against the bundled upstream file inventory; no monkey-patching is used.

| Official HF repository | Immutable revision | Served alias |
| --- | --- | --- |
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` | Qwen3.5-0.8B-BF16 |
| Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Qwen3.5-2B-BF16 |
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Qwen3.5-4B-BF16 |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Qwen3.5-9B-BF16 |
| Qwen/Qwen3.5-27B | `fc05daec18b0a78c049392ed2e771dde82bdf654` | Qwen3.5-27B-BF16 |

Full safetensors snapshots are downloaded through Hugging Face's pinned-revision
cache. All serving configuration/tokenizer/weight files are inventoried and
SHA-256 checked. Standard HF blob symlinks are supported. Weight indices must
match the actual shard tensors. Some official model parameters are stored as
FP32; the server explicitly loads the model with `--dtype bfloat16`. No FP8,
quantization, MTP, or speculative decoding is enabled.

The manifest field `sha256` has kind `hf-snapshot-tree-sha256`: SHA-256 of a canonical
sorted JSON file inventory. It is not interchangeable with an old GGUF file hash.
Files and digests are retained so the exact model artifacts can be checked.

## Entry points

The public login-node entry point is:

```text
bash scripts/clm_vllm.sh check [--config FILE]
bash scripts/clm_vllm.sh submit [--config FILE]
bash scripts/clm_vllm.sh status SUITE
bash scripts/clm_vllm.sh report SUITE
```

Resource overrides are trusted, user-authored Bash configuration. They are copied
as resolved exported values into each suite, hashed and checked by workers. The
scientific grid and sampling do not change through this resource configuration.

The launcher snapshots source, records the setup/array/report IDs, then releases
the held setup job. The array's `afterok` dependency blocks evaluation when setup
fails. The report's `afterany` dependency retains failed/missing coverage. Invalid
dependencies are cancelled by Slurm. An uncertain scheduler response stops
submission and retains evidence under `submission/`; there is no blind retry.

Compute-only internal commands:

```text
python -m ctxmgr.harness.clm_vllm_suite setup --suite PATH
python -m ctxmgr.harness.clm_vllm_suite cell --suite PATH --index 0..19
python -m ctxmgr.harness.clm_vllm_suite report --suite PATH
python -m ctxmgr.harness.clm_vllm_models prepare --size 27B --out PATH
python -m ctxmgr.harness.clm_vllm_models verify --manifest PATH/model.json
python -m ctxmgr.harness.clm_vllm_proxy --port PORT --upstream http://127.0.0.1:PORT
```

These commands must not be run directly on a Bouchet login node. The sbatch worker
checks the actual hostname against `SLURM_JOB_NODELIST` and rejects login nodes.

## Process and API contract

Each GPU task owns one vLLM process group and one loopback compatibility gateway.
The server serves a local pinned snapshot under a BF16 alias, never a mutable HF
branch name. Calls are sequential (`--max-num-seqs 1`). Server flags are saved in
`server-config.json`; the health check also verifies the served alias.

vLLM 0.30.0 calls response reasoning `reasoning`; the pinned CLM/LiteLLM code uses
`reasoning_content`. The gateway forwards non-streaming requests byte-for-byte and
renames that one response field. It leaves content, tools, usage, and sampling
unchanged; it rejects conflicting aliases. Incoming tool history is accepted by
vLLM under either alias. Streaming is explicitly unsupported because neither
experiment client uses it. Only loopback health/models/chat endpoints are exposed.

The original `none` agent intentionally retains only assistant content. Native
CLM retains and counts its own reasoning history. The gateway does not make those
protocols identical; that difference is explicit in the report. The native
600-second model-call timeout remains unchanged. GPU readiness has a separate
3600-second timeout for initial model loading/compilation.

An actual native-client transport probe precedes each cell. Tool-call refusal
is recorded as a model/probe warning; it does not exclude small models. A failed
HTTP/protocol exchange fails the cell, preserving its logs. A probe is not a
benchmark grade, and a successful probe does not prove semantic task correctness.

Each episode runs in a separate subprocess with stdin disconnected so that native
tools cannot consume the scheduling list. Cell cleanup targets complete process
groups, including workers left after their leaders exit. Slurm allocation cleanup
remains the final boundary for node processes and forced termination.

## Reporting contract

`clm_official` schema 3 records vLLM/BF16, immutable model identity, official CLM
settings, frozen team source hashes, shared task identities, and transport version.
The suite reporter verifies each cell's exact size/seed/12-episode selection against
the shared 240-episode plan. It uses read-only cell summaries to avoid overwriting
live progress files. Reports are serialized with a lock; files are atomically
replaced and the final status records the report-file hashes.

Only rows with complete provenance and a valid saved reward are graded. An
interrupted job can still contribute earlier valid grades. Missing, corrupt,
failed and mismatched rows remain present with blank rewards. Means and paired
comparisons have explicit coverage; no unmatched seed is substituted into a pair.

This validator checks saved metadata, file identities and native receipts; it does
not rerun the grader over every snapshot. Raw snapshots and scores remain available
for an independent audit. No automatic GitHub message or email is sent.

## Primary serving references

- [vLLM Qwen3.5-27B recipe: BF16 on one H200](https://recipes.vllm.ai/Qwen/Qwen3.5-27B)
- [Official Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [vLLM 0.30 reasoning output interface](https://docs.vllm.ai/en/v0.30.0/features/reasoning_outputs/)
- [vLLM 0.30 OpenAI-compatible server and generation config](https://docs.vllm.ai/en/v0.30.0/serving/online_serving/openai_compatible_server/)
- [vLLM 0.30 Qwen parser aliases](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/tool_parsers/__init__.py)
- [vLLM 0.30 request reasoning aliases](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/entrypoints/openai/chat_completion/protocol.py)

Documentation/source checks do not replace the user's actual H200 execution.
