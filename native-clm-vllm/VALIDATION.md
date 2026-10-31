# Validation: five-model native CLM vLLM/BF16 revision

## Source and scope

This revision adds official Qwen3.5-9B BF16 to the previous four-model package.
The five models are 0.8B, 2B, 4B, 9B and 27B. Six official task levels, task
seeds 42–45 and conditions `none` / native `clm_self` produce 240 planned
episodes in 20 model/seed GPU cells. The user-selected official sampling is
temperature 0.7, top_p 0.95, max_tokens 8192 and thinking enabled. The 32,768
raw o200k budget, 2,048 reserve and 81,920 served-model context are unchanged.

The 9B repository pin is the verified official Hugging Face commit
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The model verifier checks an
immutable revision, complete BF16 safetensors weights, config and tokenizer,
inventory SHA-256 and model identity before serving. The pre-existing four
model pins are unchanged.

## Local validation

- A plain Python run of `python3 -m unittest discover -s tests -p
  'test_vllm_*.py' -q` collected 73 tests: 70 passed, two transport tests
  could not start because this local Python lacks pinned LiteLLM 1.105.1, and
  one real-upstream integration test skipped because `clm_contextbench` is not
  installed in that Python. The prior four-model package recorded 73 passing
  tests in its locked native-client environment; that result does not certify
  this five-model revision.
- A focused run of the 51 grid/model/launcher/native/integration tests completed
  with 50 passed and one real-upstream skip. Bash syntax checks for the launcher,
  compute runtime and sbatch worker passed. An exact-settings assertion verified
  five model IDs, 20 cells, 240 rows and the unmodified 0.7/0.95/8192/thinking
  scientific profile.
- Synthetic grid and launcher tests check 240 distinct episode identities,
  48 episodes each for 9B and 27B, 20 model/seed cells, frozen submission,
  dependency ordering, partial grades and missing versus zero rewards.
- The backend remains vLLM 0.30.0 / BF16 on one H200 per cell. The original
  `none` agent, native CLM prompts/skills/Bash-file tools, task generator and
  scorer are preserved from the four-model package.

## Still required on Bouchet

- Run all 73 tests in the locked native-client environment on an allocated
  compute node and verify the pinned CLM/ContextBench source hashes.
- Confirm the course account's H200 allocation, all five pinned model
  downloads and full-weight hashes, and actual 9B/27B vLLM inference with
  the native tool protocol. Source-level and mocked transport tests do not
  replace this GPU check.
- Final benchmark coverage is established by `report/report-status.json`
  showing 240/240 validated grades, not by Slurm job completion alone.

No H200 model inference, Bouchet job, model-weight download or accuracy result
has been performed by this local revision.
