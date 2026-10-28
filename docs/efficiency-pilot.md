# Efficiency and public multi-document pilot

Ruthvik requested implementation and submission after the round-25 collaborator review. This is exploratory follow-up, not an extension of the frozen confirmatory analysis.

## Questions and comparisons

Compare Qwen3.5 2B/4B/9B with no management, self management and a fixed 27B manager against a 27B actor with none/self/same-size peer management. All use bf16, vLLM 0.30, one H200, served length 81,920, greedy decoding, thinking off, actor output cap 2,048, manager cap 4,096, trigger 18,432 o200k, usable benchmark window 30,720 o200k. Each configuration runs serially. Report quality alongside resources, including failed and partial episodes.

ContextBench timing pilot: Needle K4-C24, seeds 50–53, all four actor sizes (48 episodes). KV R1000, seeds 50–53, 9B none_r/self_arc/peer_27b_arc/mask_arc and 27B none_r/self_arc/mask_arc (28 episodes). No grader or edit semantics changed; only runner/evaluation elapsed time was added. These are previously observed instances, appropriate for instrumentation, not independent validation.

Each episode resets both servers' prefix caches, then allows reuse within that episode. Conditions are deterministically shuffled. Server startup and scoring time are separate from runner time. One dedicated H200 allocation per job is measured; small-actor jobs retain both resident servers for their baselines, so resident-memory/throughput conclusions about a standalone small actor require a separate serving-layout experiment. A small actor and manager share one GPU and serial calls; this is not a batched multi-actor throughput study.

## Metrics

`src/ctxmgr/accounting/efficiency.py` extracts tokens and API latency by role, official reward, optional lenient KV, termination and provenance. Nominal model size × tokens (with/without observed prefix reuse) is explicitly a **proxy**, not architecture-aware FLOPs. Missing cache metadata stays missing. GPU telemetry records total allocated-device memory, utilization and power every two seconds; job metadata separates startup. Instrumented `episode.json` records runner and evaluation time, and `timing.json` records each episode's absolute interval. Dollar costs require an explicit hardware rental rate and are not inferred from the university allocation. Cost per success requires a specified success criterion; partial reward is not binary success.

History job reads group-readable team results without modifying them. Historical latency is observational and must remain separate from the controlled cache-reset pilot. No patient data is involved.

## Public multi-document pilot

Source: HotpotQA distractor dev set (https://hotpotqa.github.io/), CC BY-SA 4.0. Verified SHA256: `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`. The official URL is tried first; a mirror is accepted only for identical bytes. Eight cases are selected by `sha256('pilot-v1:'+id)` order, without performance filtering. All configurations receive the same cases and document order.

Short control: the original ten paragraphs, streamed one at a time. Long variant: add approximately 40K o200k tokens of unique unrelated paragraphs from other dev cases, then shuffle all paragraphs and packet them into roughly 3.5K-token inputs. Added paragraphs are a stress-test modification, not the official benchmark. Original paragraphs are never chosen/placed using answer or supporting-fact labels. The actor sees the question from the start, reads each packet and returns READY; only after FINAL does it answer and identify supporting sentence indices. Actor replies and manager notes count against the same context wall. Managers use the existing editor and may only retain/rewrite context, not make extra task-tool calls.

Question answering and sentence-level support are scored after inference; evaluator-only answer/support labels are never passed into agent prompts. Report normalized answer EM/F1, support F1, JSON parse validity, packet completion, management-event count, elapsed time and per-role usage. Gold-perfect answer plus support on a completed stream is a separate strict success metric. Budget-wall runs receive no final answer and remain in the denominator. No LLM judge is used. It is an exploratory HotpotQA-derived streaming QA pilot, not an official HotpotQA result or a general autonomous-agent benchmark. Familiar public facts may be memorized; this pilot cannot establish that correct answers depended on retained evidence. Inspect citations and compare short/long controls before expanding.

Full pilot: 8 cases × 2 lengths × 3 conditions × 4 actor sizes = 192 QA episodes. Smoke: 2B, one ContextBench seed under none/peer_27b plus one QA case at both lengths under three conditions. A failed infrastructure smoke blocks full GPU jobs via `afterok`. Low task quality is retained and does not silently trigger sample replacement.

## Submission and outputs

`slurm/efficiency_submit.sh` submits setup, existing-log analysis, a 2B smoke, and four full GPU jobs with at most two of this experiment's full GPU jobs running concurrently. Full GPU jobs depend on smoke success. Setup/test failure blocks all dependent work. The submission manifest is append-only; the submit script refuses reuse of its manifest path. Work occurs in a dedicated project-directory checkout, leaving existing branches/runs untouched. GPU jobs use pinned, group-readable team model snapshots read-only; all generated artifacts/caches are in Ruthvik's course project/scratch paths.

AI assistance: Codex prepared instrumentation, pilot scripts, tests and documentation; experimental claims still require human review.
