# When/how analysis

Aggregate-only scripts for the when-versus-how study. Extraction runs on the Bouchet
login node with shell tools only (no Python); the statistics run anywhere.

- `extract.sh ROOT LABEL` — one TSV row per ContextBench episode: label, actor, condition,
  episode, official reward, termination. Run it on the native suite (`native`) and on
  each when/how root (`whenhow`), concatenated into `cb.tsv`.
- `paired.py cb.tsv` — paired differences by (actor, level, seed): mean difference,
  seed-level bootstrap 95% CI, better/worse counts, exact two-sided sign test. Low-pressure
  controls (K4-C4, R130) are excluded.
- `bcp_an.py` — BrowseComp-Plus accuracy per condition (Wilson 95% CI), context-limit
  hits, exact McNemar tests on the same questions. Input `bcp.tsv`: actor, condition,
  question id, judged correct, judged, hit-the-limit flag (no question or answer text).
- `bcp_cost.py` — nominal compute, 2 × parameters × (uncached prompt + completion
  tokens) per role, from `usage.json` and the manager summary in `whenhow.json`.
  Ignores attention FLOPs and Qwen3.5's linear-attention layers; a size-scaled token
  proxy, not a measured cost.
