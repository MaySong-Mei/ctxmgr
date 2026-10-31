These assets are copied verbatim from
https://github.com/RulinShao/ContextBench/tree/fb533c18c78a82d3f1b2047a9c1700735cf829ae
(Apache-2.0; see LICENSE.contextbench):

- skills/clm/kv_store.md: clm_exp/skills/clm/kv_store.md
- skills/clm/needle_retention.md: clm_exp/skills/clm/needle_retention.md
- contextbench.def: environment/contextbench.def

The ContextBench wheel omits these method skill files. Bundling just these assets
lets the unchanged official ClmStream load and hash-check the paper's skills.

The CLM implementation itself is a pinned Git dependency, not vendored or edited:
https://github.com/facebookresearch/context-language-models/tree/483198229ab7b90c0eed4651306ab3b3505aaf50
It carries its own CC BY-NC 4.0 license. upstream.lock.json records SHA-256 of the
installed upstream Python/YAML files and copied ContextBench assets; `check`
verifies them before every run. Keep the two licenses distinct.
