# Phase 6 — Shared-prefix candidate trie on Apple M4

## Experiment

DecisionLens previously scored every multi-token candidate separately,
cloning the frozen prefix cache and teacher-forcing all candidate tokens.
The new **opt-in** `--sequence-scorer trie` builds a token-prefix trie,
shares common prefix continuations, compresses single-child edges into
one forward call, and branches the MLX KV cache only where necessary.
The previous `reference` scorer remains the default and is used as a
numerical and decision-parity baseline.

**This is inference processing, not new model weights or a standalone
model.** It still requires the local ~2.1 GiB 4-bit Qwen backbone.

## Paired warm in-process results

All measurements below were executed locally on the 24 GiB Apple M4.
Model loading, server transport, concurrent load, and thermal soak are
excluded. The benchmark alternates scorer order.

| Dataset | Candidate pattern | Reference median | Trie median | Relative speed |
|---|---|---:|---:|---:|
| 24 rule-labeled diagnostic cases × 2 rounds | Mixed, limited prefix sharing | 0.3036 s | 0.2913 s | ~1.04× |
| Synthetic 16-option stress × 3 rounds | Heavy shared prefixes | 1.0392 s | 0.5844 s | **~1.78×** |
| Synthetic 32-option stress × 3 rounds | Heavy shared prefixes | 1.9633 s | 0.9843 s | **~1.99×** |

For the 16-option stress case, the tokenized candidates contain 128
total tokens but only 24 distinct trie prefix nodes. For the 32-option
case, 256 total tokens reduce to 42 distinct trie prefix nodes. Those
counts represent *logical prefix nodes*, not exact GPU operations.

Across all 48 paired diagnostic runs, the two implementations selected
identical complete decisions, with 42/48 exact matches to the
author-defined rules (21/24 repeated). The largest difference in a
normalized candidate score was **0.00713**. In the 16- and 32-option
stress tests, all paired decisions agreed, with maximum normalized
score differences of 0.00132 and 0.00179 respectively.

The score difference is not zero, likely reflecting differences in
MLX floating-point evaluation paths and sequence segmentation; its
source has not been exhaustively isolated. Scores are uncalibrated
heuristics, **not** probabilities of real-world events. These tests
do not prove parity for every tokenizer, candidate set or model.

## Reproduce

~~~bash
python3 -m unittest discover -q

.venv-student/bin/python -m orinth_clef.trie_benchmark \
  --rounds 2 --output runs/phase6/trie-benchmark.json

.venv-student/bin/python -m orinth_clef.prefix_stress \
  --options 16 --rounds 3

.venv-student/bin/python -m orinth_clef.prefix_stress \
  --options 32 --rounds 3 \
  --output runs/phase6/prefix-stress-32.json

.venv-student/bin/python -m orinth_clef.decision_lens \
  --model models/qwen3-4b-60iter-4bit --no-adapter \
  --sequence-scorer trie \
  --output runs/phase6/teacher-regression-trie.json
~~~

## Engineering constraints and next gates

- The trie scorer is **opt-in**. It is beneficial when allowed
  candidate IDs share token prefixes, not a universal latency win.
- JSON-escaped option IDs still fail closed; token-boundary-safe
  escaping requires dedicated work.
- The reference scorer and trie may differ slightly numerically.
  More adversarial prefix-of-another, Unicode and tokenizer cases
  are needed before switching the default.
- For production, benchmark p50/p95, peak unified memory, concurrent
  throughput, and long-sequence behavior under sustained thermal load.
- Independently adjudicated real-world decision gold and calibration
  remain release blockers. Do not present these synthetic results
  as general accuracy or SOTA.
