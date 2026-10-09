# Phase 9 — selective micro-head cascade: negative result

## Hypothesis

Use the existing trained rank-32 question-conditioned micro-head as a
fast first stage. If **every** question has a top-two raw score margin
above a predeclared threshold, accept its typed decision. Otherwise
invoke the more accurate DecisionLens trie with short-system prompt.
Do not claim that uncalibrated logit margins are probabilities.

## Implemented

- `micro_head.decide_with_margins`: exactly one micro-head backbone
  prefill, one top-two raw margin per question, typed outputs.
- `selective_cascade.route`: all-question gating, strict finite
  nonnegative margins; fail closed.
- `choose_threshold`: *empirical* validation-only threshold utility.
  It has no coverage/risk guarantee and was **not used to tune** the
  experiment below.
- `collect` and `summarize`: paired measurements and **serial
  cost accounting**. Fallback cost is added to first-stage cost.
- CPU tests for gating, invalid scores, threshold selection, and
  correct fallback latency accounting.

## Measured M4 diagnostic

Previously inspected author-created 32-case rule fixture, Qwen3-4B
fused 4-bit model, trained 24-epoch rank-32 head. Warm, loaded model,
no server overhead. Predeclared raw margin threshold = 1.0.

| Runtime | Exact complete tasks | Median latency |
|---|---:|---:|
| DecisionLens trie / short system | **26/32** | **270.75 ms** |
| One-pass micro-head | 9/32 | 169.99 ms |
| Margin-gated cascade | 21/32 | 423.94 ms |

Only 8/32 tasks accepted the micro-head prediction; the other 24
needed a second backbone evaluation. This makes the cascade
**worse in both accuracy and latency** than DecisionLens.

The fixture has already been inspected during earlier experiments,
so it is **not** an untouched or independent benchmark. No claim
of calibration, SOTA, or real-world risk control is justified.
The rule-based ProofRoute path is not part of this comparison.

## Decision

**Reject the serial micro-head -> full-model fallback cascade.**
Do not promote this to default or publish the margin as confidence.
The failure is architectural: fallback performs another expensive
backbone pass. A new proposal must share computation, use an
actually smaller first-stage model, or exit early from intermediate
layers without a second full prefill.

## Next credible research gate

1. Train a layer-conditional decision head that reuses activations
   already produced by the final path; quantify quality by exit depth.
2. Evaluate modern sub-100M encoder / classifier baselines on
   human-labeled, domain-disjoint ambiguous tasks, not just rules.
3. Compare **complete system** against a conventional Python/SQL
   rules + small-classifier + LLM hybrid.
4. Include accuracy/coverage, abstention risk, p50/p95, cold and
   warm latency, peak memory, energy, and artifact portability.
5. Preserve failed experiments in the public research log.

To reproduce:

~~~bash
.venv-student/bin/python -m orinth_clef.selective_cascade \
  --threshold 1.0 --fixture examples/holdout_rules_v1.jsonl
~~~
