# Phase 5 — Schema coverage before further distillation

## The observed generalization failure

We built 24 *new* deterministic rule-labeled diagnostics, not Clef pseudo-labels:
6 support routing controls, 6 weather threshold decisions, 6 inventory
decisions and 6 financial-risk decisions. Each case has two questions,
an explicit rule and a complete expected decision object.

The fixture is **author-created**, not independently human-audited gold.
It is useful for detecting regressions and failure modes, not a
publishable general reasoning benchmark.

The original one-prefill micro-head achieved **7/24 exact complete-case
matches**, though all 24 responses were schema-valid. It failed badly
on the new support control cases (0/6). We should not claim its small
in-domain pseudo-label agreement implies generalization.

## Candidate-sequence extension

The original DecisionLens scored only single-token candidate IDs, refusing
any candidate with more than one token. This is an avoidable schema
limitation, not a model intelligence problem.

We now score *every* allowed candidate's token sequence by teacher-forcing
it against a **copy** of the same prompt KV cache. For a candidate with
multiple tokens, the score is its mean conditional token log-likelihood.
This is a heuristic normalization, **not** a calibrated class probability.
The selected continuation advances the original cache before the next
question. One-token choices retain the fast existing path.

No unconstrained autoregressive answer generation occurs. For long
candidate strings the extra branch forwards consume real computation and
memory. The method does not yet use a prefix trie to share candidate
subsequences, and candidate tokenization may not exactly match every
possible context-sensitive tokenizer boundary. Both warrant further tests.

## Actual Apple M4 results, same 24 diagnostic cases

| Runtime | Exact complete decisions | Schema-valid | Median warm case latency |
|---|---:|---:|---:|
| Micro-head, 24 epochs | 7/24 | 24/24 | ~0.152 s |
| DecisionLens, single-token-only | 11/24 | 12/24 | ~0.115 s* |
| **DecisionLens, candidate sequences** | **21/24** | **24/24** | **0.288 s** |

*The old 0.115-second median is **not a valid speed comparison**: 12
unsupported cases failed immediately without inference. On the new
version, single-token families measured medians of ~0.231 s (support)
and ~0.275 s (weather); multi-token families ~0.301 s (inventory)
and ~0.302 s (risk).

Per-family exact scores: support 6/6, weather 5/6, inventory 5/6,
risk 5/6. The three misses are weather-3, inventory-1 and risk-2.
This improvement is mostly **coverage**, not a demonstrated increase in
underlying model reasoning capability.

## Run

~~~bash
.venv-student/bin/python -m orinth_clef.challenge \
  --fixture examples/challenge_rules.jsonl \
  --model models/qwen3-4b-60iter-4bit \
  --head runs/phase4/micro-head-24.safetensors \
  --output runs/phase5/challenge-sequences.json
~~~

Model weights, head weights and case-level predictions are local and
ignored by Git. Source fixture, tests, aggregate results and the
evaluation harness are committed.

## Next gates

1. Make the challenge fixture truly independent: external annotators,
   task-family-disjoint cases and independently adjudicated ground truth.
2. Build a prefix trie for multi-token candidates to reuse shared
   candidate prefixes. Compare exact scores and measured latency against
   this reference implementation.
3. Verify tokenizer boundary correctness and Unicode / escaping /
   adversarial candidate IDs. Fail closed on unsupported cases.
4. Measure both quality and latency on identical supported case sets,
   not median latency including instant validation failures.
5. Reconsider an adaptive cascade only when a confidence/abstention gate
   is validated on an independent dataset. A faster but substantially
   less reliable micro-head is not a production shortcut.
