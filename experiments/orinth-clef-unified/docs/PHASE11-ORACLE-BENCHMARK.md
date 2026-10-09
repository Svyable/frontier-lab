# Phase 11 — frozen executable-rule oracle benchmark

## Purpose and protocol

Previous phases relied on teacher pseudo-labels and repeatedly inspected
author-created fixtures. Phase 11 freezes a new 64-task benchmark before
model evaluation, using a deterministic seeded generator. Each task has
two linked decisions (typed Boolean and binary choice) and an explicit
executable policy; the oracle label is computed by the independent
ProofRoute policy interpreter, **not by any neural model**.

Four families have 16 cases each: numeric thresholds, exact category
matches, conjunctions, and comparisons of two state fields. The fixture
is committed to git at `examples/oracle_rules_v2.jsonl`, seed 20261009,
SHA-256:
`42552be8fa1256132f57089636ee11985878e665924742048304d8af476619f4`.

The fixture was generated and committed **before** the first neural
evaluation. The tasks are procedurally authored and share templates;
this is **not independently human-adjudicated gold**. It is a narrow
executable-policy test, not a general reasoning benchmark.

## Apple M4, warm model, single paired pass

| Runtime | Exact complete tasks | Median latency | Mean latency |
|---|---:|---:|---:|
| Explicit symbolic policy | **64/64** | **0.016 ms** | 0.016 ms |
| DecisionLens trie, short system | **55/64** | 233.5 ms | 235.4 ms |
| 36-layer micro-head | 40/64 | 186.3 ms | 178.3 ms |
| Shared-backbone depth-24 early exit | 40/64 | **125.1 ms** | 119.6 ms |

The symbolic interpreter has the **answer-bearing structured policy**
as input; the neural runtimes only receive task text/state. The
symbolic runtime is therefore not a fair apples-to-apples neural
comparison and does not solve natural-language policy extraction.

| Family | Cases | DecisionLens correct | Micro/exit correct |
|---|---:|---:|---:|
| Exact category | 16 | 16 | 13 |
| Conjunction | 16 | 11 | 13 |
| Relative field comparison | 16 | 15 | 10 |
| Numeric threshold | 16 | 13 | **4** |

**Critical failure:** the early head accepted **64/64** tasks, of which
**24/64 were incorrect** against the explicit oracle. Its raw logit
margin is not calibrated; the empirical threshold selected on
12 Clef pseudo-label validation tasks transferred badly to these
out-of-distribution policy questions. The architecture saves
computation but has unacceptable reliability.

The full micro-head made the same 40/64 exact matches as the
early-exit route. The better-performing DecisionLens should remain
the neural default. This test is a single small warm run with no
confidence intervals, no power/energy measures, and no public
hardware-independent comparisons. No SOTA claim.

## Reproduction

~~~bash
python3 -m orinth_clef.oracle_benchmark generate \
  --output examples/oracle_rules_v2.jsonl --seed 20261009 --per-family 16
# Generate only into a new path; committed fixture refuses overwrite.

.venv-student/bin/python -m orinth_clef.oracle_benchmark evaluate \
  --fixture examples/oracle_rules_v2.jsonl \
  --output runs/phase11/oracle_v2.json

# Expected nonzero exit: diagnostic early-head promotion gate FAILS.
.venv-student/bin/python -m orinth_clef.oracle_benchmark gate \
  --report runs/phase11/oracle_v2.json
~~~

The evaluator verifies every executable policy and label before
loading MLX, and rotates method order between tasks. It records
per-task outputs, routing, wall-clock latency, p50/p95, mean, and
family breakdown. Source and fixture are included in the offline
Hugging Face-style research bundle.

## Gate for next iteration

- Do not enable the early head by default or describe raw margins
  as confidence. It fails this frozen oracle suite.
- Build **new training and validation data** with executable or
  independently adjudicated labels; freeze a distinct unseen
  benchmark before model selection. Do not tune to these 64 cases.
- Test exact numeric comparisons, negative examples, multi-question
  consistency, calibration, OOD abstention, and family-level shifts.
- Compare against compact non-generative classifiers and the
  rule-engine-plus-LLM hybrid, on identical information inputs.
- A public Hugging Face release still needs rights review, a
  portable inference interface, model-card caveats and external
  evaluation.
