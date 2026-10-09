# Phase 10 — Shared-backbone early exits on MLX

**Status: implemented, reproducible, modest mean-latency result; no SOTA.**

## Architectural change

A rank-32 question-conditioned head reads the hidden state after a
configurable depth D of the **36-layer frozen Qwen3-4B** backbone.
If the lowest per-question top-two logit margin meets a validation-
selected threshold, the whole task exits. Otherwise the **same**
intermediate activation is passed through layers D+1..36, and the
existing 24-epoch final micro-head returns the decisions. Neither
path generates JSON tokens.

Unlike the Phase 9 serial cascade, fallback does not run the
transformer prefix twice. It still runs the early head and the
remaining layers, so savings require sufficiently high early coverage.

Numerical M4 check: for an ordinary task, comparing
`model.model(inputs)` against the 18-layer prefix plus continuation
produced **maximum and mean absolute hidden-state difference 0.0**.

The intermediate head is separately trained using only 96
teacher-pseudo-labeled train cases (24 epochs, rank 32). Threshold
selection uses 12 validation cases; 12 family-disjoint test cases
were **previously examined** in earlier experiments, so the
measurements are exploratory, not a fresh holdout.

## M4 experiments (24 GiB, warm)

| Early layer | Validation agreement target | Test exits | Test pseudo-label agreement | Median task latency |
|---|---:|---:|---:|---:|
| 12/36 | 80% empirical | 0/12 | 7/12 | 189.04 ms |
| 18/36 | 95% empirical | 0/12 | 7/12 | 189.22 ms |
| **24/36** | **80% empirical** | **4/12** | **7/12** | **189.15 ms** |

The 24-layer candidate was compared to the full final micro-head
in **three paired rounds**, alternating order for each case.

| Metric | Shared-backbone exit | Full 36-layer micro-head |
|---|---:|---:|
| Teacher pseudo-label matches | 21/36 | 21/36 |
| Mean warm latency | **167.99 ms** | 188.59 ms |
| Median warm latency | 188.61 ms | **187.98 ms** |
| Early exits | 12/36 (4 distinct tasks) | n/a |

Mean latency decreased by **10.9%**; median latency did **not**
improve. The 12 accepted early decisions in the paired benchmark
were all pseudo-label matches; this is **only four distinct cases**
and cannot justify calibration or reliable abstention.

The always-on DecisionLens trie is a **different, more accurate**
baseline on many tasks. Matching the full micro-head is not matching
DecisionLens, and this result does not establish a new quality/latency
Pareto frontier against that stronger baseline.

## Reproduction

~~~bash
.venv-student/bin/python -m orinth_clef.early_exit \
  --depth 24 --epochs 24 --target-validation-agreement 0.8 \
  --output runs/phase10/early24-target80.json \
  --weights runs/phase10/early24.safetensors

.venv-student/bin/python -m orinth_clef.early_exit_benchmark \
  --training-report runs/phase10/early24-target80.json \
  --early-weights runs/phase10/early24.safetensors \
  --rounds 3

.venv-student/bin/python -m orinth_clef.hf_release export \
  --include-weights --head runs/phase4/micro-head-24.safetensors \
  --early-head runs/phase10/early24.safetensors \
  --early-report runs/phase10/early24-target80.json \
  --output runs/phase10/hf-early-exit-v1
~~~

The HF-style research bundle contains the optional early-head
weights and minimal configuration, plus the full MLX model and final
micro-head. All are needed for this experimental route. The raw
teacher training data is **not** distributed.

**Packaged-model smoke test:** the 45-file checksum-verified bundle
loaded all three components and executed the depth-24 early-exit
route. For an out-of-suite single Boolean task with
`state.severity=4` and the instruction `True iff severity >= 4`,
the early head returned **false**, which is **incorrect**. The
manifest still verified after inference. This one case is not an
accuracy estimate, but it directly demonstrates why this optional
head must not be promoted to a reliable default.

## Research gates

1. A new, independently adjudicated dataset of diverse multi-question
   decisions; freeze all thresholds before test evaluation.
2. Confidence calibration and risk/coverage curves, including
   abstention on distribution shifts and adversarial inputs.
3. More expressive early heads, training across layer depths,
   and a baseline that directly runs fewer layers.
4. Compare end-to-end p50/p95, mean, peak memory, throughput and
   energy to the stronger DecisionLens, modern compact classifiers
   and rule-engine hybrids on identical tasks.
5. A portable Transformers/PyTorch implementation and licensing
   review before any public Hugging Face upload.

**Conclusion:** the computation-sharing mechanism is verified and
has a small exploratory mean-latency gain, but the learned early
head is not yet good enough for a moonshot or a SOTA claim.
