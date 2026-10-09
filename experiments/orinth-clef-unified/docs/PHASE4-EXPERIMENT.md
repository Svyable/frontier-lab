# Phase 4 — Frozen-backbone question-conditioned micro-head

## Hypothesis and implementation

Instead of generating JSON or performing one transformer forward per question,
encode the full task with **one** frozen Qwen3-4B transformer pass. Reuse its
last-token hidden representation, plus quantized embedding lookups for each
question and candidate description. A three-projection, rank-32 MLX head
scores all candidate descriptions and the runtime assembles typed JSON.

The backbone is the local fused 4-bit Phase 3 checkpoint (~2.1 GiB).
The head is ~960 KiB. **The head cannot operate independently of the backbone.**
All model weights and raw predictions remain on the local Mac.

## First M4 experiments (24 synthetic held-out cases)

| Candidate | Train epochs | Valid teacher match | Test teacher match | Combined | Median end-to-end |
|---|---:|---:|---:|---:|---:|
| Fused 4-bit DecisionLens | n/a | 8/12 | 9/12 | 17/24 | 0.2789 s |
| Micro-head rank 32 | 4 | 8/12 | 4/12 | 12/24 | ~0.186 s |
| **Micro-head rank 32** | **24** | **9/12** | **7/12** | **16/24** | **~0.186 s** |

The 24-epoch head has approximately **1.5x lower median end-to-end latency**
than DecisionLens on this small warm M4 test, but one fewer teacher
pseudo-label exact match. All 24 outputs are schema-valid by construction.

**Decision:** retain the 24-epoch head as an experimental *speed candidate*,
not a quality-approved replacement. There is no evidence of SOTA on
external tasks or independent accuracy. The teacher labels are pseudo-labels.
Both validation and test have now been examined during iteration, so neither
can serve as an untouched final holdout for subsequent tuning.

## Experiment details

- Hardware: Apple M4, 24 GiB unified memory.
- Feature extraction: ~22.3 seconds for all 120 tasks; backbone features
  cached in process for head training only.
- 24-epoch head training, rank 32, Adam learning rate 1e-4, fixed sample
  order, seed 42, 96 training cases. Validation/test labels are never used
  in gradient updates.
- Final reported latency includes backbone forward, embedding lookups,
  head inference and JSON schema validation, but excludes loading weights,
  server overhead and dataset IO. These are warm in-process medians, not
  sustained request-throughput measurements.
- Head output scores are not calibrated probabilities, and do not replicate
  Clef's probabilistic SystemOne head.
- Locally reproducible run:

~~~bash
.venv-student/bin/python -m orinth_clef.micro_head \
  --model models/qwen3-4b-60iter-4bit --data data/mlx_120 \
  --rank 32 --epochs 24 \
  --output runs/phase4/micro-head-24.json \
  --head-weights runs/phase4/micro-head-24.safetensors
~~~

## Next gated research

1. Create a **new**, independently audited benchmark with unrelated domains,
   arbitrary multi-token candidate names, conflicting criteria, negation,
   adversarial distractors and open-ended ordinary-language regressions.
2. Benchmark paired end-to-end latency distributions under identical load,
   sequence lengths, batch sizes and thermal conditions.
3. Investigate multi-query conditioning, query-aware token pooling and
   contrastive candidate embeddings to close the quality gap without
   additional backbone passes.
4. Measure micro-head quality by question type, scenario family, calibration
   and abstention behavior. Require independent human gold labels.
5. Only package and publish after held-out reliability and repeatable
   performance are demonstrated.
