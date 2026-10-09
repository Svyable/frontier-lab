# Phase 4: One-pass decision head

Experiment goal: freeze the 4-bit Qwen backbone, extract one task representation per request, and train a compact question-conditioned head to score variable option descriptions. Evaluate teacher imitation on held-out families and measure complete end-to-end latency separately from head-only latency. Do not report pseudo-label agreement as accuracy or candidate logits as calibrated probabilities.

Acceptance gates: strict schema validity, independently verified outcomes, and faster end-to-end median than the 0.279-second DecisionLens baseline. The micro-head is an experimental candidate, not a production release.
