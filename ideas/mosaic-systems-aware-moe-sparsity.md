# Idea: mosaic-systems-aware-moe-sparsity

- **Status:** seed
- **Constraint:** architecture–systems co-design for sparse MoE; sparsity chosen under cluster deliverable FLOPs, not model-FLOPs alone
- **Related work:** [Compute-Optimal Is Not Cluster-Optimal (MOSAIC)](https://arxiv.org/abs/2608.10605) — Sarkar, Tang, Zha (Aug 2026); systems-aware scaling for sparse MoE

## Hypothesis

Under calibrated sparsity ranges, efficiency-agnostic model-FLOPs budgets have no interior optimal sparsity (loss keeps preferring sparser models), so the sparsity that actually wins emerges from MFU, communication, memory, and parallel layout — a design rule we can stress-test at much smaller scales.

## Why it might matter

Stupidly useful framing: “pick sparsity from the cluster, not the FLOPs equation.” Even without 79B runs, a miniature scaling sweep + a crude MFU/comms model could falsify or support the qualitative claim.

## Minimal test

Literature synthesis first (no invented SOTA). Optional tiny MoE sweep varying expert count / active fraction at fixed token budget on one machine; log loss vs a hand-calibrated “deliverable FLOPs” proxy (MFU × peak). Mark any numbers `UNVERIFIED` until reproduced.

## Risks / measurement traps

Paper’s active-parameter range starts at ~104M — far above our AMX-class toys; qualitative transfer may fail. Easy to overfit a toy performance model. Never quote paper tables as Frontier Lab results.

## Next step

Parked as a seed. If revisited: one-page writeup extracting the sparsity-boundary claim + a checklist for a small-scale falsification attempt.
