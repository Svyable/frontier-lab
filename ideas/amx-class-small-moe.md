# Idea: amx-class-small-moe

- **Status:** seed (inspirational; not the sole charter track)
- **Constraint:** single-core / AMX-class throughput; tiny active parameter count
- **Related work:** [gdiamos/amx-reasoning-v1-instruct](https://huggingface.co/gdiamos/amx-reasoning-v1-instruct) — *Outrageously Small Neural Networks* (Diamos et al., Sep 2026)

## Hypothesis

Basic reasoning probes (induction, positional shift, simple arithmetic) can appear at surprisingly low token budgets when architecture is derived from a single-core roofline and data is strong curated/generated corpora — but only if routing and vocab failure modes are diagnosed, because they hide under a smooth loss.

## Why it might matter

Pushes the frontier toward **inference-optimal tiny models** and reusable diagnostics, not just bigger fleets.

## Minimal test

Reproduce or adapt a small block-routed MoE; run early probes at a fixed token budget; log expert participation and eval-token histograms beside loss.

## Risks / measurement traps

Expert collapse, non-function-preserving expert insertion, routing stats dominating design, untrained vocab rows winning argmax under uncertainty.

## Next step

Charter broadened beyond AMX-only work. Active next bet is `ideas/undertrained-token-decode-mask.md` (generalizing the vocab-mask diagnostic). Keep this note as inspiration for later tiny/hardware-aware runs (needs compute + HF auth).
