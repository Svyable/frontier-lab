# Idea: cox-moe-amx-inference

- **Status:** seed
- **Constraint:** throughput-oriented MoE inference on AMX-enabled CPU–GPU co-execution; memory-bound expert execution
- **Related work:** [CoX-MoE](https://arxiv.org/abs/2605.17889) — Son, Chen, Yoo, Choi, Kim (May 2026); AMX CPU–GPU collaborative MoE inference with coalesced expert execution

## Hypothesis

Coalescing expert execution (ordinary batches instead of fragmented micro-batches) plus static expert-aware GPU/CPU stratification can recover operational intensity that micro-batch offloading destroys — a systems trick that may transfer to smaller sparse MoEs, not only large serving stacks.

## Why it might matter

Paragraph-simple systems nip: batch shape and expert placement dominate throughput when experts spill to CPU. Ties hardware-aware MoE design to measurable toks/s, not just loss.

## Minimal test

Reproduce or re-implement the coalescing + stratification ideas on a small public MoE (or a tiny block-routed toy); ablate (a) micro-batch vs coalesced expert batches, (b) with/without frequency-based expert pinning; report throughput and PCIe/transfer proxies. Do **not** invent FlexGen/MoE-Lightning comparisons — cite paper numbers as upstream only.

## Risks / measurement traps

Paper claims are on large serving setups; gains may vanish at tiny expert counts. Confounding GPU memory pressure with AMX CPU speedups. License and reimplementation cost before claiming a Frontier Lab artifact.

## Next step

Parked as a seed. If revisited: skim HTML/PDF for exact coalescing policy details, then decide whether a writeup-only note or a toy ablation is worth it.
