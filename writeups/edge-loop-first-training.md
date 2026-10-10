# Edge-loop: first trained-network results on Apple Silicon

**Status:** first reproducible MLX training pilot, **not** a language-model benchmark.
**Predecessor:** [PR #20](https://github.com/Svyable/frontier-lab/pull/20)
measured *random, untrained* recurrent blocks. Here we train tied-weight
attention/MLP blocks and test whether the observed dynamical dials survive learning.
**Sources:** Clark, [*Lyapunov spectrum of random neural networks*
(arXiv:2610.12426)](https://arxiv.org/abs/2610.12426); the earlier
[edge-loop design](../ideas/edge-loop-design.md). This does not implement
Clark's large-N full-spectrum cavity theory on the trained transformer.

## Protocol

- Machine: user's Apple M4 Pro MacBook Pro, 24 GB unified memory; MLX 0.32.3,
  NumPy 2.5.3; Metal backend.
- Task: input is three shuffled pairs of distinct key tokens (0,1,2) and
  binary value tokens (3,4), a marker (5), then the queried key. Predict the
  value associated with the queried key. Chance accuracy ~50%.
- The task has only **144 possible input/target combinations**. Test data
  use a separate RNG stream but sample the **same finite domain**; combinations
  can recur during training. There is no claim of out-of-distribution generalization.
- Architecture: 48 hidden dimensions, 4 causal attention heads, 2x GELU MLP,
  tied recurrent block, delta=0.5, 8 training loops; learned token/position
  embeddings and output head. No pretrained weights.
- Optimizer: Adam 0.002, 800 updates, batch 48; 1024 held-out RNG samples.
  Five matched initialization seeds: 2026, 711, 42, 88, 1024. For each seed,
  minibatches and initialization are matched across injection settings.
- Independent variable: fixed input-injection gain kappa = 0.3, 1.0, 3.0.
- Same trained checkpoint also evaluated at 0, 1, 4 and 16 loops.
  One-input JVP tangent-rate/settling probe for 28 loops is diagnostic only.
- Source: `experiments/lyapunov-spectrum/mlx_train_loop.py`. Raw run:
  `experiments/lyapunov-spectrum/results/mlx_training_sweep.json`.
  An earlier **400-step / two-seed** pilot is separately retained in
  `mlx_training_pilot.json` (not pooled with the 800-step run).

## Main measured result: strength helps learning here, not reliably

Test accuracy (mean across 5 seeds, range in parentheses):

| kappa | 0 loops | 1 loop | 4 loops | **8 trained loops** | 16 loops |
| --- | ---: | ---: | ---: | ---: | ---: |
| **0.3** | 51.9% | 52.9% | 52.7% | **51.9%** (47.0–64.2%) | 52.1% |
| **1.0** | 50.3% | 55.5% | 61.1% | **57.2%** (48.4–73.1%) | 56.8% |
| **3.0** | 50.1% | 59.3% | 73.3% | **77.6%** (59.9–100%) | 74.3% |

The strongest condition beat the weakest by **25.65 percentage points in
mean 8-loop accuracy**, on the matched five-seed pilot. This is an exploratory
difference, not a statistically established general result. Many seeds fail.

At kappa=3, two strong seeds reached 92.3% and 100.0% at eight loops, but
degraded to 80.7% and 97.1% respectively at sixteen loops. A learned
eight-loop solution is **not** necessarily invariant to additional loops.
One kappa=3 seed showed a **positive** measured finite-time tangent rate
(+0.082/unit time) and did not satisfy the 1e-3 settling criterion by
28 loops; it was the seed with perfect 8-loop held-out performance.
For other seeds, strongly negative tangent rates (e.g. approximately -1.3)
coincided with collapsed near-chance solutions. Local contraction can
indicate quick convergence toward an **uninformative** state.

The early 400-step pilot had one kappa=3 seed at 84.3% and one at chance.
The separate 800-step run with the same nominal seed ended at 59.9%.
These were independently initialized runs rather than one continued training
trajectory, and their intermediate losses diverged. Thus the comparison
does **not** establish late-training forgetting; it highlights sensitivity
and the need to test exact-run reproducibility.

## Interpretation and limits

1. **Supported narrowly:** injection strength is a meaningful optimization
   dial in this toy trained recurrence, not just in random-weight spectra.
   Larger injection helped the mean under the tested settings.
2. **Refuted as a blanket design rule:** maximizing contraction and then
   halting on a small step norm is not sufficient for accurate reasoning.
   Collapse can be fast; successful computation need not settle quickly.
3. **Not established:** benefits to pretrained transformers, LLM reasoning,
   language modeling, token efficiency, hidden-state attractor dimensions,
   Kolmogorov–Sinai entropy, or Clark's full-spectrum result.
4. **Probe caution:** the recorded `mean_tangent_rate_after_8` is one
   input, one tangent vector, 20 sampled steps after an 8-step transient.
   It is neither a converged top Lyapunov exponent nor a distribution.

## Next experimental decision

Before scaling up, add a held-out **compositional** task with more unique
combinations, a standard untied transformer baseline, stronger stopping
criteria measured against *accuracy*, and learning-rate sensitivity.
Evaluate on a pretrained looped checkpoint if its license permits. Measure
local tangent rates on many inputs, not one. Do not enable adaptive
halting until task performance and stopping reliability are both verified.

**Reproduce on Apple Silicon:** see
[experiments/lyapunov-spectrum/README.md](../experiments/lyapunov-spectrum/README.md).
Local weights are saved in ignored `checkpoints/`, not published to GitHub
or Hugging Face. No paid training services or external GPU resources were used.
