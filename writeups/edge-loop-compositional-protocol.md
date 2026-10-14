# Edge-loop: group-held-out compositional training protocol

**Status: RUN NOT YET PERFORMED.** This document specifies the next
experiment, not its outcome. Execution requires the authorized user's
Apple Silicon MacBook (MLX/Metal). No remote GPU spend is approved.

## Why this experiment exists

[PR #21](https://github.com/Svyable/frontier-lab/pull/21) trained tied
recurrence on a task with just 144 possible input/label combinations. Its
best injection setting had higher average accuracy but high seed variance,
and extra loops sometimes degraded accuracy. Training and evaluation sampled
the same finite domain, so this could be memorization.

## More demanding split, still deliberately modest

Four distinct keys chosen from eight, with one binary value per key, yield
C(8,4) * 2^4 = **1,120 canonical mappings**. Each has 4! * 4 = 96
possible orderings/query presentations. All presentations of a mapping
are assigned to train *or* test by a deterministic hash of the sorted key/
bit pairs. Current split: **895 training and 225 test mappings**, zero
mapping overlap by construction (checked by unit tests). Atomic keys and
bit values are reused across both splits; the holdout is novel
*combinations of associations*, not novel tokens.

Test examples are fixed at one RNG seed, train data are resampled from
train mappings only, and independent runs match minibatch RNG streams
for the same seed. Reporting held-out mapping accuracy is stronger than
the previous same-domain evaluation. This does **not** establish
general-purpose compositional reasoning or OOD robustness on real tasks.

## Conditions and controls

| Condition | Weights | Injection kappa | Executed blocks |
|---|---|---:|---:|
| tied_weak | One shared block | 0.3 | 8 |
| tied_strong | One shared block | 3.0 | 8 |
| untied_strong | Eight independent blocks | 3.0 | 8 |

All have hidden size 48, four attention heads, a two-layer GELU MLP,
Adam optimizer at learning rate 0.002, batch size 64 and 1,200 optimizer
updates for each seed 2026, 711 and 42. Same test inputs and training
sample sequence by seed. The untied baseline matches number of executed
blocks but **not parameter count**, so improvement over it cannot isolate
weight sharing from capacity without further parameter-matched runs.

The run writes both train-domain and group-held-out accuracy and cross
entropy, tied-model accuracy at depths 0/1/4/8/16, and untied accuracy
at depths 0/1/4/8. These are diagnostic depth ablations, not training
at variable depth.

## Dynamics and halting audit

For tied models only, an autodiff Jacobian-vector product estimates
finite-time tangent amplification over 28 steps on **12 separate test
inputs**, including the spread and fraction with positive mean rate.
A 28-step mean log tangent norm is not the asymptotic Lyapunov exponent,
and this does not implement Clark's full-spectrum cavity method.

For each test input we also record its first threshold crossing,
||h(t+1)-h(t)|| / ||h(t+1)|| < eps, with eps=1e-2 and 1e-3,
up to 8 loops. Report stopped accuracy, share exiting early, mean
stopping step, and the **number of correct 8-loop predictions harmed by
early stopping**. All trajectories are computed to depth 8; claimed
wall-clock savings require a separate adaptive execution benchmark.

## Go / no-go interpretation (set before seeing results)

- **Useful transfer:** a tied model exceeds 50% accuracy on held-out
  mappings consistently across seeds and beats a simple no-loop
  control; compare fairly against the untied model, disclosing parameter
  counts and all seeds.
- **Halting-safe:** early exit cannot be called safe if even a useful
  model loses correct predictions through the tested stopping rule.
  Inspect false exits by input, not just aggregate accuracy.
- **No positive signal:** unstable or chance-level models, or differences
  confined to one seed, are publishable *negative* findings but not a
  basis for scaling training.
- **Next controls if promising:** matched-parameter untied model,
  larger split families, different random splits, longer training/learning
  rate sweeps, and a true pretrained recurrent-depth checkpoint.

Upstream: David G. Clark, [Lyapunov spectrum of random neural networks
(2026)](https://arxiv.org/abs/2610.12426); Universal Transformers
(Dehghani et al., 2019); Geiping et al. recurrent-depth work. This
study neither proves nor directly applies the large-N analytic spectrum
of the random network to trained transformers.

**Execution state:** CPU data tests 3/3 passed on 2026-10-14; MLX/Metal
forward, gradient, smoke and full training remain unverified while the
MacBook is disconnected. Do not substitute invented numbers.
