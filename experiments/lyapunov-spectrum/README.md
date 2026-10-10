# lyapunov-spectrum

Checks for `ideas/lyapunov-spectrum-intake.md` (Clark, arXiv:2610.12426).

- `check_identity.py` — numerically checks the paper's finite-N identity F(s) = (1/N) tr ∂v(k+1)/∂I(k) against QR Lyapunov exponents, plus a Hutchinson trace estimate. numpy only; results recorded in the intake doc §8.

- `phase_diagram.py` — max Lyapunov exponent of random rate nets (± input, ± normalization) and a random looped transformer block, swept over g, δ, norm gain, injection; DMFT closed form at δ=1. Outputs in `results/` (the `N=` header is ignored for `looptf`, which is fixed at d=64, L=16, 4 heads).
- `halting_check.py` — checks that the looped block's settling rate equals λ₁ (predicts loop count).

Design writeup: `ideas/edge-loop-design.md`.

Pins: Python 3, numpy 2.5.3, CPU (cloud container, 4 cores).

## MLX trained loop pilot (Apple Silicon)

Unlike the random-weight experiments above, `mlx_train_loop.py` **trains** a tied-weight
causal-attention/MLP recurrent block on a synthetic 3-pair binary key/value retrieval task.
Input is `key, bit, key, bit, key, bit, marker, query-key`; predict the
bit associated with the queried key. The input key/value pairs are shuffled.
A fixed random test stream uses the same finite synthetic domain as training,
so **held-out here means a different random draw, not unseen combinations or
out-of-distribution generalization**. The chance accuracy is approximately 50%.

Controlled ablation: initial weights and minibatch streams are matched between
input-injection strengths, with 5 random seeds per condition. We train at fixed
8 loops, then evaluate at 0, 1, 4, 8, and 16 loops. A separate Jacobian-vector
product (JVP) probe measures tangent amplification over 28 steps for one held-out
input; this is a *finite-time local diagnostic*, not the full Lyapunov spectrum.
Never treat the resulting numbers as pretrained-language-model benchmarks.

Run from repository root, on Apple Silicon with `mlx` and `numpy` installed:

```sh
python experiments/lyapunov-spectrum/test_mlx_train_loop.py
python experiments/lyapunov-spectrum/mlx_train_loop.py \
  --steps 800 --batch 48 --width 48 --loops 8 \
  --injections 0.3 1.0 3.0 \
  --output experiments/lyapunov-spectrum/results/mlx_training_sweep.json
```

Checkpoints are **local only** under `checkpoints/` (ignored by Git). Results
JSON and methodology are committed; no weights or models are published.
This trains from scratch, not on a downloaded pretrained checkpoint.

Research constraints: the tiny task has just 144 unique input/label
combinations; training examples repeat combinations. Every claim should
state the 5-seed spread and failure cases. Small near-zero settling differences
are at machine precision and do not establish a globally negative exponent.
