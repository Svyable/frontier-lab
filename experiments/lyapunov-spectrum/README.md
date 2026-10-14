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

## Group-held-out compositional follow-up (PR #22; training pending)

The original three-pair pilot could reuse the same 144 examples. The follow-up
`compositional_data.py` instead samples **four distinct keys from eight**,
assigns each key a binary value, shuffles the four key/value pairs, and asks for
one queried value. Its deterministic BLAKE2s hash partitions by the **entire
canonical key-to-value mapping**: every permutation and queried key of a given
mapping are excluded from the other split. There are 1,120 possible mappings,
and the split currently yields 895 train and 225 test mappings. **This remains
a synthetic finite-domain transfer test, not a language-model benchmark.**

`mlx_compositional.py` trains 3 alternatives at the same 8-block execution
depth: tied weak injection (0.3), tied strong injection (3.0), and an **untied**
8-block transformer (injection 3.0). The untied baseline has more parameters
and is **compute/depth matched, not parameter matched**. It reports mapping-
disjoint test accuracy, training-domain accuracy, evaluation at variable loop
depths, a finite-time JVP probe on multiple inputs, and whether fixed thresholds
halt before a correct prediction is reached.

CPU-only dataset tests and syntax checks (can run anywhere):

```sh
cd experiments/lyapunov-spectrum
python -m unittest -v test_compositional_data
python -m py_compile compositional_data.py mlx_compositional.py test_mlx_compositional.py
```

Once the MacBook is online, use its existing MLX virtual environment from
the **repository root** and run:

```sh
experiments/orinth-clef-unified/.venv-student/bin/python \
  experiments/lyapunov-spectrum/test_mlx_compositional.py

# Smoke before sustained training:
experiments/orinth-clef-unified/.venv-student/bin/python \
  experiments/lyapunov-spectrum/mlx_compositional.py \
  --steps 2 --batch 8 --seeds 42 --variants tied_strong untied_strong \
  --test-size 64 --probe-inputs 2 --output /tmp/compositional-smoke.json

# Three conditions × three seeds; only after successful smoke:
experiments/orinth-clef-unified/.venv-student/bin/python \
  experiments/lyapunov-spectrum/mlx_compositional.py \
  --steps 1200 --batch 64 --seeds 2026 711 42 \
  --output experiments/lyapunov-spectrum/results/compositional_mlx.json
```

The runner writes results after each completed condition so interrupted runs
leave a partial record marked `in_progress`. It does **not** save checkpoints
unless `--save-checkpoints` is explicitly passed. No MLX training results have
been recorded for this follow-up yet. See
`writeups/edge-loop-compositional-protocol.md` for decision criteria.
