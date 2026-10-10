# lyapunov-spectrum

Checks for `ideas/lyapunov-spectrum-intake.md` (Clark, arXiv:2610.12426).

- `check_identity.py` — numerically checks the paper's finite-N identity F(s) = (1/N) tr ∂v(k+1)/∂I(k) against QR Lyapunov exponents, plus a Hutchinson trace estimate. numpy only; results recorded in the intake doc §8.

- `phase_diagram.py` — max Lyapunov exponent of random rate nets (± input, ± normalization) and a random looped transformer block, swept over g, δ, norm gain, injection; DMFT closed form at δ=1. Outputs in `results/` (the `N=` header is ignored for `looptf`, which is fixed at d=64, L=16, 4 heads).
- `halting_check.py` — checks that the looped block's settling rate equals λ₁ (predicts loop count).

Design writeup: `ideas/edge-loop-design.md`.

Pins: Python 3, numpy 2.5.3, CPU (cloud container, 4 cores).
