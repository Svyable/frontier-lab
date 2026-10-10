# lyapunov-spectrum

Checks for `ideas/lyapunov-spectrum-intake.md` (Clark, arXiv:2610.12426).

- `check_identity.py` — numerically checks the paper's finite-N identity F(s) = (1/N) tr ∂v(k+1)/∂I(k) against QR Lyapunov exponents, plus a Hutchinson trace estimate. numpy only; results recorded in the intake doc §8.

Pins: Python 3, numpy 2.5.3, CPU (cloud container, 4 cores).
