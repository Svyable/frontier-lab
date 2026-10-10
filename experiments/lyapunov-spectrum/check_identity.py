"""Numerical check of the Part 1 identity in Clark, "Lyapunov spectrum of random
neural networks" (arXiv:2610.12426):

    F(s) = (1/N) tr dv(k+1)/dI(k)

for the minimum-norm solution of the shifted tangent dynamics, compared against the
standard QR (Benettin) Lyapunov spectrum. Also reports a Hutchinson trace estimate,
the matrix-free variant we would need for large models.

Usage: python check_identity.py --N 64 --g 4 --delta 0.5 --window 100 --probes 16
Dense pinv on an (mN x (m+1)N) matrix: N=64, m=100 takes ~6 min on 4 CPU cores.
"""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--g", type=float, default=4.0)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--probes", type=int, default=16)
    p.add_argument("--qr-steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s", type=float, nargs="+", default=[-1.0, -0.4, -0.1, 0.05, 0.15, 0.3])
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    N, d = a.N, a.delta
    J = rng.normal(0, a.g / np.sqrt(N), (N, N))
    step = lambda x: (1 - d) * x + d * J @ np.tanh(x)
    jac = lambda x: (1 - d) * np.eye(N) + d * J * (1 - np.tanh(x) ** 2)[None, :]

    x = rng.normal(size=N)
    for _ in range(2000):
        x = step(x)

    # QR Lyapunov spectrum (rates per unit time, Eq. 2)
    Q, acc, xs = np.eye(N), np.zeros(N), x.copy()
    for _ in range(a.qr_steps):
        Q, R = np.linalg.qr(jac(xs) @ Q)
        acc += np.log(np.abs(np.diag(R)))
        xs = step(xs)
    lam = np.sort(acc / (a.qr_steps * d))[::-1]
    print("lambda_1 = %.3f, #positive = %d / %d" % (lam[0], (lam > 0).sum(), N))

    m, k = a.window, a.window // 2
    Ms = []
    for _ in range(m):
        Ms.append(jac(x))
        x = step(x)

    for s in a.s:
        alpha = np.exp(-d * s)
        K = np.zeros((m * N, (m + 1) * N))  # Eq. 16 on a finite window
        for n in range(m):
            K[n * N:(n + 1) * N, n * N:(n + 1) * N] = -alpha * Ms[n]
            K[n * N:(n + 1) * N, (n + 1) * N:(n + 2) * N] = np.eye(N)
        I = np.zeros((m * N, N))
        I[k * N:(k + 1) * N, :] = np.eye(N)
        V = np.linalg.pinv(K) @ I  # minimum-norm solution per unit source
        P = V[(k + 1) * N:(k + 2) * N, :]  # one-step response = projector onto E^s
        z = rng.choice([-1.0, 1.0], size=(N, a.probes))
        hutch = np.mean(np.sum(z * (P @ z), axis=0)) / N
        print("s=%5.2f  QR F(s)=%.3f  identity tr/N=%.3f  hutch(%d)=%.3f"
              % (s, (lam < s).mean(), np.trace(P) / N, a.probes, hutch))


if __name__ == "__main__":
    main()
