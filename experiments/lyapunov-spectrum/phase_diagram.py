"""First-principles design calculator for a looped recurrent core (no training).

Measures the maximum Lyapunov exponent lambda_1 (per unit time, i.e. per-step rate / delta)
of random, untrained maps along the loop axis:

  rate     x <- (1-d) x + d [ J tanh(x) + sigma * xi ]                      (Clark 2026 model + input)
  ratenorm x <- (1-d) x + d [ J tanh(a * x / rms(x)) + sigma * xi ]         (normalized pre-activation)
  looptf   h <- (1-d) h + d [ Attn(LN(h)) + MLP(LN(h)) ]  with fixed input injection (looped transformer)

For 'rate' at d=1 it also prints the DMFT closed form lambda_1 = log(g sqrt(<tanh'(x)^2>)),
x ~ N(0, q), q = g^2 <tanh(x)^2> + sigma^2 (Molgedey et al. 1992; Clark 2026 Table 1).
"""
import argparse

import numpy as np

# Dense grid, not Gauss-Hermite: tanh'^2 is a narrow peak relative to sqrt(q) at large g,
# and 101 Hermite nodes undersample it (gave lambda_1 = 0.608 vs 0.53 simulated at g=5).
GZ = np.linspace(-12, 12, 20001)
GW = np.exp(-GZ ** 2 / 2)
GW = GW / GW.sum()


def gauss(f, q):
    return float(np.sum(GW * f(np.sqrt(q) * GZ)))


def dmft_lambda1_delta1(g, sigma):
    q = 1.0
    for _ in range(2000):
        q = g * g * gauss(lambda x: np.tanh(x) ** 2, q) + sigma ** 2 + 1e-300
    cd = gauss(lambda x: (1 - np.tanh(x) ** 2) ** 2, q)
    return np.log(g * np.sqrt(cd)), q


def lyap_max(step, x0, d, rng, time_units=300.0, transient=100.0, eps=1e-7):
    """Benettin with one tangent vector via finite differences (float64)."""
    n_tr, n = int(transient / d), int(time_units / d)
    x = x0
    for i in range(n_tr):
        x = step(x, i)
    v = rng.normal(size=x.shape)
    v /= np.linalg.norm(v)
    acc = 0.0
    for i in range(n):
        xn = step(x, n_tr + i)
        vn = (step(x + eps * v, n_tr + i) - xn) / eps
        nv = np.linalg.norm(vn)
        acc += np.log(nv)
        v, x = vn / nv, xn
    return acc / (n * d)


def make_rate(N, g, d, sigma, rng, norm_gain=None, frozen_input=True):
    J = rng.normal(0, g / np.sqrt(N), (N, N))
    # frozen_input: one fixed input pattern per step index drawn from a seeded stream, so the
    # perturbed and unperturbed copies see the same drive (common-noise / driven setting).
    seeds = rng.integers(1 << 31)

    def drive(i):
        return np.random.default_rng((seeds, i)).normal(size=N) * sigma if sigma > 0 else 0.0

    def step(x, i):
        u = x if norm_gain is None else norm_gain * x / np.sqrt(np.mean(x * x) + 1e-12)
        return (1 - d) * x + d * (J @ np.tanh(u) + drive(i))

    return step, rng.normal(size=N)


def make_looptf(dm, L, H, g, d, rng, inj=1.0):
    def W(i, o, scale=1.0):
        return rng.normal(0, scale / np.sqrt(i), (i, o))

    Wq, Wk, Wv = W(dm, dm), W(dm, dm), W(dm, dm)
    Wo = W(dm, dm, g)
    W1, W2 = W(dm, 4 * dm), W(4 * dm, dm, g)
    e = rng.normal(size=(L, dm)) * inj  # fixed injected input (embedded tokens)
    mask = np.triu(np.full((L, L), -np.inf), 1)
    hd = dm // H

    def ln(z):
        z = z - z.mean(-1, keepdims=True)
        return z / np.sqrt((z * z).mean(-1, keepdims=True) + 1e-6)

    def gelu(z):
        return 0.5 * z * (1 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

    def step(h, i):
        z = ln(h + e)
        q, k, v = z @ Wq, z @ Wk, z @ Wv
        out = np.zeros_like(z)
        for a in range(H):
            s = slice(a * hd, (a + 1) * hd)
            att = q[:, s] @ k[:, s].T / np.sqrt(hd) + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            out[:, s] = att @ v[:, s]
        h1 = out @ Wo
        z2 = ln(h + e + h1)
        return (1 - d) * h + d * (h1 + gelu(z2 @ W1) @ W2)

    return step, rng.normal(size=(L, dm))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rate", "ratenorm", "looptf"], default="rate")
    p.add_argument("--N", type=int, default=400)
    p.add_argument("--g", type=float, nargs="+", default=[0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    p.add_argument("--delta", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--norm-gain", type=float, nargs="+", default=[1.0])
    p.add_argument("--inj", type=float, nargs="+", default=[1.0],
                   help="looptf: injected-input scale(s); ratenorm: also swept via --norm-gain")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--time", type=float, default=300.0)
    a = p.parse_args()

    print("model=%s N=%d sigma=%.2f seeds=%d  (lambda_1 per unit time; mean +- spread over seeds)"
          % (a.model, a.N, a.sigma, a.seeds))
    header = "g     " + "".join("  d=%-12s" % dl for dl in a.delta)
    if a.model == "rate" and 1.0 in a.delta:
        header += "  DMFT(d=1)"
    print(header)
    knobs = a.inj if a.model == "looptf" else a.norm_gain if a.model == "ratenorm" else [None]
    for g in a.g:
        for kn in knobs:
            row = "%-5.2f " % g + ("" if kn is None else "k=%-5.2f " % kn)
            for d in a.delta:
                vals = []
                for sd in range(a.seeds):
                    rng = np.random.default_rng(1000 * sd + 7)
                    if a.model == "looptf":
                        step, x0 = make_looptf(64, 16, 4, g, d, rng, inj=kn)
                    else:
                        ng = kn
                        step, x0 = make_rate(a.N, g, d, a.sigma, rng, norm_gain=ng)
                    vals.append(lyap_max(step, x0, d, rng, time_units=a.time))
                row += "  %+.3f+-%.3f " % (np.mean(vals), np.ptp(vals) / 2)
            if a.model == "rate" and 1.0 in a.delta:
                row += "  %+.3f" % dmft_lambda1_delta1(g, a.sigma)[0]
            print(row, flush=True)


if __name__ == "__main__":
    main()
