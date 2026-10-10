"""Does the loop's convergence rate equal lambda_1? (random looped transformer, no training)

If the looped map is contracting (lambda_1 < 0), ||h_{t+1} - h_t|| should decay like
exp(lambda_1 * delta * t), giving a first-principles loop count to reach tolerance eps:
    T(eps) ~ log(1/eps) / (|lambda_1| * delta).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from phase_diagram import lyap_max, make_looptf  # noqa: E402

d = 0.5
print("inj   lambda_1   decay_rate(||dh||)   T_pred(eps=1e-3)  T_meas(eps=1e-3)")
for inj in [0.3, 1.0, 3.0]:
    for seed in [7, 1007]:
        rng = np.random.default_rng(seed)
        step, h = make_looptf(64, 16, 4, 1.0, d, rng, inj=inj)
        lam = lyap_max(step, h.copy(), d, np.random.default_rng(seed), time_units=150)
        rng = np.random.default_rng(seed)
        step, h = make_looptf(64, 16, 4, 1.0, d, rng, inj=inj)
        dh = []
        for t in range(2000):
            hn = step(h, t)
            dh.append(np.linalg.norm(hn - h) / (np.linalg.norm(hn) + 1e-12))
            h = hn
        dh = np.array(dh)
        # fit decay over the window where 1e-9 < dh < 1e-2 (asymptotic regime, above roundoff)
        idx = np.where((dh < 1e-2) & (dh > 1e-9))[0]
        rate = np.polyfit(idx * d, np.log(dh[idx]), 1)[0] if len(idx) > 5 else float("nan")
        t_meas = int(np.argmax(dh < 1e-3)) if (dh < 1e-3).any() else -1
        t_pred = np.log(1e3) / (abs(lam) * d) if lam < 0 else float("inf")
        print("%-5.1f %+.3f     %+.3f               %7.1f           %5d" % (inj, lam, rate, t_pred, t_meas))
