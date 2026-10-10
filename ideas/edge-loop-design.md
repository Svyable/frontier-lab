# Idea: edge-loop (first-principles design of a looped core, no training)

- **Status:** design (CPU numerics on random, untrained networks only; **no training, no benchmark claims**)
- **Constraint:** zero training budget. Every design rule below comes from either the paper's theory or a measurement on random networks that anyone can rerun on a laptop CPU in minutes.
- **Related work:** Clark, *Lyapunov spectrum of random neural networks*, arXiv:2610.12426 (intake: [`lyapunov-spectrum-intake.md`](lyapunov-spectrum-intake.md)). Prior art for looped/recurrent-depth models: Universal Transformers (Dehghani et al. 2019) and recurrent-depth latent reasoning (Geiping et al. 2025, arXiv:2502.05171). Edge-of-chaos trainability: Poole et al. 2016, Schoenholz et al. 2017. Input suppresses chaos: Molgedey et al. 1992, Schuecker et al. 2018. Gradient flossing: Engelken 2023. Transient chaos in looped reasoning: Lai et al., arXiv:2609.04963.

Code and raw outputs: `experiments/lyapunov-spectrum/` (`phase_diagram.py`, `halting_check.py`, `results/*.txt`).

---

## 1. One-paragraph pitch

A looped transformer is a dynamical system along its loop axis. The paper gives exact tools for measuring and predicting the chaos of such systems. On random networks we find that the chaos of a modern pre-norm looped block is **not** set by weight scale, which normalization cancels. It is set by two scalars: the **norm gain** and the **input-injection strength**. Both act as monotone dials. When the loop is contracting, its convergence rate **equals** the top Lyapunov exponent λ₁. So one dial fixes trainability across loops, the speed at which the loop settles, and therefore how much compute each token needs. *edge-loop* is a looped core whose dials are set analytically, before any training, to a target λ₁ chosen from the loop budget.

## 2. Measurements behind the design

All results are λ₁ **per unit time** (per-step rate ÷ δ, as in the paper), measured with the Benettin method using a finite-difference tangent vector in float64. They are averages over 2 seeds, with ± giving half the range between seeds. These are random, untrained weights.

### M1. The paper's closed form predicts λ₁ (rate network, δ=1)

`x ← (1−δ)x + δ[J tanh(x) + σξ]`, N=1000, with closed form λ₁ = log(g·√⟨tanh′(x)²⟩) and x ~ N(0, q), q = g²⟨tanh²⟩ + σ².

| g | σ=0 sim / theory | σ=0.5 sim / theory | σ=1.0 sim / theory |
|---|---|---|---|
| 1.0 | −0.005 / 0.000 | −0.271 / −0.272 | −0.460 / −0.460 |
| 1.5 | +0.058 / +0.065 | −0.018 / −0.017 | −0.145 / −0.142 |
| 2.0 | +0.152 / +0.155 | +0.115 / +0.120 | +0.042 / +0.046 |
| 3.0 | +0.310 / +0.310 | +0.295 / +0.299 | +0.268 / +0.269 |
| 5.0 | +0.526 / +0.532 | +0.517 / +0.529 | +0.513 / +0.520 |

The closed form agrees with simulation to within about 0.01 everywhere. For small δ the values also match the paper's figures: at δ=0.1, N=2000 we get λ₁ = 0.229 at g=3 and 0.411 at g=5, against ≈0.22 and ≈0.42 read off its Fig. 1.

*Bug we hit, for the record:* a 101-node Gauss–Hermite quadrature undersampled the narrow tanh′² peak at large q. It gave 0.608 at g=5 and looked like a theory/simulation disagreement. A dense grid fixed it (0.532).

### M2. Normalization removes g as a knob; the norm gain is the dial

`x ← (1−δ)x + δ J tanh(a·x/rms(x))`, N=500–1000.

| | δ=0.1 | δ=0.5 | δ=1.0 |
|---|---|---|---|
| a=1, g ∈ [0.5, 3] | +0.037 … +0.054 (flat) | +0.052 … +0.059 (flat) | +0.076 … +0.077 (flat) |
| g=1, a=0.3 | −0.008 | −0.008 | −0.011 |
| g=1, a=0.6 | +0.020 | +0.018 | +0.008 |
| g=1, a=1 | +0.057 | +0.058 | +0.074 |
| g=1, a=2 | +0.170 | +0.174 | +0.234 |
| g=1, a=4 | +0.379 | +0.345 | +0.466 |

Why: the update is scale-equivariant. Multiplying J by c multiplies the state by c, and the normalization divides it back out. So **in pre-norm blocks, initial weight scale does not set chaos. The norm gain (RMSNorm γ, or a fixed scalar) does.** The edge sits near a ≈ 0.3–0.6 here.

Also: with normalization λ₁ never goes strongly negative (it bottoms out near 0). On its own, a normalized loop cannot contract onto a fixed point; something else must provide contraction (M3).

### M3. In a looped transformer, input-injection strength is the contraction dial

A random pre-LN causal block: d=64, L=16, 4 heads, GELU MLP, residual outputs scaled by g. The input e is re-injected every loop: `h ← (1−δ)h + δ·Block(h + κ·e)`.

| | δ=0.1 | δ=0.5 | δ=1.0 |
|---|---|---|---|
| g=1, κ=0 | −0.09 ± 0.12 | −0.09 ± 0.13 | −0.09 ± 0.10 |
| g=1, κ=0.1 | −0.10 ± 0.10 | −0.10 ± 0.11 | −0.09 ± 0.08 |
| g=1, κ=0.3 | −0.17 ± 0.01 | −0.18 ± 0.01 | −0.12 ± 0.05 |
| g=1, κ=1 | −0.35 ± 0.07 | −0.37 ± 0.08 | −0.33 ± 0.13 |
| g=1, κ=3 | −0.62 ± 0.05 | −0.71 ± 0.06 | −0.90 ± 0.10 |
| κ=1, g=0.25 → 8 | −0.71 → −0.09 | −0.84 → −0.08 | −1.14 → −0.01 |

Unifying reading: what matters is the **ratio of injected input to state**. Stronger injection (κ↑) or a smaller state (g↓) gives more contraction. This is the transformer version of "input suppresses chaos" (refs above). The random looped block is in the ordered (contracting) regime everywhere we looked. Near κ≈0 the seed spread is large, so the operating point should sit where seeds agree (κ ≳ 0.3).

### M4. Settling rate = λ₁, which predicts loop count

Same block, δ=0.5, g=1. We fit the exponential decay rate of ‖h_{t+1}−h_t‖/‖h_t‖ and compare it with λ₁. T_pred = ln(1/ε)/(|λ₁|δ) with ε=10⁻³.

| κ | λ₁ | decay rate | T_pred | T_measured |
|---|---|---|---|---|
| 0.3 | −0.185 / −0.151 | −0.179 / −0.159 | 75 / 91 | 49 / 64 |
| 1.0 | −0.431 / −0.274 | −0.428 / −0.269 | 32 / 50 | 31 / 32 |
| 3.0 | −0.768 / −0.650 | −0.727 / −0.659 | 18 / 21 | 16 / 15 |

(Pairs are two seeds.) The decay rate matches λ₁ to within 0.04 in all 6 runs. T_pred is a conservative bound, 1.0–1.6× the measured count; the gap is the initial-distance prefactor. **The chaos dial is also the compute dial.**

### M5. The time-step δ barely changes λ₁ per unit time for δ ≤ 0.5

Rate network at g=3, σ=0: 0.211 (δ=0.1) vs 0.203 (δ=0.5) vs 0.310 (δ=1). The looped transformer shows the same pattern. This matches the paper, where the spectra for δ=0.05 and δ=0.1 nearly coincide. So for small δ the loop approximates an ODE. **Loop count T and step δ can then be traded at fixed "time" τ = δT.**

## 3. The design

```
tokens ─► Prelude (2 untied pre-norm blocks) ─► e
h₀ = 0
for t = 1..T:                       # tied core, T variable, δ = τ / T
    h ← (1−δ)·h + δ·Core( RMSNorm_a(h) + κ ⊙ RMSNorm(e) )
    halt when ‖h_t − h_{t−1}‖ / ‖h_t‖ < ε          (only meaningful because λ₁ < 0 by design)
Coda (1–2 untied blocks) ─► LM head
Core = pre-norm causal attention + SiLU/GELU MLP (smooth activations only)
```

| Component | Rule | Comes from |
|---|---|---|
| Tied core, looped | Depth becomes a dynamical system, so λ₁, F(s), D_KY and h_KS are well defined and can be measured. Test-time compute scales with T. | Paper §1; prior art on looped models |
| Leaky update with δ = τ/T, δ ≤ 0.5 | Behavior per unit time is nearly δ-independent, so T can change at inference with δ = τ/T without changing the dynamics being approximated. | M5; paper Fig. 1 |
| **Don't tune weight-init scale for dynamics** | Pre-norm cancels it. Use any standard init. | M2 |
| **Norm gain a = chaos ceiling** | Initialize the core's RMSNorm γ so the uninjected loop sits at λ₁ ≈ 0 (a ≈ 0.3–0.6 in M2; **re-measure for the real block**). | M2 |
| **Injection gain κ = contraction dial** (learnable, per channel) | Initialize κ so that λ₁ = λ* < 0, with λ* chosen in §4. | M3 |
| Convergence halting | Valid only when λ₁ < 0. Expected loops T ≈ ln(1/ε)/(|λ₁|δ) (conservative). Inputs whose local λ₁ is closer to 0 run longer, which gives adaptive compute for free. That is consistent with the transient-chaos-on-hard-tasks observation in Lai et al. | M4 |
| Smooth activations (SiLU/GELU/tanh), no ReLU | The paper's identity and theory assume invertible Jacobians. | Paper §2.1 |
| Ship a "spectrum card" | λ₁ (Benettin, 1 extra forward per loop), top-k exponents → D_KY and h_KS, measured along the loop axis at init and at every checkpoint. | Paper §1.2; intake L1 |

## 4. Choosing the operating point λ* from first principles

Two constraints pull in opposite directions. Both are exponential in λ₁·τ:

1. **Gradient and signal flow through the loop.** Perturbations, and backprop-through-loops gradients, scale like e^{λ₁τ} over τ = δT units of time. Keeping that within e^{±1} requires |λ₁| ≲ 1/τ. Too negative and the loop forgets its starting state, so extra loops add nothing. Positive values explode.
2. **Compute per token.** From M4, loops ≈ ln(1/ε)/(|λ₁|δ). Values closer to 0 cost more loops.

Worked example (numbers only, nothing trained): suppose we budget τ = 8 time units and δ = 0.25, i.e. T_max = 32 loops. Constraint 1 gives λ* ∈ [−0.125, 0). At λ* = −0.125 and ε = 10⁻², M4 predicts ≈ 4.6/(0.125·0.25) ≈ 147 loops to converge, far above 32. **So for this budget the loop cannot both preserve gradients over τ and converge within T_max.** Pick one:

- (a) **Gradient-first:** λ* ≈ −1/τ, train with a fixed T and no halting.
- (b) **Halting-first:** λ* ≈ −ln(1/ε)/(δ·T_max) ≈ −0.58, which in M3 means κ ≈ 2–3. Accept the faster forgetting, and train with truncated backprop through the last few loops (as recurrent-depth models do).

This trade-off is the main design decision. It was invisible before λ₁ was measured, and it should be decided before any training run. **Recommendation: (b), halting-first.** It is the regime where M4's prediction is reliable and where compute is bounded. Under (a) the loop sits near marginal, where M3 seeds disagree by ±0.1.

## 5. What could break (and how we'd know cheaply)

- **Trained weights aren't random.** Attention at random init is close to uniform averaging. Trained attention is sharp and input-dependent and may change λ₁ a lot. *Cheap check (CPU, no training):* run `phase_diagram.py`-style measurements on a public pretrained looped model (e.g. the Geiping et al. recurrent-depth checkpoint on HF, if its license allows) along its loop axis. Do our dials still order λ₁ monotonically?
- **Toy scale.** d=64, L=16. *Check:* d ∈ {128, 256}, L ∈ {64, 256}. Still CPU-feasible.
- **The finite-difference tangent vector** is fine in float64 at ε=1e-7. In bf16 it is not, so the spectrum card must run in fp32 or fp64.
- **Two seeds.** Each design-critical number (the edge a, and κ for λ*) should be re-measured with ≥5 seeds and a CI before it is frozen into an init recipe.
- **Prior art overlap.** Looped/recurrent-depth models, convergence halting and edge-of-chaos init each exist separately. The claim to novelty is narrower: **in pre-norm loops, chaos is set by norm gain and injection ratio, not weight scale, and the settling rate equals λ₁, which links trainability to compute.** That is UNVERIFIED as novel; we need a literature check before calling it new.

## 6. Next steps (all zero-budget)

1. Run the M3/M4 dials on a real pretrained looped checkpoint (CPU inference only). This is the single most informative next measurement.
2. Repeat M2–M4 at larger d and L with 5 seeds, and add an attention-temperature sweep (a third dial we haven't measured).
3. Add top-k QR exponents to report D_KY/N and h_KS/N for the looped block, not just λ₁.
4. Literature check on the novelty claim in §5.
5. If 1–2 hold up: write `writeups/edge-loop.md` and propose an HF Space "spectrum card" calculator. That needs HF auth and your OK to publish as Svyable.
