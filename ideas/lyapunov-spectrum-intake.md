# Intake: Lyapunov spectrum of random neural networks (arXiv:2610.12426)

- **Status:** exploring (intake). Not yet an experiment.
- **Constraint:** start CPU-only (numpy/JAX, N ≤ 4096). Escalate before any GPU training spend.
- **Related work:** see [References](#references). The paper is the primary source. Its code is at [davidclark1/rnn-lyapunov](https://github.com/davidclark1/rnn-lyapunov) (license **TODO**: check before reuse).

> **Credit.** David G. Clark (Flatiron Institute, Center for Computational Neuroscience), *Lyapunov spectrum of random neural networks*, arXiv:2610.12426 [cond-mat.dis-nn], 8 Oct 2026. The paper says the work was done with GPT-6 Astra (initial derivation) and Claude Opus 5.5 (code, figures, text revision). The problem was proposed by Ashok Litwin-Kumar. Peer-review status: **UNVERIFIED** (v1 preprint, two days old at intake).

---

## 0. TL;DR (read this first)

**What the paper is.** It is a theoretical physics result, not a model-architecture paper. It finally computes the **full Lyapunov spectrum** of the classic Sompolinsky–Crisanti–Sommers (SCS) chaotic rate network at large N, which had been an open problem since 1988. The method is new: an exact finite-N identity turns "how many exponents are below s?" into a **trace of a one-step response**. A cavity calculation then makes that response solvable at large N.

**What it is not.** It does not give a better transformer, a training recipe, or a benchmark. Nothing in it shows a model getting more accurate. Any architecture we build from it is **our hypothesis**, and we have to earn it with measurements.

**What carries over to building models.** Three things, from most to least certain:

1. **A diagnostic.** We can measure how "chaotic" or "contracting" a recurrent, looped or deep network is, using numbers that don't depend on the choice of coordinates: the attractor dimension D_KY and the entropy rate h_KS. The paper's identity also suggests a **matrix-free** estimator that doesn't need N tangent vectors. We checked the identity below: it holds, but the cheap estimator is noisy.
2. **Design knobs with theory behind them.** The paper gives closed-form or single-site predictions of the spectrum as a function of the **gain g** (init scale), the **leak/step δ** (residual branch scale) and the nonlinearity. Those can set init and step size *before* training instead of by sweep.
3. **A training-time lever.** We could regularize the *shape* of the spectrum, not just the top exponent. That would generalize "gradient flossing" (Engelken 2023, ref [48]), which already showed that pushing exponents toward 0 helps RNNs learn long-range dependencies.

**Candidate architecture (hypothesis only).** A **looped recurrent core**: one tied block iterated T times with an explicit leak δ. g and δ are chosen from the theory, a spectrum-shaping regularizer is applied, and an optional halting signal comes from the local expansion rate. See §7.

---

## 1. The model studied

Euler-discretized random rate network (Eq. 5–7):

```
x_i(n+1) = (1-δ) x_i(n) + δ Σ_j J_ij φ(x_j(n)),   J_ij ~ N(0, g²/N) i.i.d., φ = tanh
M(n) = (1-δ) I + δ J D(n),   D(n) = diag(φ'(x_i(n)))   ("gains")
```

- **g**: coupling strength. Chaos sets in at **g = 1** (as N→∞).
- **δ ∈ (0,1]**: time step / leak. δ→0 recovers continuous time. δ=1 means no leak.
- Lyapunov exponents are rates **per unit time** (divided by mδ), so they are comparable across δ.
- The couplings are **asymmetric** (J_ij and J_ji independent), and the network is **autonomous** (no input), **untrained** and **dense** (all-to-all).

## 2. Method in three parts

| Part | Statement | Exact? | Why we care |
|---|---|---|---|
| 1. Cumulative distribution as response (§2.1) | Shift every exponent down by s (multiply Jacobians by e^{-δs}). Drive the shifted tangent dynamics v(n+1)=M_s(n)v(n)+I(n) with a source at one time. The **minimum-norm** solution over all n∈ℤ has one-step response ∂v(k+1)/∂I(k) = P(k+1), the projector onto the decaying subspace. So **F(s) = (1/N) tr ∂v(k+1)/∂I(k)** = fraction of exponents < s (Eq. 22). | **Exact at finite N** | Model-agnostic. It applies to any differentiable map with invertible Jacobians: RNNs, looped transformers, deep residual nets read as depth-time maps. |
| 2. Regularized forward–backward system (§2.2) | Write the min-norm solution as the η→0⁺ limit of ridge least squares, argmin ‖K V − I‖² + η²‖V‖². Its stationarity conditions form a **2N-dim forward–backward** system (forward field v, backward field u). This is "Hermitization" (Feinberg–Zee). | Exact for η>0 | Computable with JVPs (forward) and VJPs (backward), i.e. ordinary autodiff, using CG on KᵀK+η². |
| 3. Cavity / single-site theory (§2.3) | Add one neuron and take N→∞. The rest of the network acts on it through the standard DMFT field ξ₀ and **two kernels A(n,n'), P(n,n')** found self-consistently (Eqs. 54–66). F(s) is read off as ⟨R^vv₀₀(k+1,k)⟩. | Main assumption: **the limits N→∞ and η→0⁺ can be exchanged** (backed by simulations, not proven) | A cheap predictor: the cost of solving it doesn't grow with N, so it can choose (g, δ) at any width. |

Code-level note (App. D.1): the single-site solve uses a periodic window of m steps (mδ = 32–64 time units) and 1024–4096 gain trajectories sampled with scrambled Sobol. η is annealed from 0.1δ down to 1.6×10⁻⁴δ. Near s_u = δ⁻¹log(1−δ) it switches to twisted boundary conditions. Each gain trajectory needs an m×m inverse.

## 3. Results we can use

Checks against known results (Table 1, App. A–B):

- **Trivial fixed point (g<1):** F(s) is the fraction of the circular-law disk |μ|<g that falls inside the disk |μ − c_δ| < r_s, with c_δ = 1 − 1/δ and r_s = e^{δs}/δ. At δ=1, F(s) = e^{2s}/g². As δ→0 it becomes a semicircle law. The paper also gives an electrostatic (Gauss's law) proof.
- **Maximum exponent in the chaotic phase:**
  - δ=1: **λ₁ = log(g·√C^d(0))**, with C^d(0) = ⟨φ'(x)²⟩ from DMFT. This is closed form and cheap.
  - δ→0: λ₁ = −1 + √(1−E₀), where E₀ is the ground state of the Schrödinger operator −d²/dτ² + 1 − g²C^d(τ).
  - Finite δ: λ₁ = s*, the largest s at which 0 is in the spectrum of the operator T_s (Eq. B.8).

Numbers from the figures (theory matches N=4096 simulations; **values read off the plots by eye, approximate**):

| Quantity | δ→0 | δ=0.25 | δ=0.5 | Source |
|---|---|---|---|---|
| Fraction of positive exponents, g=3 | ≈0.033 | ≈0.037 | ≈0.046 | Fig. 1b |
| Fraction of positive exponents, g=5 | ≈0.040 | ≈0.046 | ≈0.062 | Fig. 1d |
| λ₁ at g=3 / g=5 | ≈0.22 / ≈0.42 | similar | similar | Fig. 1b,d |
| D_KY/N plateau at large g | ≈0.09 | ≈0.11 | peaks ≈0.14 near g≈10, then falls | Fig. 2a |
| h_KS/N at g=20 | ≈0.016 | ≈0.022 | ≈0.036 | Fig. 2b |

Qualitative findings to carry forward:

1. **Chaos is extensive.** D_KY and h_KS grow linearly with N. Even so, the attractor fills only about 10% of state space in continuous time.
2. **The step size δ is a real knob.** Larger δ (a larger residual step) gives more positive exponents, a higher entropy rate and a higher D_KY. At large g, D_KY can *decrease* again (δ=0.5).
3. **Participation ratio ≠ attractor dimension.** PR depends on coordinates (PR^φ ≠ PR^x). D_KY and h_KS don't. *Don't use PCA dimension as a proxy for "how much the dynamics compute".*
4. **The linear-equivalence trap.** The covariance of this nonlinear network equals the covariance of a *stable linear* network driven by noise (refs [35–37]), yet the two have opposite λ₁. Second-order statistics can't tell you whether a network is chaotic.

## 4. Assumptions and limits: what does NOT transfer directly

Before using this result in any architecture, keep these in mind:

- **Random, i.i.d., untrained weights.** Trained weights have structure. The paper lists extensions as *straightforward but not done*: multiple populations, **a low-rank component** and correlated reciprocal couplings J_ij/J_ji. The low-rank case matters most for us, because trained RNNs are often modeled as random plus low-rank (ref [13]).
- **Autonomous dynamics.** Real models are driven by input, and input is known to **suppress chaos** (refs [18–20]). The input-driven spectrum isn't computed here.
- **Ergodicity, a single invariant measure, and invertible Jacobians** are assumed. ReLU with dead units makes Jacobians singular. tanh/GELU-like smooth activations are safer for this theory.
- **N→∞ and η→0⁺ are exchanged.** Not proven. A finite window with finite η introduces bias (we see it, §8).
- **Transformers are not recurrent in time.** The theory applies to (a) RNN/LSTM-style token recurrence, (b) **looped/universal transformers** iterating a tied block (ref [49]), and (c) depth read as time *if* weights are tied or statistically stationary across layers. Attention maps that depend on the input aren't i.i.d. couplings.
- **Linear SSMs (Mamba, RWKV, linear attention) have trivial spectra.** The state Jacobian is a diagonal gate, so the exponents are just average log gates. The chaotic-phase theory has nothing to add there. It applies only once the recurrence is nonlinear in the state.

## 5. Translation table: paper → modern architecture

| Paper | Architecture analog | Design implication |
|---|---|---|
| Euler step x ← (1−δ)x + δ·Jφ(x) | Residual block with scaled branch x ← x + δ·f(x) (DeepNorm, 1/√L or 1/L branch scaling, ReZero) | δ is the residual-branch scale. The theory predicts how δ changes chaos/expansion *per unit "time"*. |
| g (coupling std ×√N) | Init scale / spectral norm of the recurrent or tied weight | g≈1 is the edge of chaos. g sets λ₁ through the closed form at δ=1. |
| Gains d_i(n) = φ'(x_i) | Activation derivatives, gating | C^d(0) = ⟨φ'²⟩ enters λ₁. Nonlinearity choice is a spectrum knob. |
| Time n | Token step (RNN), loop iteration (looped transformer), layer index (tied deep net) | Pick the axis we probe explicitly. Never mix them. |
| F(s), D_KY, h_KS | Probe metrics on a trained or initialized model | Log them next to loss (lab rule: diagnostics over vibes). |
| Min-norm solution ↔ dichotomy projector | Ridge least-squares over a window, using JVP + VJP + CG | Matrix-free F(s) estimator (Hutchinson trace). |
| Forward–backward 2N system | Same structure as backprop-through-time (forward state, backward adjoint) | Fits existing autodiff stacks. |

## 6. Candidate levers, ranked by simplicity × leverage

| # | Lever | Paragraph-simple? | What would count as signal | Risk |
|---|---|---|---|---|
| L1 | **Spectrum probe toolkit.** QR spectrum for small N. Matrix-free F(s) via the Part-1 identity (ridge LS + Hutchinson) for large N. Report λ₁, F(0), D_KY/N, h_KS/N. | Yes | Matches the paper's Fig. 1–2 at N=4096. The matrix-free estimate agrees with QR within a stated error. | Low. This is a diagnostic. Hutchinson variance is the main issue (§8). |
| L2 | **Theory-set init.** Choose (g, δ, φ) so that λ₁ at init is ≈ 0⁺ (the edge of chaos), using the DMFT closed form at δ=1 or T_s at finite δ. | Yes | Same or better loss at a fixed token budget compared with standard init, on a tiny looped/RNN LM, over ≥3 seeds. | Medium. "Edge of chaos" init is already known (refs [45–47]); what's new is that the *full* spectrum and δ are targeted. Novelty may be small. |
| L3 | **Spectrum-shaping regularizer.** Penalize the deviation of F(s) (or of the top-k exponents) from a target profile during training, generalizing gradient flossing [48]. | Mostly | Better long-range dependency probes (copy/induction at long lag) at equal compute. | Medium–high. Extra compute per step, and the regularizer could fight the task loss. |
| L4 | **δ as an architectural hyperparameter in looped models.** Sweep the residual-branch scale with theory as a prior. Expect a peak in D_KY at intermediate g for large δ. | Yes | A δ-vs-quality curve whose shape the theory predicts *before* training. | Low. A cheap ablation. |
| L5 | **Random + low-rank core.** Freeze a random bulk J at the theory-chosen g and train only a low-rank component plus a readout (reservoir-style, refs [13–15]). | Yes | Large reduction in trainable parameters at matched quality on small tasks. | High. Reservoirs tend to underperform trained nets on language. Keep it as an efficiency/edge-device bet. |
| L6 | **Lyapunov halting for looped reasoning.** Use the local expansion rate (a short-window F(s) or λ₁ estimate) as an adaptive-compute stop signal. Ref [49] reports transient chaos on harder tasks. | Mostly | Accuracy vs. loop-count Pareto improves over fixed-T or learned-halting baselines. | High. Speculative, and the estimator may cost more than the loop iterations it saves. |

## 7. Working architecture hypothesis

> Superseded by the first-principles design in [`edge-loop-design.md`](edge-loop-design.md), which replaces the g-based init with norm-gain / injection dials (weight scale is cancelled by pre-norm).: "edge-loop" (name TBD)

*Hypothesis, not a result. Every claim here is UNVERIFIED until §9 gates pass.*

```
embed(tokens) → h₀
for t in 1..T:                       # tied weights: a dynamical system in t
    h_t = (1−δ)·h_{t−1} + δ·Block(h_{t−1}; input injection)
    [optional] probe: local expansion estimate → halt?
readout(h_T) → logits
```

- **Block:** a small transformer or MLP block with a smooth activation (tanh/GELU, not ReLU, so Jacobians stay invertible as §4 requires).
- **Init:** g and δ chosen from the single-site theory so that λ₁(init) ≈ 0⁺ along the loop axis (L2).
- **Training:** standard LM loss + λ_reg × spectrum-shape penalty, applied to a sparse subset of steps to cap cost (L3).
- **Measurement:** every checkpoint logs λ₁, F(0), D_KY/N and h_KS/N along the loop axis, next to the loss (L1).
- **Scale for first tests:** ≤10M params, CPU/MLX-friendly. This fits the lab's existing tiny/AMX track (`ideas/amx-class-small-moe.md`) and the M4 benchmarks in `experiments/orinth-clef-unified`.

Falsifiable predictions:

1. P1: At init, the measured λ₁ and D_KY/N of the looped block track the theory's (g, δ) predictions to within the error of the i.i.d. approximation. *Expected to fail partially*, because attention isn't i.i.d. How large the gap is is itself a finding.
2. P2: Compared with a standard-init baseline with matched params and tokens, theory-set init reaches the same loss in fewer steps, or improves a long-lag probe, over ≥3 seeds.
3. P3: The spectrum regularizer improves long-lag probes without hurting short-context loss by more than X% (**TODO**: set X before running).

If P2 and P3 both fail, we keep L1 (the diagnostic) as the publishable artifact and park the architecture.

## 8. Our own verification so far (this intake)

Script: `experiments/lyapunov-spectrum/check_identity.py` (numpy only). It builds K (Eq. 16) on a finite window of m steps around a chaotic trajectory, takes the min-norm solution with `pinv`, and compares tr(P)/N with the fraction of QR exponents below s.

Run: `N=64, g=4, δ=0.5, window m=100, 16 Rademacher probes, seed 0` (~6 min, CPU):

```
lambda_1 = 0.188, #positive = 4 / 64
s=-1.00  QR F(s)=0.672  identity tr/N=0.643  hutch(16)=0.649
s=-0.40  QR F(s)=0.844  identity tr/N=0.834  hutch(16)=0.800
s=-0.10  QR F(s)=0.922  identity tr/N=0.893  hutch(16)=0.831
s= 0.05  QR F(s)=0.953  identity tr/N=0.935  hutch(16)=0.982
s= 0.15  QR F(s)=0.984  identity tr/N=0.955  hutch(16)=0.891
s= 0.30  QR F(s)=1.000  identity tr/N=0.996  hutch(16)=0.996
```

A non-chaotic control (N=32, g=3, δ=0.5; this finite network settled into a non-chaotic state with λ₁ = −0.118) matched exactly at s ∈ {−0.8, −0.3, 0, 0.1}. It was off by 0.025 at s=−1.5.

Takeaways:

- The **identity holds**: within ~0.03 on a finite 100-step window. The remaining gap is consistent with window truncation (the infinite-window statement is exact). **TODO**: confirm the gap shrinks as m grows.
- A **16-probe Hutchinson estimate is too noisy** (errors up to ~0.09) to resolve the few-percent fraction of positive exponents that matters here. The matrix-free path needs variance reduction (more probes, Hutch++, or deflating the top-k directions), or else we use QR for the top-k exponents and the identity only for the bulk.
- At N=64, g=4, δ=0.5 we get 4/64 ≈ 6% positive exponents, the same order as the paper's ~4–6% at g=3–5 (N=4096). Finite-size effects are expected; this isn't a reproduction.

## 9. Experiment plan with go/no-go gates

| Phase | Goal | Compute | Gate to proceed |
|---|---|---|---|
| 0 | Reproduce Fig. 1 (spectrum) and Fig. 4c (λ₁ vs g) for δ ∈ {0.25, 0.5, 1}, N ≤ 4096, using QR simulation plus the closed-form λ₁ at δ=1. Pull upstream code if its license allows. | CPU, hours | Our curves overlap the paper's within visual tolerance. Discrepancies get documented. |
| 1 | Build L1: a matrix-free F(s) estimator (JAX: JVP/VJP + CG on KᵀK+η², Hutch++). Validate against QR at N ∈ {256, 1024, 4096}. | CPU/1 GPU, hours | Error ≤ 0.01 in F(0) at an affordable probe count. If it misses, fall back to QR-top-k. |
| 2 | Probe real models: (a) a tiny GRU/tanh-RNN LM along tokens, (b) a looped transformer along loop iterations, (c) a tied-depth residual MLP along depth. At init and over training. | CPU/1 GPU | We find spectrum changes during training that the loss curve doesn't show. That alone is publishable as a diagnostic. |
| 3 | L2 + L4 ablation on a ≤10M looped LM: theory-set (g, δ) vs standard init, ≥3 seeds, fixed token budget, long-lag probes. | **Escalate:** small GPU budget | P2 holds at p<0.05 across seeds, or we write up a clean negative. |
| 4 | L3 regularizer ablation. | **Escalate** | P3 holds. |
| 5 | Publish per `hub/PUBLISH.md`: Space (spectrum probe demo), dataset (measured spectra of public small models), model only if P2/P3 hold. | — | License chosen, cards complete, human OK for the Svyable org. |

## 10. Measurement traps

- **Units.** Report exponents per unit time (÷ δ) when comparing across δ, as the paper does. Per-step exponents mislead.
- **Transients.** The paper discards ≥300 time units, re-orthonormalizes every 0.5 time units, and averages over 2000–8000 time units. Short runs bias λ₁.
- **Near onset (g ≲ 1.5)** only a handful of exponents are positive, even at N=4096. The paper goes to N=8192–16384 and takes medians over networks.
- **PR vs D_KY.** Don't report the PCA participation ratio as "dimension of computation"; it depends on coordinates.
- **Linear equivalence.** Matching covariances says nothing about chaos.
- **Input drive.** Spectra measured with input are not the autonomous spectra. Always state which one was measured.
- **Window bias** in the identity estimator (§8). Report m and η.
- **Hutchinson variance.** Always report the probe count and a CI.
- **Non-invertible Jacobians** (ReLU dead units, hard gates) break Part 1's assumption.

## 11. Decisions for the user / Chief of Staff

1. **Compute budget** for Phases 3–4 (GPU). Not needed for Phases 0–2.
2. **License** for our code and any reuse of `davidclark1/rnn-lyapunov` (license **TODO**).
3. **HF publishing as Svyable**: needs an authenticated HF connector. None has been tested in this session yet.
4. **Track priority** relative to the current `undertrained-token-decode-mask` bet.

## 12. Honesty note on "leapfrogging"

This paper is a deep, clean theoretical result. A frontier-beating model does not follow from it. The realistic first win is a **reusable diagnostic that is invariant to coordinate choice** (L1) plus a **cheap, theory-set init/step-size ablation** (L2/L4) at tiny scale. The architecture in §7 is worth building only because each piece can be falsified cheaply. We'll make claims about beating anyone only after §9 gates pass, with seeds and baselines.

## References

Numbers are the paper's reference numbers. Every entry is as cited in arXiv:2610.12426; we have not independently checked these papers yet.

- [1] Sompolinsky, Crisanti, Sommers. Chaos in random neural networks. PRL 1988.
- [13] Mastrogiuseppe, Ostojic. Low-rank recurrent networks. Neuron 2018.
- [14] Sussillo, Abbott. FORCE learning. Neuron 2009.
- [18–20] Schuecker et al. 2018; Engelken et al. 2022; Molgedey et al. 1992 (input suppresses chaos).
- [22] Engelken, Wolf, Abbott. Lyapunov spectra of chaotic recurrent neural networks. PRR 2023 (numerics baseline).
- [23] Benettin et al. 1980 (QR method).
- [30, 31] Clark et al. 2023, 2025 (participation-ratio dimension).
- [35–37] Shen & Hu 2025; Wakhloo 2026; Clark 2026 (linear equivalence).
- [45–47] Poole et al. 2016; Schoenholz et al. 2017; Cowsik et al. 2025 (edge of chaos / signal propagation and trainability).
- [48] Engelken. Gradient flossing. NeurIPS 2023.
- [49] Lai, Bao, Quinn, Gilpin. Fractal basins trap latent reasoning. arXiv:2609.04963 (looped models, transient chaos).
- [50] Gurnee et al. Verbalizable representations form a global workspace in language models. Transformer Circuits 2026.
- [51–54] Hüls; Froyland et al.; Sacker–Sell (dichotomy projectors and spectra).
- [55] Deninger 2011 (Fuglede–Kadison determinant, used in the original AI derivation).
- [56] Feinberg, Zee 1997 (Hermitization).
