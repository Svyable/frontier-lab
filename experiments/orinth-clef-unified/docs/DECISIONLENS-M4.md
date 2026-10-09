# DecisionLens — schema-native logit selection (M4 prototype)

## Moonshot hypothesis

For narrow typed decisions, an autoregressive language decoder may be unnecessary.
Instead, compile the caller's allowed values into candidate token IDs, score those
IDs from the frozen language model's next-token logits, and construct the typed
JSON in the runtime. This is a **processing innovation**, not a new trained model
architecture, not a new general-purpose reasoning SOTA claim.

## Current implementation

- Qwen3-4B 4-bit + 60-iteration LoRA, ~2.1 GiB base weights + ~7 MiB adapter.
- Dynamic schema keys; arbitrary **single-token** choice IDs and Boolean noul.
- One prompt prefill plus a shared MLX KV cache to score subsequent question fields.
- **Zero autoregressively generated output tokens**. A single model forward pass
  per question scores the next allowed token; the next field consumes only
  the selected token and its JSON delimiter through the cache.
- Deterministic selection. Strict schema parsing of the constructed response.
- Multi-token choice IDs fail closed; this prototype does not silently score
  only the first token or claim support for all possible schemas.
- Normalized candidate logits are token preference scores, **not calibrated
  probabilities of events**, Clef-compatible joint probabilities, or uncertainty.
- No native custom Metal kernel yet; MLX executes standard kernels.

## M4 benchmark (24 held-out synthetic cases; 12 valid, 12 test)

| Runtime | Valid schema | Teacher pseudo-label exact agreement | Median warm per-case |
|---|---:|---:|---:|
| Autoregressive JSON, 60-step LoRA | 24/24 | 16/24 | 0.4855 s |
| DecisionLens, naive full re-prefill per field | 24/24 | 16/24 | 0.5412 s |
| **DecisionLens, shared KV-cache** | **24/24** | **16/24** | **0.2823 s** |

Observed **1.72× lower median per-case latency** for cached DecisionLens vs
generative JSON on this workload, or ~42% less latency. Warm in-process
latency excludes model loading, process startup, disk IO, batching and server
overhead. Small, narrow synthetic benchmark; no claim of generalized SOTA.

Run:

~~~bash
.venv-student/bin/python -m orinth_clef.decision_lens \
  --data data/mlx_120 --adapter runs/phase2/qwen3-4b-60iter \
  --splits valid test --output runs/phase3/decision-lens-kvcache.json
~~~

## Compression ablation — quality matters more than bit count

The 60-step adapter was fused to a local bf16 checkpoint (~7.5 GiB),
then independently quantized using MLX-LM. These are **local experimental
artifacts**, not releases.

| Variant | Local model folder | Schema | Teacher pseudo-label matches | DecisionLens median |
|---|---:|---:|---:|---:|
| Original Qwen 4-bit + 7 MiB LoRA | ~2.1 GiB + adapter | 24/24 | 16/24 | 0.2823 s |
| **Fused 4-bit** | **~2.1 GiB** | **24/24** | **17/24** | **0.2789 s** |
| Fused 3-bit (3.501 effective bpw) | ~1.7 GiB | 24/24 | **8/24** | 0.2797 s |
| Fused mixed 3/4-bit (3.624 effective bpw) | ~1.7 GiB | 24/24 | **6/24** | 0.2797 s |

**Do not ship 3-bit or mixed 3/4-bit.** The ~19% size reduction caused
large disagreement on these 24 teacher pseudo-labels without measured
latency benefit. A 3-bit autoregressive evaluation was stopped because
it was taking disproportionately long; no completed generative metric
is claimed for that variant. Fused 4-bit is the provisional candidate
for additional validation. The +1 exact teacher match vs unfused adapter
is within noise and not a quality improvement claim.

## Most promising next experiments

1. **Learned micro-head**: cache hidden representations and train a compact
   question-conditioned decision projection. Target one prefill for all
   questions; use truly held-out task families to detect overfit.
2. **Trie-based multi-token scoring**: support arbitrary schema option IDs
   with a shared KV prefix and per-branch scoring, without JSON decoding.
3. **Early exit / dynamic depth**: identify cases where intermediate layers
   preserve held-out decisions, and fall back to full depth when uncertain.
   Requires access to intermediate activations and reliable routing gates.
4. **Mixed 2/3/4-bit quantization**: measure exact memory, latency and held-out
   quality with immutable model revisions and reversible artifacts.
5. **Fused Metal / batched field scoring**: profile prefill vs candidate scoring
   and benchmark against existing MLX optimizations. Speed must be measured,
   not assumed from compression ratios.
6. **Native compact model package**: include schema ABI, tokenizer revision,
   quantization map, calibration provenance and benchmark fixture hashes.

Related external research: Apple QuantSpec (2025), MLX-LM prompt caching and
quantization, and independent experimental MLX KV compression projects. The
reported speedups in those papers/projects are not measurements of DecisionLens.
