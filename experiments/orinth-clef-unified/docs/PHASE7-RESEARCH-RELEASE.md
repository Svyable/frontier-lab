# Phase 7 — Faster *and* smarter: failed challengers, reproducible model bundle

## Research question

Can we replace the multi-token DecisionLens trie with a faster inference
path **without losing rule-level correctness**, while preparing the
actual MLX weights and runtime for third-party benchmarking?

## Controlled M4 diagnostic

24 author-created, deterministic rule-labeled cases across support,
weather, inventory and risk. Two paired repetitions (48 observations).
These cases were previously examined; **not independent gold**.

| Runtime | Exact complete decisions | Warm median latency |
|---|---:|---:|
| DecisionLens trie | **42/48** | ~0.288–0.296 s |
| Reversible compact aliases | 36/48 | 0.312 s |
| Per-question direct classifier | 30/48 | **0.249 s** |

The compact-label scheme mapped long option names to single-token
letters while keeping the original names in the criteria. It was
**worse in both correctness and latency** on the mixed diagnostic.
On a 32-option prefix-heavy stress test it was also slower than the
trie (1.179s versus 0.975s), though both picked the intended option
in all three runs. The larger prompt negated the single-token benefit.

The direct classifier evaluated a separate short prompt for each
question and selected a one-token letter without JSON generation.
It was ~14% faster than the trie on this diagnostic, but lost 12
exact matches out of 48. It is **not suitable as a replacement**.

**Decision:** retain DecisionLens trie as the strongest measured
quality candidate; keep reference as default and trie as opt-in.
Do not cherry-pick speed-only experiments or call these moonshot
quality results.

## Hugging Face-compatible local research package

`python -m orinth_clef.hf_release export` builds a local folder with:

- Real fused Qwen3-4B 4-bit MLX safetensors and matching tokenizer/config
  (when `--include-weights` is supplied)
- Frozen DecisionLens reference/trie runtime, experimental compact
  runtime and separately marked micro-head weights (optional)
- HF-style model card with explicit caveats and no unsupported
  general SOTA claim
- `provenance.json` (source revision, teacher revision, data fingerprint,
  base model, runtime version)
- `manifest.json` with SHA-256 hashes and byte counts for each file
- `.gitattributes` configured for safetensors Git LFS
- Author-created rule diagnostic fixture and a standalone evaluation CLI

The export is **local-only**: no API uploads, repository creation,
license selection or public publishing. The model card explicitly
requires a rights review for the base, teacher-derived supervision
and this repository's own code license. Apache-2.0 metadata is
present on the Qwen and local Clef source references, but we
have not established all redistribution permissions.

### Build

~~~bash
.venv-student/bin/python -m orinth_clef.hf_release export \
  --model-dir models/qwen3-4b-60iter-4bit \
  --include-weights --head runs/phase4/micro-head-24.safetensors \
  --output runs/phase7/hf-weighted-v2

.venv-student/bin/python -m orinth_clef.hf_release verify \
  --directory runs/phase7/hf-weighted-v2

cd runs/phase7/hf-weighted-v2
PYTHONPATH=. ../../../.venv-student/bin/python -m orinth_clef.hf_benchmark \
  --model . --fixture challenge_rules.jsonl
~~~

The packaged model is an **MLX model repository**, not a native
Transformers checkpoint. Researchers on other platforms need a
separately tested conversion path; do not label it cross-platform
without verification.

## Prompt compression: a modest Pareto improvement

After testing compact option IDs and direct classification, we
compared three alternative prompt representations while keeping
the trie scorer and model weights fixed. On the previously inspected
24-case author diagnostic, two paired rounds:

| Prompt | Exact matches | Median latency |
|---|---:|---:|
| Baseline | 42/48 | 0.288 s |
| Short system + compact JSON | 40/48 | 0.210 s |
| Baseline system + compact JSON | 38/48 | 0.248 s |
| **Short system + baseline JSON** | **44/48** | **0.249 s** |

We selected the last variant before running a new 32-case suite
of library, vehicle, access and shipment decisions. On this suite,
the short-system variant achieved **52/64** exact complete matches
versus **50/64** for the baseline, and **0.263s** versus **0.305s**
warm median latency (two paired repetitions). This is a small
improvement in both quality and speed, but **not** independent
human-adjudicated accuracy or evidence of general SOTA. The new
suite was examined once after selection, and should not be reused
to tune further variants.

Run:

~~~bash
.venv-student/bin/python -m orinth_clef.prompt_benchmark \
  --style short_system --fixture examples/holdout_rules_v1.jsonl \
  --rounds 2 --output runs/phase7/holdout-short-system.json
~~~

On the previously inspected 24-case Clef teacher pseudo-label
regression, the shorter prompt also measured **18/24** teacher
agreement versus **17/24** for the baseline, with 24/24 schema
validity and ~0.277s median. These labels are not human gold.

The original prompt remains the default; the shorter one is opt-in
through `prompt_style="short_system"`.

## In-place KV-cache rollback: parity, not a speed breakthrough

We prototyped speculative in-place cache updates with offset
rewind rather than deepcopy at each trie fork. The original cache
offsets are restored even on exceptions. On the 24-case diagnostic
(48 paired runs), the rollback scorer matched all reference
decisions with maximum normalized score difference 0.00713;
median latency was 0.28718s versus 0.28772s reference.

On the 32-option shared-prefix stress, rollback measured
**0.967s** versus **0.971s** for the copying trie, with
identical decisions and zero score drift across three rounds.
That is not a meaningful speed improvement. It is explicitly
experimental, non-thread-safe, dependent on MLX KVCache internals,
and **not** the recommended default.

## Genuine moonshot gates

1. Collect independently annotated, domain-diverse human gold, with
   family-disjoint untouched test sets. Separate teacher imitation,
   true task accuracy, calibration and out-of-distribution behavior.
2. Train a query-aware multi-question decision head against gold
   and soft teacher distributions, not only argmax pseudo-labels.
3. Evaluate a shared-prefix trie with batched KV branches and
   memory-aware kernels; measure p50/p95, peak RAM, tokens/s and
   correctness on the same supported tasks.
4. Publish only after model rights review, reproducible artifacts,
   independently runnable benchmark scripts and honest model cards.
5. A new model must improve **both** correctness and latency against
   the same reference under comparable hardware and task conditions,
   or explicitly describe its Pareto trade-off.
