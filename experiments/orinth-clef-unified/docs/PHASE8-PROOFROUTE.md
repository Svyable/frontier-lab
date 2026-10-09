# Phase 8 — ProofRoute: audited symbolic fast path + neural fallback

**Research status:** working prototype, not an AI-model breakthrough or SOTA.
The primary innovation hypothesis is an *adaptive decision system* that
spends model compute only on decisions not covered by trusted, explicit
machine-checkable policies. Symbolic execution is established prior art.
The question is whether a hybrid architecture can dominate modern
classifier, LLM, and rule-engine baselines on **real mixed workloads**.

## Semantics

`orinth_clef.proofroute` accepts the existing typed SystemOne task
schema and an optional separately supplied policy AST. The policy must
cover **every** question; an incomplete or invalid policy is rejected.
Supported Boolean operators: `and`, `or`, `not`; comparisons:
`eq`, `ne`, `gt`, `ge`, `lt`, `le`, with typed literal values or
explicit same-state field references. Choice policies map true/false
to allowed choice IDs; Boolean questions return real Booleans.

Safety boundaries:
- No Python `eval`, imports, dynamic function calls or arbitrary code.
- Bounded expression depth and node count; no implicit string coercion.
- Missing fields, invalid types or unsupported data lead to **abstention**
  or an explicitly provided neural fallback, not invented decisions.
- All-or-nothing execution avoids silently mixing partial results.
- Policy authorship and review are external; **natural-language
  instructions are never interpreted as authoritative rules**.
- Returned policy decisions are deterministic conditional on trusted
  policy and state. They are not neural probabilities or calibrated
  confidence scores.

## Benchmark design

`python -m orinth_clef.proofroute_benchmark --rounds 2`

Uses the previously inspected 32-case author-rule fixture across four
families. The four rule templates are **hand-coded from the task
specification**, not learned or inferred by the model, and the same
template is reused for each family's eight cases. The baseline is
the fused 4-bit Qwen3 DecisionLens trie with the short-system prompt.
Model is loaded before timing; paired orders alternate. The rule
measurement includes validation + compilation + execution on every
case (not just an already compiled policy). Results **do not measure
rule discovery cost, policy authoring, cold-start, model loading or
real-world rule maintenance**.

### Measured result on Apple M4, 24 GiB

| Path | Complete exact | Median per case |
|---|---:|---:|
| Hand-written Python rules | **64/64** | **0.00071 ms** |
| Explicit policy, validate + compile + execute | **64/64** | **0.01321 ms** |
| DecisionLens trie, short system prompt | 52/64 | 270.10 ms |

The measured policy fast path is about **20,000x** faster per
rule-covered case than the always-on neural model, but about
**19x slower than straightforward hand-written Python** on these
microsecond-scale measurements. This is **not a faster neural model**. It
is a deterministic rule engine executing a rule already supplied
by the caller. Both timings exclude cold model loading, and the
policy timing excludes human rule creation, audit and maintenance.
No advantage over optimized direct Python/SQL rule execution has
been measured.

This is not an independent gold benchmark and not a fair claim that
an AI model has beaten another model. A correct explicit program
should beat an LLM at its own deterministic rule.

## Genuine next-step experiment

Create an independently sourced mixed workload with explicit
machine-verifiable policies for some tasks and genuinely ambiguous
human-annotated tasks for others. Evaluate against:
1. Direct compiled Python/SQL rule engine (critical baseline).
2. Small fine-tuned encoder classifier (CPU and M4).
3. DecisionLens 4-bit LLM, always-on.
4. ProofRoute hybrid, including policy coverage/maintenance cost.
5. Modern 4B model and grammar-constrained generation.

Publish coverage, **selective accuracy at coverage**, abstention,
end-to-end p50/p95, peak memory, energy, and model bytes. Train a
real uncertainty-aware router using train/valid data; reserve a
family-disjoint, untouched human-labeled test set. Only call this a
breakthrough if it beats the *best conventional hybrid baseline*,
not merely the always-on LLM.

The Hugging Face-style MLX research bundle includes the reference
ProofRoute runtime as an **optional** policy interface; the MLX model
itself remains the same fine-tuned Qwen checkpoint.
