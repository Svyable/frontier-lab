# Idea: undertrained-token-decode-mask

- **Status:** exploring
- **Constraint:** decode-time / eval-only; zero training spend
- **Related work:**
  - Land & Bartolo, *Fishing for Magikarp* (EMNLP 2024) — detecting under-trained tokens
  - [gdiamos/amx-reasoning-v1-instruct](https://huggingface.co/gdiamos/amx-reasoning-v1-instruct) vocabulary mask (Diamos et al., Sep 2026) — masking untrained rows fixed ~29% of DROP answers being garbage tokens and lifted EM 15.3% → 18.2% on that checkpoint (upstream numbers; not ours)

## Hypothesis

Many HF causal LMs have undertrained vocab rows that win greedy argmax under uncertainty. A reusable detect → mask → measure probe can show large measured gains for almost no complexity — a paragraph-simple trick that loss curves hide.

## Why it might matter

Applies across models and sizes; no architecture change; clear A/B probe. Fits Frontier Lab’s bar: simple, measurable, public.

## Minimal test

On 2–3 small public instruct models:
1. Detect candidate undertrained tokens (embedding / unembedding indicators; Magikarp-style).
2. Build a fixed short-answer / extractive probe set.
3. Compare greedy decode **mask off vs mask on**: garbage-token rate + task EM/F1.
4. Report per-model deltas only — no invented universal multipliers.

## Risks / measurement traps

- False positives that ban rare-but-useful tokens
- Tied vs untied embeddings need different indicators
- Gains may be tiny on well-covered vocabularies; that is a valid negative result
- Never claim “10×” without measured evidence

## Next step

See `experiments/undertrained-token-mask/` for the probe plan. Implement detection + eval scripts after plan review; Hub Space only once HF auth is ready.
