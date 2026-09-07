# Experiment: undertrained-token-mask

**Idea:** `ideas/undertrained-token-decode-mask.md`  
**Status:** plan only (no runs yet)  
**Spend:** none planned until scripts exist and a tiny eval is approved

## Goal

Ship a reusable probe that (1) finds likely undertrained token ids on a HF causal LM and (2) measures greedy quality with those ids masked vs unmasked.

## Models (candidates)

Pick small, license-clean instruct checkpoints already on Hub. Exact ids TBD at run time; prefer &lt;3B so the probe is cheap. Record `revision` pins.

## Probe set

- Fixed N short-answer / extractive items (target N≈100–200)
- Same prompts for mask-off and mask-on
- Greedy decode only for the primary comparison

## Metrics (honest)

| Metric | Definition |
| --- | --- |
| Garbage rate | Fraction of generations whose first content token is in the undertrained set (or non-digit when digits required) |
| EM / F1 | Task-appropriate exact match / token F1 on answerable rows |
| Ban list size | Number of masked ids |

Mark anything not yet measured as `TODO` / `UNVERIFIED`.

## Detection sketch

1. Load `embed_tokens` and `lm_head` (handle tied weights).
2. Rank candidate undertrained rows (norm / cosine-to-mean style indicators per Magikarp).
3. Optional verification prompts for low-prob candidates.
4. Export `banned_token_ids` JSON (same role as upstream `generation.json` mask).

## Deliverables

- [ ] `detect_undertrained.py` (or notebook) + pinned deps
- [ ] `eval_mask_ab.py` writing a small JSONL results table
- [ ] This README filled with measured numbers or explicit negatives
- [ ] Optional HF Space later (needs HF connector auth)

## Non-goals

- Retraining embeddings
- Claiming SOTA
- Burning GPU budget before CoS/user OK
