# Phase 2 measured results — Apple M4, 24 GiB (2026-10-09)

**Status: 120-case teacher capture, base/smoke/60-step adapter comparison complete.**

## Data

- 30 hand-authored synthetic scenario families × 4 paraphrase templates = **120 cases**, with 240 individual typed decisions.
- Pinned Clef-Flash teacher revision: `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1`.
- 120/120 teacher calls completed in **74.39 seconds** (including orchestration).
- Group-held-out split: **96 train, 12 validation, 12 test**. No family is shared between splits.
- Compared with *unverified synthetic construction intent*: choice matches **120/120**; noul matches **111/120**. This is NOT a human-verified accuracy result.
- Mean probability of selected choice: **0.96856**; mean noul probability: **0.46307**. These are teacher outputs, not calibrated performance metrics.
- Raw teacher responses, construction intents, and all trained weights remain on the owner's Mac in ignored directories.

## Controlled student comparison (same 24 held-out cases)

| Candidate | Strict schema valid | Exact teacher pseudo-label agreement | Median generation time |
|---|---:|---:|---:|
| Unadapted Qwen3-4B 4-bit | **21/24 (87.5%)** | **15/24 (62.5%)** | **0.550 s** |
| Previous 10-step LoRA | **21/24 (87.5%)** | **13/24 (54.2%)** | **0.526 s** |
| **New 60-step LoRA** | **24/24 (100%)** | **16/24 (66.7%)** | **0.486 s** |

The previous 10-step LoRA **did not outperform the unadapted base** on this dataset. The new 60-step LoRA improves schema validity by **3 cases** and teacher pseudo-label exact agreement by **1 case** compared with the base. Paired outcomes: 2 newly matching cases, 1 case regressed. On validation and test separately the new adapter achieved 12/12 schema-valid in each, with 8/12 teacher matches in each. The base achieved 11/12 and 10/12 schema-valid, with 9/12 and 6/12 teacher matches respectively.

The strict evaluator rejects malformed JSON schemas instead of silently repairing them. These figures are **teacher imitation** and schema-format measures, not independent accuracy or model-calibration measures. The evaluation set is small, synthetic, and domain-limited. A one-case gain in teacher agreement is too small to establish superiority.

## New 60-step adapter configuration

~~~bash
.venv-student/bin/mlx_lm.lora --model models/qwen3-4b-4bit \
  --train --data data/mlx_120 --adapter-path runs/phase2/qwen3-4b-60iter \
  --batch-size 1 --grad-accumulation-steps 4 --num-layers 4 \
  --max-seq-length 512 --learning-rate 1e-5 --mask-prompt \
  --grad-checkpoint --iters 60 --steps-per-report 10 \
  --steps-per-eval 20 --save-every 60
~~~

- **60/60 iterations completed** on Apple M4 in **38.99 seconds**, 1.835M trainable parameters, adapter ~7 MB.
- Validation loss: **2.235 initially → 1.052 at step 20 → 0.166 at step 40 → 0.061 at step 60**. The dataset has repeated templates, so low loss does not demonstrate generalization.
- Student teacher-agreement and schema-validity reports stored locally under `runs/phase2/`; adapter under `runs/phase2/qwen3-4b-60iter/`.
- Clef server stopped after capture; memory free was ~72% after inference/training processes exited.

## Next gated step

Preserve the new adapter as a **candidate**, not a shipped model. Expand to genuinely different tasks and independently audited labels, add general-language regression tests, and benchmark reliability under schema variations before fusing weights or exploring further compression. A dedicated student-compatible probabilistic decision head remains a separate research milestone.
