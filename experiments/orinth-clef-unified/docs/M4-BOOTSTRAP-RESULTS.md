# Apple M4 / 24 GiB bootstrap measurement — 2026-10-09

**Result: teacher and student MLX execution proven; typed student output gate FAILED.**
These measurements are from the owner's Mac, not synthetic timing estimates.

## Environment and provenance

- Machine: `Mac16,7`, Apple Silicon arm64, macOS 26.1; 24 GiB unified memory.
- Disk headroom before downloads: ~59 GiB; remaining after setup: ~46 GiB.
- Teacher: `mlx-community/clef-flash-4bit` revision `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1`, ~5.8 GiB local folder.
- Student: `mlx-community/Qwen3-4B-4bit` revision `4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, ~2.1 GiB local folder.
- `mlx==0.32.3`, `mlx-vlm==0.7.4`, `mlx-lm==0.32.0`; Python 3.12.12 in independent virtual environments.
- No upload of teacher predictions, user personal files, model weights, or local adapters.
- Clef answer protocol includes `{"type":"noul","noul":p}`, confirmed by the actual `clef_mlx.py` and fixed in PR #5.

## Observed teacher results

- Real `/health` and `/v1/systemone` succeeded on `127.0.0.1:8001`.
- Warm-up probe: team=technical probability 0.7402; urgent true probability 0.9621; reported latency 620.9 ms.
- Twelve hand-written synthetic input scenarios collected without errors; all 12 had unique case and group IDs.
- Request latencies across those twelve: 560.6 ms minimum, 576.15 ms median, 610.2 ms maximum (model-reported).
- Data split: train=10, valid=1, test=1, held out by scenario group. Teacher outputs are **pseudo-labels**, not verified gold.
- Stopping the teacher released shared memory; macOS reported free-memory percentage rising from 34% to 64%.

## Observed student smoke training

```bash
.venv-student/bin/mlx_lm.lora \
  --model models/qwen3-4b-4bit --train --data data/mlx \
  --adapter-path runs/qwen3-4b-smoke \
  --batch-size 1 --grad-accumulation-steps 4 \
  --num-layers 4 --max-seq-length 512 \
  --learning-rate 1e-5 --mask-prompt \
  --grad-checkpoint --iters 10 --steps-per-report 1 \
  --steps-per-eval 10 --save-every 10
```

- Completed: 10/10 iterations, no Metal error. 1.835 million trainable parameters (0.046% of 4.022B).
- Validation loss: 2.721 initially, 2.099 at the last evaluation. The dataset is far too small to infer generalization.
- Adapter artifacts saved locally under `runs/qwen3-4b-smoke` (~7 MB each for final and checkpoint).
- Single held-out CLI generation loaded successfully, with MLX reporting 2.748 GB peak memory and 82.4 output tokens/s. **It produced malformed typed schema.**

## Held-out typed-output evaluation

```bash
.venv-student/bin/python -m orinth_clef.eval_student --splits valid test
```

- `python3 -m unittest discover -v`: **25 passing** on the M4 after adding the schema evaluator.
- Both held-out scenarios generated syntactically valid JSON with **wrong decision field shapes**.
- Valid schema: **0 / 2**. Exact match to teacher pseudo-labels under strict schema: **0 / 2**. Median per-item generation time ~0.702 s excluding model load.
- Examples of errors: a `choice` mapping where a string option ID was required; decision keys `choice` / `noul` instead of `team` / `urgent`.
- Do not report these as model accuracy or calibration, and do not claim the 4B checkpoint is a working Clef replacement.

## Next experiments, gated

1. Produce a larger, diverse teacher-labeled data corpus and add independent human-verified gold cases. Keep scenario/family holdouts.
2. Reassess whether structured constrained decoding and explicit per-schema validation can reduce formatting failures *without fabricating probability calibration*.
3. Train a longer reproducible adapter only after preparing meaningful data; benchmark against the original teacher and the unadapted Qwen baseline.
4. Do **not** fuse, ship or brand a unified trained model before schema correctness and held-out decision metrics pass.
5. A real joint-schema student head would be separate training; cannot transplant Clef's 9B head into Qwen3-4B.
