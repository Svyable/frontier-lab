# Phase 2 — reproducible teacher/student benchmark on 24 GiB M4

## Goal and constraints

Increase scenario coverage before more training; compare an unadapted 4B base with the 10-step LoRA adapter under identical decoding and strict JSON schema checks. Do not confuse pseudo-label agreement with task accuracy.

**Construction data is synthetic and is not independently verified.** `orinth_clef.build_cases` creates 30 scenario families × four phrasings, each with one choice and one noul question. It writes construction intent separately from teacher requests. Family IDs prevent simple paraphrase leakage between train/validation/test.

## Commands

Run from `experiments/orinth-clef-unified`:

~~~bash
python3 -m unittest discover -v
python3 -m orinth_clef.build_cases
# Start the teacher in a separate terminal:
bash scripts/mac_bootstrap.sh serve
# In another terminal:
REV="$(cat data/teacher_revision.txt)"
.venv-teacher/bin/python -m orinth_clef.collect_resume \
  --revision "$REV" --cases data/cases_120.jsonl \
  --output data/raw/clef_120.jsonl
python3 -m orinth_clef.analyze_teacher \
  --raw data/raw/clef_120.jsonl \
  --references data/construction_intent_120.jsonl \
  --output data/teacher_audit_120.json
python3 -m orinth_clef export --raw data/raw/clef_120.jsonl \
  --output data/mlx_120 --seed orinth-clef-v1
# Stop the teacher before loading Qwen; shared unified memory is limited.
.venv-student/bin/python -m orinth_clef.eval_student \
  --no-adapter --data data/mlx_120 --output runs/phase2/base.json
.venv-student/bin/python -m orinth_clef.eval_student \
  --adapter runs/qwen3-4b-smoke --data data/mlx_120 \
  --output runs/phase2/smoke-adapter.json
~~~

A collection interrupted midway can be rerun against the same output file. Previously completed cases must match their input request, scenario group and teacher revision; otherwise the collector stops rather than mixing data. Newly completed rows are flushed to disk individually.

## Interpret results correctly

- Schema validity is strict: `choice` must be a valid option ID string, `noul` a JSON boolean, with exactly the requested question keys.
- Teacher pseudo-label agreement is an imitation metric, **not accuracy**.
- Agreement with the separate synthetic construction intent is **not independent gold accuracy**.
- Four phrasings per family are near-duplicates. Held-out families are independent by construction, but 30 families remain far too small to establish broad generalization.
- Only train a larger student adapter after both baseline evaluations and the teacher audit are recorded.
- Do not fuse weights, quantize further, claim calibration, or publish a Clef replacement on these results.

## Model and artifact hygiene

Model weights, teacher predictions, student adapters, training/evaluation reports, and construction-intent reference files are stored under ignored `models/`, `data/` and `runs/`. No personal documents or external customer data are needed for this experiment.
