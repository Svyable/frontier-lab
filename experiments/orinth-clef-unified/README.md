# Orinth × Clef — local unified-model experiment

**Status: Phase 1 bootstrap, 2026-10-08. Mac Metal tests, teacher model execution and student training are UNVERIFIED.**

This contained [Frontier Lab](https://github.com/Svyable/frontier-lab) experiment captures real Clef-Flash teacher predictions, validates their typed probabilities and exports group-separated MLX-LM chat training data. It is **not yet a unified model** and does not change the upstream Orinth repository.

## Research design

**Student candidate:** mlx-community/Qwen3-4B-4bit, 4-bit MLX. **Teacher:** mlx-community/clef-flash-4bit, joint-head SystemOne decision model. Use the teacher *sequentially* before student training on the user's 24 GB Apple M4; never load two models plus training activations together.

Supported in this milestone: **text-only choice and noul**, strict schema validation, pinned teacher revision, synthetic unlabeled seed cases, local-only HTTP, no truncation, deterministic scenario-group holdouts, manifest describing pseudo-labels. Unsupported: score, image, video, model fusion, schema-head training, actual on-device model runs.

The teacher's probabilities are retained in the raw capture; the MLX SFT data contains only its top selections. That is *label imitation*, not probability distillation or evidence of calibration.

## Runtime baseline

- CPU tests: Python 3.13.5 on Linux, standard library only.
- Intended target: macOS arm64, Apple M4, 24 GB unified memory — **not accessed during bootstrap**.
- Clef MLX conversion model card tested on mlx-vlm 0.7.4, mlx-lm 0.32.0, mlx 0.32.3 (upstream figures, not measured here).
- At runtime, record exact model snapshot revision; do not silently follow an evolving branch.

## Run tests anywhere

~~~bash
cd experiments/orinth-clef-unified
python3 -m unittest discover -v
python3 -m orinth_clef doctor
~~~

Doctor exits 2 on non-Apple machines; tests remain portable.

## Apple Silicon: download and run the teacher

Install uv from https://docs.astral.sh/uv/ if needed.

~~~bash
cd experiments/orinth-clef-unified
uv venv --python 3.12 .venv-teacher
uv pip install --python .venv-teacher/bin/python 'mlx-vlm==0.7.4' 'huggingface_hub[hf_xet]'
source .venv-teacher/bin/activate
REV=$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("mlx-community/clef-flash-4bit").sha)')
echo "Pinned Clef revision: $REV"
hf download mlx-community/clef-flash-4bit --revision "$REV" --local-dir models/clef-flash-4bit
(cd models/clef-flash-4bit && python clef_mlx.py serve --port 8001)
~~~

The server listens on localhost by default, without authentication. Never expose port 8001. In a **second terminal** with the teacher virtual environment active:

~~~bash
python -m orinth_clef probe --url http://127.0.0.1:8001
python -m orinth_clef collect --revision "$REV" --cases examples/cases.jsonl --output data/raw/clef_12.jsonl
python -m orinth_clef export --raw data/raw/clef_12.jsonl --output data/mlx
~~~

The REV environment variable must be set to the exact revision printed by the first terminal. None of these commands invent teacher responses. An unreachable teacher or malformed prediction aborts collection. The 12 handwritten seed scenarios are only for functional checks, not scientifically meaningful training.

## Proposed student smoke test (NOT RUN)

Stop the teacher process first. A fresh 3.12 environment with mlx-lm[train] is recommended.

~~~bash
uv venv --python 3.12 .venv-student
uv pip install --python .venv-student/bin/python 'mlx-lm[train]'
source .venv-student/bin/activate
mlx_lm.lora --model mlx-community/Qwen3-4B-4bit --train --data data/mlx \
 --adapter-path runs/qwen3-4b-smoke --iters 10 --batch-size 1 \
 --grad-accumulation-steps 4 --num-layers 4 --max-seq-length 512 \
 --learning-rate 1e-5 --mask-prompt --grad-checkpoint
~~~

Monitor real memory pressure with memory_pressure -Q and vm_stat. Never call a successful 10-iteration training run a quality result.

## Observed Apple M4 results and student typed-output evaluator

See [docs/M4-BOOTSTRAP-RESULTS.md](docs/M4-BOOTSTRAP-RESULTS.md) for the measured 2026-10-09 run.
The 12-case teacher capture and 10-iteration student LoRA smoke test both ran locally, but **0 of 2 held-out outputs passed strict decision schema**. This checkpoint is not a validated Clef replacement.

After training, run:

~~~bash
.venv-student/bin/python -m orinth_clef.eval_student --splits valid test
~~~

The local ignored report is written to `runs/qwen3-4b-smoke/eval.json`. It explicitly distinguishes schema validity and **teacher pseudo-label** agreement from independent task accuracy. Strict parsing never repairs malformed model output.

## Phase 2 — larger reproducible synthetic benchmark

See [Phase 2 method](docs/PHASE2-METHOD.md) and [measured M4 results](docs/PHASE2-M4-RESULTS.md).
The new dataset generator produces 120 synthetic cases from 30 scenario families with
four phrasings per family; family-held-out splits are 96 train / 12 validation /
12 test. Resumable teacher capture pins the same model revision and validates
previously collected rows. The teacher's decision probabilities remain in raw
local files, while MLX SFT exports use pseudo-labels only.

Run `python3 -m orinth_clef.build_cases` then follow the Phase 2 method.
The evaluator supports `--no-adapter` for a true base-model control. Neither
teacher agreement nor comparison with synthetic construction intents constitutes
independently validated accuracy.

## Release gates

1. Real SystemOne response contains all requested choice probabilities and noul probabilities.
2. Exact HF model revision is saved in every collected raw row. No invented labels.
3. No scenario group is split across train / valid / test.
4. Gold labels and a held-out benchmark are added before claiming task accuracy, teacher parity or calibration.
5. Retain general-language evaluations before adapter fusion or dedicated head training.
6. No model publication or license choice without owner approval.

Local data, models, runs and virtual environments are ignored by Git.

## Provenance / licensing

- https://github.com/Onestep-AI-Labs/orinth — Apache 2.0; current serving backend is GGUF/llama.cpp, not MLX.
- https://huggingface.co/Cloudflare/clef-flash — Apache 2.0; teacher's joint schema head.
- https://huggingface.co/mlx-community/clef-flash-4bit — MLX conversion with clef_mlx.py and SystemOne server.
- https://huggingface.co/Qwen/Qwen3-4B — Apache 2.0 student base candidate.
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md — train/fuse interface.
- https://github.com/SamsungLabs/LittleBit — CC BY-NC 4.0; its CUDA training code is **not** incorporated here.

See docs/ORINTH-INTEGRATION.md for next scoped upstream extension.
