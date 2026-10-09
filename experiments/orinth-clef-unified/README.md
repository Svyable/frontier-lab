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

## Phase 3 — DecisionLens: zero-autoregressive-token decision path

See [DecisionLens measured M4 report](docs/DECISIONLENS-M4.md). The runtime
scores allowed **single-token** choice IDs and JSON booleans directly from
Qwen3-4B logits, reusing an MLX KV cache across fields. It never generates
JSON tokens autoregressively; the runtime assembles strictly typed output.

On 24 synthetic held-out scenarios, cached DecisionLens with the 60-step
adapter achieved 24/24 schema validity and 16/24 teacher pseudo-label
matches at 0.2823s median per case, compared with 0.4855s for ordinary
generation (1.72x median latency improvement). A locally fused 4-bit
candidate reached 17/24 matches at 0.2789s. 3-bit variants degraded badly;
see report. No general SOTA or independent accuracy claim.

~~~bash
.venv-student/bin/python -m orinth_clef.decision_lens \
  --model models/qwen3-4b-60iter-4bit --no-adapter \
  --data data/mlx_120 --output runs/phase3/decision-lens-fused4bit.json
python3 -m orinth_clef.package_lens
python3 -m orinth_clef.package_lens --verify
~~~

The ~1 KiB **thin manifest** is a content-addressed overlay referencing
**separately installed ~2.1 GiB model weights**; it is not a standalone
1 KiB model. Normalized candidate token scores are not calibrated
probabilities. Multi-token choice IDs are supported experimentally via
KV-cached candidate-sequence scoring (see Phase 5 below).

## Phase 4 — One-pass micro-head (experimental)

See [M4 micro-head benchmark](docs/PHASE4-EXPERIMENT.md).
A frozen fused 4-bit Qwen backbone encodes the entire task in one pass.
A separately trained ~960 KiB question-conditioned head scores the choices.
On the same 24 synthetic held-out cases, the 24-epoch head produced 24/24
schema-valid responses and 16/24 teacher pseudo-label matches at ~0.186s
median end-to-end, compared with 17/24 and 0.2789s for DecisionLens.
The head is ~1.5x faster on this workload but has lower agreement;
**do not treat it as a quality-approved replacement or standalone model**.

~~~bash
.venv-student/bin/python -m orinth_clef.micro_head \
  --model models/qwen3-4b-60iter-4bit --data data/mlx_120 \
  --rank 32 --epochs 24 \
  --output runs/phase4/micro-head-24.json \
  --head-weights runs/phase4/micro-head-24.safetensors
~~~

## Phase 5 — Multi-token constrained decisions

See [rule-labeled M4 challenge report](docs/PHASE5-CHALLENGE.md).
DecisionLens now supports multi-token choice IDs by teacher-forcing
candidate continuations against cloned MLX KV caches, rather than
generating arbitrary text. On 24 new author-labeled deterministic-rule
diagnostics, schema validity improved from **12/24 to 24/24** and exact
complete-case rule agreement from **11/24 to 21/24**. The 24-epoch
micro-head reached 7/24 on the same diagnostics, exposing weak
out-of-domain generalization. The multi-token path costs additional
compute; it is not a speedup claim.

~~~bash
.venv-student/bin/python -m orinth_clef.challenge \
  --fixture examples/challenge_rules.jsonl \
  --output runs/phase5/challenge-sequences.json
~~~

## Phase 6 — Prefix-trie candidate scorer (opt-in)

See [measured M4 trie benchmark](docs/PHASE6-TRIE.md).
The candidate-sequence scorer can share token prefixes and KV-cache
continuations between allowed choice IDs. On a synthetic 32-option
shared-prefix task, warm M4 median latency fell from **1.963s to
0.984s (~1.99x)**; on the 24-case mixed diagnostic, the change was
only ~1.04x. Paired decisions agreed on those tested workloads, with
small nonzero numerical score drift. The original scorer remains the
default pending broader validation.

~~~bash
.venv-student/bin/python -m orinth_clef.trie_benchmark --rounds 2
.venv-student/bin/python -m orinth_clef.prefix_stress --options 32 --rounds 3
.venv-student/bin/python -m orinth_clef.decision_lens \
  --model models/qwen3-4b-60iter-4bit --no-adapter \
  --sequence-scorer trie --output runs/phase6/teacher-regression-trie.json
~~~

## Phase 7 — Research-grade MLX model bundle (local, unpublished)

See [Phase 7 experiment and release report](docs/PHASE7-RESEARCH-RELEASE.md).
Two additional decision runtimes were tested on the 24 author-labeled
rule cases (two paired rounds). The trie retained **42/48** complete
matches at ~0.29s median. Compact aliasing achieved 36/48 at 0.312s;
a per-question direct classifier achieved 30/48 at 0.249s.
Neither is a quality-and-speed improvement; both remain experiments.
A separate short-system-prompt variant **did** improve both metrics
on two author-created rule diagnostics: 44/48 vs 42/48 exact at
0.249s vs 0.288s median on the original suite, and 52/64 vs
50/64 at 0.263s vs 0.305s on a newly written 32-case suite.
This is a promising but small opt-in Pareto improvement, **not**
independent accuracy or general SOTA. Experimental KV rollback
showed no meaningful speedup over the copying trie.

Build a local Hugging Face-style MLX release with actual weights,
a runnable inference package, SHA-256 manifest, provenance, model card,
and a reproducible diagnostic CLI:

~~~bash
.venv-student/bin/python -m orinth_clef.hf_release export \
  --include-weights --head runs/phase4/micro-head-24.safetensors \
  --output runs/phase7/hf-weighted-v2
.venv-student/bin/python -m orinth_clef.hf_release verify \
  --directory runs/phase7/hf-weighted-v2
~~~

**No automatic upload or model-license assertion.** Independent
human gold and rights review are required before public publication.
The artifact is MLX/Apple Silicon, not a Transformers/PyTorch model.

## Phase 8 — ProofRoute: explicit verified policy fast path

[Phase 8 research report](docs/PHASE8-PROOFROUTE.md). This
**optional**, CPU-only path evaluates typed, externally supplied
policy expressions against task state and returns complete decisions
without model inference. Invalid policies fail closed; missing data
abstains or invokes a separately provided validated neural fallback.

On the **previously inspected** 32-case author-rule diagnostic
(two repetitions), it achieved 64/64 complete matches with
0.01346 ms median per-case time (including validation and compilation),
versus 52/64 and 270.41 ms for DecisionLens. This ~20,000x
**rule-path** speed difference is not a faster AI model, and does
not beat conventional rule-engine baselines. Human policy authoring,
maintenance, cold start and model loading are excluded.

~~~bash
.venv-student/bin/python -m orinth_clef.proofroute_benchmark --rounds 2
~~~

The path is included as optional source in the local Hugging Face-style
MLX bundle, not substituted for the learned Qwen checkpoint.

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
