"""Build and verify a local Hugging Face-style MLX research release.

No network calls, no automatic upload, and no license assertions about
Clef-derived training data. Manifest verifies every included file.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .schema import SchemaError

MODEL_FILES = (
    "config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "model.safetensors", "model.safetensors.index.json"
)
SOURCE_FILES = (
    "__init__.py", "schema.py", "data.py", "eval_student.py",
    "candidate_sequences.py", "prefix_trie.py", "decision_lens.py",
    "compact_labels.py", "challenge.py", "trie_benchmark.py",
    "prefix_stress.py", "micro_head.py", "hf_release.py",
    "hf_benchmark.py", "rollback_trie.py", "rollback_benchmark.py",
    "rollback_stress.py", "prompt_benchmark.py",
    "direct_classifier.py", "direct_benchmark.py",
    "compact_benchmark.py", "compact_stress.py",
    "proofroute.py", "proofroute_benchmark.py"
)
CARD = """---
language:
- en
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3-4B
tags:
- mlx
- decision-making
- constrained-decoding
- research
---

# DecisionLens — Experimental Qwen3-4B 4-bit MLX decision model

**Research-only candidate, not a validated SOTA model.** This is a
Qwen3-4B 4-bit MLX checkpoint fine-tuned to imitate decisions from
Clef-flash. It is not Clef itself and does not reproduce Clef's
probability head. The constrained runtime selects valid JSON decisions
without unconstrained text generation.

## Model and training provenance

- Base: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
  (Apache-2.0 base license; does not automatically resolve rights to
  derivative training data or all release components).
- MLX conversion reference:
  [mlx-community/Qwen3-4B-4bit](https://huggingface.co/mlx-community/Qwen3-4B-4bit).
- Fine-tuned with a local 60-iteration LoRA run (rank 8, 4 layers,
  learning rate 1e-5, gradient accumulation 4, max sequence length 512),
  then fused into a 4-bit Qwen checkpoint. See `training_config.json`
  and `training_dataset_manifest.json` for reproducibility metadata.
- Supervision: synthetic tasks pseudo-labeled by a Clef-flash teacher.
  **Teacher agreement is not independent correctness.**
- No model release or license grant has been authorized yet.
  **Do not publish or redistribute this bundle until licensing and
  teacher-data redistribution rights are independently verified.**

## Local MLX usage (Apple Silicon)

Install with `pip install -r requirements.txt`. In this directory:

```python
from mlx_lm import load
from orinth_clef.decision_lens import decide

model, tokenizer = load(".")
task = {
    "state": {"issue": "billing", "severity": 4},
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Route by issue.",
            "criteria": {"billing": "Billing issue", "technical": "Technical issue"}
        },
        "urgent": {"type": "noul", "instructions": "True iff severity >= 4"}
    }
}
decisions, uncalibrated_scores = decide(model, tokenizer, task, sequence_scorer="trie")
print(decisions)
```

The default reference scorer is also available as `sequence_scorer="reference"`.
The trie scorer helps when option IDs share token prefixes. These
normalized scores are **not calibrated probabilities**.
The opt-in `prompt_style="short_system"` was slightly faster and
more accurate on two author-created rule diagnostics; see below.
The experimental `sequence_scorer="rollback"` mutates KV-cache
storage in place and is **not thread-safe or recommended for deployment**.

Reproduce the included *author-labeled* diagnostic locally:

```bash
python -m orinth_clef.hf_release verify --directory .
python -m orinth_clef.hf_benchmark --model . --fixture challenge_rules.jsonl
python -m orinth_clef.hf_benchmark --model . --fixture holdout_rules_v1.jsonl \
  --prompt-style short_system
```

An optional explicit-policy route is available through
`orinth_clef.proofroute.decide(task, policy, fallback)`. This is
a conventional symbolic fast path, **not a learned model capability**.
It never infers rules from natural language and abstains if policy
coverage or required state is missing. See the source repository's
`docs/PHASE8-PROOFROUTE.md` for limitations.

The optional `micro-head-experimental.safetensors` is a separate
~960 KiB experimental head. It still requires the full Qwen backbone
and has substantially weaker out-of-domain decision accuracy; it is
**not** the recommended default.

## Evaluation evidence

- 24 synthetic Clef pseudo-label tasks (previously inspected):
  baseline 17/24 exact teacher agreement; short-system prompt
  18/24, both 24/24 schema-valid. Not independent accuracy.
- 24 author-created deterministic rule cases: 21/24 exact
  complete-case matches, 24/24 schema-valid with DecisionLens.
  **Not independently audited gold.**
- Shorter system prompt, existing 24-case author diagnostic (two rounds):
  44/48 complete matches vs 42/48 baseline, 0.249s vs 0.288s median.
- Shorter system prompt, new 32-case author rule suite (two rounds):
  52/64 complete matches vs 50/64 baseline, 0.263s vs 0.305s median.
  These are not independent human-adjudicated accuracy measurements.
- Prefix-heavy 32-option synthetic stress on Apple M4:
  1.963s reference vs 0.984s trie warm median (~1.99x);
  ordinary mixed tasks showed only ~1.04x. No general SOTA claim.
- All figures are local, small-sample, hardware- and task-specific.
  Full evidence and limitations: see `EVALUATION.md` and source repo.

## Limitations and intended use

Experimental text-only structured choice/boolean decisions. Do not use
as the sole basis for consequential financial, medical, legal or safety
decisions. Unknown robustness, bias, general reasoning quality,
probability calibration, and out-of-domain accuracy. Candidate IDs
requiring JSON escaping fail closed. MLX runtime targets Apple Silicon;
this bundle is not a standard Transformers/PyTorch checkpoint.
No independent evaluation or broad SOTA claim is available.

## Reproducibility and integrity

`manifest.json` contains SHA-256 hashes of every bundled file.
`python -m orinth_clef.hf_release verify --directory .` checks integrity.
The local training dataset and teacher raw outputs are **not** included;
their fingerprint may be included in `provenance.json`. Independent
researchers need a separately licensed dataset for replication.

Source: https://github.com/Svyable/frontier-lab/tree/main/experiments/orinth-clef-unified
"""
EVALUATION = """# Evidence and reproducibility

This release has **no independently adjudicated benchmark**.
All measurements are experimental and may not generalize.

| Evaluation | Exact complete decisions | Schema valid |
|---|---:|---:|
| Clef pseudo-labels (24 synthetic cases) | 17/24 agreement | 24/24 |
| Author-rule diagnostic (24 new cases) | 21/24 | 24/24 |

On a 24 GiB Apple M4, a 32-option shared-prefix synthetic task
measured 1.963s reference versus 0.984s trie median over 3
paired warm rounds. On a mixed 24-case diagnostic, the gain
was only ~1.04x. Not independent accuracy, not general SOTA.

A short-system-prompt experiment selected on the 24-case diagnostic
improved exact matches from 42/48 to 44/48 and warm median latency
from 0.288s to 0.249s. One subsequent 32-case author-created
rule suite (library, vehicle, access, shipment) showed 50/64 vs
52/64 exact matches and 0.305s vs 0.263s median. The cases were
not independently annotated; repeated runs do not create 64
independent samples. No independent general accuracy claim.

In-place KV-cache rollback matched 48/48 decisions on the original
mixed diagnostic, but showed no meaningful latency gain versus the
copying trie in a 32-option stress test (~0.967s vs ~0.971s). It is
experimental, not thread-safe, and not the recommended default.

The research source contains benchmark runners and author-labeled
fixture. The package includes the fixture for reproducibility,
**not as an independent test set**. Evaluators should supply fresh
held-out examples, human adjudication, confidence calibration,
paired p50/p95, peak memory, throughput and licensing review.
"""


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export(source, model_dir, output, include_weights, head_path=None):
    source, model_dir, output = Path(source), Path(model_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise SchemaError("output directory must be empty; refuse overwrite")
    for name in MODEL_FILES:
        if not (model_dir / name).is_file():
            raise SchemaError(f"missing model file: {name}")
    for name in SOURCE_FILES:
        if not (source / "orinth_clef" / name).is_file():
            raise SchemaError(f"missing runtime file: {name}")
    if head_path is not None and not Path(head_path).is_file():
        raise SchemaError("missing micro-head weights")
    output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if name == "model.safetensors" and not include_weights:
            continue
        shutil.copy2(model_dir / name, output / name)
    package = output / "orinth_clef"
    package.mkdir()
    for name in SOURCE_FILES:
        shutil.copy2(source / "orinth_clef" / name, package / name)
    (output / "README.md").write_text(CARD)
    (output / "EVALUATION.md").write_text(EVALUATION)
    (output / "requirements.txt").write_text("mlx-lm==0.32.0\n")
    (output / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    for fixture_name in ("challenge_rules.jsonl", "holdout_rules_v1.jsonl"):
        fixture = source / "examples" / fixture_name
        if fixture.is_file():
            shutil.copy2(fixture, output / fixture_name)
    if head_path is not None:
        shutil.copy2(head_path, output / "micro-head-experimental.safetensors")
    training_config = source / "runs" / "phase2" / "qwen3-4b-60iter" / "adapter_config.json"
    if training_config.is_file():
        shutil.copy2(training_config, output / "training_config.json")
    training_manifest = source / "data" / "mlx_120" / "manifest.json"
    if training_manifest.is_file():
        shutil.copy2(training_manifest, output / "training_dataset_manifest.json")
    provenance = {
        "artifact_type": "mlx_qwen3_4b_fused4bit_decision_runtime",
        "weights_included": include_weights,
        "base_model": "Qwen/Qwen3-4B",
        "conversion_reference": "mlx-community/Qwen3-4B-4bit",
        "teacher_reference": "mlx-community/clef-flash-4bit",
        "training_label_type": "teacher_pseudolabels_not_gold",
        "training_iterations": 60,
        "runtime_version": "mlx-lm==0.32.0",
        "research_only": True,
        "distribution_rights_review_required": True,
        "independent_gold_benchmark_available": False,
        "micro_head_included": head_path is not None,
    }
    data = source / "data" / "mlx_120" / "train.jsonl"
    if data.exists():
        provenance["training_split_sha256"] = sha256(data)
    revision = source / "data" / "teacher_revision.txt"
    if revision.exists():
        provenance["teacher_revision"] = revision.read_text().strip()
    try:
        provenance["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        provenance["source_commit"] = None
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    paths = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {
        "format": "decisionlens-mlx-research-bundle-v1",
        "weights_included": include_weights,
        "files": {str(p.relative_to(output)): {
            "sha256": sha256(p), "bytes": p.stat().st_size
        } for p in paths}
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify(output)


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format") != "decisionlens-mlx-research-bundle-v1":
        raise SchemaError("unknown bundle format")
    files = manifest["files"]
    for name, metadata in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise SchemaError("invalid manifest path")
        path = directory / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != metadata["sha256"]:
            raise SchemaError(f"missing or modified artifact: {name}")
        if path.stat().st_size != metadata["bytes"]:
            raise SchemaError(f"incorrect artifact size: {name}")
    # Importing bundled Python modules creates __pycache__ automatically.
    # Those generated bytecode files are not release artifacts and must
    # not invalidate the signed source/weights manifest after first use.
    actual = {
        str(p.relative_to(directory)) for p in directory.rglob("*")
        if p.is_file() and p.name not in ("manifest.json", ".DS_Store")
        and "__pycache__" not in p.relative_to(directory).parts
        and p.suffix != ".pyc"
    }
    if actual != set(files):
        raise SchemaError("unexpected or missing files in bundle")
    if manifest["weights_included"] != ("model.safetensors" in files):
        raise SchemaError("weight flag mismatch")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    if manifest["weights_included"] and not shards.issubset(files):
        raise SchemaError("model index references missing shards")
    return {"files_verified": len(files), "weights_included": manifest["weights_included"],
            "total_bytes": sum(meta["bytes"] for meta in files.values())}


def main():
    p = argparse.ArgumentParser(description="Build or verify an offline MLX research bundle")
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("export")
    make.add_argument("--source", type=Path, default=Path("."))
    make.add_argument("--model-dir", type=Path, default=Path("models/qwen3-4b-60iter-4bit"))
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--include-weights", action="store_true")
    make.add_argument("--head", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    result = (export(args.source, args.model_dir, args.output,
                     args.include_weights, args.head)
              if args.command == "export" else verify(args.directory))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
