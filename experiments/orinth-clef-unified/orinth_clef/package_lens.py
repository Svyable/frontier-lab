"""Thin content-addressed DecisionLens manifest; no model weights copied.

This is an *overlay* package: a few KB of metadata reference separately
installed local weights. It is NOT a standalone 7 MB foundation model.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .schema import SchemaError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model: Path, *, revision: str, benchmark: Path) -> dict:
    if not model.is_dir() or not benchmark.is_file():
        raise SchemaError("model directory and benchmark report must exist")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise SchemaError("revision must be a pinned 40-character lowercase SHA")
    report = json.loads(benchmark.read_text(encoding="utf-8"))
    if report.get("model_path") != str(model) or report.get("adapter_path") is not None:
        raise SchemaError("benchmark must evaluate this exact standalone model without adapter")
    if report.get("count", 0) < 1 or report.get("schema_valid_count") != report["count"]:
        raise SchemaError("benchmark must show complete schema validity")
    files = {}
    for path in sorted(model.iterdir()):
        if path.is_file() and (path.suffix == ".safetensors" or path.name in {
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "generation_config.json", "special_tokens_map.json"
        }):
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if not any(name.endswith(".safetensors") for name in files):
        raise SchemaError("no model safetensors found")
    if "tokenizer.json" not in files or "config.json" not in files:
        raise SchemaError("model missing tokenizer or config")
    return {
        "format": "decisionlens.thin.v1",
        "standalone": False,
        "weight_source": "separately_installed_local_model",
        "model_path_relative_to_workspace": str(model),
        "source_model_revision": revision,
        "runtime": "orinth_clef.decision_lens",
        "schema_abi": "choice-single-token-and-noul-boolean-v1",
        "probabilities": "normalized_token_scores_not_calibrated",
        "benchmark_report_sha256": sha256_file(benchmark),
        "benchmark_case_count": report["count"],
        "benchmark_teacher_pseudolabel_agreement_count": report["teacher_pseudolabel_agreement_count"],
        "files": files,
    }


def verify_manifest(manifest: dict, workspace: Path) -> bool:
    if manifest.get("format") != "decisionlens.thin.v1" or manifest.get("standalone") is not False:
        raise SchemaError("invalid thin package format")
    rel = Path(manifest.get("model_path_relative_to_workspace", ""))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise SchemaError("model path must be workspace-relative")
    root = workspace.resolve()
    model = (root / rel).resolve()
    if not model.is_relative_to(root) or not model.is_dir():
        raise SchemaError("model must resolve within workspace")
    for name, record in manifest["files"].items():
        if name != Path(name).name or name.startswith("."):
            raise SchemaError("invalid package filename")
        path = model / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]:
            raise SchemaError(f"missing or changed file {name}")
        if sha256_file(path) != record["sha256"]:
            raise SchemaError(f"checksum mismatch: {name}")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Build or verify a thin DecisionLens package")
    p.add_argument("--model", type=Path, default=Path("models/qwen3-4b-60iter-4bit"))
    p.add_argument("--revision-file", type=Path, default=Path("data/student_revision.txt"))
    p.add_argument("--benchmark", type=Path, default=Path("runs/phase3/decision-lens-fused4bit.json"))
    p.add_argument("--output", type=Path, default=Path("runs/phase3/decisionlens.thin.json"))
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    if args.verify:
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        verify_manifest(manifest, Path.cwd())
        print("Verified local DecisionLens package integrity")
    else:
        manifest = build_manifest(args.model, revision=args.revision_file.read_text().strip(),
                                  benchmark=args.benchmark)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"format": manifest["format"], "standalone": False,
                          "files": len(manifest["files"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
