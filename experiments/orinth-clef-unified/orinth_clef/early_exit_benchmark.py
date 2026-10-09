"""Paired M4 benchmark of shared-backbone exit versus full 36-layer head.

Reuses validation-selected threshold from a frozen training report.
Test labels are read ONLY to score, never to set thresholds.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from .early_exit import decide as early_decide
from .micro_head import build_head, decide as full_decide, read_cases


def run(model, tokenizer, early_head, final_head, test_cases, depth, threshold, rounds):
    rows = []
    for repeat in range(rounds):
        for i, (task, expected) in enumerate(test_cases):
            results = {}
            order = ("early", "full") if (repeat + i) % 2 else ("full", "early")
            for method in order:
                start = time.perf_counter()
                if method == "early":
                    answer, route, _, _ = early_decide(
                        model, tokenizer, task, depth, early_head, final_head, threshold)
                else:
                    answer = full_decide(final_head, model, tokenizer, task)
                    route = "full"
                results[method] = {"seconds": time.perf_counter() - start,
                                   "match": answer == expected, "route": route}
            rows.append({"index": i, **results})
    return {
        "rounds": rounds, "cases": len(test_cases), "depth": depth,
        "threshold": threshold,
        "early_accepted": sum(r["early"]["route"] == "early" for r in rows),
        "early_matches": sum(r["early"]["match"] for r in rows),
        "full_matches": sum(r["full"]["match"] for r in rows),
        "early_median_seconds": statistics.median(r["early"]["seconds"] for r in rows),
        "full_median_seconds": statistics.median(r["full"]["seconds"] for r in rows),
        "early_mean_seconds": statistics.mean(r["early"]["seconds"] for r in rows),
        "full_mean_seconds": statistics.mean(r["full"]["seconds"] for r in rows),
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--data", type=Path, default=Path("data/mlx_120"))
    p.add_argument("--training-report", type=Path, default=Path("runs/phase10/early24-target80.json"))
    p.add_argument("--early-weights", type=Path, default=Path("runs/phase10/early24.safetensors"))
    p.add_argument("--final-weights", type=Path, default=Path("runs/phase4/micro-head-24.safetensors"))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("runs/phase10/paired24.json"))
    args = p.parse_args()
    if not 1 <= args.rounds <= 20:
        p.error("rounds must be 1..20")
    from mlx_lm import load
    model, tokenizer = load(args.model)
    report = json.loads(args.training_report.read_text())
    if report["dataset_sha256"] != __import__(
        "orinth_clef.micro_head", fromlist=["dataset_hash"]).dataset_hash(args.data):
        p.error("training report dataset hash does not match")
    early = build_head(2560, report["rank"])
    early.load_weights(str(args.early_weights))
    final = build_head(2560, 32)
    final.load_weights(str(args.final_weights))
    result = run(model, tokenizer, early, final, read_cases(args.data)["test"],
                 report["depth"], report["threshold"], args.rounds)
    result["evaluation_kind"] = "teacher_pseudolabels_previously_inspected_not_independent_gold"
    result["threshold_source"] = "training_report_validation_only"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
