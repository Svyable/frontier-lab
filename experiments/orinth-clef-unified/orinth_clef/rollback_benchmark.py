"""Paired M4 benchmark: reference candidate scoring vs shared-prefix rollback."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .decision_lens import decide


def benchmark(model_path, fixture, rounds=2):
    from mlx_lm import load

    if not 1 <= rounds <= 10:
        raise ValueError("rounds must be 1..10")
    cases = read_challenges(fixture)
    model, tokenizer = load(model_path)
    records = []
    for repetition in range(rounds):
        for i, case in enumerate(cases):
            task = case["task"]
            # Alternate order to mitigate warm-cache / thermal bias.
            order = ("reference", "rollback") if (i + repetition) % 2 == 0 else ("rollback", "reference")
            runs = {}
            for mode in order:
                start = time.perf_counter()
                decisions, distributions = decide(model, tokenizer, task, sequence_scorer=mode)
                runs[mode] = {
                    "seconds": time.perf_counter() - start,
                    "decisions": decisions, "distributions": distributions
                }
            identical = runs["reference"]["decisions"] == runs["rollback"]["decisions"]
            drift = max(
                abs(runs["reference"]["distributions"][qid][label] -
                    runs["rollback"]["distributions"][qid][label])
                for qid in runs["reference"]["distributions"]
                for label in runs["reference"]["distributions"][qid]
            )
            records.append({
                "id": case["id"], "family": case["family"], "round": repetition,
                "reference_seconds": round(runs["reference"]["seconds"], 5),
                "rollback_seconds": round(runs["rollback"]["seconds"], 5),
                "decision_parity": identical,
                "max_normalized_score_difference": drift,
                "gold_exact": runs["rollback"]["decisions"] == case["gold"]
            })
    return {
        "benchmark": "paired_warm_in_process_m4_candidate_scoring_not_sota",
        "model": model_path, "fixture": str(fixture), "rounds": rounds,
        "count": len(records),
        "decision_parity_count": sum(r["decision_parity"] for r in records),
        "max_score_difference": max(r["max_normalized_score_difference"] for r in records),
        "reference_median_seconds": statistics.median(r["reference_seconds"] for r in records),
        "rollback_median_seconds": statistics.median(r["rollback_seconds"] for r in records),
        "rollback_gold_exact_count": sum(r["gold_exact"] for r in records),
        "records": records,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--fixture", type=Path, default=Path("examples/challenge_rules.jsonl"))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("runs/phase6/rollback-benchmark.json"))
    args = p.parse_args()
    result = benchmark(args.model, args.fixture, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
    if result["decision_parity_count"] != result["count"] or result["max_score_difference"] > 0.01:
        raise SystemExit("rollback/reference parity gate failed")


if __name__ == "__main__":
    main()
