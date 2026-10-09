"""Controlled prefix-heavy stress test; measures speed, NOT task accuracy."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .candidate_sequences import candidate_sequences
from .decision_lens import decide
from .prefix_trie import build_trie, trie_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--options", type=int, default=16)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("runs/phase6/prefix-stress.json"))
    args = p.parse_args()
    if not 2 <= args.options <= 50 or not 1 <= args.rounds <= 10:
        p.error("options 2..50; rounds 1..10")
    from mlx_lm import load
    model, tokenizer = load("models/qwen3-4b-60iter-4bit")
    question = {
        "type": "choice",
        "instructions": "Pick the first option by lexicographic order.",
        "criteria": {
            f"warehouse_restock_priority_{i:03d}": f"Queue {i} priority bucket"
            for i in range(args.options)
        }
    }
    task = {"state": {"note": "Synthetic prefix sharing throughput stress."},
            "questions": {"routing": question}}
    sequences = candidate_sequences(tokenizer, question)
    records = []
    for repeat in range(args.rounds):
        pair = {}
        for mode in (("reference", "trie") if repeat % 2 == 0 else ("trie", "reference")):
            start = time.perf_counter()
            decisions, distributions = decide(model, tokenizer, task, sequence_scorer=mode)
            pair[mode] = {
                "seconds": round(time.perf_counter() - start, 5),
                "decisions": decisions,
                "distribution": distributions["routing"]
            }
        records.append({
            "reference_seconds": pair["reference"]["seconds"],
            "trie_seconds": pair["trie"]["seconds"],
            "same_decision": pair["reference"]["decisions"] == pair["trie"]["decisions"],
            "max_distribution_drift": max(abs(
                pair["reference"]["distribution"][k] - pair["trie"]["distribution"][k]
            ) for k in pair["reference"]["distribution"])
        })
    result = {
        "benchmark": "synthetic_shared_prefix_latency_not_accuracy",
        "options": args.options, "rounds": args.rounds,
        "trie_stats": trie_stats(build_trie(sequences)),
        "sum_candidate_tokens": sum(map(len, sequences.values())),
        "reference_median_seconds": statistics.median(r["reference_seconds"] for r in records),
        "trie_median_seconds": statistics.median(r["trie_seconds"] for r in records),
        "decision_parity": all(r["same_decision"] for r in records),
        "max_score_drift": max(r["max_distribution_drift"] for r in records),
        "records": records
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["decision_parity"] or result["max_score_drift"] > 0.01:
        raise SystemExit("parity gate failed")


if __name__ == "__main__":
    main()
