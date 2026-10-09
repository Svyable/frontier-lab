"""Paired prefix-heavy benchmark of copied-KV trie versus rollback trie."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .decision_lens import decide


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--options", type=int, default=32)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("runs/phase7/rollback-stress.json"))
    args = p.parse_args()
    if not 2 <= args.options <= 50 or not 1 <= args.rounds <= 10:
        p.error("options 2..50, rounds 1..10")
    from mlx_lm import load
    model, tokenizer = load("models/qwen3-4b-60iter-4bit")
    options = {f"warehouse_restock_priority_{i:03d}": f"Queue {i} priority bucket"
               for i in range(args.options)}
    task = {
        "state": {"note": "Synthetic prefix sharing throughput stress."},
        "questions": {"routing": {
            "type": "choice", "instructions": "Pick the first option by lexicographic order.",
            "criteria": options
        }}
    }
    records = []
    for i in range(args.rounds):
        pair = {}
        for mode in (("rollback", "trie") if i % 2 == 0 else ("trie", "rollback")):
            start = time.perf_counter()
            decisions, distribution = decide(model, tokenizer, task, sequence_scorer=mode)
            pair[mode] = {
                "seconds": time.perf_counter() - start,
                "decision": decisions["routing"],
                "distribution": distribution["routing"]
            }
        records.append({
            "rollback_seconds": pair["rollback"]["seconds"],
            "trie_seconds": pair["trie"]["seconds"],
            "same": pair["rollback"]["decision"] == pair["trie"]["decision"],
            "max_score_difference": max(abs(
                pair["rollback"]["distribution"][key] - pair["trie"]["distribution"][key]
            ) for key in pair["trie"]["distribution"]),
            "rollback_correct": pair["rollback"]["decision"] == sorted(options)[0]
        })
    result = {
        "benchmark": "synthetic_prefix_stress_not_accuracy",
        "options": args.options, "rounds": args.rounds,
        "rollback_median_seconds": statistics.median(r["rollback_seconds"] for r in records),
        "trie_median_seconds": statistics.median(r["trie_seconds"] for r in records),
        "decision_parity": all(r["same"] for r in records),
        "max_score_difference": max(r["max_score_difference"] for r in records),
        "rollback_correct_count": sum(r["rollback_correct"] for r in records),
        "records": records
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["decision_parity"] or result["max_score_difference"] > 0.01:
        raise SystemExit("rollback/trie parity gate failed")


if __name__ == "__main__":
    main()
