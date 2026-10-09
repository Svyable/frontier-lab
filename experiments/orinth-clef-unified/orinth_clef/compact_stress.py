"""Prefix-heavy benchmark comparing compact aliases and trie scoring."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .compact_labels import decide_compact
from .decision_lens import decide


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--options", type=int, default=32)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("runs/phase7/compact-stress.json"))
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
        runs = {}
        for mode in (("compact", "trie") if i % 2 == 0 else ("trie", "compact")):
            start = time.perf_counter()
            decisions, _ = (decide_compact(model, tokenizer, task) if mode == "compact"
                            else decide(model, tokenizer, task, sequence_scorer="trie"))
            runs[mode] = {"seconds": time.perf_counter() - start,
                          "decision": decisions["routing"]}
        records.append({
            "compact_seconds": runs["compact"]["seconds"],
            "trie_seconds": runs["trie"]["seconds"],
            "same": runs["compact"]["decision"] == runs["trie"]["decision"],
            "compact_correct": runs["compact"]["decision"] == sorted(options)[0],
            "trie_correct": runs["trie"]["decision"] == sorted(options)[0]
        })
    report = {
        "options": args.options, "rounds": args.rounds,
        "compact_median_seconds": statistics.median(x["compact_seconds"] for x in records),
        "trie_median_seconds": statistics.median(x["trie_seconds"] for x in records),
        "compact_correct_count": sum(x["compact_correct"] for x in records),
        "trie_correct_count": sum(x["trie_correct"] for x in records),
        "records": records
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
