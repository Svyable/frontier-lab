"""Paired diagnostic benchmark of compact IDs against reference/trie scoring."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .compact_labels import decide_compact
from .decision_lens import decide


def benchmark(model_path, fixture, rounds):
    from mlx_lm import load
    cases = read_challenges(fixture)
    model, tokenizer = load(model_path)
    records = []
    for repetition in range(rounds):
        for index, case in enumerate(cases):
            order = ("compact", "trie") if (repetition + index) % 2 == 0 else ("trie", "compact")
            outputs = {}
            for mode in order:
                start = time.perf_counter()
                result = (decide_compact(model, tokenizer, case["task"])
                          if mode == "compact" else
                          decide(model, tokenizer, case["task"], sequence_scorer="trie"))
                outputs[mode] = {
                    "decisions": result[0],
                    "seconds": round(time.perf_counter() - start, 5),
                }
            records.append({
                "id": case["id"], "family": case["family"], "round": repetition,
                "compact_exact": outputs["compact"]["decisions"] == case["gold"],
                "trie_exact": outputs["trie"]["decisions"] == case["gold"],
                "same_decision": outputs["compact"]["decisions"] == outputs["trie"]["decisions"],
                "compact_seconds": outputs["compact"]["seconds"],
                "trie_seconds": outputs["trie"]["seconds"]
            })
    return {
        "benchmark": "author_labeled_diagnostic_not_independent_accuracy",
        "cases": len(cases), "rounds": rounds,
        "compact_exact": sum(x["compact_exact"] for x in records),
        "trie_exact": sum(x["trie_exact"] for x in records),
        "decision_parity": sum(x["same_decision"] for x in records),
        "compact_median_seconds": statistics.median(x["compact_seconds"] for x in records),
        "trie_median_seconds": statistics.median(x["trie_seconds"] for x in records),
        "records": records
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--fixture", type=Path, default=Path("examples/challenge_rules.jsonl"))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("runs/phase7/compact-benchmark.json"))
    args = p.parse_args()
    result = benchmark(args.model, args.fixture, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
