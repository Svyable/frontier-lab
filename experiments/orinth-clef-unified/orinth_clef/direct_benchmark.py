"""Paired diagnostic benchmark of question-local direct classification against reference/trie scoring."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .direct_classifier import direct_decide
from .decision_lens import decide


def benchmark(model_path, fixture, rounds):
    from mlx_lm import load
    cases = read_challenges(fixture)
    model, tokenizer = load(model_path)
    records = []
    for repetition in range(rounds):
        for index, case in enumerate(cases):
            order = ("direct", "trie") if (repetition + index) % 2 == 0 else ("trie", "direct")
            outputs = {}
            for mode in order:
                start = time.perf_counter()
                result = (direct_decide(model, tokenizer, case["task"])
                          if mode == "direct" else
                          decide(model, tokenizer, case["task"], sequence_scorer="trie"))
                outputs[mode] = {
                    "decisions": result[0],
                    "seconds": round(time.perf_counter() - start, 5),
                }
            records.append({
                "id": case["id"], "family": case["family"], "round": repetition,
                "direct_exact": outputs["direct"]["decisions"] == case["gold"],
                "trie_exact": outputs["trie"]["decisions"] == case["gold"],
                "same_decision": outputs["direct"]["decisions"] == outputs["trie"]["decisions"],
                "direct_seconds": outputs["direct"]["seconds"],
                "trie_seconds": outputs["trie"]["seconds"]
            })
    return {
        "benchmark": "author_labeled_diagnostic_not_independent_accuracy",
        "cases": len(cases), "rounds": rounds,
        "direct_exact": sum(x["direct_exact"] for x in records),
        "trie_exact": sum(x["trie_exact"] for x in records),
        "decision_parity": sum(x["same_decision"] for x in records),
        "direct_median_seconds": statistics.median(x["direct_seconds"] for x in records),
        "trie_median_seconds": statistics.median(x["trie_seconds"] for x in records),
        "records": records
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--fixture", type=Path, default=Path("examples/challenge_rules.jsonl"))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("runs/phase7/direct-benchmark.json"))
    args = p.parse_args()
    result = benchmark(args.model, args.fixture, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
