"""Paired diagnostic of short prompt versus baseline with the same trie."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .decision_lens import decide


def benchmark(model_path, fixture, rounds, challenger="compact"):
    from mlx_lm import load
    cases = read_challenges(fixture)
    model, tokenizer = load(model_path)
    rows = []
    for repeat in range(rounds):
        for i, case in enumerate(cases):
            pair = {}
            for style in (("baseline", challenger) if (i + repeat) % 2 == 0
                          else (challenger, "baseline")):
                start = time.perf_counter()
                result, _ = decide(model, tokenizer, case["task"],
                                   sequence_scorer="trie", prompt_style=style)
                pair[style] = {
                    "seconds": round(time.perf_counter() - start, 5),
                    "exact": result == case["gold"],
                    "decisions": result,
                }
            rows.append({
                "id": case["id"], "round": repeat,
                "baseline_exact": pair["baseline"]["exact"],
                "compact_exact": pair[challenger]["exact"],
                "same": pair["baseline"]["decisions"] == pair[challenger]["decisions"],
                "baseline_seconds": pair["baseline"]["seconds"],
                "compact_seconds": pair[challenger]["seconds"],
            })
    return {
        "kind": "author_rule_diagnostic_not_independent_accuracy",
        "challenger_style": challenger,
        "cases": len(cases), "rounds": rounds,
        "baseline_exact": sum(r["baseline_exact"] for r in rows),
        "compact_exact": sum(r["compact_exact"] for r in rows),
        "same": sum(r["same"] for r in rows),
        "baseline_median_seconds": statistics.median(r["baseline_seconds"] for r in rows),
        "compact_median_seconds": statistics.median(r["compact_seconds"] for r in rows),
        "rows": rows
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--fixture", type=Path, default=Path("examples/challenge_rules.jsonl"))
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--style", choices=("compact", "compact_json", "short_system"), default="compact")
    p.add_argument("--output", type=Path, default=Path("runs/phase7/prompt-benchmark.json"))
    args = p.parse_args()
    result = benchmark(args.model, args.fixture, args.rounds, args.style)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
