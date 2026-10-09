"""Reproducible on-device research diagnostic for a packaged MLX checkpoint."""
import argparse
import json
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .decision_lens import decide


def evaluate(model_path, fixture, scorer="trie", prompt_style="baseline"):
    from mlx_lm import load
    model, tokenizer = load(str(model_path))
    cases = read_challenges(Path(fixture))
    records = []
    for row in cases:
        start = time.perf_counter()
        decisions, _ = decide(
            model, tokenizer, row["task"],
            sequence_scorer=scorer, prompt_style=prompt_style
        )
        records.append({
            "id": row["id"], "family": row["family"],
            "exact": decisions == row["gold"],
            "seconds": time.perf_counter() - start,
        })
    return {
        "benchmark_kind": "author_created_rule_diagnostic_not_independent_gold",
        "model": str(model_path), "scorer": scorer,
        "prompt_style": prompt_style,
        "count": len(records),
        "exact_count": sum(x["exact"] for x in records),
        "median_seconds": statistics.median(x["seconds"] for x in records),
        "by_family": {
            family: {
                "count": sum(x["family"] == family for x in records),
                "exact": sum(x["exact"] for x in records if x["family"] == family),
            } for family in sorted({x["family"] for x in records})
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("."))
    p.add_argument("--fixture", type=Path, default=Path("challenge_rules.jsonl"))
    p.add_argument("--scorer", choices=("reference", "trie", "rollback"), default="trie")
    p.add_argument("--prompt-style", choices=("baseline", "short_system", "compact", "compact_json"), default="baseline")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = evaluate(args.model, args.fixture, args.scorer, args.prompt_style)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
