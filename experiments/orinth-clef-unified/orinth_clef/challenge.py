"""Out-of-distribution, rule-labeled benchmark for the two decision runtimes.

Labels come from explicit deterministic rules in the fixture, NOT from Clef.
These author-created cases are diagnostic, not independent human gold.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from .eval_student import parse_student_output
from .schema import SchemaError, validate_request


def read_challenges(path):
    cases = []
    ids = set()
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        case_id = row["id"]
        if case_id in ids:
            raise SchemaError(f"duplicate case id: {case_id}")
        ids.add(case_id)
        task = row["task"]
        validate_request({"model": "clef-flash", **task})
        gold = row["gold"]
        parse_student_output(json.dumps({"decisions": gold}), task["questions"])
        if not row.get("rule"):
            raise SchemaError(f"missing explicit rule: {case_id}")
        cases.append(row)
    if not cases:
        raise SchemaError("empty challenge fixture")
    return cases


def evaluate(path, model_path, head_path):
    from mlx_lm import load
    from .decision_lens import decide as lens_decide
    from .micro_head import build_head, decide as micro_decide

    cases = read_challenges(path)
    model, tokenizer = load(model_path)
    head = build_head(2560, 32)
    head.load_weights(str(head_path))
    outcomes = []
    # Paired cases, both paths on the same warm loaded backbone.
    for case in cases:
        task = case["task"]
        gold = case["gold"]
        row = {"id": case["id"], "family": case["family"], "gold": gold}
        for name, func in (
            ("micro", lambda: micro_decide(head, model, tokenizer, task)),
            ("lens", lambda: lens_decide(model, tokenizer, task)[0]),
        ):
            start = time.perf_counter()
            try:
                result = func()
                row[name] = {
                    "decisions": result,
                    "exact": result == gold,
                    "schema_valid": True,
                    "seconds": round(time.perf_counter() - start, 5),
                }
            except SchemaError as exc:
                row[name] = {
                    "decisions": None, "exact": False, "schema_valid": False,
                    "seconds": round(time.perf_counter() - start, 5),
                    "error": str(exc),
                }
        outcomes.append(row)
    return {
        "benchmark": "author_created_rule_labeled_diagnostic_not_independent_gold",
        "model": model_path, "head": str(head_path), "count": len(outcomes),
        "summary": {
            name: {
                "exact": sum(row[name]["exact"] for row in outcomes),
                "schema_valid": sum(row[name]["schema_valid"] for row in outcomes),
                "median_seconds": statistics.median(row[name]["seconds"] for row in outcomes),
            } for name in ("micro", "lens")
        },
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path("examples/challenge_rules.jsonl"))
    parser.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    parser.add_argument("--head", type=Path, default=Path("runs/phase4/micro-head-24.safetensors"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase5/challenge.json"))
    args = parser.parse_args()
    report = evaluate(args.fixture, args.model, args.head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"count": report["count"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
