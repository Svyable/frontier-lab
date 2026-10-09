"""Paired M4 benchmark: explicit deterministic policies versus MLX neural scorer.

Policy definitions are hand-authored and supplied out of band; no natural
language rule induction and no gold-label lookup. This measures the value
of routing verifiable decisions away from a language model.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .proofroute import compile_policy, execute


def cmp(op, field, value):
    return {"op": op, "field": field, "value": value}


def any_of(*args):
    return {"op": "or", "args": list(args)}


POLICIES = {
    "library": {
        "charge": {"when": cmp("gt", "days_overdue", 0),
                   "then": "fine_required", "else": "no_fine"},
        "block_renewal": {"when": cmp("ge", "renewal_count", 3)},
    },
    "vehicle": {
        "status": {"when": any_of(cmp("lt", "tire_depth_mm", 3),
                                  cmp("eq", "engine_check", True)),
                   "then": "maintenance_due", "else": "roadworthy"},
        "danger": {"when": cmp("lt", "tire_depth_mm", 2)},
    },
    "access": {
        "decision": {"when": {"op": "and", "args": [
            cmp("eq", "badge_active", True),
            cmp("ge", "clearance", {"field": "required"})]},
                     "then": "access_granted", "else": "access_denied"},
        "escalate": {"when": cmp("eq", "clearance", 0)},
    },
    "shipment": {
        "handling": {"when": any_of(cmp("eq", "fragile", True),
                                    cmp("gt", "weight_kg", 20)),
                     "then": "special_handling", "else": "standard_handling"},
        "long_haul": {"when": cmp("ge", "distance_km", 500)},
    },
}


def _policy(case):
    policy = json.loads(json.dumps(POLICIES[case["family"]]))
    return policy


def run(fixture, model_path, rounds):
    from mlx_lm import load
    from .decision_lens import decide as neural_decide
    cases = read_challenges(fixture)
    model, tokenizer = load(model_path)
    rows = []
    for repeat in range(rounds):
        for i, case in enumerate(cases):
            task, expected = case["task"], case["gold"]
            policy = _policy(case)
            paths = ("neural", "rule") if (repeat + i) % 2 else ("rule", "neural")
            outputs = {}
            for name in paths:
                start = time.perf_counter()
                if name == "rule":
                    if policy is None:
                        answer, route = None, "abstain"
                    else:
                        answer = execute(compile_policy(task, policy), task["state"])
                        route = "verified_rule"
                else:
                    answer, _ = neural_decide(model, tokenizer, task,
                                              sequence_scorer="trie",
                                              prompt_style="short_system")
                    route = "neural"
                outputs[name] = {
                    "seconds": time.perf_counter() - start,
                    "exact": answer == expected, "route": route,
                }
            rows.append({"id": case["id"], "family": case["family"], **outputs})
    result = {
        "benchmark": "author_labeled_explicit_policy_vs_neural_not_SOTA",
        "cases": len(cases), "rounds": rounds,
        "rule_coverage": sum(r["rule"]["route"] == "verified_rule" for r in rows),
        "rule_exact": sum(r["rule"]["exact"] for r in rows),
        "neural_exact": sum(r["neural"]["exact"] for r in rows),
        "rule_median_seconds": statistics.median(r["rule"]["seconds"] for r in rows),
        "neural_median_seconds": statistics.median(r["neural"]["seconds"] for r in rows),
        "rows": rows
    }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, default=Path("examples/holdout_rules_v1.jsonl"))
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("runs/phase8/proofroute.json"))
    args = p.parse_args()
    report = run(args.fixture, args.model, args.rounds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
