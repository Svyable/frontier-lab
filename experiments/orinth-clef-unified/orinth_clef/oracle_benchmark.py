"""Reproducible, model-independent exact-rule benchmark and paired MLX runner.

Rules are authored as structured policies; labels come from the policy
interpreter, not an LLM or teacher. This is NOT independent human gold.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from .challenge import read_challenges
from .eval_student import parse_student_output
from .proofroute import compile_policy, execute
from .schema import SchemaError


def cases(seed=20261009, per_family=16):
    if type(seed) is not int or type(per_family) is not int or not 1 <= per_family <= 100:
        raise SchemaError("invalid deterministic fixture parameters")
    rng = random.Random(seed)
    output = []
    for family in ("threshold", "category", "compound", "relative"):
        for i in range(per_family):
            severity = rng.randrange(-5, 11)
            limit = rng.randrange(-4, 10)
            group = rng.choice(("billing", "technical", "legal", "general"))
            other = rng.randrange(-5, 11)
            state = {"severity": severity, "limit": limit, "group": group, "reference": other}
            if family == "threshold":
                predicate = {"op": "ge", "field": "severity", "value": limit}
                instruction = f"True exactly when severity is greater than or equal to {limit}."
            elif family == "category":
                target = rng.choice(("billing", "technical", "legal", "general"))
                predicate = {"op": "eq", "field": "group", "value": target}
                instruction = f"True only if group is exactly '{target}'."
            elif family == "compound":
                target = rng.choice(("billing", "technical", "legal", "general"))
                predicate = {"op": "and", "args": [
                    {"op": "ge", "field": "severity", "value": limit},
                    {"op": "eq", "field": "group", "value": target}]}
                instruction = (f"True only when severity >= {limit} AND group equals '{target}'.")
            else:
                predicate = {"op": "gt", "field": "severity", "value": {"field": "reference"}}
                instruction = "True only when severity is strictly greater than reference."
            # Both typed outputs must agree with the same explicit rule.
            task = {"state": state, "questions": {
                "flag": {"type": "noul", "instructions": instruction},
                "route": {"type": "choice", "instructions": instruction + " Choose yes if true.",
                          "criteria": {"yes": "Condition is satisfied",
                                       "no": "Condition is not satisfied"}}}}
            policy = {"flag": {"when": predicate},
                      "route": {"when": predicate, "then": "yes", "else": "no"}}
            gold = execute(compile_policy(task, policy), state)
            parse_student_output(json.dumps({"decisions": gold}), task["questions"])
            output.append({"id": f"{family}-{i:03d}", "family": family, "task": task,
                           "policy": policy, "gold": gold, "rule": instruction})
    return output


def write_fixture(path, seed=20261009, per_family=16):
    path = Path(path)
    if path.exists():
        raise SchemaError("fixture exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = cases(seed, per_family)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_fixture(path):
    rows = read_challenges(Path(path))
    for row in rows:
        if "policy" not in row:
            raise SchemaError("missing independent executable policy")
        expected = execute(compile_policy(row["task"], row["policy"]), row["task"]["state"])
        if expected != row["gold"]:
            raise SchemaError("gold/policy disagreement: " + row["id"])
    return rows


def evaluate(path, model_dir, early_weights, final_weights, report_path):
    from mlx_lm import load
    from .decision_lens import decide as lens_decide
    from .early_exit import decide as early_decide
    from .micro_head import build_head, decide as full_decide
    rows = verify_fixture(path)
    model, tokenizer = load(str(model_dir))
    report = json.loads(Path(report_path).read_text())
    early = build_head(2560, report["rank"])
    early.load_weights(str(early_weights))
    final = build_head(2560, 32)
    final.load_weights(str(final_weights))
    results = []
    for i, row in enumerate(rows):
        task, gold = row["task"], row["gold"]
        record = {"id": row["id"], "family": row["family"]}
        # Rotate order so a method never systematically benefits from warmth.
        methods = ("exit", "full", "lens") if i % 2 else ("lens", "full", "exit")
        for method in methods:
            start = time.perf_counter()
            if method == "exit":
                answer, route, _, _ = early_decide(
                    model, tokenizer, task, report["depth"], early, final, report["threshold"])
            elif method == "full":
                answer = full_decide(final, model, tokenizer, task)
                route = "full"
            else:
                answer = lens_decide(model, tokenizer, task, sequence_scorer="trie",
                                     prompt_style="short_system")[0]
                route = "lens"
            parse_student_output(json.dumps({"decisions": answer}), task["questions"])
            record[method] = {"correct": answer == gold, "seconds": time.perf_counter() - start,
                              "route": route, "decisions": answer}
        start = time.perf_counter()
        exact = execute(compile_policy(task, row["policy"]), task["state"])
        record["symbolic"] = {"correct": exact == gold, "seconds": time.perf_counter() - start,
                              "route": "verified_rule"}
        results.append(record)
    summary = {}
    for method in ("symbolic", "exit", "full", "lens"):
        values = [r[method] for r in results]
        times = sorted(v["seconds"] for v in values)
        summary[method] = {
            "exact": sum(v["correct"] for v in values),
            "count": len(values),
            "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
            "p95_seconds": times[min(len(times) - 1, int(0.95 * len(times)))],
        }
    summary["exit"]["accepted"] = sum(r["exit"]["route"] == "early" for r in results)
    summary["exit"]["accepted_wrong"] = sum(
        r["exit"]["route"] == "early" and not r["exit"]["correct"] for r in results)
    by_family = {}
    for family in sorted({r["family"] for r in results}):
        family_rows = [r for r in results if r["family"] == family]
        by_family[family] = {
            "count": len(family_rows),
            **{name: sum(r[name]["correct"] for r in family_rows)
               for name in ("symbolic", "exit", "full", "lens")},
            "early_accepted": sum(r["exit"]["route"] == "early" for r in family_rows),
        }
    return {"benchmark": "author_programmed_rule_oracle_not_independent_human_gold",
            "family_breakdown": by_family,
            "fixture_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "training_report_sha256": hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
            "summary": summary, "rows": results}


def diagnostic_gate(report):
    """Conservative local gate, NOT statistical confidence or release approval."""
    summary = report["summary"]
    exit_stats, lens = summary["exit"], summary["lens"]
    failures = []
    if exit_stats["exact"] < lens["exact"]:
        failures.append("early exit loses exact correctness to DecisionLens")
    if "accepted_wrong" not in exit_stats or exit_stats["accepted_wrong"] > 0:
        failures.append("early exit accepted incorrect or unverified oracle decisions")
    if exit_stats["mean_seconds"] >= lens["mean_seconds"]:
        failures.append("early exit is not faster in mean latency")
    return {"passed": not failures, "failures": failures,
            "scope": "author_rule_diagnostic_only_not_public_release_approval"}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    make = sub.add_parser("generate")
    make.add_argument("--output", type=Path, default=Path("examples/oracle_rules_v2.jsonl"))
    make.add_argument("--seed", type=int, default=20261009)
    make.add_argument("--per-family", type=int, default=16)
    run = sub.add_parser("evaluate")
    run.add_argument("--fixture", type=Path, default=Path("examples/oracle_rules_v2.jsonl"))
    run.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    run.add_argument("--early-weights", type=Path, default=Path("runs/phase10/early24.safetensors"))
    run.add_argument("--final-weights", type=Path, default=Path("runs/phase4/micro-head-24.safetensors"))
    run.add_argument("--training-report", type=Path, default=Path("runs/phase10/early24-target80.json"))
    run.add_argument("--output", type=Path, default=Path("runs/phase11/oracle_v2.json"))
    gate = sub.add_parser("gate")
    gate.add_argument("--report", type=Path, default=Path("runs/phase11/oracle_v2.json"))
    args = p.parse_args()
    if args.command == "generate":
        print(json.dumps({"sha256": write_fixture(args.output, args.seed, args.per_family),
                          "cases": len(verify_fixture(args.output))}))
    elif args.command == "gate":
        report = json.loads(args.report.read_text())
        # Older reports may lack accepted_wrong; derive it from recorded rows.
        if "accepted_wrong" not in report["summary"]["exit"]:
            report["summary"]["exit"]["accepted_wrong"] = sum(
                r["exit"]["route"] == "early" and not r["exit"]["correct"]
                for r in report["rows"])
        outcome = diagnostic_gate(report)
        print(json.dumps(outcome, indent=2))
        if not outcome["passed"]:
            raise SystemExit(1)
    else:
        report = evaluate(args.fixture, args.model, args.early_weights,
                          args.final_weights, args.training_report)
        report["diagnostic_gate"] = diagnostic_gate(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2))
        print(json.dumps(report["diagnostic_gate"], indent=2))


if __name__ == "__main__":
    main()
