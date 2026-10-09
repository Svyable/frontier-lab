"""ProofRoute: a bounded, auditable typed-rule fast path for DecisionLens.

Rules are explicitly supplied as structured data, NEVER inferred from prose.
A request either has complete validated coverage or fails closed to the
neural fallback. This is conventional symbolic execution, not an ML SOTA claim.
"""
import argparse
import json
import math
import time
from pathlib import Path

from .schema import SchemaError, validate_request

OPS = frozenset({"eq", "ne", "gt", "ge", "lt", "le", "and", "or", "not"})
MAX_NODES = 256
MAX_DEPTH = 12


class RuleNotApplicable(Exception):
    """Missing data or an unsupported comparison; do not guess."""


def _scalar(value):
    return value is None or type(value) in (str, bool, int, float)


def _finite(value):
    return not isinstance(value, float) or math.isfinite(value)


def _compile(expr, depth, budget):
    budget[0] += 1
    if budget[0] > MAX_NODES or depth > MAX_DEPTH:
        raise SchemaError("rule complexity limit exceeded")
    if not isinstance(expr, dict) or not isinstance(expr.get("op"), str) or expr["op"] not in OPS:
        raise SchemaError("invalid rule operator")
    op = expr["op"]
    if op in ("and", "or"):
        if set(expr) != {"op", "args"} or not isinstance(expr["args"], list) or not 2 <= len(expr["args"]) <= 16:
            raise SchemaError("and/or require 2-16 arguments")
        return (op, tuple(_compile(item, depth + 1, budget) for item in expr["args"]))
    if op == "not":
        if set(expr) != {"op", "arg"}:
            raise SchemaError("not requires one argument")
        return (op, _compile(expr["arg"], depth + 1, budget))
    if set(expr) != {"op", "field", "value"}:
        raise SchemaError("comparison requires op, field, value")
    field, value = expr["field"], expr["value"]
    if not isinstance(field, str) or not field or len(field) > 128 or "." in field or field.startswith("_"):
        raise SchemaError("field must be a simple nonprivate state key")
    if isinstance(value, dict):
        if set(value) != {"field"} or not isinstance(value["field"], str):
            raise SchemaError("field reference must contain one field key")
        other = value["field"]
        if not other or len(other) > 128 or "." in other or other.startswith("_"):
            raise SchemaError("invalid field reference")
        return (op, field, ("field", other))
    if not _scalar(value) or not _finite(value):
        raise SchemaError("comparison value must be a finite JSON scalar")
    if op in ("gt", "ge", "lt", "le") and (type(value) not in (int, float)):
        raise SchemaError("ordered comparison requires numeric constant")
    return (op, field, ("literal", value))


def compile_policy(task, policy):
    """Validate full question coverage and compile once for repeated requests."""
    validate_request({"model": "clef-flash", **task})
    if not isinstance(policy, dict) or set(policy) != set(task["questions"]):
        raise SchemaError("policy must cover exactly all questions")
    budget = [0]
    compiled = {}
    for qid, q in task["questions"].items():
        spec = policy[qid]
        if not isinstance(spec, dict):
            raise SchemaError("question policy must be an object")
        if q["type"] == "noul":
            if set(spec) != {"when"}:
                raise SchemaError("boolean policy requires when")
            compiled[qid] = ("noul", _compile(spec["when"], 0, budget))
        else:
            if set(spec) != {"when", "then", "else"}:
                raise SchemaError("choice policy requires when, then, else")
            if (not isinstance(spec["then"], str) or not isinstance(spec["else"], str)
                    or spec["then"] not in q["criteria"] or spec["else"] not in q["criteria"]):
                raise SchemaError("choice policy outputs must be allowed IDs")
            compiled[qid] = ("choice", _compile(spec["when"], 0, budget),
                             spec["then"], spec["else"])
    return compiled


def _eval(expr, state):
    op = expr[0]
    if op == "and":
        # Evaluate all arguments to avoid hiding a missing field behind false.
        values = [_eval(arg, state) for arg in expr[1]]
        return all(values)
    if op == "or":
        values = [_eval(arg, state) for arg in expr[1]]
        return any(values)
    if op == "not":
        return not _eval(expr[1], state)
    _, field, rhs = expr
    if not isinstance(state, dict) or field not in state:
        raise RuleNotApplicable("missing state field")
    actual = state[field]
    if rhs[0] == "field":
        if rhs[1] not in state:
            raise RuleNotApplicable("missing referenced field")
        constant = state[rhs[1]]
    else:
        constant = rhs[1]
    if not _scalar(actual) or not _finite(actual) or not _scalar(constant) or not _finite(constant):
        raise RuleNotApplicable("unsupported state value")
    if op in ("eq", "ne"):
        equal = (type(actual) is type(constant) and actual == constant)
        # Numeric int/float equality is safe; bool is deliberately distinct.
        if type(actual) in (int, float) and type(constant) in (int, float):
            equal = actual == constant
        return equal if op == "eq" else not equal
    if type(actual) not in (int, float) or type(constant) not in (int, float):
        raise RuleNotApplicable("ordered comparison requires numeric operands")
    if op == "gt":
        return actual > constant
    if op == "ge":
        return actual >= constant
    if op == "lt":
        return actual < constant
    return actual <= constant


def execute(compiled, state):
    if not isinstance(state, dict):
        raise RuleNotApplicable("rules require object state")
    answers = {}
    for qid, spec in compiled.items():
        result = _eval(spec[1], state)
        answers[qid] = result if spec[0] == "noul" else spec[2] if result else spec[3]
    return answers


def decide(task, policy=None, fallback=None):
    """Return (decisions, route). No partial rule/model mixing.

    An invalid policy is a developer error and raises SchemaError.
    A valid but inapplicable policy falls back or explicitly abstains.
    """
    validate_request({"model": "clef-flash", **task})
    if policy is not None:
        compiled = compile_policy(task, policy)
        try:
            return execute(compiled, task["state"]), "verified_rule"
        except RuleNotApplicable:
            pass
    if fallback is None:
        return None, "abstain"
    decisions = fallback(task)
    from .eval_student import parse_student_output
    parse_student_output(json.dumps({"decisions": decisions}), task["questions"])
    return decisions, "neural_fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    policy = json.loads(args.policy.read_text())
    start = time.perf_counter()
    answers, route = decide(task, policy)
    print(json.dumps({"decisions": answers, "route": route,
                      "seconds": time.perf_counter() - start}, indent=2))


if __name__ == "__main__":
    main()
