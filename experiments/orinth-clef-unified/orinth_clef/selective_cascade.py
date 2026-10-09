"""Experimental selective cascade: one-pass micro-head, optional neural fallback.

Raw logit margins are not calibrated confidence. Threshold must be selected
on separate validation data; this module does not claim risk guarantees.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .schema import SchemaError


def route(margins, threshold):
    """Whole-task gating: one weak answer sends the entire task to fallback."""
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold) or threshold < 0:
        raise SchemaError("threshold must be finite and nonnegative")
    if not isinstance(margins, dict) or not margins:
        raise SchemaError("missing margins")
    if not all(type(m) in (float, int) and math.isfinite(m) and m >= 0 for m in margins.values()):
        raise SchemaError("nonfinite or invalid margin")
    return "micro" if min(margins.values()) >= threshold else "fallback"


def choose_threshold(validation_rows, min_agreement=0.95):
    """Select max coverage under an *empirical* validation agreement floor.

    Empty acceptance means fallback always. This is not a conformal guarantee.
    """
    if not 0 <= min_agreement <= 1:
        raise SchemaError("invalid agreement target")
    if not validation_rows:
        raise SchemaError("empty validation")
    margins = [min(r["margins"].values()) for r in validation_rows]
    candidates = sorted(set(margins))
    candidates.append(max(margins) + 1.0)
    eligible = []
    for threshold in candidates:
        accepted = [r for r, m in zip(validation_rows, margins) if m >= threshold]
        if accepted and sum(r["micro"] == r["expected"] for r in accepted) / len(accepted) >= min_agreement:
            eligible.append((len(accepted), -threshold, threshold))
    return max(eligible)[2] if eligible else max(margins) + 1.0


def collect(head, model, tokenizer, cases, gold_key):
    from .micro_head import decide_with_margins
    from .decision_lens import decide as neural_decide
    rows = []
    for item in cases:
        task = item["task"]
        start = time.perf_counter()
        micro, margins = decide_with_margins(head, model, tokenizer, task)
        micro_seconds = time.perf_counter() - start
        start = time.perf_counter()
        fallback, _ = neural_decide(model, tokenizer, task, sequence_scorer="trie",
                                    prompt_style="short_system")
        fallback_seconds = time.perf_counter() - start
        rows.append({"id": item["id"], "expected": item[gold_key],
                     "micro": micro, "margins": margins, "fallback": fallback,
                     "micro_seconds": micro_seconds, "fallback_seconds": fallback_seconds})
    return rows


def summarize(rows, threshold):
    outcomes = []
    for row in rows:
        path = route(row["margins"], threshold)
        answer = row["micro"] if path == "micro" else row["fallback"]
        # The fallback is invoked *after* the micro-head in a real cascade.
        elapsed = row["micro_seconds"] + (row["fallback_seconds"] if path == "fallback" else 0)
        outcomes.append({"id": row["id"], "path": path, "exact": answer == row["expected"],
                         "seconds": elapsed})
    return {
        "threshold": threshold, "count": len(outcomes),
        "micro_accepted": sum(x["path"] == "micro" for x in outcomes),
        "cascade_exact": sum(x["exact"] for x in outcomes),
        "micro_alone_exact": sum(r["micro"] == r["expected"] for r in rows),
        "fallback_alone_exact": sum(r["fallback"] == r["expected"] for r in rows),
        "median_cascade_seconds": statistics.median(x["seconds"] for x in outcomes),
        "median_micro_seconds": statistics.median(r["micro_seconds"] for r in rows),
        "median_fallback_seconds": statistics.median(r["fallback_seconds"] for r in rows),
        "outcomes": outcomes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--head", type=Path, default=Path("runs/phase4/micro-head-24.safetensors"))
    p.add_argument("--fixture", type=Path, default=Path("examples/holdout_rules_v1.jsonl"))
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Predeclared raw margin; not calibrated, do not tune on test")
    p.add_argument("--output", type=Path, default=Path("runs/phase9/selective-cascade.json"))
    args = p.parse_args()
    from mlx_lm import load
    from .challenge import read_challenges
    from .micro_head import build_head
    cases = read_challenges(args.fixture)
    model, tokenizer = load(args.model)
    head = build_head(2560, 32)
    head.load_weights(str(args.head))
    rows = collect(head, model, tokenizer, cases, "gold")
    result = summarize(rows, args.threshold)
    result["evaluation_kind"] = "author_created_rule_diagnostic_not_independent_gold"
    result["threshold_selection"] = "predeclared_cli_margin_not_calibrated_not_tuned_on_fixture"
    result["latency_note"] = "warm per-path serial timings, cascade adds both if fallback"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
