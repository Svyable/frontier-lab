"""Audit teacher predictions against *unverified construction intent*.

Not a human gold benchmark, accuracy claim, or calibration measurement.
"""
import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .data import read_jsonl
from .schema import SchemaError, pseudo_labels, validate_response


def summarize_teacher(rows: list[dict], references: list[dict]) -> dict:
    ref_by_id = {}
    for ref in references:
        if ref.get("reference_kind") != "construction_intent_unverified":
            raise SchemaError("only explicit unverified construction intents accepted")
        ident = ref.get("id")
        if not ident or ident in ref_by_id:
            raise SchemaError("duplicate or missing construction reference")
        ref_by_id[ident] = ref
    if not rows or len(rows) != len(ref_by_id):
        raise SchemaError("teacher and construction reference sample counts differ")
    observed = set()
    label_agree = Counter()
    total = Counter()
    choice_confidences = []
    noul_probs = []
    teacher_revisions = set()
    for row in rows:
        ident = row.get("id")
        if ident not in ref_by_id or ident in observed:
            raise SchemaError(f"unknown or duplicate teacher case: {ident}")
        observed.add(ident)
        ref = ref_by_id[ident]
        if row.get("group_id") != ref.get("group_id"):
            raise SchemaError(f"mismatched group for {ident}")
        request = row["request"]
        response = validate_response(request, row["response"])
        labels = pseudo_labels(request, response)
        expected = ref["intent_decisions"]
        if set(labels) != set(expected):
            raise SchemaError(f"mismatched question keys for {ident}")
        for qid, label in labels.items():
            kind = request["questions"][qid]["type"]
            total[kind] += 1
            label_agree[kind] += int(label == expected[qid])
            answer = response["answers"][qid]
            if kind == "choice":
                choice_confidences.append(answer["probabilities"][answer["choice"]])
            else:
                noul_probs.append(answer["noul"])
        teacher_revisions.add(row["teacher_revision"])
    return {
        "evaluation_kind": "teacher_vs_unverified_synthetic_construction_intent_not_accuracy",
        "sample_count": len(rows),
        "family_count": len({row["group_id"] for row in rows}),
        "teacher_revisions": sorted(teacher_revisions),
        "decision_counts": dict(total),
        "construction_intent_agreement": {k: {"matched": label_agree[k], "total": total[k]} for k in sorted(total)},
        "mean_selected_choice_probability": statistics.mean(choice_confidences),
        "mean_noul_probability": statistics.mean(noul_probs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize_teacher(read_jsonl(args.raw), read_jsonl(args.references))
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
