"""Measure whether an MLX student outputs *valid* typed decisions.

The teacher labels are pseudo-labels, not correctness ground truth. This
evaluation never reports teacher agreement as accuracy or probability
calibration, and keeps heavy MLX imports out of the CPU-only tests.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from .data import read_jsonl
from .schema import SchemaError, validate_request


def parse_student_output(raw: str, questions: dict) -> dict:
    """Fail closed: require exact IDs and JSON types; no repair/coercion."""
    try:
        payload = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise SchemaError("student output is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"decisions"}:
        raise SchemaError("output must contain only a decisions object")
    decisions = payload["decisions"]
    if not isinstance(decisions, dict) or set(decisions) != set(questions):
        raise SchemaError("decision IDs do not match question IDs")
    for qid, question in questions.items():
        answer = decisions[qid]
        if question["type"] == "choice":
            if not isinstance(answer, str) or answer not in question["criteria"]:
                raise SchemaError(f"{qid}: choice must be a valid option ID string")
        elif question["type"] == "noul":
            if not isinstance(answer, bool):
                raise SchemaError(f"{qid}: noul must be a JSON boolean")
        else:
            raise SchemaError(f"{qid}: unsupported question type")
    return decisions


def evaluate_student(
    *,
    model_path: str,
    adapter_path: str,
    data_dir: Path,
    splits: tuple[str, ...] = ("valid", "test"),
    max_tokens: int = 128,
) -> dict:
    """Load student once, generate deterministically, and report schema parity."""
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    rows = []
    for split in splits:
        dataset = read_jsonl(data_dir / f"{split}.jsonl")
        if not dataset:
            raise SchemaError(f"empty {split} split")
        for index, sample in enumerate(dataset):
            messages = sample["messages"]
            if [m["role"] for m in messages] != ["system", "user", "assistant"]:
                raise SchemaError(f"{split}/{index}: expected system, user, assistant messages")
            task = json.loads(messages[1]["content"])
            validate_request({"model": "clef-flash", **task})
            expected = json.loads(messages[2]["content"])["decisions"]
            parse_student_output(messages[2]["content"], task["questions"])
            rows.append((split, index, task, messages, expected))

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0)
    observations = []
    for split, index, task, messages, expected in rows:
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        start = time.perf_counter()
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        elapsed = time.perf_counter() - start
        try:
            predicted = parse_student_output(raw, task["questions"])
            schema_ok, error = True, None
        except SchemaError as exc:
            predicted, schema_ok, error = None, False, str(exc)
        observations.append({
            "split": split,
            "index": index,
            "schema_valid": schema_ok,
            "teacher_pseudolabel_exact_match": schema_ok and predicted == expected,
            "seconds": round(elapsed, 3),
            "error": error,
            "generated": raw,
            "teacher_pseudolabels": expected,
        })
    valid = sum(bool(x["schema_valid"]) for x in observations)
    parity = sum(bool(x["teacher_pseudolabel_exact_match"]) for x in observations)
    return {
        "evaluation_kind": "student_schema_and_teacher_pseudolabel_agreement_not_accuracy",
        "model_path": model_path,
        "adapter_path": adapter_path,
        "splits": list(splits),
        "count": len(observations),
        "schema_valid_count": valid,
        "schema_valid_rate": valid / len(observations),
        "teacher_pseudolabel_agreement_count": parity,
        "teacher_pseudolabel_agreement_rate": parity / len(observations),
        "median_latency_seconds": statistics.median(x["seconds"] for x in observations),
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MLX student typed decisions")
    parser.add_argument("--model", default="models/qwen3-4b-4bit")
    parser.add_argument("--adapter", default="runs/qwen3-4b-smoke")
    parser.add_argument("--data", type=Path, default=Path("data/mlx"))
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["valid", "test"])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen3-4b-smoke/eval.json"))
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 1024:
        parser.error("--max-tokens must be in [1,1024]")
    report = evaluate_student(
        model_path=args.model,
        adapter_path=args.adapter,
        data_dir=args.data,
        splits=tuple(args.splits),
        max_tokens=args.max_tokens,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "observations"}, indent=2))
    print(f"Saved detailed predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
