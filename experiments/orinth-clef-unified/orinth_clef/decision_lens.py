"""DecisionLens: schema-native logit selection without autoregressive JSON.

Research prototype. A shared KV cache advances one question at a time.
Multi-token candidates are teacher-forced along separate prefix-cache branches.
Normalized candidate scores are NOT calibrated event probabilities.
No external services.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .data import read_jsonl
from .eval_student import parse_student_output
from .schema import SchemaError, validate_request


def candidate_values(question: dict) -> list[str | bool]:
    if question["type"] == "choice":
        options = sorted(question["criteria"])
        if not 2 <= len(options) <= 50:
            raise SchemaError("choice requires 2-50 options")
        return options
    if question["type"] == "noul":
        return [False, True]
    raise SchemaError("unsupported decision type")


def candidate_token_ids(tokenizer, question: dict) -> dict:
    """Only single-token candidates; refuse ambiguous token IDs or silent truncation."""
    tokens = {}
    for value in candidate_values(question):
        spelling = value if isinstance(value, str) else ("true" if value else "false")
        ids = tokenizer.encode(spelling, add_special_tokens=False)
        if len(ids) != 1:
            raise SchemaError(f"candidate {spelling!r} is not a single token")
        if ids[0] in tokens.values():
            raise SchemaError("ambiguous candidate tokenization")
        tokens[value] = ids[0]
    return tokens


def select_from_logits(candidate_ids: dict, scores: list[float]) -> tuple:
    """Select argmax deterministically; return normalized *token scores*, not confidence."""
    if not candidate_ids:
        raise SchemaError("no candidates")
    if len(scores) != len(candidate_ids) or not all(math.isfinite(x) for x in scores):
        raise SchemaError("invalid candidate scores")
    labels = list(candidate_ids)
    winner = max(range(len(scores)), key=lambda i: scores[i])
    ceiling = max(scores)
    weights = [math.exp(s - ceiling) for s in scores]
    total = sum(weights)
    return labels[winner], {str(labels[i]): weights[i] / total for i in range(len(labels))}


def decide(model, tokenizer, task: dict, *, sequence_scorer="reference",
           prompt_style="baseline") -> tuple[dict, dict]:
    """Select typed candidates with KV-cached, constrained token scoring."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from .candidate_sequences import candidate_sequences, score_sequences
    if sequence_scorer not in ("reference", "trie", "rollback"):
        raise SchemaError("unsupported sequence scorer")
    if sequence_scorer == "trie":
        from .prefix_trie import score_sequences_trie
    elif sequence_scorer == "rollback":
        from .rollback_trie import score_sequences_rollback

    if prompt_style not in ("baseline", "compact", "compact_json", "short_system"):
        raise SchemaError("unsupported prompt style")
    validate_request({"model": "clef-flash", **task})
    questions = task["questions"]
    if prompt_style in ("baseline", "compact_json"):
        system = (
            "Return only compact JSON with a single 'decisions' object. "
            "For 'choice' questions return exactly one allowed option ID; "
            "for 'noul' questions return a JSON boolean. Do not add explanation."
        )
    else:
        system = 'Return JSON {"decisions":{...}}. choice: allowed ID; noul: boolean. No prose.'
    content = json.dumps(
        task, ensure_ascii=False, sort_keys=True,
        separators=(",", ":") if prompt_style in ("compact", "compact_json") else None
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ordered = sorted(questions)
    first_key = ordered[0]
    first_prefix = ('{"decisions": {' + json.dumps(first_key) + ': ' +
                    ('"' if questions[first_key]["type"] == "choice" else ''))
    # Cache the entire shared prompt and then advance only the selected answer
    # and next field name. This avoids a second full-context prefill.
    cache = make_prompt_cache(model)
    tokens = tokenizer.encode(prompt + first_prefix, add_special_tokens=False)
    answers = {}
    score_distributions = {}
    for index, qid in enumerate(ordered):
        question = questions[qid]
        sequences = candidate_sequences(tokenizer, question)
        logits = model(mx.array([tokens]), cache=cache)[0, -1, :]
        mx.eval(logits)
        if all(len(ids) == 1 for ids in sequences.values()):
            # Keep the fast, previously benchmarked single-token path.
            raw_scores = [float(logits[ids[0]].item()) for ids in sequences.values()]
            answer, distribution = select_from_logits(sequences, raw_scores)
        else:
            # Multi-token options use the selected experimental scorer.
            # Only the chosen continuation advances the original prefix.
            if sequence_scorer == "reference":
                scorer = score_sequences
            elif sequence_scorer == "trie":
                scorer = score_sequences_trie
            else:
                scorer = score_sequences_rollback
            answer, distribution = scorer(model, cache, logits, sequences)
        answers[qid] = answer
        score_distributions[qid] = distribution
        if index + 1 < len(ordered):
            next_qid = ordered[index + 1]
            tail = ('"' if isinstance(answer, str) else '') + ', ' + json.dumps(next_qid) + ': '
            if questions[next_qid]["type"] == "choice":
                tail += '"'
            tokens = sequences[answer] + tokenizer.encode(tail, add_special_tokens=False)
    payload = '{"decisions": ' + json.dumps(answers, sort_keys=True) + '}'
    parse_student_output(payload, questions)
    return answers, score_distributions


def evaluate(*, model_path: str, adapter_path: str | None, data_dir: Path,
             splits: tuple[str, ...] = ("valid", "test"),
             sequence_scorer: str = "reference",
             prompt_style: str = "baseline") -> dict:
    from mlx_lm import load

    cases = []
    for split in splits:
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        if not rows:
            raise SchemaError(f"empty {split} split")
        for index, row in enumerate(rows):
            messages = row["messages"]
            task = json.loads(messages[1]["content"])
            expected = json.loads(messages[2]["content"])["decisions"]
            cases.append((split, index, task, expected))
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    observations = []
    for split, index, task, expected in cases:
        start = time.perf_counter()
        try:
            decisions, distributions = decide(
                model, tokenizer, task, sequence_scorer=sequence_scorer,
                prompt_style=prompt_style
            )
            valid, error = True, None
        except SchemaError as exc:
            decisions, distributions, valid, error = None, None, False, str(exc)
        elapsed = time.perf_counter() - start
        observations.append({
            "split": split, "index": index, "schema_valid": valid,
            "teacher_pseudolabel_exact_match": valid and decisions == expected,
            "seconds": round(elapsed, 4), "error": error,
            "decisions": decisions, "normalized_candidate_scores_not_calibrated": distributions,
            "teacher_pseudolabels": expected,
        })
    return {
        "evaluation_kind": "decision_lens_token_logit_selection_not_calibrated_not_accuracy",
        "model_path": model_path, "adapter_path": adapter_path,
        "sequence_scorer": sequence_scorer,
        "prompt_style": prompt_style,
        "splits": list(splits), "count": len(observations),
        "schema_valid_count": sum(x["schema_valid"] for x in observations),
        "teacher_pseudolabel_agreement_count": sum(x["teacher_pseudolabel_exact_match"] for x in observations),
        "median_latency_seconds": statistics.median(x["seconds"] for x in observations),
        "observations": observations,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate zero-autoregressive-token DecisionLens")
    p.add_argument("--model", default="models/qwen3-4b-4bit")
    p.add_argument("--adapter", default="runs/phase2/qwen3-4b-60iter")
    p.add_argument("--no-adapter", action="store_true")
    p.add_argument("--sequence-scorer", choices=("reference", "trie", "rollback"), default="reference")
    p.add_argument("--prompt-style", choices=("baseline", "compact", "compact_json", "short_system"), default="baseline")
    p.add_argument("--data", type=Path, default=Path("data/mlx_120"))
    p.add_argument("--splits", nargs="+", choices=["valid", "test", "train"], default=["valid", "test"])
    p.add_argument("--output", type=Path, default=Path("runs/phase3/decision-lens.json"))
    args = p.parse_args()
    report = evaluate(model_path=args.model, adapter_path=None if args.no_adapter else args.adapter,
                      data_dir=args.data, splits=tuple(args.splits),
                      sequence_scorer=args.sequence_scorer,
                      prompt_style=args.prompt_style)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
