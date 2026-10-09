"""Experimental one-prefill question-conditioned decision head for MLX.

A frozen quantized transformer encodes each task once. Tiny learned projections
score option descriptions. This imitates Clef pseudo-labels, NOT gold accuracy.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from .data import read_jsonl
from .eval_student import parse_student_output
from .schema import SchemaError, validate_request


def candidate_descriptions(question):
    if question["type"] == "choice":
        return [(key, f"{key}: {question['criteria'][key]}")
                for key in sorted(question["criteria"])]
    if question["type"] == "noul":
        criteria = question.get("criteria", {})
        return [(False, "false: " + criteria.get("false", "Condition not met")),
                (True, "true: " + criteria.get("true", "Condition met"))]
    raise SchemaError("unsupported question type")


def read_cases(data_dir):
    result = {}
    for split in ("train", "valid", "test"):
        rows = read_jsonl(data_dir / f"{split}.jsonl")
        if not rows:
            raise SchemaError(f"empty {split} split")
        parsed = []
        for i, row in enumerate(rows):
            messages = row.get("messages")
            if not isinstance(messages, list) or [m.get("role") for m in messages] != [
                "system", "user", "assistant"
            ]:
                raise SchemaError(f"{split}/{i}: malformed messages")
            task = json.loads(messages[1]["content"])
            validate_request({"model": "clef-flash", **task})
            labels = parse_student_output(messages[2]["content"], task["questions"])
            parsed.append((task, labels))
        result[split] = parsed
    return result


def dataset_hash(data_dir):
    digest = hashlib.sha256()
    for split in ("train", "valid", "test"):
        digest.update(split.encode())
        digest.update((data_dir / f"{split}.jsonl").read_bytes())
    return digest.hexdigest()


def make_prompt(tokenizer, task):
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "Represent this decision task. Do not answer."},
        {"role": "user", "content": json.dumps(task, sort_keys=True, ensure_ascii=False)},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)


def features(model, tokenizer, task):
    """Exactly one transformer forward. Candidate vectors use embedding lookup."""
    import mlx.core as mx
    validate_request({"model": "clef-flash", **task})
    ids = tokenizer.encode(make_prompt(tokenizer, task), add_special_tokens=False)
    if not ids:
        raise SchemaError("empty prompt")
    hidden = model.model(mx.array([ids]))[0, -1, :].astype(mx.float32)
    hidden = mx.stop_gradient(hidden)
    entries = []
    embed = model.model.embed_tokens
    for qid, q in sorted(task["questions"].items()):
        descriptions = candidate_descriptions(q)
        query_ids = tokenizer.encode(
            qid + ": " + q.get("instructions", "Decide from the state"),
            add_special_tokens=False
        )
        if not query_ids:
            raise SchemaError("empty question")
        query = mx.mean(embed(mx.array(query_ids)).astype(mx.float32), axis=0)
        option_vectors = []
        for _, text in descriptions:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if not tokens:
                raise SchemaError("empty candidate")
            option_vectors.append(mx.mean(embed(mx.array(tokens)).astype(mx.float32), axis=0))
        vectors = mx.stack(option_vectors)
        query, vectors = mx.stop_gradient(query), mx.stop_gradient(vectors)
        entries.append((qid, [v for v, _ in descriptions], query, vectors))
    mx.eval(hidden, *[entry[2] for entry in entries], *[entry[3] for entry in entries])
    return hidden, entries


def build_head(hidden_size, rank):
    import mlx.nn as nn

    class MicroHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = nn.Linear(hidden_size, rank, bias=False)
            self.question = nn.Linear(hidden_size, rank, bias=False)
            self.option = nn.Linear(hidden_size, rank, bias=False)

        def __call__(self, hidden, query, options):
            context = self.context(hidden) + self.question(query)
            keys = self.option(options)
            return (keys * context).sum(axis=-1) / math.sqrt(rank)

    return MicroHead()


def capture(model, tokenizer, partitions):
    cache = {}
    for split, items in partitions.items():
        captured = []
        for task, labels in items:
            hidden, entries = features(model, tokenizer, task)
            indexed = []
            for qid, options, query, vectors in entries:
                indexed.append((qid, options, query, vectors, options.index(labels[qid])))
            captured.append((hidden, indexed, labels))
        cache[split] = captured
    return cache


def train(cache, hidden_size, rank, epochs, rate):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(42)
    head = build_head(hidden_size, rank)
    optimizer = optim.Adam(learning_rate=rate)

    def loss_fn(net, hidden, query, options, target):
        logits = net(hidden, query, options)
        return mx.logsumexp(logits) - logits[target]

    value_and_grad = nn.value_and_grad(head, loss_fn)
    losses = []
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for hidden, entries, _ in cache["train"]:
            for _, _, query, options, target in entries:
                loss, grads = value_and_grad(head, hidden, query, options, target)
                optimizer.update(head, grads)
                mx.eval(head.parameters(), optimizer.state)
                total += float(loss.item())
                n += 1
        losses.append({"epoch": epoch + 1, "train_loss": total / n})
    return head, losses


def decide(head, model, tokenizer, task):
    import mlx.core as mx
    hidden, entries = features(model, tokenizer, task)
    decisions = {}
    for qid, options, query, vectors in entries:
        logits = head(hidden, query, vectors)
        decisions[qid] = options[int(mx.argmax(logits).item())]
    # Ensure schema-valid response by construction.
    parse_student_output(json.dumps({"decisions": decisions}), task["questions"])
    return decisions


def evaluate(head, model, tokenizer, partitions, splits=("valid", "test")):
    result = {}
    for split in splits:
        observations = []
        for index, (task, labels) in enumerate(partitions[split]):
            start = time.perf_counter()
            decision = decide(head, model, tokenizer, task)
            seconds = time.perf_counter() - start
            observations.append({
                "index": index, "decisions": decision, "teacher_pseudolabels": labels,
                "teacher_pseudolabel_exact_match": decision == labels,
                "end_to_end_seconds": round(seconds, 5)
            })
        result[split] = {
            "count": len(observations),
            "schema_valid_count": len(observations),
            "teacher_pseudolabel_agreement_count": sum(
                row["teacher_pseudolabel_exact_match"] for row in observations
            ),
            "median_end_to_end_seconds": statistics.median(
                row["end_to_end_seconds"] for row in observations
            ),
            "observations": observations,
        }
    return result


def main():
    p = argparse.ArgumentParser(description="Frozen Qwen + one-prefill micro-head")
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--data", type=Path, default=Path("data/mlx_120"))
    p.add_argument("--output", type=Path, default=Path("runs/phase4/micro-head.json"))
    p.add_argument("--head-weights", type=Path, default=Path("runs/phase4/micro-head.safetensors"))
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    args = p.parse_args()
    if not 1 <= args.rank <= 256 or not 1 <= args.epochs <= 100:
        p.error("rank must be 1..256 and epochs 1..100")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01:
        p.error("invalid learning rate")
    from mlx_lm import load

    partitions = read_cases(args.data)
    model, tokenizer = load(args.model)
    capture_start = time.perf_counter()
    cache = capture(model, tokenizer, partitions)
    capture_seconds = time.perf_counter() - capture_start
    hidden_size = int(cache["train"][0][0].shape[-1])
    head, losses = train(cache, hidden_size, args.rank, args.epochs, args.learning_rate)
    results = evaluate(head, model, tokenizer, partitions)
    report = {
        "evaluation_kind": "one_prefill_micro_head_teacher_imitation_not_accuracy",
        "backbone": args.model, "dataset_sha256": dataset_hash(args.data),
        "rank": args.rank, "epochs": args.epochs, "learning_rate": args.learning_rate,
        "feature_capture_seconds": round(capture_seconds, 3),
        "training_losses": losses, "splits": results,
        "probability_note": "head scores uncalibrated; teacher labels are not gold",
        "latency_note": "end_to_end_seconds includes frozen backbone and head, excludes model load"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.head_weights.parent.mkdir(parents=True, exist_ok=True)
    head.save_weights(str(args.head_weights))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("splits", "training_losses")}, indent=2))
    print(json.dumps({s: {k: v for k, v in r.items() if k != "observations"}
                      for s, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
