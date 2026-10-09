"""Shared-backbone early exit for frozen Qwen3/MLX.

At a chosen intermediate layer, a trained head may accept a task. Otherwise
the *same hidden state* continues through the remaining transformer layers.
No second prefill. Experimental pseudo-label evaluation, not gold accuracy.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from .eval_student import parse_student_output
from .micro_head import (build_head, candidate_descriptions, dataset_hash,
                         make_prompt, read_cases, train)
from .schema import SchemaError, validate_request


def check_depth(depth, total):
    if type(depth) is not int or not 1 <= depth <= total:
        raise SchemaError("exit depth must be within transformer layers")


def option_features(model, tokenizer, task):
    import mlx.core as mx
    embed = model.model.embed_tokens
    entries = []
    for qid, q in sorted(task["questions"].items()):
        query_ids = tokenizer.encode(qid + ": " + q.get("instructions", "Decide from the state"),
                                     add_special_tokens=False)
        if not query_ids:
            raise SchemaError("empty query")
        query = mx.mean(embed(mx.array(query_ids)).astype(mx.float32), axis=0)
        descriptions = candidate_descriptions(q)
        vectors = []
        for _, desc in descriptions:
            ids = tokenizer.encode(desc, add_special_tokens=False)
            if not ids:
                raise SchemaError("empty candidate")
            vectors.append(mx.mean(embed(mx.array(ids)).astype(mx.float32), axis=0))
        entries.append((qid, [o for o, _ in descriptions], mx.stop_gradient(query),
                        mx.stop_gradient(mx.stack(vectors))))
    mx.eval(*[entry[2] for entry in entries], *[entry[3] for entry in entries])
    return entries


def backbone_until(model, tokenizer, task, depth):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    validate_request({"model": "clef-flash", **task})
    total = len(model.model.layers)
    check_depth(depth, total)
    ids = tokenizer.encode(make_prompt(tokenizer, task), add_special_tokens=False)
    if not ids:
        raise SchemaError("empty prompt")
    h = model.model.embed_tokens(mx.array([ids]))
    mask = create_attention_mask(h, None)
    for layer in model.model.layers[:depth]:
        h = layer(h, mask, None)
    mx.eval(h)
    return h, mask


def continue_backbone(model, hidden, mask, depth):
    """Resume from intermediate activation; never repeat earlier layers."""
    import mlx.core as mx
    check_depth(depth, len(model.model.layers))
    h = hidden
    for layer in model.model.layers[depth:]:
        h = layer(h, mask, None)
    h = model.model.norm(h)[0, -1, :].astype(mx.float32)
    mx.eval(h)
    return mx.stop_gradient(h)


def hidden_at(model, h):
    import mlx.core as mx
    result = model.model.norm(h)[0, -1, :].astype(mx.float32)
    mx.eval(result)
    return mx.stop_gradient(result)


def predict(head, hidden, entries, questions):
    import mlx.core as mx
    answers, margins = {}, {}
    for qid, options, query, vectors in entries:
        logits = head(hidden, query, vectors)
        mx.eval(logits)
        scores = [float(x) for x in logits.tolist()]
        if not all(math.isfinite(x) for x in scores):
            raise SchemaError("nonfinite head logits")
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        answers[qid] = options[order[0]]
        margins[qid] = scores[order[0]] - scores[order[1]]
    parse_student_output(json.dumps({"decisions": answers}), questions)
    return answers, margins


def decide(model, tokenizer, task, depth, early_head, final_head, threshold):
    """Run early head; if rejected resume SAME backbone activation."""
    from .selective_cascade import route
    start = time.perf_counter()
    h, mask = backbone_until(model, tokenizer, task, depth)
    entries = option_features(model, tokenizer, task)
    early, margins = predict(early_head, hidden_at(model, h), entries, task["questions"])
    if route(margins, threshold) == "micro":
        return early, "early", time.perf_counter() - start, margins
    final_hidden = continue_backbone(model, h, mask, depth)
    final, _ = predict(final_head, final_hidden, entries, task["questions"])
    return final, "final", time.perf_counter() - start, margins


def capture_early(model, tokenizer, partitions, depth):
    """Capture early-layer frozen representations for train/valid only."""
    import mlx.core as mx
    cache = {}
    for split in ("train", "valid"):
        captured = []
        for task, labels in partitions[split]:
            h, _ = backbone_until(model, tokenizer, task, depth)
            hidden = hidden_at(model, h)
            entries = option_features(model, tokenizer, task)
            indexed = [(qid, opts, query, vec, opts.index(labels[qid]))
                       for qid, opts, query, vec in entries]
            captured.append((hidden, indexed, labels))
        cache[split] = captured
    return cache


def validation_rows(head, cache):
    rows = []
    for hidden, entries, expected in cache["valid"]:
        simple = [(qid, opts, query, vectors) for qid, opts, query, vectors, _ in entries]
        task_questions = {qid: {"type": "noul"} if type(opts[0]) is bool else
                          {"type": "choice", "criteria": {str(o): str(o) for o in opts}}
                          for qid, opts, _, _ in simple}
        answer, margins = predict(head, hidden, simple, task_questions)
        rows.append({"margins": margins, "micro": answer, "expected": expected})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/qwen3-4b-60iter-4bit")
    p.add_argument("--data", type=Path, default=Path("data/mlx_120"))
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--target-validation-agreement", type=float, default=0.95)
    p.add_argument("--final-head", type=Path, default=Path("runs/phase4/micro-head-24.safetensors"))
    p.add_argument("--output", type=Path, default=Path("runs/phase10/early-exit.json"))
    p.add_argument("--weights", type=Path, default=Path("runs/phase10/early-head.safetensors"))
    args = p.parse_args()
    from mlx_lm import load
    from .selective_cascade import choose_threshold
    partitions = read_cases(args.data)
    model, tokenizer = load(args.model)
    check_depth(args.depth, len(model.model.layers))
    if args.rank != 32 or not 1 <= args.epochs <= 100:
        p.error("this experiment uses rank 32 and epochs 1..100")
    cache = capture_early(model, tokenizer, partitions, args.depth)
    size = int(cache["train"][0][0].shape[-1])
    early, losses = train(cache, size, args.rank, args.epochs, 1e-4)
    final = build_head(size, 32)
    final.load_weights(str(args.final_head))
    threshold = choose_threshold(validation_rows(early, cache),
                                 args.target_validation_agreement)
    observations = []
    for index, (task, expected) in enumerate(partitions["test"]):
        answer, path, seconds, margins = decide(model, tokenizer, task, args.depth,
                                                 early, final, threshold)
        observations.append({"index": index, "path": path, "seconds": seconds,
                             "teacher_agreement": answer == expected,
                             "margins": margins})
    report = {
        "evaluation_kind": "family_disjoint_teacher_pseudolabels_previously_inspected_not_gold",
        "dataset_sha256": dataset_hash(args.data),
        "depth": args.depth, "total_layers": len(model.model.layers),
        "rank": args.rank, "epochs": args.epochs,
        "threshold": threshold,
        "threshold_selection": "validation_only_empirical_not_calibrated",
        "target_validation_agreement": args.target_validation_agreement,
        "test_cases": len(observations),
        "test_teacher_agreement": sum(r["teacher_agreement"] for r in observations),
        "test_early_accepted": sum(r["path"] == "early" for r in observations),
        "median_test_seconds": statistics.median(r["seconds"] for r in observations),
        "training_losses": losses,
        "test_observations": observations,
    }
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    early.save_weights(str(args.weights))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("training_losses", "test_observations")}, indent=2))


if __name__ == "__main__":
    main()
