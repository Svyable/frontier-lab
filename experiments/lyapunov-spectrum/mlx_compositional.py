"""MLX/Metal follow-up: group-disjoint relational retrieval, tied vs untied.

Run on Apple Silicon from repository root. No pretrained data or paid services.
This script reports measurements only; the train/test map groups are disjoint.
The untied model matches executed block count, NOT parameter count.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from compositional_data import (SEQ_LEN, VOCAB_SIZE, canonical_mapping,
                                describe_split, mapping_pools, sample)
from mlx_train_loop import TiedBlock

VARIANTS = {"tied_weak": (True, 0.3),
            "tied_strong": (True, 3.0),
            "untied_strong": (False, 3.0)}


class RetrievalModel(nn.Module):
    def __init__(self, width, depth, variant):
        super().__init__()
        tied, injection = VARIANTS[variant]
        self.emb = nn.Embedding(VOCAB_SIZE, width)
        self.pos = nn.Embedding(SEQ_LEN, width)
        self.tied = tied
        self.injection = injection
        self.delta = 0.5
        self.depth = depth
        self.width = width
        if tied:
            self.core = TiedBlock(width, 4)
        else:
            self.blocks = [TiedBlock(width, 4) for _ in range(depth)]
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)

    def encode(self, tokens):
        return self.emb(tokens) + self.pos(mx.arange(SEQ_LEN)[None, :])

    def logits(self, h, e):
        return self.head(self.norm(h[:, -1, :] + e[:, -1, :]))

    def __call__(self, tokens, depth=None):
        depth = self.depth if depth is None else depth
        if not self.tied and depth > self.depth:
            raise ValueError("untied model has no blocks beyond training depth")
        e = self.encode(tokens)
        h = mx.zeros_like(e)
        for i in range(depth):
            block = self.core if self.tied else self.blocks[i]
            h = block(h, e, self.injection, self.delta)
        return self.logits(h, e)


def parameter_count(tree):
    if isinstance(tree, dict):
        return sum(parameter_count(v) for v in tree.values())
    if isinstance(tree, (tuple, list)):
        return sum(parameter_count(v) for v in tree)
    return int(tree.size) if hasattr(tree, "size") else 0


def classification_loss(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))


def accuracy_and_ce(model, x, y, depth):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    for offset in range(0, len(y), 128):
        bx, by = x[offset:offset + 128], y[offset:offset + 128]
        logits = model(bx, depth=depth)
        mx.eval(logits)
        correct += int(mx.sum(mx.argmax(logits, axis=-1) == by).item())
        loss += float(mx.sum(nn.losses.cross_entropy(logits, by)).item())
        total += len(by)
    return {"accuracy": round(correct / total, 5),
            "cross_entropy": round(loss / total, 5)}


def finite_time_probe(model, test_tokens, probes=12, steps=28):
    """Many-input finite-time JVP rates; NOT stationary Lyapunov exponents."""
    if not model.tied:
        return None
    model.eval()
    rng = np.random.default_rng(9507)
    rates, settle8, settle28 = [], [], []
    for i in range(min(probes, len(test_tokens))):
        e = model.encode(test_tokens[i:i + 1])
        h = mx.zeros_like(e)
        v = mx.array(rng.normal(size=(1, SEQ_LEN, model.width)).astype("float32"))
        v = v / mx.sqrt(mx.sum(v * v))
        logs, diffs = [], []
        for _ in range(steps):
            [new_h], [new_v] = mx.jvp(
                lambda z: model.core(z, e, model.injection, model.delta),
                [h], [v])
            norm = mx.sqrt(mx.sum(new_v * new_v))
            change = mx.sqrt(mx.sum((new_h - h) ** 2)) / (
                mx.sqrt(mx.sum(new_h * new_h)) + 1e-10)
            mx.eval(norm, change)
            logs.append(float(mx.log(norm + 1e-15).item()) / model.delta)
            diffs.append(float(change.item()))
            h, v = new_h, new_v / (norm + 1e-15)
        rates.append(float(np.mean(logs[8:])))
        settle8.append(diffs[7])
        settle28.append(diffs[-1])
    return {"inputs": len(rates), "rate_definition": "JVP mean of steps 9..28",
            "rate_mean": round(float(np.mean(rates)), 5),
            "rate_p10": round(float(np.percentile(rates, 10)), 5),
            "rate_p90": round(float(np.percentile(rates, 90)), 5),
            "fraction_positive": round(float(np.mean(np.asarray(rates) > 0)), 4),
            "step8_change_median": round(float(np.median(settle8)), 6),
            "step28_change_median": round(float(np.median(settle28)), 6)}


def early_exit_audit(model, tokens, labels, thresholds=(1e-2, 1e-3)):
    """First threshold crossing up to training depth vs fixed-depth accuracy.

    All states continue running to the trained depth; only the first eligible
    prediction is recorded. This is an offline audit, not an acceleration claim.
    """
    if not model.tied:
        return None
    model.eval()
    stats = {}
    for threshold in thresholds:
        n, correct_ref, correct_halt, early, harmed, steps_total = 0, 0, 0, 0, 0, 0
        for offset in range(0, len(labels), 128):
            x, y = tokens[offset:offset + 128], labels[offset:offset + 128]
            e = model.encode(x)
            h = mx.zeros_like(e)
            size = len(y)
            choices = np.zeros(size, dtype=np.int32)
            step_at = np.full(size, model.depth, dtype=np.int32)
            chosen = np.zeros(size, dtype=bool)
            for step in range(1, model.depth + 1):
                hn = model.core(h, e, model.injection, model.delta)
                delta = mx.sqrt(mx.sum((hn - h) ** 2, axis=(1, 2))) / (
                    mx.sqrt(mx.sum(hn * hn, axis=(1, 2))) + 1e-10)
                pred = mx.argmax(model.logits(hn, e), axis=-1)
                mx.eval(delta, pred)
                delta_np = np.array(delta)
                pred_np = np.array(pred)
                hit = (~chosen) & (delta_np < threshold)
                choices[hit] = pred_np[hit]
                step_at[hit] = step
                chosen |= hit
                h = hn
            choices[~chosen] = pred_np[~chosen]
            ref = pred_np
            actual = np.array(y)
            n += size
            correct_ref += int(np.sum(ref == actual))
            correct_halt += int(np.sum(choices == actual))
            early += int(np.sum(step_at < model.depth))
            harmed += int(np.sum((ref == actual) & (choices != actual)))
            steps_total += int(step_at.sum())
        stats[str(threshold)] = {
            "fixed_depth_accuracy": round(correct_ref / n, 5),
            "halt_accuracy": round(correct_halt / n, 5),
            "early_fraction": round(early / n, 5),
            "mean_steps": round(steps_total / n, 3),
            "fixed_correct_but_halt_wrong": harmed,
            "evaluated": n}
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--seeds", type=int, nargs="+", default=[2026, 711, 42])
    p.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument("--probe-inputs", type=int, default=12)
    p.add_argument("--output", default="experiments/lyapunov-spectrum/results/compositional_mlx.json")
    p.add_argument("--save-checkpoints", action="store_true")
    p.add_argument("--log-every", type=int, default=150)
    args = p.parse_args()
    if args.width % 4 or args.depth < 4 or args.log_every < 1 or args.probe_inputs < 1 or min(args.steps, args.batch, args.test_size) < 1:
        raise ValueError("invalid dimensions or counts")
    train_pool, test_pool = mapping_pools()
    tx_np, ty_np = sample(np.random.default_rng(91523), test_pool, args.test_size)
    assert all(len(set(tx[0:8:2])) == 4 for tx in tx_np)
    train_maps = set(map(lambda x: tuple(map(tuple, x)), train_pool))
    assert all(canonical_mapping(row) not in train_maps for row in tx_np)
    test_x, test_y = mx.array(tx_np), mx.array(ty_np)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"status": "in_progress", "backend": "MLX Metal",
               "task": "four-pair key/value retrieval, group-held-out mappings",
               "split": describe_split(), "steps": args.steps, "batch": args.batch,
               "width": args.width, "depth": args.depth, "lr": args.lr,
               "test_size": args.test_size, "seeds": args.seeds,
               "variants": args.variants, "runs": [],
               "warning": "untied matches depth/compute, not parameter count; no pretrained claims"}
    for variant in args.variants:
        for seed in args.seeds:
            mx.random.seed(seed)
            model = RetrievalModel(args.width, args.depth, variant)
            optimizer = optim.Adam(learning_rate=args.lr)
            grad = nn.value_and_grad(model, classification_loss)
            train_rng = np.random.default_rng(44000 + seed)
            t0 = time.monotonic()
            train_history = []
            model.train()
            for step in range(1, args.steps + 1):
                x_np, y_np = sample(train_rng, train_pool, args.batch)
                loss, grads = grad(model, mx.array(x_np), mx.array(y_np))
                optimizer.update(model, grads)
                mx.eval(loss, model.parameters(), optimizer.state)
                if step == 1 or step % args.log_every == 0:
                    value = round(float(loss.item()), 5)
                    train_history.append([step, value])
                    print(f"{variant} seed={seed} step={step} loss={value}", flush=True)
            model.eval()
            eval_train_np, labels_train_np = sample(np.random.default_rng(800 + seed), train_pool, 512)
            train_eval = accuracy_and_ce(model, mx.array(eval_train_np), mx.array(labels_train_np), args.depth)
            test_eval = {str(depth): accuracy_and_ce(model, test_x, test_y, depth)
                         for depth in ([0, 1, 4, args.depth, 2 * args.depth]
                                       if model.tied else [0, 1, 4, args.depth])}
            probe = finite_time_probe(model, test_x, args.probe_inputs)
            halt = early_exit_audit(model, test_x, test_y)
            record = {"variant": variant, "seed": seed,
                      "trainable_parameters": parameter_count(model.parameters()),
                      "training_losses": train_history,
                      "train_eval": train_eval, "test_by_depth": test_eval,
                      "finite_time_probe": probe, "halting_audit": halt,
                      "elapsed_seconds": round(time.monotonic() - t0, 2)}
            if args.save_checkpoints:
                ckpts = Path("experiments/lyapunov-spectrum/checkpoints")
                ckpts.mkdir(parents=True, exist_ok=True)
                checkpoint = ckpts / f"compositional_{variant}_s{args.steps}_seed{seed}.safetensors"
                model.save_weights(str(checkpoint))
                record["local_checkpoint"] = str(checkpoint)
            results["runs"].append(record)
            out.write_text(json.dumps(results, indent=2) + "\n")
            print("RESULT", json.dumps(record), flush=True)
    results["status"] = "complete"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
