"""Apple Silicon MLX pilot: does input injection survive supervised learning?

Task: 3 randomly ordered key/value pairs (one binary value per unique key),
then a query key; output its associated bit. Train/test examples have disjoint
RNG streams; all conditions use matched initialization and minibatches.
The toy benchmark tests in-distribution relational retrieval, not reasoning SOTA.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

SEQ = 8
KEYS = 3
VAL0 = 3
MARK = 5


def examples(rng, batch):
    keys = np.argsort(rng.random((batch, KEYS)), axis=1).astype(np.int32)
    bits = rng.integers(0, 2, (batch, KEYS), dtype=np.int32)
    pick = rng.integers(0, KEYS, (batch,), dtype=np.int32)
    x = np.empty((batch, SEQ), dtype=np.int32)
    x[:, 0:6:2] = keys
    x[:, 1:6:2] = bits + VAL0
    x[:, 6] = MARK
    x[:, 7] = keys[np.arange(batch), pick]
    y = bits[np.arange(batch), pick]
    return mx.array(x), mx.array(y)


class TiedBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, width)

    def __call__(self, h, e, injection, delta):
        b, n, dim = h.shape
        z = self.norm1(h + injection * e)
        q, k, v = mx.split(self.qkv(z), 3, axis=-1)
        head_dim = dim // self.heads
        def split_heads(a):
            return mx.transpose(mx.reshape(a, (b, n, self.heads, head_dim)), (0, 2, 1, 3))
        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        logits = (q @ mx.swapaxes(k, -1, -2)) / math.sqrt(head_dim)
        mask = mx.triu(mx.full((n, n), -1e4), k=1)
        attention = mx.softmax(logits + mask, axis=-1)
        a = mx.transpose(attention @ v, (0, 2, 1, 3))
        a = self.out(mx.reshape(a, (b, n, dim)))
        z2 = self.norm2(h + injection * e + a)
        update = a + self.fc2(nn.gelu(self.fc1(z2)))
        return (1 - delta) * h + delta * update


class LoopModel(nn.Module):
    def __init__(self, width, heads, loops, injection, delta):
        super().__init__()
        self.emb = nn.Embedding(6, width)
        self.pos = nn.Embedding(SEQ, width)
        self.core = TiedBlock(width, heads)
        self.head_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 2)
        self.width = width
        self.loops = loops
        self.injection = injection
        self.delta = delta

    def encode(self, tokens):
        return self.emb(tokens) + self.pos(mx.arange(SEQ)[None, :])

    def evolve(self, e, loops=None):
        h = mx.zeros_like(e)
        for _ in range(self.loops if loops is None else loops):
            h = self.core(h, e, self.injection, self.delta)
        return h

    def __call__(self, tokens, loops=None):
        e = self.encode(tokens)
        h = self.evolve(e, loops=loops)
        return self.head(self.head_norm(h[:, -1, :] + e[:, -1, :]))


def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))


def evaluate(model, test_x, test_y, loops=None):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    for offset in range(0, len(test_y), 128):
        x = test_x[offset:offset + 128]
        y = test_y[offset:offset + 128]
        logits = model(x, loops=loops)
        mx.eval(logits)
        correct += int(mx.sum(mx.argmax(logits, axis=1) == y).item())
        loss += float(mx.sum(nn.losses.cross_entropy(logits, y)).item())
        total += len(y)
    return {"accuracy": round(correct / total, 5), "cross_entropy": round(loss / total, 5)}


def probe(model, tokens, steps=28):
    """Post-training along-loop finite-time tangent amplification and settling.
    This is not a proof of a stationary Lyapunov exponent or full spectrum.
    """
    model.eval()
    e = model.encode(tokens[:1])
    h = mx.zeros_like(e)
    rng = np.random.default_rng(1904)
    v = mx.array(rng.normal(size=(1, SEQ, model.width)).astype("float32"))
    v = v / mx.sqrt(mx.sum(v * v))
    deltas, logs = [], []
    for t in range(steps):
        f = lambda z: model.core(z, e, model.injection, model.delta)
        [hn], [vn] = mx.jvp(f, [h], [v])
        norm = mx.sqrt(mx.sum(vn * vn))
        step_diff = mx.sqrt(mx.sum((hn - h) ** 2)) / (mx.sqrt(mx.sum(hn * hn)) + 1e-10)
        mx.eval(norm, step_diff)
        logs.append(float(mx.log(norm + 1e-15).item()) / model.delta)
        deltas.append(float(step_diff.item()))
        v = vn / (norm + 1e-15)
        h = hn
    return {"mean_tangent_rate_after_8": round(float(np.mean(logs[8:])), 5),
            "settling_relative_step_8": round(deltas[7], 6),
            "settling_relative_step_28": round(deltas[-1], 6),
            "first_step_under_1e-3": next((i + 1 for i, x in enumerate(deltas) if x < 1e-3), None)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--injections", type=float, nargs="+", default=[0.3, 3.0])
    p.add_argument("--output", default="experiments/lyapunov-spectrum/results/mlx_training_pilot.json")
    p.add_argument("--checkpoints", default="experiments/lyapunov-spectrum/checkpoints")
    args = p.parse_args()
    if args.width % 4:
        raise ValueError("width must be divisible by 4")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpts = Path(args.checkpoints)
    ckpts.mkdir(parents=True, exist_ok=True)
    test_x, test_y = examples(np.random.default_rng(90210), 1024)
    results = {"backend": "MLX Metal Apple Silicon", "task": "3-pair binary key lookup",
               "training_examples": "fresh synthetic samples each step; disjoint test RNG",
               "steps": args.steps, "batch": args.batch, "width": args.width,
               "loops": args.loops, "delta": 0.5, "lr": args.lr, "seeds": [2026, 711, 42, 88, 1024],
               "test_examples": 1024, "conditions": []}
    print(f"MLX pilot: {args.steps} steps / condition; width={args.width} loops={args.loops}", flush=True)
    for injection in args.injections:
        for seed in results["seeds"]:
            mx.random.seed(seed)
            model = LoopModel(args.width, 4, args.loops, injection, 0.5)
            optimizer = optim.Adam(learning_rate=args.lr)
            grad_fn = nn.value_and_grad(model, loss_fn)
            train_rng = np.random.default_rng(8000 + seed)
            t0 = time.monotonic()
            history = []
            model.train()
            for step in range(1, args.steps + 1):
                x, y = examples(train_rng, args.batch)
                loss, grads = grad_fn(model, x, y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)
                if step % 60 == 0 or step == 1:
                    history.append([step, round(float(loss.item()), 5)])
                    print(f"injection={injection} seed={seed} step={step} loss={history[-1][1]}", flush=True)
            eval0 = evaluate(model, test_x, test_y, loops=0)
            eval1 = evaluate(model, test_x, test_y, loops=1)
            eval4 = evaluate(model, test_x, test_y, loops=4)
            eval8 = evaluate(model, test_x, test_y)
            eval16 = evaluate(model, test_x, test_y, loops=16)
            diag = probe(model, test_x)
            checkpoint_path = ckpts / f"loop_s{args.steps}_k{injection:g}_seed{seed}.safetensors"
            model.save_weights(str(checkpoint_path))
            result = {"injection": injection, "seed": seed, "loss_history": history,
                      "test_0_loops": eval0, "test_1_loop": eval1, "test_4_loops": eval4,
                      "test_8_loops": eval8, "test_16_loops": eval16,
                      "probe": diag, "runtime_sec": round(time.monotonic() - t0, 1),
                      "checkpoint_local": str(checkpoint_path)}
            results["conditions"].append(result)
            out.write_text(json.dumps(results, indent=2) + "\n")
            print("RESULT:", json.dumps(result), flush=True)
    print("DONE", str(out), flush=True)


if __name__ == "__main__":
    main()
