"""Lightweight Apple-Silicon tests for the looped-network training pilot."""
import math
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_train_loop import LoopModel, examples, loss_fn, probe


class MLXLoopTests(unittest.TestCase):
    def test_key_lookup_labels(self):
        x, y = examples(np.random.default_rng(42), 96)
        x, y = np.asarray(x), np.asarray(y)
        for row, label in zip(x, y):
            keys = row[0:6:2]
            values = row[1:6:2] - 3
            self.assertEqual(sorted(keys.tolist()), [0, 1, 2])
            self.assertEqual(int(label), int(values[np.where(keys == row[7])[0][0]]))
            self.assertEqual(int(row[6]), 5)

    def test_forward_and_training(self):
        mx.random.seed(2026)
        model = LoopModel(16, 4, 4, 1.0, 0.5)
        x, y = examples(np.random.default_rng(21), 12)
        logits = model(x)
        self.assertEqual(logits.shape, (12, 2))
        self.assertEqual(model(x, loops=0).shape, (12, 2))
        optimizer = optim.Adam(learning_rate=0.001)
        f = nn.value_and_grad(model, loss_fn)
        loss, grads = f(model, x, y)
        optimizer.update(model, grads)
        mx.eval(loss, model.parameters())
        self.assertTrue(math.isfinite(float(loss.item())))

    def test_jvp_probe(self):
        mx.random.seed(2026)
        model = LoopModel(16, 4, 4, 1.0, 0.5)
        x, _ = examples(np.random.default_rng(21), 4)
        result = probe(model, x, steps=12)
        self.assertTrue(math.isfinite(result["mean_tangent_rate_after_8"]))


if __name__ == "__main__":
    unittest.main()
