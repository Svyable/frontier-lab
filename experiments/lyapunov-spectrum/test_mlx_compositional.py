"""Run on Apple Silicon with MLX installed; no external datasets required."""
import math
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from compositional_data import mapping_pools, sample
from mlx_compositional import (RetrievalModel, VARIANTS, accuracy_and_ce,
                               classification_loss, early_exit_audit,
                               finite_time_probe, parameter_count)


class MLXCompositionalTests(unittest.TestCase):
    def test_forward_gradient_and_depth_controls(self):
        train, _ = mapping_pools()
        x_np, y_np = sample(np.random.default_rng(11), train, 8)
        x, y = mx.array(x_np), mx.array(y_np)
        params = {}
        for variant in VARIANTS:
            mx.random.seed(17)
            model = RetrievalModel(width=16, depth=4, variant=variant)
            logits = model(x)
            self.assertEqual(logits.shape, (8, 2))
            self.assertEqual(model(x, depth=0).shape, (8, 2))
            params[variant] = parameter_count(model.parameters())
            optimizer = optim.Adam(learning_rate=0.001)
            value, gradients = nn.value_and_grad(model, classification_loss)(model, x, y)
            optimizer.update(model, gradients)
            mx.eval(value, model.parameters(), optimizer.state)
            self.assertTrue(math.isfinite(float(value.item())))
            self.assertEqual(accuracy_and_ce(model, x, y, 4)["accuracy"] >= 0, True)
        self.assertEqual(params["tied_weak"], params["tied_strong"])
        self.assertGreater(params["untied_strong"], params["tied_strong"])

    def test_multi_input_probe_and_halt_audit(self):
        _, test = mapping_pools()
        x_np, y_np = sample(np.random.default_rng(55), test, 8)
        mx.random.seed(19)
        model = RetrievalModel(width=16, depth=4, variant="tied_strong")
        x, y = mx.array(x_np), mx.array(y_np)
        rates = finite_time_probe(model, x, probes=2, steps=12)
        self.assertEqual(rates["inputs"], 2)
        self.assertTrue(math.isfinite(rates["rate_mean"]))
        audit = early_exit_audit(model, x, y, thresholds=(1e-2,))
        self.assertEqual(audit["0.01"]["evaluated"], 8)
        self.assertGreaterEqual(audit["0.01"]["mean_steps"], 1)
        self.assertLessEqual(audit["0.01"]["mean_steps"], 4)


if __name__ == "__main__":
    unittest.main()
