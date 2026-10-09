import math
import unittest

from orinth_clef.schema import SchemaError
from orinth_clef.selective_cascade import route, choose_threshold, summarize


class CascadeTests(unittest.TestCase):
    def test_all_questions_must_clear_margin(self):
        self.assertEqual(route({"a": 3.0, "b": 0.1}, 1.0), "fallback")
        self.assertEqual(route({"a": 3.0, "b": 1.0}, 1.0), "micro")

    def test_fail_closed_on_invalid_margins(self):
        for values in ({}, {"a": math.nan}, {"a": -1}, {"a": "2"}):
            with self.assertRaises(SchemaError):
                route(values, 0.5)
        for threshold in (-1, math.nan, math.inf, True):
            with self.assertRaises(SchemaError):
                route({"a": 1.0}, threshold)

    def test_empirical_validation_threshold_not_probability(self):
        rows = [
            {"margins": {"q": 0.1}, "micro": {"q": 0}, "expected": {"q": 1}},
            {"margins": {"q": 0.8}, "micro": {"q": 1}, "expected": {"q": 1}},
            {"margins": {"q": 1.3}, "micro": {"q": 1}, "expected": {"q": 1}},
        ]
        self.assertEqual(choose_threshold(rows, 1.0), 0.8)
        self.assertGreater(choose_threshold(rows[:1], 1.0), 0.1)

    def test_fallback_latency_is_added_and_not_free(self):
        rows = [{"id": "x", "expected": {"a": 1}, "micro": {"a": 0},
                 "fallback": {"a": 1}, "margins": {"a": 0.1},
                 "micro_seconds": 0.2, "fallback_seconds": 0.3}]
        result = summarize(rows, 1.0)
        self.assertEqual(result["cascade_exact"], 1)
        self.assertEqual(result["micro_accepted"], 0)
        self.assertAlmostEqual(result["median_cascade_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
