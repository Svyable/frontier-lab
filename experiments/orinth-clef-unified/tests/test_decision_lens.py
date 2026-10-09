import math
import unittest

from orinth_clef.decision_lens import candidate_token_ids, candidate_values, select_from_logits
from orinth_clef.schema import SchemaError


class FakeTokenizer:
    def encode(self, value, add_special_tokens=False):
        return {"billing": [10], "technical": [20], "true": [30], "false": [40],
                "multiple": [1, 2], "duplicate": [10]}.get(value, [55])


class DecisionLensTests(unittest.TestCase):
    def test_candidate_values_sorted(self):
        self.assertEqual(candidate_values({"type": "choice", "criteria": {
            "technical": "Bug", "billing": "Payments"
        }}), ["billing", "technical"])
        self.assertEqual(candidate_values({"type": "noul"}), [False, True])

    def test_token_ids_exact_and_boolean_type(self):
        t = FakeTokenizer()
        self.assertEqual(candidate_token_ids(t, {"type": "choice", "criteria": {
            "billing": "x", "technical": "y"
        }}), {"billing": 10, "technical": 20})
        self.assertEqual(candidate_token_ids(t, {"type": "noul"}), {False: 40, True: 30})

    def test_refuse_multitoken_candidate(self):
        with self.assertRaisesRegex(SchemaError, "not a single token"):
            candidate_token_ids(FakeTokenizer(), {"type": "choice", "criteria": {
                "billing": "x", "multiple": "y"
            }})

    def test_refuse_colliding_candidate_tokens(self):
        with self.assertRaisesRegex(SchemaError, "ambiguous"):
            candidate_token_ids(FakeTokenizer(), {"type": "choice", "criteria": {
                "billing": "x", "duplicate": "y"
            }})

    def test_argmax_and_normalization_not_confidence(self):
        winner, weights = select_from_logits({"billing": 10, "technical": 20}, [0.0, 2.0])
        self.assertEqual(winner, "technical")
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertGreater(weights["technical"], 0.8)

    def test_invalid_logits_fail_closed(self):
        with self.assertRaisesRegex(SchemaError, "invalid candidate scores"):
            select_from_logits({False: 1, True: 2}, [0.0, math.nan])


if __name__ == "__main__":
    unittest.main()
