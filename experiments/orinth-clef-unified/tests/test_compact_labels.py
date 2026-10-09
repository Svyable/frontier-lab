"""CPU-only contracts for reversible compact choice labels."""
import copy
import unittest

from orinth_clef.compact_labels import compact_task
from orinth_clef.schema import SchemaError


class SimpleTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class CollidingTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1]


class CompactLabelsTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "state": {"issue": "billing"},
            "questions": {
                "team": {
                    "type": "choice", "instructions": "Route by issue",
                    "criteria": {"technical": "Technical issue", "billing": "Billing issue"}
                },
                "urgent": {"type": "noul", "instructions": "True if urgent"}
            }
        }

    def test_round_trip_and_immutability(self):
        before = copy.deepcopy(self.task)
        transformed, mappings = compact_task(self.task, SimpleTokenizer())
        self.assertEqual(self.task, before)
        self.assertEqual(mappings["team"], {"A": "billing", "B": "technical"})
        self.assertIn("Original option ID", transformed["questions"]["team"]["criteria"]["A"])
        self.assertEqual(transformed["questions"]["urgent"], self.task["questions"]["urgent"])

    def test_colliding_aliases_fail_closed(self):
        with self.assertRaisesRegex(SchemaError, "insufficient"):
            compact_task(self.task, CollidingTokenizer())


if __name__ == "__main__":
    unittest.main()
