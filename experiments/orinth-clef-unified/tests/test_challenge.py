"""CPU-only regression tests for rule-labeled diagnostic cases."""
import json
import tempfile
import unittest
from pathlib import Path

from orinth_clef.challenge import read_challenges
from orinth_clef.schema import SchemaError


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "case-1", "family": "logic", "rule": "true iff enabled",
            "task": {
                "state": {"enabled": True},
                "questions": {"enabled": {"type": "noul", "instructions": "True if enabled"}}
            },
            "gold": {"enabled": True}
        }

    def write(self, root, cases):
        path = root / "cases.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in cases))
        return path

    def test_valid_case(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(read_challenges(self.write(Path(temp), [self.case]))[0]["id"], "case-1")

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(SchemaError, "duplicate"):
                read_challenges(self.write(Path(temp), [self.case, self.case]))

    def test_invalid_gold_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            row = dict(self.case, gold={"enabled": "true"})
            with self.assertRaises(SchemaError):
                read_challenges(self.write(Path(temp), [row]))

    def test_rule_required(self):
        with tempfile.TemporaryDirectory() as temp:
            row = dict(self.case, rule="")
            with self.assertRaisesRegex(SchemaError, "missing explicit rule"):
                read_challenges(self.write(Path(temp), [row]))

    def test_empty_fixture_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(SchemaError, "empty"):
                read_challenges(self.write(Path(temp), []))


if __name__ == "__main__":
    unittest.main()
