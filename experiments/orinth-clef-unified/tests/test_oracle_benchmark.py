"""Deterministic oracle benchmark contracts, no MLX required."""
import json
import tempfile
import unittest
from pathlib import Path

from orinth_clef.oracle_benchmark import cases, diagnostic_gate, verify_fixture, write_fixture
from orinth_clef.schema import SchemaError


class OracleBenchmarkTests(unittest.TestCase):
    def test_deterministic_and_four_families(self):
        first = cases(seed=9, per_family=8)
        self.assertEqual(first, cases(seed=9, per_family=8))
        self.assertEqual(len(first), 32)
        self.assertEqual({r["family"] for r in first},
                         {"threshold", "category", "compound", "relative"})
        self.assertEqual(len({r["id"] for r in first}), 32)
        self.assertTrue(all(len(r["task"]["questions"]) == 2 for r in first))

    def test_rule_labels_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "oracle.jsonl"
            digest = write_fixture(p, seed=19, per_family=4)
            self.assertEqual(len(digest), 64)
            self.assertEqual(len(verify_fixture(p)), 16)
            with self.assertRaisesRegex(SchemaError, "exists"):
                write_fixture(p)
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rows[0]["gold"]["flag"] = not rows[0]["gold"]["flag"]
            p.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(SchemaError, "disagreement"):
                verify_fixture(p)

    def test_diagnostic_gate_rejects_wrong_accepted_exits(self):
        report = {"summary": {
            "exit": {"exact": 40, "accepted_wrong": 24, "mean_seconds": 0.12},
            "lens": {"exact": 55, "mean_seconds": 0.23}}}
        outcome = diagnostic_gate(report)
        self.assertFalse(outcome["passed"])
        self.assertEqual(len(outcome["failures"]), 2)
        report["summary"]["exit"].update({"exact": 55, "accepted_wrong": 0})
        self.assertTrue(diagnostic_gate(report)["passed"])
        del report["summary"]["exit"]["accepted_wrong"]
        self.assertFalse(diagnostic_gate(report)["passed"])

    def test_reject_invalid_sizes(self):
        for count in (0, -1, 101, 1.0, True):
            with self.assertRaises(SchemaError):
                cases(per_family=count)


if __name__ == "__main__":
    unittest.main()
