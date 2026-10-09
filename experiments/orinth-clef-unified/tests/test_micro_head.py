"""CPU-only contracts for the experimental question-conditioned micro-head."""
import json
import tempfile
import unittest
from pathlib import Path

from orinth_clef.micro_head import candidate_descriptions, dataset_hash, read_cases
from orinth_clef.schema import SchemaError


class MicroHeadContractTests(unittest.TestCase):
    def test_choice_candidate_order_and_descriptions(self):
        options = candidate_descriptions({"type": "choice", "criteria": {
            "zeta": "Last", "alpha": "First"
        }})
        self.assertEqual(options, [("alpha", "alpha: First"), ("zeta", "zeta: Last")])

    def test_boolean_candidate_types(self):
        options = candidate_descriptions({"type": "noul"})
        self.assertIs(options[0][0], False)
        self.assertIs(options[1][0], True)

    def test_unsupported_type_fails(self):
        with self.assertRaises(SchemaError):
            candidate_descriptions({"type": "ranking"})

    def test_missing_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception):
                read_cases(Path(tmp))

    def test_dataset_fingerprint_changes_with_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "valid", "test"):
                (root / f"{split}.jsonl").write_text("first\n")
            before = dataset_hash(root)
            (root / "train.jsonl").write_text("second\n")
            self.assertNotEqual(before, dataset_hash(root))


if __name__ == "__main__":
    unittest.main()
