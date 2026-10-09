"""CPU contract tests for constrained multi-token candidate sequences."""
import unittest

from orinth_clef.candidate_sequences import candidate_sequences
from orinth_clef.decision_lens import candidate_values
from orinth_clef.schema import SchemaError


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class EmptyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return []


class SequenceContractTests(unittest.TestCase):
    def test_multi_token_choice_ids(self):
        question = {"type": "choice", "criteria": {
            "warehouse_restock": "Restock", "ship_now": "Ship"
        }}
        result = candidate_sequences(Tokenizer(), question)
        self.assertEqual(list(result), ["ship_now", "warehouse_restock"])
        self.assertGreater(len(result["warehouse_restock"]), 1)

    def test_boolean_ids_are_typed(self):
        result = candidate_sequences(Tokenizer(), {"type": "noul"})
        self.assertEqual(list(result), [False, True])
        self.assertEqual(len(result[True]), 4)

    def test_empty_tokenization_rejected(self):
        with self.assertRaisesRegex(SchemaError, "empty candidate"):
            candidate_sequences(EmptyTokenizer(), {
                "type": "choice", "criteria": {"alpha": "A", "beta": "B"}
            })

    def test_json_escape_choice_ids_fail_closed(self):
        for unsafe in ('quoted"key', "back\\slash", "line\\nbreak"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaisesRegex(SchemaError, "JSON escaping"):
                    candidate_sequences(Tokenizer(), {
                        "type": "choice",
                        "criteria": {unsafe: "Unsafe", "safe": "Safe"}
                    })

    def test_schema_limit_fifty_options(self):
        question = {"type": "choice", "criteria": {
            f"option_{i}": f"Option {i}" for i in range(50)
        }}
        self.assertEqual(len(candidate_values(question)), 50)
        question["criteria"]["option_50"] = "Too many"
        with self.assertRaises(SchemaError):
            candidate_values(question)


if __name__ == "__main__":
    unittest.main()
