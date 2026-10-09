import unittest

from orinth_clef.eval_student import parse_student_output
from orinth_clef.schema import SchemaError

QUESTIONS = {
    "team": {
        "type": "choice",
        "criteria": {"technical": "Bug triage", "billing": "Payments"},
    },
    "urgent": {"type": "noul", "instructions": "Is it urgent?"},
}


class StudentSchemaTests(unittest.TestCase):
    def test_exact_typed_json_passes(self):
        parsed = parse_student_output(
            '{"decisions": {"team": "billing", "urgent": false}}', QUESTIONS
        )
        self.assertEqual(parsed, {"team": "billing", "urgent": False})

    def test_nested_choice_schema_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "option ID string"):
            parse_student_output(
                '{"decisions": {"team": {"choice": "billing"}, "urgent": false}}',
                QUESTIONS,
            )

    def test_wrong_question_keys_are_rejected(self):
        with self.assertRaisesRegex(SchemaError, "decision IDs"):
            parse_student_output(
                '{"decisions": {"choice": "billing", "noul": false}}',
                QUESTIONS,
            )

    def test_boolean_string_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "JSON boolean"):
            parse_student_output(
                '{"decisions": {"team": "billing", "urgent": "false"}}', QUESTIONS
            )

    def test_non_json_is_rejected(self):
        with self.assertRaisesRegex(SchemaError, "valid JSON"):
            parse_student_output("The answer is billing.", QUESTIONS)

    def test_extra_fields_are_rejected(self):
        with self.assertRaisesRegex(SchemaError, "only a decisions"):
            parse_student_output(
                '{"decisions": {"team": "billing", "urgent": false}, "reason": "x"}',
                QUESTIONS,
            )


if __name__ == "__main__":
    unittest.main()
