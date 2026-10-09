import copy
import unittest

from orinth_clef.schema import SchemaError, pseudo_labels, validate_request, validate_response


REQUEST = {
    "model": "clef-flash",
    "state": "Checkout is offline",
    "questions": {
        "team": {"type": "choice", "criteria": {"billing": "Billing issues", "technical": "Service issues"}},
        "urgent": {"type": "noul", "instructions": "Immediate response?"},
    },
}
RESPONSE = {
    "model": "clef-flash",
    "answers": {
        "team": {"type": "choice", "choice": "technical", "confidence": 0.9, "probabilities": {"billing": 0.1, "technical": 0.9}},
        "urgent": {"type": "noul", "noul": 0.8},
    },
    "usage": {"latency_ms": 100},
}


class SchemaTests(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate_request(REQUEST), REQUEST)
        self.assertEqual(validate_response(REQUEST, RESPONSE), RESPONSE)
        self.assertEqual(pseudo_labels(REQUEST, RESPONSE), {"team": "technical", "urgent": True})

    def test_reject_unsupported_score(self):
        request = copy.deepcopy(REQUEST)
        request["questions"]["urgent"] = {"type": "score", "criteria": ["bad", "good"]}
        with self.assertRaisesRegex(SchemaError, "unsupported type"):
            validate_request(request)

    def test_reject_images(self):
        with self.assertRaisesRegex(SchemaError, "text-only"):
            validate_request({**REQUEST, "images": []})

    def test_reject_nonfinite_state(self):
        with self.assertRaisesRegex(SchemaError, "finite JSON"):
            validate_request({**REQUEST, "state": {"score": float("nan")}})

    def test_reject_invalid_probability(self):
        response = copy.deepcopy(RESPONSE)
        response["answers"]["team"]["probabilities"]["technical"] = 1.5
        with self.assertRaisesRegex(SchemaError, "finite"):
            validate_response(REQUEST, response)

    def test_reject_choice_not_in_schema(self):
        response = copy.deepcopy(RESPONSE)
        response["answers"]["team"]["choice"] = "sales"
        with self.assertRaisesRegex(SchemaError, "selected choice"):
            validate_response(REQUEST, response)

    def test_reject_missing_answer(self):
        with self.assertRaisesRegex(SchemaError, "question IDs"):
            validate_response(REQUEST, {"answers": {"urgent": 0.5}})

    def test_reject_mismatched_answer_type(self):
        response = copy.deepcopy(RESPONSE)
        response["answers"]["urgent"]["type"] = "choice"
        with self.assertRaisesRegex(SchemaError, "answer type"):
            validate_response(REQUEST, response)

    def test_reject_boolean_as_probability(self):
        response = copy.deepcopy(RESPONSE)
        response["answers"]["urgent"]["noul"] = True
        with self.assertRaisesRegex(SchemaError, "number"):
            validate_response(REQUEST, response)

    def test_reject_bad_distribution_sum(self):
        response = copy.deepcopy(RESPONSE)
        response["answers"]["team"]["probabilities"] = {"billing": .2, "technical": .2}
        with self.assertRaisesRegex(SchemaError, "sum to 1"):
            validate_response(REQUEST, response)


if __name__ == "__main__":
    unittest.main()
