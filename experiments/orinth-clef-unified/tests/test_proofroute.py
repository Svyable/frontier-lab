"""ProofRoute typed symbolic fast-path safety and semantics."""
import copy
import unittest

from orinth_clef.proofroute import (RuleNotApplicable, compile_policy, decide,
                                    execute)
from orinth_clef.schema import SchemaError


class ProofRouteTests(unittest.TestCase):
    def setUp(self):
        self.task = {
            "state": {"n": 4, "limit": 3, "active": True},
            "questions": {
                "choice": {"type": "choice", "criteria": {
                    "yes": "Grant", "no": "Deny"}},
                "flag": {"type": "noul"},
            }
        }
        self.policy = {
            "choice": {"when": {"op": "and", "args": [
                {"op": "eq", "field": "active", "value": True},
                {"op": "gt", "field": "n", "value": {"field": "limit"}}
            ]}, "then": "yes", "else": "no"},
            "flag": {"when": {"op": "not", "arg": {
                "op": "le", "field": "n", "value": 3}}},
        }

    def test_full_typed_execution_and_immutable_inputs(self):
        task, policy = copy.deepcopy(self.task), copy.deepcopy(self.policy)
        answers, route = decide(task, policy)
        self.assertEqual((answers, route), ({"choice": "yes", "flag": True}, "verified_rule"))
        self.assertEqual(task, self.task)
        self.assertEqual(policy, self.policy)

    def test_missing_field_abstains_or_falls_back(self):
        task = copy.deepcopy(self.task)
        del task["state"]["limit"]
        self.assertEqual(decide(task, self.policy), (None, "abstain"))
        self.assertEqual(decide(task, self.policy, lambda _: {
            "choice": "no", "flag": False
        }), ({"choice": "no", "flag": False}, "neural_fallback"))

    def test_no_policy_abstains(self):
        self.assertEqual(decide(self.task), (None, "abstain"))

    def test_partial_policy_fails_closed(self):
        with self.assertRaisesRegex(SchemaError, "exactly all"):
            decide(self.task, {"choice": self.policy["choice"]})

    def test_invalid_choice_and_operators_fail(self):
        policy = copy.deepcopy(self.policy)
        policy["choice"]["then"] = "invented"
        with self.assertRaisesRegex(SchemaError, "allowed IDs"):
            compile_policy(self.task, policy)
        policy = copy.deepcopy(self.policy)
        policy["flag"]["when"]["arg"]["op"] = "__import__"
        with self.assertRaisesRegex(SchemaError, "invalid rule operator"):
            compile_policy(self.task, policy)

    def test_numeric_bool_separation(self):
        policy = copy.deepcopy(self.policy)
        policy["flag"]["when"] = {"op": "eq", "field": "active", "value": 1}
        self.assertFalse(execute(compile_policy(self.task, policy), self.task["state"])["flag"])

    def test_wrong_type_abstains(self):
        task = copy.deepcopy(self.task)
        task["state"]["n"] = "four"
        self.assertEqual(decide(task, self.policy), (None, "abstain"))

    def test_fallback_invalid_output_rejected(self):
        with self.assertRaises(SchemaError):
            decide(self.task, None, lambda _: {"choice": "not_allowed", "flag": True})

    def test_complexity_limit(self):
        policy = copy.deepcopy(self.policy)
        nested = {"op": "eq", "field": "n", "value": 4}
        for _ in range(14):
            nested = {"op": "not", "arg": nested}
        policy["flag"]["when"] = nested
        with self.assertRaisesRegex(SchemaError, "complexity"):
            compile_policy(self.task, policy)

    def test_200_boundary_and_randomized_oracle_cases(self):
        import random
        from orinth_clef.proofroute_benchmark import POLICIES
        rng = random.Random(20261009)
        for i in range(200):
            depth = rng.choice([0, 1, 2, 3, 4, 20])
            engine = rng.choice([True, False])
            task = {
                "state": {"tire_depth_mm": depth, "engine_check": engine},
                "questions": {
                    "status": {"type": "choice", "criteria": {
                        "maintenance_due": "Service", "roadworthy": "Pass"}},
                    "danger": {"type": "noul"},
                },
            }
            got, route = decide(task, POLICIES["vehicle"])
            self.assertEqual(route, "verified_rule")
            self.assertEqual(got, {
                "status": "maintenance_due" if depth < 3 or engine else "roadworthy",
                "danger": depth < 2
            }, f"case {i}")

    def test_inapplicable_branch_does_not_silently_skip(self):
        task = copy.deepcopy(self.task)
        del task["state"]["active"]
        self.assertEqual(decide(task, self.policy), (None, "abstain"))


if __name__ == "__main__":
    unittest.main()
