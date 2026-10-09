import unittest

from orinth_clef.analyze_teacher import summarize_teacher
from orinth_clef.build_cases import create_cases
from orinth_clef.data import collect_cases
from orinth_clef.schema import SchemaError
from tests.test_collection_v2 import FakeTeacher, REV


class TeacherAuditTests(unittest.TestCase):
    def test_audit_does_not_claim_gold_accuracy(self):
        cases, refs = create_cases()
        rows = collect_cases(cases[:4], FakeTeacher(), revision=REV)
        report = summarize_teacher(rows, refs[:4])
        self.assertIn("not_accuracy", report["evaluation_kind"])
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["decision_counts"], {"choice": 4, "noul": 4})
        self.assertNotIn("accuracy", report)

    def test_audit_fails_on_missing_reference(self):
        cases, refs = create_cases()
        rows = collect_cases(cases[:2], FakeTeacher(), revision=REV)
        with self.assertRaisesRegex(SchemaError, "counts differ"):
            summarize_teacher(rows, refs[:1])


if __name__ == "__main__":
    unittest.main()
