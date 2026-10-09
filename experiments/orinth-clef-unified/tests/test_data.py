import json
import tempfile
import unittest
from pathlib import Path

from orinth_clef.data import collect_cases, export_student_data, read_jsonl, split_rows
from orinth_clef.schema import SchemaError
from tests.test_schema import REQUEST, RESPONSE


class FakeTeacher:
    def predict(self, request):
        return RESPONSE


def fake_rows(count=12):
    cases = [
        {"id": f"case-{i:02d}", "group_id": f"scenario-{i:02d}", "request": REQUEST}
        for i in range(count)
    ]
    return collect_cases(cases, FakeTeacher(), revision="0123456789abcdef")


class DataTests(unittest.TestCase):
    def test_teacher_data_keeps_probabilities_and_revision(self):
        row = fake_rows(1)[0]
        self.assertEqual(row["response"]["answers"]["team"]["probabilities"]["technical"], .9)
        self.assertEqual(row["teacher_revision"], "0123456789abcdef")
        self.assertIs(row["pseudo_labels"]["urgent"], True)

    def test_requires_exact_revision(self):
        with self.assertRaisesRegex(ValueError, "revision"):
            collect_cases([{"id": "x", "group_id": "g", "request": REQUEST}], FakeTeacher(), revision="")

    def test_splits_are_deterministic_order_independent(self):
        rows = fake_rows(20)
        left = split_rows(rows)
        right = split_rows(list(reversed(rows)))
        self.assertEqual(left, right)
        self.assertTrue(all(left[s] for s in left))
        self.assertEqual(sum(len(x) for x in left.values()), 20)

    def test_same_group_cannot_cross_splits(self):
        rows = fake_rows(14)
        rows[1]["group_id"] = rows[0]["group_id"]
        # Give each sample a distinct state so its split can be observed.
        for i, row in enumerate(rows):
            row["request"] = {**row["request"], "state": f"unique state {i}"}
        splits = split_rows(rows)
        observed = {}
        for split, samples in splits.items():
            for sample in samples:
                observed[json.loads(sample["messages"][1]["content"])["state"]] = split
        self.assertEqual(len(observed), 14)
        self.assertEqual(observed["unique state 0"], observed["unique state 1"])

    def test_export_contains_no_unverified_gold_or_teacher_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            rows = fake_rows(12)
            manifest = export_student_data(rows, output)
            self.assertEqual(manifest["sample_count"], 12)
            self.assertEqual(manifest["status"], "teacher_pseudolabels_not_human_gold")
            student = read_jsonl(output / "train.jsonl")[0]
            text = json.dumps(student)
            self.assertNotIn("teacher_revision", text)
            self.assertNotIn("0.9", text)
            self.assertIn('"decisions"', student["messages"][-1]["content"])

    def test_needs_independent_groups(self):
        with self.assertRaisesRegex(SchemaError, "three independent"):
            split_rows(fake_rows(2))


if __name__ == "__main__":
    unittest.main()
