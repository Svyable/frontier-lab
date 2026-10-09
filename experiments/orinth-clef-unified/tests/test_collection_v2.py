import tempfile
import unittest
from pathlib import Path

from orinth_clef.build_cases import create_cases
from orinth_clef.collect_resume import collect_incremental
from orinth_clef.data import read_jsonl, split_rows
from orinth_clef.schema import SchemaError

REV = "f" * 40


class FakeTeacher:
    def __init__(self):
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        answers = {}
        for qid, question in request["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[0],
                    "confidence": 0.65,
                    "probabilities": {keys[0]: 0.65, keys[1]: 0.35},
                }
            else:
                answers[qid] = {"type": "noul", "noul": 0.25}
        return {"model": "clef-flash", "answers": answers}


class CorpusTests(unittest.TestCase):
    def test_deterministic_balanced_scenario_held_out(self):
        cases, refs = create_cases()
        self.assertEqual((len(cases), len(refs)), (120, 120))
        self.assertEqual(len({case["group_id"] for case in cases}), 30)
        self.assertEqual(len({case["id"] for case in cases}), 120)
        self.assertEqual({r["reference_kind"] for r in refs}, {"construction_intent_unverified"})
        self.assertNotIn("reference", cases[0])
        self.assertNotIn("gold", cases[0])

    def test_incremental_resume_does_not_requery_completed(self):
        cases, _ = create_cases()
        cases = cases[:5]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "teacher.jsonl"
            teacher = FakeTeacher()
            first = collect_incremental(cases, teacher, revision=REV, output=path, max_new=2)
            self.assertEqual(first, {"existing": 0, "new": 2, "total": 2, "requested": 5})
            second = collect_incremental(cases, teacher, revision=REV, output=path)
            self.assertEqual(second, {"existing": 2, "new": 3, "total": 5, "requested": 5})
            self.assertEqual(teacher.calls, 5)
            third = collect_incremental(cases, teacher, revision=REV, output=path)
            self.assertEqual(third["new"], 0)
            self.assertEqual(len(read_jsonl(path)), 5)

    def test_changed_revision_fails_without_writing(self):
        cases, _ = create_cases()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "teacher.jsonl"
            teacher = FakeTeacher()
            collect_incremental(cases[:1], teacher, revision=REV, output=path)
            with self.assertRaisesRegex(SchemaError, "revision"):
                collect_incremental(cases[:1], teacher, revision="a" * 40, output=path)
            self.assertEqual(teacher.calls, 1)
            self.assertEqual(len(read_jsonl(path)), 1)

    def test_changed_request_and_duplicate_id_fail_closed(self):
        cases, _ = create_cases()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "teacher.jsonl"
            collect_incremental(cases[:1], FakeTeacher(), revision=REV, output=path)
            altered = {**cases[0], "request": {**cases[0]["request"], "state": "altered"}}
            with self.assertRaisesRegex(SchemaError, "differs"):
                collect_incremental([altered], FakeTeacher(), revision=REV, output=path)
            with self.assertRaisesRegex(SchemaError, "distinct"):
                collect_incremental([cases[0], cases[0]], FakeTeacher(), revision=REV, output=path)
            self.assertEqual(len(read_jsonl(path)), 1)

    def test_three_way_family_holdout(self):
        cases, _ = create_cases()
        teacher = FakeTeacher()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "teacher.jsonl"
            collect_incremental(cases, teacher, revision=REV, output=path)
            splits = split_rows(read_jsonl(path))
            self.assertEqual({k: len(v) for k, v in splits.items()}, {
                "train": 96, "valid": 12, "test": 12
            })


if __name__ == "__main__":
    unittest.main()
