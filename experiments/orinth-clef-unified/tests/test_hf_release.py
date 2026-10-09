"""Offline Hugging Face bundle integrity contracts (no MLX dependency)."""
import json
import tempfile
import unittest
from pathlib import Path

from orinth_clef.hf_release import export, verify
from orinth_clef.schema import SchemaError


class HFReleaseTests(unittest.TestCase):
    def make_source(self, root):
        source = root / "source"
        model = root / "model"
        source.mkdir()
        model.mkdir()
        (source / "orinth_clef").mkdir()
        (source / "examples").mkdir()
        from orinth_clef.hf_release import MODEL_FILES, SOURCE_FILES
        for name in MODEL_FILES:
            (model / name).write_text(
                '{"weight_map":{"a":"model.safetensors"}}'
                if name.endswith(".index.json") else "{}"
            )
        for name in SOURCE_FILES:
            (source / "orinth_clef" / name).write_text("# dummy\n")
        (source / "examples" / "challenge_rules.jsonl").write_text("{}\n")
        return source, model

    def test_metadata_bundle_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            out = root / "export"
            result = export(source, model, out, False)
            self.assertFalse(result["weights_included"])
            self.assertFalse((out / "model.safetensors").exists())
            self.assertTrue((out / "README.md").exists())
            self.assertTrue((out / "orinth_clef" / "hf_release.py").exists())
            self.assertEqual(verify(out), result)

    def test_weight_bundle_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            out = root / "export"
            result = export(source, model, out, True)
            self.assertTrue(result["weights_included"])
            (out / "README.md").write_text("tampered")
            with self.assertRaisesRegex(SchemaError, "modified"):
                verify(out)

    def test_verify_after_python_import_and_reject_extra_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            out = root / "export"
            export(source, model, out, False)
            cache = out / "orinth_clef" / "__pycache__"
            cache.mkdir()
            (cache / "decision_lens.cpython-312.pyc").write_bytes(b"generated")
            self.assertFalse(verify(out)["weights_included"])
            (out / "unexpected.py").write_text("unexpected")
            with self.assertRaisesRegex(SchemaError, "unexpected"):
                verify(out)

    def test_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            out = root / "export"
            export(source, model, out, False)
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["files"]["../outside"] = {"sha256": "x", "bytes": 1}
            (out / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(SchemaError, "invalid manifest path"):
                verify(out)

    def test_early_exit_head_requires_valid_metadata_and_both_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            head = root / "early.safetensors"
            head.write_bytes(b"fake-weights-for-manifest-test")
            report = root / "early.json"
            data = {
                "depth": 24, "total_layers": 36, "rank": 32, "epochs": 24,
                "threshold": 0.9,
                "threshold_selection": "validation_only_empirical_not_calibrated",
                "target_validation_agreement": 0.8,
                "dataset_sha256": "abc",
                "evaluation_kind": "pseudo_labels_not_gold",
            }
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(SchemaError, "together"):
                export(source, model, root / "missing", False, early_head_path=head)
            result = export(source, model, root / "valid", False,
                            early_head_path=head, early_report_path=report)
            self.assertEqual(result["files_verified"], verify(root / "valid")["files_verified"])
            self.assertTrue((root / "valid" / "early-head-experimental.safetensors").exists())
            self.assertEqual(json.loads((root / "valid" / "early-exit-experimental.json").read_text())["depth"], 24)
            data["threshold"] = float("nan")
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(SchemaError, "invalid early"):
                export(source, model, root / "invalid", False,
                       early_head_path=head, early_report_path=report)

    def test_refuse_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            out = root / "export"
            out.mkdir()
            (out / "existing").write_text("do not delete")
            with self.assertRaisesRegex(SchemaError, "empty"):
                export(source, model, out, False)
            self.assertEqual((out / "existing").read_text(), "do not delete")

    def test_reject_missing_weight_shard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, model = self.make_source(root)
            (model / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"missing.safetensors"}}'
            )
            with self.assertRaisesRegex(SchemaError, "missing shards"):
                export(source, model, root / "export", True)


if __name__ == "__main__":
    unittest.main()
