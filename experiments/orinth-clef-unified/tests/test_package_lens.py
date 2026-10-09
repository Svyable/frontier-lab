import json
import tempfile
import unittest
from pathlib import Path

from orinth_clef.package_lens import build_manifest, verify_manifest
from orinth_clef.schema import SchemaError


class ThinPackageTests(unittest.TestCase):
    def setup_model(self, root):
        model = root / "models" / "tiny"
        model.mkdir(parents=True)
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (model / name).write_text("test", encoding="utf-8")
        benchmark = root / "benchmark.json"
        benchmark.write_text(json.dumps({
            "model_path": "models/tiny", "adapter_path": None,
            "count": 2, "schema_valid_count": 2,
            "teacher_pseudolabel_agreement_count": 1
        }), encoding="utf-8")
        return model, benchmark

    def test_content_addressed_overlay_verifies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, report = self.setup_model(root)
            # Work from workspace-relative paths, as the real CLI does.
            import os
            previous = Path.cwd()
            try:
                os.chdir(root)
                manifest = build_manifest(Path("models/tiny"), revision="a" * 40,
                                          benchmark=Path("benchmark.json"))
                self.assertFalse(manifest["standalone"])
                self.assertTrue(verify_manifest(manifest, root))
                (model / "model.safetensors").write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(SchemaError, "changed file|checksum"):
                    verify_manifest(manifest, root)
            finally:
                os.chdir(previous)

    def test_benchmark_must_be_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, report = self.setup_model(root)
            payload = json.loads(report.read_text())
            payload["schema_valid_count"] = 1
            payload["model_path"] = str(model)
            report.write_text(json.dumps(payload))
            with self.assertRaisesRegex(SchemaError, "complete schema validity"):
                build_manifest(model, revision="a" * 40, benchmark=report)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(SchemaError, "workspace-relative"):
                verify_manifest({"format": "decisionlens.thin.v1", "standalone": False,
                                 "model_path_relative_to_workspace": "../escape",
                                 "files": {}}, root)


if __name__ == "__main__":
    unittest.main()
