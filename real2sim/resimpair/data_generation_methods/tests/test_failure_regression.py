"""Fail-closed provenance/split checks for reusable visual regression evidence."""
from pathlib import Path
import tempfile
import unittest

from data_generation_methods.r2h_pipeline.common import file_sha256
from data_generation_methods.r2h_pipeline.failure_regression import (
    validate_lineage_splits, verify_all_bindings,
)


class FailureRegressionTest(unittest.TestCase):
    def test_changed_nested_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "frame.png"
            p.write_bytes(b"before")
            manifest = {"cases": [{"frames": [{"path": str(p), "sha256": file_sha256(p)}]}]}
            self.assertEqual(verify_all_bindings(manifest), 1)
            p.write_bytes(b"after")
            with self.assertRaises(ValueError):
                verify_all_bindings(manifest)

    def test_same_source_different_execution_cannot_cross_split(self):
        with self.assertRaises(ValueError):
            validate_lineage_splits([
                {"split": "development", "lineage_groups": ["human:a", "physical:x"]},
                {"split": "validation", "lineage_groups": ["human:a", "physical:y"]},
            ])

    def test_same_execution_different_source_cannot_cross_split(self):
        with self.assertRaises(ValueError):
            validate_lineage_splits([
                {"split": "development", "lineage_groups": ["human:a", "physical:x"]},
                {"split": "validation", "lineage_groups": ["human:b", "physical:x"]},
            ])

    def test_derivative_visual_variants_stay_in_single_partition(self):
        result = validate_lineage_splits([
            {"split": "development", "lineage_groups": ["human:a", "physical:x"]},
            {"split": "development", "lineage_groups": ["human:a", "physical:x"]},
            {"split": "validation", "lineage_groups": ["human:b", "physical:y"]},
        ])
        self.assertEqual(result["physical:x"], "development")


if __name__ == "__main__":
    unittest.main()
