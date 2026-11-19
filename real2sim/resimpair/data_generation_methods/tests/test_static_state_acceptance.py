"""Binding/evidence semantics only; fixtures are not visual qualifications."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data_generation_methods.r2h_pipeline.common import file_sha256
from data_generation_methods.r2h_pipeline.static_state_acceptance import CHECKS, RULE_VERSION, gate_state


class StaticStateAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / "fixture.png"
        cv2.imwrite(str(self.image), np.zeros((16, 16, 3), np.uint8))
        self.package = self.root / "package.json"
        self.package.write_text('{"fixture":true}\n')
        self.context = {
            "condition_role": "initial", "condition_packet_sha256": "a" * 64,
            "reference_frame_sha256": "b" * 64, "execution_id": "fixture-execution",
            "camera_id": "opposite", "human_source_id": "fixture-source",
            "human_session_id": "fixture-session", "human_video_sha256": "c" * 64,
            "human_window_frames": [0, 42], "human_fps": 30., "human_split": "train",
            "human_window_interval": "start_inclusive_end_exclusive",
            "role_ids": {"selected_object": "apple:0", "source_container": "bowl:0",
                         "destination_container": "bowl:1"},
        }
        self.report = {
            "schema": "r2h_static_state_review_v1", "rule_version": RULE_VERSION,
            "artifact": {"path": str(self.image), "sha256": file_sha256(self.image)},
            "package_sha256": file_sha256(self.package), "context": deepcopy(self.context),
            "reviewer": {"id": "fixture-only", "kind": "human"},
            "evidence": [{"id": "e0", "kind": "human_annotation",
                          "artifact_sha256": file_sha256(self.image),
                          "observation": "Synthetic test annotation, not actual visual evidence."}],
            "checks": {key: {"status": "pass", "evidence_ids": ["e0"],
                             "note": "Fixture only"} for key in CHECKS},
            "limitations": ["No visual quality claim from test fixtures."],
        }

    def gate(self):
        p = self.root / "review.json"
        p.write_text(json.dumps(self.report))
        return gate_state(p, self.image, self.package, self.context)

    def test_complete_evidence_clears_only_static_state(self):
        result = self.gate()
        self.assertTrue(result["clears_static_state"])
        for name in ("can_generate_video", "visual_inference_performed_by_gate",
                     "actor_identity_verified", "action_order_verified", "physical_contact_verified",
                     "quality_release_approved", "training_admitted"):
            self.assertFalse(result[name])

    def test_each_context_binding_is_mandatory(self):
        for key in self.context:
            with self.subTest(key=key):
                self.report["context"] = deepcopy(self.context)
                value = self.context[key]
                self.report["context"][key] = (
                    [0, 43] if isinstance(value, list) else
                    {**value, "selected_object": "apple:99"} if isinstance(value, dict) else
                    60. if isinstance(value, float) else
                    "terminal" if key == "condition_role" else
                    "d" * 64 if key.endswith("sha256") else "different")
                with self.assertRaises(ValueError):
                    self.gate()

    def test_each_missing_dimension_fails_schema(self):
        for name in CHECKS:
            check = self.report["checks"].pop(name)
            with self.assertRaisesRegex(ValueError, "schema violation"):
                self.gate()
            self.report["checks"][name] = check

    def test_all_vlm_yes_is_unknown(self):
        self.report["reviewer"]["kind"] = "vlm"
        self.report["evidence"][0]["kind"] = "vlm"
        result = self.gate()
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(all(v["effective_status"] == "unknown" for v in result["checks"].values()))

    def test_direct_reviewer_cannot_relabel_vlm_evidence_as_direct(self):
        self.report["evidence"][0]["kind"] = "vlm"
        self.assertEqual(self.gate()["status"], "needs_review")

    def test_each_failed_dimension_rejects(self):
        for name in CHECKS:
            self.report["checks"][name]["status"] = "fail"
            self.assertEqual(self.gate()["status"], "reject")
            self.report["checks"][name]["status"] = "pass"

    def test_unknown_or_missing_evidence_never_passes(self):
        self.report["checks"]["object_identity"]["status"] = "unknown"
        self.assertEqual(self.gate()["status"], "needs_review")
        self.report["checks"]["object_identity"]["status"] = "pass"
        self.report["checks"]["object_identity"]["evidence_ids"] = []
        self.assertEqual(self.gate()["status"], "needs_review")

    def test_other_image_evidence_rejected(self):
        self.report["evidence"][0]["artifact_sha256"] = "d" * 64
        with self.assertRaises(ValueError): self.gate()

    def test_package_change_rejected_even_when_plan_unchanged(self):
        self.package.write_text('{"fixture":"other-source","same_plan":true}\n')
        with self.assertRaisesRegex(ValueError, "different package"): self.gate()

    def test_image_change_rejected(self):
        self.image.write_bytes(b"changed")
        with self.assertRaises(ValueError): self.gate()

    def test_undecodable_image_rejected_even_with_current_hash(self):
        self.image.write_bytes(b"not image bytes")
        digest = file_sha256(self.image)
        self.report["artifact"]["sha256"] = digest
        self.report["evidence"][0]["artifact_sha256"] = digest
        with self.assertRaisesRegex(ValueError, "decodable"): self.gate()

    def test_schema_rejects_unknown_nested_fields(self):
        for target in (self.report, self.report["context"], self.report["context"]["role_ids"],
                       self.report["checks"]["object_identity"], self.report["evidence"][0]):
            target["override"] = True
            with self.assertRaisesRegex(ValueError, "schema violation"): self.gate()
            del target["override"]

    def test_evidence_reference_and_duplicate_id_validation(self):
        self.report["checks"]["object_identity"]["evidence_ids"] = ["absent"]
        with self.assertRaises(ValueError): self.gate()
        self.report["checks"]["object_identity"]["evidence_ids"] = ["e0"]
        self.report["evidence"].append(deepcopy(self.report["evidence"][0]))
        with self.assertRaises(ValueError): self.gate()

    def test_invalid_window_or_duplicate_roles_rejected(self):
        self.context["human_window_frames"] = [42, 0]
        self.report["context"] = deepcopy(self.context)
        with self.assertRaisesRegex(ValueError, "interval"): self.gate()
        self.context["human_window_frames"] = [0, 42]
        self.context["role_ids"]["destination_container"] = "bowl:0"
        self.report["context"] = deepcopy(self.context)
        with self.assertRaisesRegex(ValueError, "distinct"): self.gate()

    def test_nonfinite_fps_rejected(self):
        self.context["human_fps"] = float("nan")
        self.report["context"] = deepcopy(self.context)
        with self.assertRaises(ValueError): self.gate()

    def test_report_cannot_supply_its_own_expected_context_by_omission(self):
        with self.assertRaises(TypeError): gate_state("unused", self.image, self.package)


if __name__ == "__main__":
    unittest.main()
