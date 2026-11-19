"""Regression tests for media/plan binding and conservative evidence decisions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data_generation_methods.r2h_pipeline.common import file_sha256
from data_generation_methods.r2h_pipeline.single_actor_acceptance import (
    CHECKS, RULE_VERSION, evaluate, gate_anchor,
)


class SingleActorAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.anchor = self.root / "anchor.png"
        cv2.imwrite(str(self.anchor), np.zeros((16, 16, 3), dtype=np.uint8))
        self.plan = self.root / "plan.json"
        self.plan.write_text('{"fixture":true}\n')
        self.report = {
            "schema": "r2h_single_actor_review_v2", "rule_version": RULE_VERSION,
            "artifact": {"path": str(self.anchor), "sha256": file_sha256(self.anchor)},
            "plan_sha256": file_sha256(self.plan), "scope": "anchor",
            "reviewer": {"id": "unit-test-fixture", "kind": "human"},
            "evidence": [{"id": "e0", "kind": "human_annotation",
                          "artifact_sha256": file_sha256(self.anchor),
                          "frame_indices": [0], "observation": "Synthetic test annotation"}],
            "checks": {k: {"status": "pass", "evidence_ids": ["e0"],
                            "note": "Synthetic test annotation"} for k in CHECKS},
            "limitations": ["No real image observation is represented by this fixture."],
        }

    def gate(self, report=None):
        p = self.root / "review.json"
        p.write_text(json.dumps(report or self.report))
        return gate_anchor(p, self.anchor, self.plan)

    def test_complete_bound_anchor_clears_only_actor_gate(self):
        result = self.gate()
        self.assertTrue(result["can_generate_video"])
        self.assertFalse(result["training_admitted"])
        self.assertFalse(result["quality_release_approved"])
        self.assertFalse(result["visual_inference_performed_by_gate"])

    def test_two_hands_with_two_people_rejected(self):
        self.report["checks"]["person_count"]["status"] = "fail"
        self.assertEqual(self.gate()["status"], "reject")

    def test_unknown_ownership_blocks(self):
        self.report["checks"]["left_hand_ownership"]["status"] = "unknown"
        self.assertFalse(self.gate()["can_generate_video"])

    def test_all_vlm_yes_cannot_release(self):
        self.report["reviewer"]["kind"] = "vlm"
        self.report["evidence"][0]["kind"] = "vlm"
        result = self.gate()
        self.assertEqual(result["status"], "needs_review")
        self.assertTrue(all(x["effective_status"] == "unknown" for x in result["checks"].values()))

    def test_missing_direct_evidence_cannot_release(self):
        self.report["checks"]["person_count"]["evidence_ids"] = []
        self.assertEqual(self.gate()["status"], "needs_review")

    def test_legacy_hand_inventory_is_not_review(self):
        with self.assertRaises(ValueError):
            evaluate({"minimum_visible_hand_count": 2, "exact_count_known": True})

    def test_changed_image_invalidates_review(self):
        self.anchor.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.gate()

    def test_changed_plan_invalidates_review(self):
        self.plan.write_text('{"fixture":"changed"}\n')
        with self.assertRaises(ValueError):
            self.gate()

    def test_evidence_from_other_media_rejected(self):
        self.report["evidence"][0]["artifact_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            self.gate()

    def test_missing_dimension_rejected(self):
        del self.report["checks"]["body_structure"]
        with self.assertRaises(ValueError):
            self.gate()

    def test_legacy_no_plan_can_reject_but_not_accept(self):
        self.report["plan_sha256"] = None
        self.assertEqual(evaluate(self.report)["status"], "needs_review")
        self.report["checks"]["person_count"]["status"] = "fail"
        self.assertEqual(evaluate(self.report)["status"], "reject")

    def test_sampled_video_not_a_generation_anchor(self):
        self.make_video_report()
        self.assertFalse(evaluate(self.report)["can_generate_video"])
        with self.assertRaises(ValueError):
            self.gate()

    def make_video_report(self):
        video = self.root / "video.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4, (16, 16))
        self.assertTrue(writer.isOpened())
        for value in (0, 80, 160):
            writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
        writer.release()
        self.report["scope"] = "sampled_video"
        self.report["artifact"] = {"path": str(video), "sha256": file_sha256(video)}
        self.report["evidence"][0]["artifact_sha256"] = file_sha256(video)

    def test_frame_index_at_decoded_count_rejected(self):
        self.make_video_report()
        self.report["evidence"][0]["frame_indices"] = [3]
        with self.assertRaisesRegex(ValueError, "beyond"):
            evaluate(self.report)

    def test_last_actual_video_frame_supported(self):
        self.make_video_report()
        self.report["evidence"][0]["frame_indices"] = [2]
        self.assertEqual(evaluate(self.report)["actual_media_frame_count"], 3)

    def test_unknown_root_field_rejected(self):
        self.report["quality_override"] = True
        with self.assertRaisesRegex(ValueError, "schema violation"):
            self.gate()

    def test_unknown_nested_field_rejected(self):
        self.report["checks"]["person_count"]["override"] = True
        with self.assertRaisesRegex(ValueError, "schema violation"):
            self.gate()

    def test_missing_nullable_plan_field_rejected(self):
        del self.report["plan_sha256"]
        with self.assertRaisesRegex(ValueError, "schema violation"):
            evaluate(self.report)

    def test_undecodable_anchor_cannot_supply_observation(self):
        self.anchor.write_bytes(b"not a PNG")
        digest = file_sha256(self.anchor)
        self.report["artifact"]["sha256"] = digest
        self.report["evidence"][0]["artifact_sha256"] = digest
        with self.assertRaises(ValueError):
            self.gate()


if __name__ == "__main__":
    unittest.main()
