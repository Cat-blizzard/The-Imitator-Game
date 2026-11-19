import copy
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from data_generation_methods.r2h_pipeline.hidden_identity_extent import (
    diagnose_pair, dimension_intervals, relative_measurements,
)
from data_generation_methods.r2h_pipeline.scene_proposals import binding


def fixture(folder):
    source, target = folder / "source.png", folder / "target.png"
    cv2.imwrite(str(source), np.zeros((256, 256, 3), np.uint8))
    cv2.imwrite(str(target), np.zeros((256, 256, 3), np.uint8))
    def side(path, box):
        return {"image": dict(binding(path), width=256, height=256),
                "object_box_xyxy_px": box, "object_visibility": "fully_visible",
                "contexts_xyxy_px": {"compartment": [20, 20, 220, 220]}}
    a = {"sample_id": "test", "source": side(source, [100, 100, 120, 120]),
         "target": side(target, [70, 70, 170, 170]),
         "correspondence": {"status": "reviewed_semantic_slot", "reviewer": "fixture",
                             "evidence": "explicit test fixture, not visual evidence",
                             "source_image_sha256": binding(source)["sha256"],
                             "target_image_sha256": binding(target)["sha256"],
                             "source_slot": "slot", "target_slot": "slot",
                             "same_context_instance": True, "comparable_view": True}}
    p = {"roi_boundary_uncertainty_px": {"source": 2., "target": 2.},
         "severe_flags": {"relative_width_factor_lower_bound_gt": 2.,
                          "relative_height_factor_lower_bound_gt": 2.,
                          "relative_area_factor_lower_bound_gt": 4.}}
    return a, p


class HiddenIdentityExtentTests(unittest.TestCase):
    def test_uncertainty_propagated_to_extent_ratios(self):
        r = relative_measurements([20, 20, 40, 50], [0, 0, 100, 100], 2.)
        self.assertEqual(r["width"], [16 / 104, 24 / 96])
        self.assertEqual(r["height"], [26 / 104, 34 / 96])
        self.assertEqual(r["area"], [16 * 26 / (104 * 104), 24 * 34 / (96 * 96)])

    def test_low_resolution_roi_is_unknown_not_a_precise_measurement(self):
        with self.assertRaisesRegex(ValueError, "unresolved"):
            dimension_intervals([2, 2, 5, 5], 2.)

    def test_large_extent_flag_never_awards_identity_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            a, p = fixture(folder)
            r = diagnose_pair(a, p, folder / "result")
            self.assertEqual(r["status"], "strong_extent_inconsistency")
            self.assertFalse(r["identity_pass"])
            self.assertEqual(r["identity_preserved"], "unknown")
            self.assertFalse(r["automatic_identity_detection"])

    def test_similar_size_is_unknown_even_with_reviewed_same_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            a, p = fixture(folder)
            a["target"]["object_box_xyxy_px"] = a["source"]["object_box_xyxy_px"]
            r = diagnose_pair(a, p, folder / "result")
            self.assertEqual(r["status"], "unknown_no_severe_extent_flag")
            self.assertFalse(r["identity_pass"])

    def test_nominal_growth_above_threshold_but_interval_overlaps_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            a, p = fixture(folder)
            a["target"]["object_box_xyxy_px"] = [80, 80, 122, 122]
            r = diagnose_pair(a, p, folder / "result")
            self.assertEqual(r["status"], "unknown_no_severe_extent_flag")

    def test_required_negative_inputs_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            base, p = fixture(folder)
            for index, mutation in enumerate((
                lambda a: a["correspondence"].update(target_slot="other"),
                lambda a: a["correspondence"].update(status="automatic_candidate"),
                lambda a: a["target"].update(object_visibility="fully_occluded"),
                lambda a: a["target"].update(object_box_xyxy_px=[170, 70, 70, 170]),
                lambda a: a["target"].update(object_box_xyxy_px=[0, 0, 250, 250]),
                lambda a: a["target"]["image"].update(sha256="0" * 64),
            )):
                a = copy.deepcopy(base)
                mutation(a)
                r = diagnose_pair(a, p, folder / f"negative_{index}")
                self.assertTrue(r["status"].startswith("unknown"))
                self.assertFalse(r["identity_pass"])

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            a, p = fixture(folder)
            diagnose_pair(a, p, folder / "result")
            with self.assertRaises(FileExistsError):
                diagnose_pair(a, p, folder / "result")


if __name__ == "__main__":
    unittest.main()
