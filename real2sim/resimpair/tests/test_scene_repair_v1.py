"""Boundary and retained-real-evidence checks; no new physics or API calls."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, write
from resimpair_scene.repair import failed_release, reordered_before, source_case, target_geometry
from resimpair_scene.revisions import apply_proposal


BASE = ROOT / "runs/production_integration_20261030_v1/final_metric_groups_closure_r1/executions"
PROPOSALS = ROOT / "runs/scene_tools_20261031_v1/proposals"


class RepairContractTests(unittest.TestCase):
    def test_required_order_cannot_be_reversed(self):
        task = {"operations": [{"id": "a"}, {"id": "b"}],
                "required_operation_precedence": [["a", "b"]]}
        with self.assertRaisesRegex(ValueError, "required_precedence"):
            reordered_before(task, "b", "a")

    def test_reordering_keeps_unrelated_precedence(self):
        task = {"operations": [{"id": i} for i in ["a", "b", "c", "d"]],
                "required_operation_precedence": [["a", "c"], ["b", "d"]]}
        self.assertEqual(reordered_before(task, "c", "b"), ["a", "c", "b", "d"])

    @unittest.skipUnless(BASE.exists(), "retained real execution evidence unavailable")
    def test_live_collisions_are_identified_without_case_specific_rules(self):
        for prefix, operation, obstacle in [("63301d", "transfer_6", "container_1"),
                                            ("b61e7", "transfer_2", "object_2")]:
            p = next(BASE.glob("metric_group_" + prefix + "*"))
            result = failed_release(p)
            self.assertEqual(result["operation"], operation)
            self.assertIn(("panda_hand", obstacle), result["pairs"])

    @unittest.skipUnless(BASE.exists(), "retained real execution evidence unavailable")
    def test_recovered_historical_rejection_is_not_a_collision_repair_target(self):
        p = next(BASE.glob("metric_group_35f5f2*"))
        with self.assertRaisesRegex(ValueError, "do_not_repair_recovered_path_rejections"):
            failed_release(p)

    @unittest.skipUnless(BASE.exists(), "retained real execution evidence unavailable")
    def test_nonzero_cavity_center_round_trip_matches_source_request(self):
        p = next(BASE.glob("metric_group_b61e7*")); case, scene, task = source_case(p)
        op = next(o for o in task["operations"] if o["id"] == "transfer_2")
        result = target_geometry(scene, task, op["id"], op["target"]["offset_xy"])
        from resimpair_grasp.common import resolve
        source = read(resolve(case["source_request"]))
        requested = next(t for t in source["tasks"] if t["id"] == case["requested_task_id"])
        placement = next(o for o in requested["placements"] if o["object"] == op["object"])
        np.testing.assert_allclose(result["source_local_offset_xy_m"], placement["target"]["offset_xy_m"], atol=1e-10)
        self.assertTrue(result["passed"])

    @unittest.skipUnless((PROPOSALS / "case_01/proposals.json").exists(), "real proposal unavailable")
    def test_revision_preserves_exact_targets_and_is_idempotent(self):
        p = PROPOSALS / "case_01/proposals.json"; report = read(p)
        proposal = next(x for x in report["proposals"] if x["edit"]["type"] == "reorder")
        from resimpair_grasp.common import resolve
        original_task = read(resolve(report["source_case"]["task"]))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "revision"
            result = apply_proposal(p, proposal["id"], out)
            self.assertEqual(result, apply_proposal(p, proposal["id"], out))
            task = read(resolve(result["case"]["task"]))
            before = {o["id"]: o for o in original_task["operations"]}
            after = {o["id"]: o for o in task["operations"]}
            self.assertEqual(before, after)
            self.assertEqual(result["scope"], "same_exact_goal_task")
            self.assertEqual(task["placement_contract"], original_task["placement_contract"])
            self.assertEqual(result["physical_success"], "unmeasured")

    @unittest.skipUnless((PROPOSALS / "case_00/proposals.json").exists(), "real proposal unavailable")
    def test_layout_edit_never_relabels_original_task_and_rejects_missing_proof(self):
        p = PROPOSALS / "case_00/proposals.json"; report = read(p)
        proposal = report["proposals"][0]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            result = apply_proposal(p, proposal["id"], out / "good")
            self.assertEqual(result["scope"], "new_related_layout_task")
            self.assertNotEqual(result["case"]["task"], report["source_case"]["task"])
            self.assertEqual(result["case"]["revision_parent"]["task"], report["source_case"]["task"])
            corrupted = deepcopy(report)
            corrupted["proposals"][0]["native_path_qualified"] = False
            write(out / "bad.json", corrupted)
            with self.assertRaisesRegex(ValueError, "native_path_evidence_required"):
                apply_proposal(out / "bad.json", proposal["id"], out / "bad")

    @unittest.skipUnless((PROPOSALS / "case_01/proposals.json").exists(), "real proposal unavailable")
    def test_report_cannot_substitute_source_case_or_edit_without_binding(self):
        p = PROPOSALS / "case_01/proposals.json"; report = read(p)
        proposal = report["proposals"][0]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            corrupted = deepcopy(report); corrupted["source_case"]["partition"] = "wrong_partition"
            write(out / "case.json", corrupted)
            with self.assertRaisesRegex(ValueError, "source_case_mismatch"):
                apply_proposal(out / "case.json", proposal["id"], out / "bad_case")
            corrupted = deepcopy(report)
            corrupted["proposals"][0]["edit"]["operation_ids"].reverse()
            write(out / "edit.json", corrupted)
            with self.assertRaisesRegex(ValueError, "id_does_not_bind_edit"):
                apply_proposal(out / "edit.json", proposal["id"], out / "bad_edit")


if __name__ == "__main__":
    unittest.main()
