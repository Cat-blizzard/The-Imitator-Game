"""CPU numerical and real media tests; physical qualification remains separate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from data_generation_methods.r2h_pipeline import sim_state_conditions as module
from data_generation_methods.r2h_pipeline.scene_proposals import binding


def camera(opposite=False):
    e = np.c_[np.diag([-1., -1., 1.]) if opposite else np.eye(3), [0., 0., 1.]]
    return {
        "width": 256, "height": 256, "near": .01, "far": 100.,
        "camera_from_world": e.tolist(),
        "intrinsic_matrix": [[100., 0., 128.], [0., 100., 128.], [0., 0., 1.]],
    }


def states_and_grasp():
    states = np.zeros((6, 2, 13), dtype=np.float32)
    states[:, :, 3] = 1
    states[:, 1, :3] = [0., .5, .02]
    states[:, 0, :3] = [[-.2, 0, .02], [-.2, 0, .02], [-.2, 0, .10],
                        [.2, 0, .10], [.2, 0, .02], [.2, 0, .02]]
    grasp = np.zeros((6, 2, 2), dtype=bool)
    grasp[1:4, 0, 0] = True
    return states, grasp


def fixture(folder):
    import imageio.v2 as imageio
    mesh = folder / "object.obj"
    bowl = folder / "bowl.obj"
    trimesh.creation.box([.02, .02, .02]).export(mesh)
    trimesh.creation.box([.17, .17, .05]).export(bowl)
    video = folder / "source.mp4"
    with imageio.get_writer(video, fps=20, codec="libx264", ffmpeg_log_level="error") as writer:
        for t in range(6):
            writer.append_data(np.full((256, 256, 3), t * 20, dtype=np.uint8))
    states, grasp = states_and_grasp()
    spec = {"uid": "test_apple", "name": "apple", "collision": str(mesh), "visual": str(mesh)}
    other = {"uid": "other", "name": "other"}
    r = {
        "asset_specs": [spec, other], "execution_id": "fixture_execution",
        "receptacle": {"uid": "bowl", "collision": str(bowl), "visual": str(bowl),
                       "floor_profile": {"rim_z_m": .05, "center_inner_floor_z_m": 0.}},
        "split": "train",
    }
    cameras = {"native": camera(), "opposite": camera(True)}
    mv = {"cameras": cameras, "object_actor_ids": [7, 8]}
    timeline = {
        "object_pixels": np.ones((6, 2, 2), int) * 20,
        "visible_bounds": np.tile([1, 2, 5, 7], (6, 2, 2, 1)),
    }
    return {
        "receipt": r, "verification": {"fixture": True}, "fresh_verification": {"fixture": True},
        "multiview": mv, "arrays": {
            "grasp": grasp, "object_states": states,
            "container_centers_world": np.array([[-.2, 0., 0.], [.2, 0., 0.]]),
            "timestamps_seconds": np.arange(6) / 20,
            # Strings intentionally conflict with measured events: they must never be used.
            "stage": np.array(["release", "retreat", "no_grasp", "idle", "grasp", "grasp"]),
        },
        "timeline": timeline, "contacts": {"support_force_N": np.ones((6, 2)) * .7},
        "camera": cameras["opposite"], "camera_index": 1, "selected": 0, "source": 0,
        "destination": 1, "video": video, "terminal_checks": {"fixture": True},
        "bindings": {"selected_camera_video": binding(video)},
    }


class StateConditionTests(unittest.TestCase):
    def test_opposite_projection_reverses_screen_order_without_switching_roles(self):
        vertices = np.array([[-.08, -.08, 0], [.08, .08, .05]])
        projections = []
        for view in (camera(), camera(True)):
            projections.append(module.camera_role_order(
                module.container_projection([-.2, 0, 0], vertices, view),
                module.container_projection([.2, 0, 0], vertices, view)))
        self.assertEqual(projections[0]["source_relative_to_destination"], "screen_left")
        self.assertEqual(projections[1]["source_relative_to_destination"], "screen_right")
        self.assertFalse(projections[1]["pixel_identity_verified"])

    def test_no_projectable_center_leaves_direction_unknown(self):
        a = {"center": {"status": "unknown"}}
        b = {"center": {"status": "projected_geometry", "center_xy_normalized": [.7, .4]}}
        self.assertEqual(module.camera_role_order(a, b)["status"], "unknown")

    def test_events_use_correct_arm_object_axes_and_measured_release(self):
        states, grasp = states_and_grasp()
        vertices = trimesh.creation.box([.02, .02, .02]).vertices
        events = module.measured_events(grasp, states, 0, vertices, .05)
        self.assertEqual(events["grasp_established"], 1)
        self.assertEqual(events["lift_clear"], 2)
        self.assertEqual(events["released"], 4)
        self.assertEqual(events["terminal"], 5)
        self.assertFalse(events["control_stage_names_used"])

    def test_other_objects_grasp_cannot_establish_selected_grasp(self):
        states, grasp = states_and_grasp()
        grasp[:, :, 0] = False
        grasp[1:4, 1, 1] = True
        with self.assertRaisesRegex(ValueError, "Complete"):
            module.measured_events(grasp, states, 0, np.zeros((2, 3)), .05)

    def test_held_terminal_cannot_be_used_as_released_anchor(self):
        states, grasp = states_and_grasp()
        grasp[-1, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "Complete"):
            module.measured_events(grasp, states, 0, np.zeros((2, 3)), .05)

    def test_lift_without_actual_grasp_is_not_a_key_contact_event(self):
        states, grasp = states_and_grasp()
        grasp[2:4, :, :] = False
        with self.assertRaisesRegex(ValueError, "rim"):
            module.measured_events(grasp, states, 0, np.zeros((2, 3)), .05)

    def test_compile_actual_media_frames_and_camera_specific_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = fixture(folder)
            with patch.object(module, "_load_verified", return_value=data):
                result = module.compile_conditions(
                    "fixture", "fixture", "opposite", folder / "packet", partition="development")
            self.assertEqual(result["status"], "prepared_for_condition_image_review")
            self.assertEqual(result["state_delta"]["screen_language"]["source_container"], "screen_right")
            self.assertEqual(result["state_delta"]["screen_language"]["destination_container"], "screen_left")
            self.assertEqual(result["roles"]["source_container"]["container_index"], 0)
            self.assertEqual(result["state_delta"]["remove"][0]["container"],
                             result["roles"]["source_container"]["id"])
            self.assertFalse(result["generation_allowed"])
            self.assertFalse(result["generated_image_identity_verified"])
            self.assertIsNone(result["roles"]["source_container"]["renderer_actor_id"])
            self.assertEqual(result["terminal_evidence"]["terminal_frame_index"], 5)
            for condition in result["conditions"]:
                image = condition["frame"]
                self.assertEqual(module.file_sha256(image["path"]), image["sha256"])
                self.assertEqual(image["video"], binding(data["video"]))
                self.assertEqual(image["timestamp_seconds"], image["frame_index_0based"] / 20)
                self.assertFalse(condition["generated_image_scale_verified"])
            with self.assertRaises(FileExistsError):
                module.compile_conditions("fixture", "fixture", "opposite", folder / "packet",
                                          partition="development")

    def test_wrong_terminal_geometry_stops_and_preserves_blocked_result(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = fixture(folder)
            data["arrays"]["object_states"][-1, 0, 0] = -.2
            with patch.object(module, "_load_verified", return_value=data):
                result = module.compile_conditions(
                    "fixture", "fixture", "opposite", folder / "packet", partition="development")
            self.assertEqual(result["status"], "blocked_missing_or_inconsistent_evidence")
            self.assertIn("destination", result["reason"])
            self.assertFalse(result["generation_allowed"])
            self.assertTrue((folder / "packet" / "conditions.json").exists())

    def test_missing_actual_input_produces_unknown_not_mock_success(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            result = module.compile_conditions(
                folder / "missing.json", folder / "also_missing.json", "opposite",
                folder / "packet", partition="development")
            self.assertEqual(result["status"], "blocked_missing_or_inconsistent_evidence")
            self.assertEqual(result["api_calls"], 0)
            self.assertNotIn("roles", result)

    def test_stale_evidence_binding_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"initial")
            evidence = binding(path)
            path.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "changed"):
                module.checked_binding(evidence)

    def test_actual_selected_video_timebase_and_count_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            data = fixture(Path(directory))
            module.verify_video_timeline(data["video"], data["camera"], np.arange(6) / 20)
            with self.assertRaises(AssertionError):
                module.verify_video_timeline(data["video"], data["camera"], np.arange(5) / 20)
            with self.assertRaises(AssertionError):
                module.verify_video_timeline(data["video"], data["camera"], np.arange(6) / 24)
            bad = dict(data["camera"], width=512)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                module.verify_video_timeline(data["video"], bad, np.arange(6) / 20)

    def test_copied_native_calibration_cannot_label_opposite_pixels(self):
        config = {"eye": [0., 0., 1.], "target": [1., 0., 1.], "fov_radians": 1.3}
        focal = 256 / (2 * np.tan(.65))
        valid = dict(camera(), camera_from_world=[[0., -1., 0., 0.],
                                                  [0., 0., -1., 1.],
                                                  [1., 0., 0., 0.]],
                     intrinsic_matrix=[[focal, 0, 128], [0, focal, 128], [0, 0, 1]])
        module.verify_camera_geometry(valid, config)
        reversed_config = dict(config, target=[-1., 0., 1.])
        with self.assertRaisesRegex(AssertionError, "look-at"):
            module.verify_camera_geometry(valid, reversed_config)


if __name__ == "__main__":
    unittest.main()
