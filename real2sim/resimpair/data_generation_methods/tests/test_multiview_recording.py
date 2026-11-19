"""CPU integration tests of the real legacy step and the opt-in camera layer.

The fake environment deliberately reuses/mutates observation buffers; no
physics qualification is inferred from this deterministic recorder fixture.
"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from data_generation_methods.human_to_robot_pipeline.multiview_recording import (
    CAMERA_IDS, SynchronizedMultiviewMixin, calibration, digest, save_multiview,
)

POLICYGEN = Path(__file__).resolve().parents[2] / "PolicyGen"
sys.path.insert(0, str(POLICYGEN))
from examples.baselines.hypericl_data_factory.rich_runtime_v2 import VisibilityRecorder, save_native


class Sensor:
    def __init__(self, side):
        size = 224 if side == "zed2i" else 512
        self.extrinsic = np.eye(4)[:3].copy()
        if side == "opposite":
            self.extrinsic[0, 3] = 1.
        self.config = SimpleNamespace(width=size, height=size, near=.01, far=100.,
                                      fov=1.3, shader_pack="rt-fast")
        self.camera = SimpleNamespace(
            get_intrinsic_matrix=lambda: np.eye(3),
            get_extrinsic_matrix=lambda: self.extrinsic)


class MutableEnvironment:
    def __init__(self, fail_step=False):
        self.unwrapped = self
        self.agent = SimpleNamespace(agents=[])
        self.elapsed_steps = np.array([20])
        self.control_freq = 20
        self.asset_specs = [{"name": "test_object", "uid": "test_only"}]
        self._sensors = {name: Sensor(name) for name in CAMERA_IDS}
        self.obs = {"sensor_data": {}}
        self.fail_step = fail_step
        for name, sensor in self._sensors.items():
            h, w = sensor.config.height, sensor.config.width
            self.obs["sensor_data"][name] = {
                "rgb": np.full((1, h, w, 3), 20, dtype=np.uint8),
                "segmentation": np.ones((1, h, w, 1), dtype=np.uint32) * 7,
            }

    def step(self, action):
        if self.fail_step:
            raise RuntimeError("Deliberate step failure after pre-command record")
        self.elapsed_steps += 1
        for data in self.obs["sensor_data"].values():
            data["rgb"][:] = int(self.elapsed_steps.item())
        return self.obs, 0, False, False, {}


class Recorder(SynchronizedMultiviewMixin, VisibilityRecorder):
    def measure(self):
        t = self.unwrapped.elapsed_steps.item()
        return {"states": np.full(18, t, dtype=np.float32),
                "objects": np.full((1, 13), t, dtype=np.float64),
                "grasp": np.array([[False], [False]])}

    def image_metrics(self):
        return {
            "object_pixels_224": [224 * 224], "object_pixels_512": [512 * 512],
            "projected_bounds_224": [[0, 0, 223, 223]],
            "visible_bounds_224": [[0, 0, 223, 223]],
            "projected_entirely_in_frame": [True],
        }


def initialized(*, fail_step=False):
    env = MutableEnvironment(fail_step)
    rec = Recorder(env)
    rec.obs = env.obs
    rec.ids = [7]
    rec.extrinsic = env._sensors["zed2i"].extrinsic
    rec.intrinsic = np.eye(3)
    rec.native_intrinsic = np.eye(3)
    rec.multiview_cameras = {name: calibration(sensor) for name, sensor in env._sensors.items()}
    rec.recording = True
    return rec


def action(t=0):
    return {f"panda_wristcam-{i}": np.full(8, t, dtype=np.float32) for i in range(2)}


def write_base(rec, folder):
    import imageio.v2 as imageio
    np.savez_compressed(
        folder / "arrays.npz", states=np.stack([r["states"] for r in rec.rows]),
        native_commands=np.stack(rec.commands),
        object_states=np.stack([r["objects"] for r in rec.rows]).astype("float32"),
        timestamps_seconds=np.arange(len(rec.rows), dtype=np.float64) / rec.control_freq)
    with imageio.get_writer(folder / "video.mp4", fps=20, codec="libx264",
                            ffmpeg_log_level="error", output_params=["-bf", "0"]) as writer:
        for frame in rec.frames:
            writer.append_data(frame)
    save_native(rec, folder)


class MultiviewRecordingTests(unittest.TestCase):
    def test_actual_legacy_step_shares_precommand_observation_and_copies_buffers(self):
        rec = initialized()
        for t in range(4):
            rec.step(action(t))
        self.assertEqual(rec.unwrapped.elapsed_steps.item(), 24)
        for t in range(4):
            self.assertEqual(rec.rows[t]["states"][0], 20 + t)
            self.assertEqual(rec.commands[t][0], t)
            self.assertEqual(rec.multiview_rows[t]["env_step_index"], 20 + t)
            for index, frames in enumerate((rec.frames, rec.native_frames, rec.opposite_frames)):
                self.assertEqual(frames[t][0, 0, 0], 20 + t)
                self.assertEqual(digest(frames[t]), rec.multiview_rows[t]["raw_rgb_sha256"][index])

    def test_missing_camera_stops_before_any_command_or_partial_row(self):
        rec = initialized()
        del rec.obs["sensor_data"]["opposite"]
        with self.assertRaises(KeyError):
            rec.step(action())
        self.assertEqual(rec.unwrapped.elapsed_steps.item(), 20)
        self.assertEqual(len(rec.rows), 0)
        self.assertEqual(len(rec.native_frames), 0)

    def test_camera_drift_stops_before_command(self):
        rec = initialized()
        rec.unwrapped._sensors["opposite"].extrinsic[0, 3] += .01
        with self.assertRaisesRegex(ValueError, "calibration changed"):
            rec.step(action())
        self.assertEqual(len(rec.rows), 0)

    def test_nonfinite_command_has_no_partial_camera_record(self):
        rec = initialized()
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            rec.step(action(float("nan")))
        self.assertEqual(len(rec.native_frames), 0)
        self.assertEqual(len(rec.multiview_rows), 0)

    def test_step_failure_retains_precommand_capture(self):
        rec = initialized(fail_step=True)
        with self.assertRaisesRegex(RuntimeError, "Deliberate"):
            rec.step(action())
        self.assertEqual(len(rec.rows), 1)
        self.assertEqual(len(rec.opposite_frames), 1)
        self.assertEqual(len(rec.multiview_rows), 1)
        self.assertEqual(rec.multiview_rows[0]["env_step_index"], 20)

    def test_export_real_codecs_pts_shared_trace_and_immutable_output(self):
        rec = initialized()
        for t in range(4):
            rec.step(action(t))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_base(rec, folder)
            saved = save_multiview(rec, folder, "test-physical-execution")
            manifest = json.loads(Path(saved["multiview"]["path"]).read_text())
            self.assertEqual(manifest["execution_id"], "test-physical-execution")
            self.assertEqual(manifest["distinct_viewpoints"], 2)
            self.assertEqual(manifest["camera_streams"], 3)
            self.assertFalse(manifest["training_admitted"])
            self.assertEqual(set(manifest["cameras"]), set(CAMERA_IDS))
            for camera in manifest["cameras"].values():
                self.assertEqual(camera["decoded_frames"], 4)
            with np.load(manifest["timeline"]["path"]) as timeline:
                np.testing.assert_array_equal(timeline["camera_frame_index"],
                                              np.repeat(np.arange(4)[:, None], 3, axis=1))
                np.testing.assert_array_equal(timeline["state_index"], timeline["action_index"])
                self.assertEqual(timeline["object_pixels"][0, 2, 0], 512 * 512)
            with self.assertRaises(FileExistsError):
                save_multiview(rec, folder, "cannot-overwrite")

    def test_wrong_shared_trace_is_rejected_before_export(self):
        rec = initialized()
        rec.step(action())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_base(rec, folder)
            with np.load(folder / "arrays.npz") as original:
                values = dict(original)
            values["states"][0, 0] += 1
            np.savez_compressed(folder / "arrays.npz", **values)
            with self.assertRaisesRegex(AssertionError, "Shared physical trace mismatch"):
                save_multiview(rec, folder, "test")
            self.assertFalse((folder / "multiview").exists())

    def test_skipped_control_step_is_rejected(self):
        rec = initialized()
        rec.step(action())
        rec.unwrapped.elapsed_steps += 1
        rec.step(action())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_base(rec, folder)
            with self.assertRaises(AssertionError):
                save_multiview(rec, folder, "test")
            self.assertFalse((folder / "multiview").exists())

    def test_failure_trace_export_preserves_failure_flag_and_shared_arrays(self):
        rec = initialized(fail_step=True)
        with self.assertRaises(RuntimeError):
            rec.step(action())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_base(rec, folder)
            result = save_multiview(rec, folder, "failed-execution", diagnostic=True)
            manifest = json.loads(Path(result["multiview"]["path"]).read_text())
            self.assertTrue(manifest["diagnostic_failure_trace"])
            self.assertFalse(manifest["training_admitted"])
            self.assertTrue(manifest["shared_arrays"]["path"].endswith("multiview_shared_arrays.npz"))
            self.assertEqual(manifest["length"], 1)


if __name__ == "__main__":
    unittest.main()
