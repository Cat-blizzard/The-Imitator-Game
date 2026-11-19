"""Opt-in, synchronized camera recording for the existing strict apple runner.

The legacy zed2i/native cameras, controller, and physics gates are unchanged.
This module adds a fixed opposite camera and records it from the very same
pre-command observation. It does not label RGB contact or physical success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .planning_contract import bind, save_new


SCHEMA = "synchronized_multiview_v1"
CAMERA_IDS = ("zed2i", "native", "opposite")
ENV_ID = "HumanGroundedAppleClutterMultiview-v1"
OPPOSITE_CAMERA = {
    "eye": [0.45, 0.0, 0.45],
    "target": [0.0, 0.0, 0.12],
    "fov_radians": 1.30,
    "width": 512,
    "height": 512,
    "near": 0.01,
    "far": 100.0,
    "fixed_during_episode": True,
}


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def register_multiview(proposal):
    """Register a separate env; the original sensor objects are retained."""
    from .experimental_apple_adapter import register
    from mani_skill.utils.registration import register_env, REGISTERED_ENVS
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils

    register(proposal)
    parent = REGISTERED_ENVS["HumanGroundedAppleClutterExperimental-v1"].cls

    @register_env(ENV_ID, max_episode_steps=3000)
    class MultiviewApple(parent):
        @property
        def _default_sensor_configs(self):
            configs = list(super()._default_sensor_configs)
            if [c.uid for c in configs] != list(CAMERA_IDS[:2]):
                raise ValueError("Unexpected parent camera set; preserve cameras explicitly")
            c = OPPOSITE_CAMERA
            configs.append(CameraConfig(
                "opposite", sapien_utils.look_at(c["eye"], c["target"]),
                c["width"], c["height"], c["fov_radians"], c["near"], c["far"]))
            return configs

    return ENV_ID


def calibration(sensor):
    c = sensor.config
    return {
        "width": int(c.width), "height": int(c.height),
        "intrinsic_matrix": array(sensor.camera.get_intrinsic_matrix()).reshape(3, 3).tolist(),
        "camera_from_world": array(sensor.camera.get_extrinsic_matrix()).reshape(3, 4).tolist(),
        "near": float(c.near), "far": float(c.far),
        "fov": float(c.fov) if c.fov is not None else None,
        "shader_pack": c.shader_pack,
        "coordinate_convention": "OpenCV camera; camera_from_world is 3x4, world in metres",
        "fixed_during_episode": True,
    }


class SynchronizedMultiviewMixin:
    """Compose before VisibilityRecorder; never render or step separately.

    Only the opposite RGB adds image memory. The two original frames remain in
    the base recorder, and their hashes are checked against this capture.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opposite_frames = []
        self.multiview_rows = []
        self.multiview_cameras = {}

    def reset(self, *args, **kwargs):
        if self.multiview_cameras or self.rows:
            raise RuntimeError("A synchronized recorder is single-episode only")
        super().reset(*args, **kwargs)
        self.multiview_cameras = {
            camera_id: calibration(self.unwrapped._sensors[camera_id])
            for camera_id in CAMERA_IDS
        }
        native = self.multiview_cameras["native"]
        opposite = self.multiview_cameras["opposite"]
        if np.allclose(native["camera_from_world"], opposite["camera_from_world"], atol=1e-6, rtol=0):
            raise ValueError("Opposite camera must differ from the preserved native view")

    def _snapshot_multiview(self):
        rgb_hashes, counts, bounds = [], [], []
        opposite_frame = None
        for camera_id in CAMERA_IDS:
            info = self.multiview_cameras[camera_id]
            live = calibration(self.unwrapped._sensors[camera_id])
            if live != info:
                raise ValueError("Fixed camera calibration changed: " + camera_id)
            height, width = info["height"], info["width"]
            data = self.obs["sensor_data"][camera_id]
            rgb = array(data["rgb"]).reshape(height, width, 3)
            if rgb.dtype != np.uint8:
                raise ValueError("Unexpected RGB dtype: " + camera_id)
            seg = array(data["segmentation"]).reshape(height, width)
            rgb_hashes.append(digest(rgb))
            camera_counts, camera_bounds = [], []
            for actor_id in self.ids:
                y, x = np.where(seg == actor_id)
                camera_counts.append(len(x))
                camera_bounds.append([int(x.min()), int(y.min()), int(x.max()), int(y.max())]
                                     if len(x) else [-1] * 4)
            counts.append(camera_counts)
            bounds.append(camera_bounds)
            if camera_id == "opposite":
                opposite_frame = rgb.copy()
        return opposite_frame, {
            "raw_rgb_sha256": rgb_hashes, "object_pixels": counts, "visible_bounds": bounds,
            "env_step_index": int(array(self.unwrapped.elapsed_steps).item()),
        }

    def step(self, action):
        if not self.recording:
            return super().step(action)
        # Validate before either recorder can append a partial command row.
        command = np.concatenate([array(action[f"panda_wristcam-{i}"]).reshape(8) for i in range(2)])
        if not np.isfinite(command).all():
            raise ValueError("Nonfinite multiview command")
        count = len(self.rows)
        frame, snapshot = self._snapshot_multiview()
        try:
            return super().step(action)
        finally:
            # Preserve the observation/command even if env.step throws after
            # recording it. An exception before a base row adds no camera row.
            if len(self.rows) == count + 1:
                self.opposite_frames.append(frame)
                self.multiview_rows.append(snapshot)
                if digest(self.frames[-1]) != snapshot["raw_rgb_sha256"][0]:
                    raise RuntimeError("zed2i capture does not match base recorder")
                if digest(self.native_frames[-1]) != snapshot["raw_rgb_sha256"][1]:
                    raise RuntimeError("native capture does not match base recorder")
            elif len(self.rows) != count:
                raise RuntimeError("Base recorder appended an unexpected number of rows")


def _video_pts(path, width, height, timestamps):
    import av
    observed = []
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            if (frame.width, frame.height) != (width, height):
                raise ValueError("Multiview video resolution differs: " + str(path))
            observed.append(float(frame.pts * frame.time_base))
    np.testing.assert_allclose(observed, timestamps, atol=1e-5, rtol=0,
                               err_msg="Multiview decoded count/PTS differ")
    return len(observed)


def save_multiview(rec, folder, execution_id, *, diagnostic=False):
    """Bind all three videos to one unchanged control/state array artifact."""
    import imageio.v2 as imageio

    folder = Path(folder)
    target = folder / "multiview"
    if target.exists():
        raise FileExistsError("Immutable multiview export exists: " + str(target))
    n = len(rec.rows)
    if not n or not (n == len(rec.multiview_rows) == len(rec.opposite_frames)
                     == len(rec.frames) == len(rec.native_frames) == len(rec.commands)):
        raise ValueError("Multiview/state/action lengths differ or recording is empty")
    timestamps = np.arange(n, dtype=np.float64) / rec.control_freq
    if diagnostic:
        # The legacy diagnostic trace uses a different schema. Preserve it and
        # add normalized shared arrays, without assigning a physical pass.
        arrays_path = folder / "multiview_shared_arrays.npz"
        if arrays_path.exists():
            raise FileExistsError(arrays_path)
        np.savez_compressed(
            arrays_path, native_commands=np.stack(rec.commands),
            states=np.stack([r["states"] for r in rec.rows]),
            object_states=np.stack([r["objects"] for r in rec.rows]).astype("float32"),
            timestamps_seconds=timestamps, stage=np.asarray(rec.stage_names, dtype="U32"),
            grasp=np.stack([r["grasp"] for r in rec.rows]))
    else:
        arrays_path = folder / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as shared:
        for key, expected in (
                ("native_commands", np.stack(rec.commands)),
                ("states", np.stack([r["states"] for r in rec.rows])),
                ("object_states", np.stack([r["objects"] for r in rec.rows]).astype("float32")),
                ("timestamps_seconds", timestamps)):
            np.testing.assert_array_equal(shared[key], expected,
                                          err_msg="Shared physical trace mismatch: " + key)
        motion_hash = hashlib.sha256()
        for key in ("native_commands", "states", "object_states", "timestamps_seconds"):
            motion_hash.update(np.ascontiguousarray(shared[key]).tobytes())
    env_steps = np.asarray([row["env_step_index"] for row in rec.multiview_rows], dtype=np.int64)
    np.testing.assert_array_equal(np.diff(env_steps), np.ones(max(n - 1, 0), dtype=np.int64))
    target.mkdir()
    opposite_video = folder / "video_opposite512.mp4"
    if opposite_video.exists():
        raise FileExistsError(opposite_video)
    with imageio.get_writer(
            opposite_video, fps=rec.control_freq, codec="libx264", macro_block_size=16,
            ffmpeg_log_level="error", output_params=[
                "-crf", "18", "-preset", "fast", "-tune", "zerolatency",
                "-g", str(rec.control_freq), "-bf", "0", "-sc_threshold", "0"]) as writer:
        for frame in rec.opposite_frames:
            writer.append_data(frame)
    timeline = target / "timeline.npz"
    np.savez_compressed(
        timeline, frame_index=np.arange(n), state_index=np.arange(n), action_index=np.arange(n),
        timestamps_seconds=timestamps, env_step_index=env_steps,
        camera_ids=np.asarray(CAMERA_IDS),
        camera_frame_index=np.repeat(np.arange(n)[:, None], len(CAMERA_IDS), axis=1),
        raw_rgb_sha256=np.asarray([r["raw_rgb_sha256"] for r in rec.multiview_rows]),
        object_pixels=np.asarray([r["object_pixels"] for r in rec.multiview_rows]),
        visible_bounds=np.asarray([r["visible_bounds"] for r in rec.multiview_rows]))
    videos = dict(zed2i=folder / "video.mp4", native=folder / "video_native512.mp4",
                  opposite=opposite_video)
    cameras = {}
    for name, info in rec.multiview_cameras.items():
        count = _video_pts(videos[name], info["width"], info["height"], timestamps)
        cameras[name] = dict(info, video=bind(videos[name]), decoded_frames=count)
    manifest = {
        "schema": SCHEMA, "execution_id": execution_id, "length": n,
        "fps": int(rec.control_freq), "one_physical_execution": True,
        "pre_command_capture": True, "same_observation_all_cameras": True,
        "video_alignment": "All camera frame[t] and measured state[t] precede command[t] in one env.step",
        "timestamp_origin": "First recorded control step; env_step_index also retains elapsed settling steps",
        "shared_arrays": bind(arrays_path), "timeline": bind(timeline), "cameras": cameras,
        "dynamic_trajectory_sha256": motion_hash.hexdigest(),
        "dynamic_hash_fields": ["native_commands", "states", "object_states", "timestamps_seconds"],
        "dynamic_hash_definition": "SHA256 concatenated C-contiguous bytes in listed order, matching 20261020E",
        "object_actor_ids": rec.ids,
        "object_identity": rec.unwrapped.asset_specs,
        "opposite_camera_config": OPPOSITE_CAMERA,
        "diagnostic_failure_trace": diagnostic,
        "segmentation_is_QA_only_and_not_model_input": True,
        "independent_motion_trajectories": 1, "distinct_viewpoints": 2,
        "camera_streams": 3, "training_admitted": False,
        "limits": [
            "Static cameras only; calibration changes abort capture.",
            "Segmentation visibility is a diagnostic, not evidence of correct human contact.",
            "Controller stages are not contact ground truth.",
            "Additional camera changes rendering overhead; no extra physics step is introduced.",
        ],
    }
    save_new(target / "manifest.json", manifest)
    return {"multiview": bind(target / "manifest.json"), "execution_id": execution_id}
