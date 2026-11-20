"""Real image/video statistics promote byte values before second moments."""
import json
from pathlib import Path
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import av
import numpy as np
from PIL import Image
import pytest

from lerobot.datasets.compute_stats import compute_episode_stats as native_stats
from imitator.dataset.lerobot_dataset import (
    LeRobotDataset, compute_episode_stats,
)
from imitator.dataset.video_utils import StreamingVideoEncoder


def _frames(count=4, height=32, width=48):
    y, x = np.indices((height, width))
    return np.stack([np.stack(((x*7+y*3+index*31) % 256,
                              (x*5+y*11+index*17) % 256,
                              np.full((height, width), 155)), axis=-1)
                     for index in range(count)]).astype(np.uint8)


def _paths(tmp_path, frames):
    paths = []
    for index, frame in enumerate(frames):
        path = tmp_path/f"frame-{index:06d}.png"
        Image.fromarray(frame).save(path)
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("dtype", ["image", "video"])
@pytest.mark.parametrize("constant", [False, True])
def test_sampled_image_stats_match_independent_float64_moments(tmp_path, dtype, constant):
    frames = np.full((4, 301, 450, 3), 220, dtype=np.uint8) if constant else _frames(4, 301, 450)
    original = frames.copy()
    paths = _paths(tmp_path, frames)
    key = "observation.images.camera"
    data = {key: paths, "action": np.array([[.1, .4], [.2, -.6], [.3, .2], [.4, -.8]], dtype=np.float32),
            "task": ["test"]*4}
    features = {key: {"dtype": dtype}, "action": {"dtype": "float32"}, "task": {"dtype": "string"}}
    actual = compute_episode_stats(data, features)
    native = native_stats(data, features)
    # Preserve the native frame selection and explicit 450/150 spatial stride.
    pixels = frames[:, ::3, ::3].astype(np.float64)/255.
    for name, expected in (("mean", pixels.mean(axis=(0, 1, 2))),
                           ("std", pixels.std(axis=(0, 1, 2))),
                           ("min", pixels.min(axis=(0, 1, 2))),
                           ("max", pixels.max(axis=(0, 1, 2)))):
        np.testing.assert_allclose(actual[key][name][:, 0, 0], expected, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(actual[key]["count"], [4])
    for name in actual[key]:
        if name != "std":
            np.testing.assert_array_equal(actual[key][name], native[key][name])
    for name in actual["action"]:
        np.testing.assert_array_equal(actual["action"][name], native["action"][name])
    assert "task" not in actual
    np.testing.assert_array_equal(frames, original)


def test_real_default_video_writer_saves_correct_channel_statistics(tmp_path):
    frames = _frames()
    key = "observation.images.camera"
    root = tmp_path/"default-video"
    dataset = LeRobotDataset.create(repo_id="default-video", fps=20, root=root,
        features={key: {"dtype": "video", "shape": (3, 32, 48),
                        "names": ["channels", "height", "width"]},
                  "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]}},
        use_videos=True, video_backend="pyav", vcodec="h264", encoder_threads=1)
    try:
        for index, frame in enumerate(frames):
            dataset.add_frame({key: frame.transpose(2, 0, 1), "task": "test",
                               "action": np.array([index, -index], dtype=np.float32)})
        dataset.save_episode()
    finally:
        dataset.finalize()
    stats = json.loads((root/"meta/stats.json").read_text())[key]
    normalized = frames.astype(np.float64)/255.
    np.testing.assert_allclose(np.asarray(stats["std"])[:, 0, 0],
                               normalized.std(axis=(0, 1, 2)), rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.asarray(stats["mean"])[:, 0, 0],
                               normalized.mean(axis=(0, 1, 2)), rtol=0, atol=1e-11)
    assert np.asarray(stats["std"])[0, 0, 0] > .1
    assert np.asarray(stats["std"])[2, 0, 0] == 0.


def test_streaming_video_keeps_pixels_and_20hz_timestamps_with_correct_raw_stats(tmp_path):
    frames = _frames()
    original = frames.copy()
    key = "observation.images.camera"
    encoder = StreamingVideoEncoder(fps=20, vcodec="h264", pix_fmt="yuv444p", encoder_threads=1)
    try:
        encoder.start_episode([key], tmp_path)
        for frame in frames:
            encoder.feed_frame(key, frame)
        path, stats = encoder.finish_episode()[key]
    finally:
        if encoder._episode_active:
            encoder.cancel_episode()
    pixels = frames.astype(np.float64)
    np.testing.assert_allclose(stats["mean"], pixels.mean(axis=(0, 1, 2)), rtol=0, atol=1e-10)
    np.testing.assert_allclose(stats["std"], pixels.std(axis=(0, 1, 2)), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(stats["count"], [4*32*48])
    np.testing.assert_array_equal(frames, original)
    with av.open(str(path)) as container:
        decoded = list(container.decode(video=0))
        assert container.streams.video[0].average_rate == 20
        np.testing.assert_allclose([frame.time for frame in decoded], np.arange(4)/20., atol=1e-12)
    assert len(decoded) == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
