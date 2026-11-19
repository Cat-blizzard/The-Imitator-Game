"""CPU data-contract tests using real, locally encoded demonstration clips."""
from __future__ import annotations

import copy
import json
import pickle

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from examples.baselines.act.native_video_data import (
    NativeVideoDataset,
    load_human_demo,
    load_human_manifest,
)


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "human.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    assert writer.isOpened()
    for value in (0, 40, 80, 120, 160, 200):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()
    payload = {"schema_version": 1, "task_id": "TwoRobotPickPillToRegions-v1", "demos": [
        {"id": "train-ep0", "repo_id": "human_H31", "episode_id": 0, "camera": "zed2i", "path": str(path),
         "role": "train", "episode_frame_range": [0, 3], "frame_indices": [0, 1, 2]},
        {"id": "eval-ep1", "repo_id": "human_H31", "episode_id": 1, "camera": "zed2i", "path": str(path),
         "role": "evaluation", "episode_frame_range": [3, 6], "num_frames": 3},
    ]}
    return payload


class NativeFixture:
    state_dim = 16
    action_dim = 14
    normalizer = object()

    def __init__(self):
        self.samples = [{"view_1": torch.arange(3 * 16 * 16).reshape(1, 3, 16, 16).float(),
                         "states": torch.arange(16).reshape(1, 16).float(),
                         "actions": torch.arange(42).reshape(3, 14).float(),
                         "action_is_pad": torch.tensor([False, False, True]),
                         "robot_task": "pill"} for _ in range(2)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_native_tensors_statistics_padding_and_batches_are_preserved(manifest):
    base = NativeFixture()
    wrapped = NativeVideoDataset(base, manifest, image_size=(8, 12), video_backend="opencv")
    assert wrapped.state_dim == 16 and wrapped.action_dim == 14
    assert wrapped.normalizer is base.normalizer
    assert len(wrapped) == 2
    for key, value in base[0].items():
        assert wrapped[0][key] is value
    assert "human_video" not in base[0]
    batch = next(iter(DataLoader(wrapped, batch_size=2)))
    assert batch["human_video"].shape == (2, 3, 3, 8, 12)
    assert batch["human_video"].dtype == torch.float32
    assert batch["states"].shape == (2, 1, 16)
    assert batch["actions"].shape == (2, 3, 14)
    assert batch["action_is_pad"].tolist() == [[False, False, True]] * 2
    assert batch["human_demo_id"][0] == batch["human_demo_id"][1]
    assert 0 <= batch["human_video"].min() <= batch["human_video"].max() <= 1


def test_episode_boundaries_and_role_selection_in_concatenated_file(manifest):
    train = load_human_demo(manifest, role="train", image_size=(16, 16), video_backend="opencv")
    evaluation = load_human_demo(manifest, image_size=(16, 16), video_backend="opencv")
    assert train["metadata"]["frame_indices"] == [0, 1, 2]
    assert evaluation["metadata"]["frame_indices"] == [3, 4, 5]
    assert train["metadata"]["video_sha256"] == evaluation["metadata"]["video_sha256"]
    assert train["human_demo_id"] != evaluation["human_demo_id"]
    torch.testing.assert_close(evaluation["human_video"][:, 0, 0, 0], torch.tensor([120, 160, 200]) / 255,
                               atol=2 / 255, rtol=0)
    assert evaluation["metadata"]["episode_frame_range"] == [3, 6]


def test_relative_path_manifest_and_roundtrip_are_portable(manifest, tmp_path):
    manifest["demos"][0]["path"] = "human.avi"
    manifest["demos"][1]["path"] = "human.avi"
    path = tmp_path / "pairing.json"
    path.write_text(json.dumps(manifest))
    parsed = load_human_manifest(path)
    assert parsed["demos"][0]["path"] == str(tmp_path / "human.avi")
    first = load_human_demo(parsed, video_backend="opencv")
    reloaded = load_human_demo(path, video_backend="opencv")
    assert first["human_demo_id"] == reloaded["human_demo_id"]
    torch.testing.assert_close(first["human_video"], reloaded["human_video"])


def test_same_repo_cache_identity_tracks_actual_frames_and_image_size(manifest):
    original = load_human_demo(manifest, role="train", video_backend="opencv")
    changed = copy.deepcopy(manifest)
    changed["demos"][0]["frame_indices"] = [0, 0, 2]
    different_frames = load_human_demo(changed, role="train", video_backend="opencv")
    assert original["human_demo_id"] != different_frames["human_demo_id"]
    changed["demos"][0]["frame_indices"] = [0, 1, 2]
    different_size = load_human_demo(changed, role="train", image_size=(16, 16), video_backend="opencv")
    assert original["human_demo_id"] != different_size["human_demo_id"]
    # Display labels and role changes alone must not change the actual clip key.
    changed["demos"][0].update(id="renamed", role="development")
    renamed = load_human_demo(changed, role="development", video_backend="opencv")
    assert renamed["human_demo_id"] == original["human_demo_id"]


def test_file_content_change_invalidates_identity_and_recorded_checksum(manifest):
    before = load_human_demo(manifest, role="train", video_backend="opencv")
    frozen = load_human_manifest(manifest)
    path = manifest["demos"][0]["path"]
    # AVI accepts trailing bytes, so pixels may remain equal while file identity changes.
    with open(path, "ab") as stream:
        stream.write(b"different-file-content")
    after = load_human_demo(manifest, role="train", video_backend="opencv")
    assert before["human_demo_id"] != after["human_demo_id"]
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_human_manifest(frozen)


def test_multiple_task_demos_are_deterministic_and_require_explicit_eval_id(manifest):
    alternate = copy.deepcopy(manifest["demos"][0])
    alternate.update(id="train-alt", frame_indices=[0, 0, 0])
    manifest["demos"].append(alternate)
    wrapped = NativeVideoDataset(NativeFixture(), manifest, video_backend="opencv")
    assert wrapped[0]["human_demo_id"] != wrapped[1]["human_demo_id"]
    assert wrapped[0]["human_demo_id"] == wrapped[0]["human_demo_id"]
    with pytest.raises(ValueError, match="exactly one"):
        load_human_demo(manifest, role="train", video_backend="opencv")
    selected = load_human_demo(manifest, role="train", demo_id="train-alt", video_backend="opencv")
    assert selected["human_demo_id"] == wrapped[1]["human_demo_id"]


def test_short_episode_repeats_only_its_own_frames(manifest):
    record = manifest["demos"][1]
    record.update(episode_frame_range=[4, 5], num_frames=3)
    item = load_human_demo(manifest, video_backend="opencv")
    assert item["metadata"]["frame_indices"] == [4, 4, 4]
    assert torch.equal(item["human_video"][0], item["human_video"][2])


@pytest.mark.parametrize("mutation,error", [
    (lambda p: p.update(schema_version=2), "schema_version"),
    (lambda p: p.update(task_id=""), "task_id"),
    (lambda p: p["demos"][0].update(role="test"), "role"),
    (lambda p: p["demos"][0].update(episode_id=-1), "episode_id"),
    (lambda p: p["demos"][0].update(num_frames=3), "exactly one"),
    (lambda p: p["demos"][0].update(frame_indices=[0, 3]), "outside"),
    (lambda p: p["demos"][0].update(frame_indices=[2, 1]), "temporal order"),
    (lambda p: p["demos"][0].update(frame_indices=[True]), "frame index"),
    (lambda p: p["demos"][0].update(episode_frame_range=[2, 2]), "exceed start"),
    (lambda p: p["demos"][1].update(id="train-ep0"), "Duplicate"),
])
def test_manifest_rejects_ambiguous_or_invalid_pairing(manifest, mutation, error):
    mutation(manifest)
    with pytest.raises(ValueError, match=error):
        load_human_manifest(manifest)


def test_video_bounds_do_not_clip_or_borrow_neighbour_frames(manifest):
    manifest["demos"][1]["episode_frame_range"] = [3, 7]
    with pytest.raises(ValueError, match="beyond video length"):
        load_human_demo(manifest, video_backend="opencv")
    manifest["demos"][1].pop("episode_frame_range")
    manifest["demos"][1].pop("num_frames")
    manifest["demos"][1]["frame_indices"] = [0, 6]
    with pytest.raises(ValueError, match="exceeds video length"):
        load_human_demo(manifest, video_backend="opencv")


def test_wrapper_task_role_frame_count_and_pickle_contract(manifest):
    with pytest.raises(ValueError, match="task_id"):
        NativeVideoDataset(NativeFixture(), manifest, task_id="another-task", video_backend="opencv")
    with pytest.raises(ValueError, match="No human demonstrations"):
        NativeVideoDataset(NativeFixture(), manifest, role="development", video_backend="opencv")
    other = copy.deepcopy(manifest["demos"][0])
    other.update(id="other", frame_indices=[0])
    manifest["demos"].append(other)
    with pytest.raises(ValueError, match="same sampled frame count"):
        NativeVideoDataset(NativeFixture(), manifest, video_backend="opencv")
    manifest["demos"].pop()
    wrapped = NativeVideoDataset(NativeFixture(), manifest, video_backend="opencv")
    restored = pickle.loads(pickle.dumps(wrapped))
    torch.testing.assert_close(restored[0]["human_video"], wrapped[0]["human_video"])
    assert restored[0]["human_demo_id"] == wrapped[0]["human_demo_id"]


def test_torchcodec_matches_exact_physical_frame_requests(manifest):
    # Exercise the same official VideoDecoderCache used by actual training.
    decoded = load_human_demo(manifest, role="evaluation", image_size=(16, 16), video_backend="torchcodec")
    reference = load_human_demo(manifest, role="evaluation", image_size=(16, 16), video_backend="opencv")
    assert decoded["human_demo_id"] == reference["human_demo_id"]
    torch.testing.assert_close(decoded["human_video"], reference["human_video"], atol=2 / 255, rtol=0)
