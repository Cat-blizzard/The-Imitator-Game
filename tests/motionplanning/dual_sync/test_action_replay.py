"""Initial-state/action replay contracts and placement-window boundaries."""

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import replay
from mani_skill.examples.motionplanning.dual.sync.runtime_policy import DEFAULT_POLICY
from mani_skill.examples.motionplanning.dual.sync.solve import phase_sequence


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "sync.h5"
    stages = [{"name": name, "status": "passed", "environment_steps": 1}
              for name in phase_sequence(DEFAULT_POLICY)]
    count = 10 + len(stages)
    metadata = {"env_info": {"env_id": replay.TASK_ID},
        "motion_planning": {"action_order": list(replay.ACTION_ORDER)},
        "episodes": [{"episode_id": 0, "episode_seed": 401,
            "dual_sync": {"schema": "dual-panda-sync-v1", "policy": DEFAULT_POLICY.to_dict(), "stages": stages}}]}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with h5py.File(path, "w") as h5:
        h5["traj_0/actions"] = np.arange(count*16, dtype=np.float32).reshape(count, 16)
        h5["traj_0/env_states/actors/target"] = np.arange((count+1)*13).reshape(count+1, 13)
    return path, metadata


def test_restores_exactly_state_zero_and_all_action_rows(recording):
    path, _ = recording
    _, _, _, phases, actions, state = replay.load_episode(path)
    assert phases[0] == ("preparation", 10)
    assert actions.shape == (sum(count for _, count in phases), 16)
    np.testing.assert_array_equal(state["actors"]["target"], np.arange(13))


def test_action_order_preserves_both_physical_arms():
    action = replay.paired_action(np.arange(16))
    np.testing.assert_array_equal(action["panda_wristcam-0"], np.arange(8))
    np.testing.assert_array_equal(action["panda_wristcam-1"], np.arange(8, 16))
    with pytest.raises(ValueError, match="finite"):
        replay.paired_action(np.full(16, np.nan))


@pytest.mark.parametrize("mutation, message", [
    (lambda data: data["episodes"][0].pop("dual_sync"), "synchronous episode metadata"),
    (lambda data: data["motion_planning"].update(action_order=list(reversed(replay.ACTION_ORDER))), "left/right"),
    (lambda data: data["episodes"][0]["dual_sync"]["stages"][0].update(status="rejected"), "partial"),
    (lambda data: data["episodes"][0]["dual_sync"]["stages"][0].update(environment_steps=2), "Action count"),
])
def test_rejects_ambiguous_phase_or_action_contract(recording, mutation, message):
    path, metadata = recording
    mutation(metadata)
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=message):
        replay.load_episode(path)


def placement_row():
    return {"max_arm_velocity_rad_s": 0., "objects": [
        {"is_grasping": False, "control_frame_linear_speed_m_s": 0.,
         "control_frame_angular_speed_rad_s": 0., "object_T_world": np.eye(4).tolist()}
        for _ in range(2)]}


def test_placement_window_uses_cumulative_drift_and_release():
    row = placement_row()
    count, reference = replay.placement_window(row, True, [True, True], 0, None)
    assert count == 1
    row["objects"][0]["object_T_world"][0][3] = .002
    count, reference = replay.placement_window(row, True, [True, True], count, reference)
    assert count == 2
    row["objects"][0]["object_T_world"][0][3] = .004
    count, reference = replay.placement_window(row, True, [True, True], count, reference)
    assert count == 1
    assert replay.placement_window(row, False, [True, True], count, reference) == (0, None)
    assert replay.placement_window(row, True, [False, True], count, reference) == (0, None)


def test_output_file_is_preserved_before_any_simulation(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("keep")
    monkeypatch.setattr(replay, "replay", lambda *a: pytest.fail("simulation started"))
    with pytest.raises(FileExistsError):
        replay.main(["--traj-path", "source.h5", "--output", str(output)])
    assert output.read_text() == "keep"


def test_zero_step_outcome_is_valid_without_quality_windows(recording):
    path, metadata = recording
    sync = metadata["episodes"][0]["dual_sync"]
    policy = replace(DEFAULT_POLICY, require_final_stability=False)
    sync["policy"] = policy.to_dict()
    sync["stages"][-1] = {"name": "observe_task_outcome", "status": "passed", "environment_steps": 0}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with h5py.File(path, "a") as h5:
        actions = h5["traj_0/actions"][()]
        del h5["traj_0/actions"]
        h5["traj_0/actions"] = actions[:-1]
    _, _, selected, phases, actions, _ = replay.load_episode(path)
    assert not selected.require_final_stability
    assert phases[-1] == ("observe_task_outcome", 0)
    assert len(actions) == sum(count for _, count in phases)
