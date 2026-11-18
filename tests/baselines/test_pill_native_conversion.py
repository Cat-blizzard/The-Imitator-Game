"""Official sync recordings retain measured native layout through conversion."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest

from examples.baselines.lerobot_dataset import h5_to_lerobot as converter
from mani_skill.examples.motionplanning.dual import two_robot_run as runner


def _joint(name):
    return NS(get_name=lambda: name)


def _live_layout():
    # Actual articulation order need not have arm joints first or match between
    # instances. Conversion must preserve recorded qpos rather than guess it.
    states = [("joint8", "joint4", "joint2", "joint7", "joint6", "joint1", "joint5", "joint3"),
              ("joint7", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint8")]
    agents = {}
    for side, names in enumerate(states):
        agents[f"piper-{side}"] = NS(single_action_space=NS(shape=(7,)),
            controller=NS(controllers={"arm": NS(joints=[_joint(f"joint{i}") for i in range(1, 7)])}),
            robot=NS(get_active_joints=lambda names=names: [_joint(name) for name in names]),
            gripper_coordinate_unit="m", gripper_open_action=1., gripper_closed_action=-1.)
    return runner._native_action_layout(NS(agent=NS(agents_dict=agents)))


def _source(tmp_path, *, layout=True):
    path = tmp_path/"pill.h5"
    actions = np.arange(42, dtype=np.float32).reshape(3, 14)/100
    states = [np.arange(32, dtype=np.float32).reshape(4, 8)+side*100 for side in range(2)]
    images = np.arange(4*4*4*3, dtype=np.uint8).reshape(4, 4, 4, 3)
    with h5py.File(path, "w") as source:
        source["traj_0/actions"] = actions
        for side, qpos in enumerate(states):
            source[f"traj_0/obs/agent/piper-{side}/qpos"] = qpos
        source["traj_0/obs/sensor_data/zed2i/rgb"] = images
    motion = {"execution_mode": "sync", "control_freq": 20,
        "action_order": ["piper-0", "piper-1"], "action_sizes": [7, 7],
        "robot_uids": ["piper", "piper"]}
    if layout:
        motion["native_action_layout"] = _live_layout()
    metadata = {"env_info": {"env_id": "TwoRobotPickPillToRegions-v1",
        "env_kwargs": {"robot_uids": ["piper", "piper"], "control_mode": "pd_joint_pos",
                       "sensor_configs": {"shader_pack": "default"}}},
        "motion_planning": motion}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    return path, metadata, actions, states, images


def test_recorder_layout_uses_live_joints_and_native_gripper_mapping():
    layout = _live_layout()
    assert [item["key"] for item in layout] == ["piper-0", "piper-1"]
    assert [item["size"] for item in layout] == [7, 7]
    assert layout[0]["state_joint_names"][0] == "joint8"
    assert layout[1]["state_joint_names"][0] == "joint7"
    for item in layout:
        assert item["arm_joint_names"] == [f"joint{i}" for i in range(1, 7)]
        assert item["arm_coordinate_unit"] == "rad" and item["gripper_coordinate_unit"] == "m"
        assert (item["gripper_open_action"], item["gripper_closed_action"]) == (1., -1.)
    rotary = NS(single_action_space=NS(shape=(7,)),
        controller=NS(controllers={"arm": NS(joints=[_joint(f"joint{i}") for i in range(1, 7)])}),
        robot=NS(get_active_joints=lambda: [_joint(f"joint{i}") for i in range(12)]),
        gripper_coordinate_unit="rad", gripper_open_action=0., gripper_closed_action=.81)
    native = runner._native_action_layout(NS(agent=NS(agents_dict={"fixture": rotary})))[0]
    assert native["gripper_coordinate_unit"] == "rad"
    assert (native["gripper_open_action"], native["gripper_closed_action"]) == (0., .81)


def test_official_piper_conversion_preserves_native_order_frequency_and_alignment(tmp_path):
    path, metadata, actions, states, images = _source(tmp_path)
    before = (path.read_bytes(), path.with_suffix(".json").read_bytes())
    with h5py.File(path) as source:
        sample = converter.load_trajectory_data(source, "traj_0", drop_terminal_frame=True)
    assert sample["data_contract"]["control_freq"] == 20
    assert sample["data_contract"]["robot_uids"] == ["piper", "piper"]
    assert sample["data_contract"]["action_order"] == metadata["motion_planning"]["native_action_layout"]
    np.testing.assert_array_equal(sample["actions"], actions)
    np.testing.assert_array_equal(sample["qpos_left"], states[0][:-1])
    np.testing.assert_array_equal(sample["qpos_right"], states[1][:-1])
    np.testing.assert_array_equal(sample["rgb_data"]["zed2i"], images[:-1])
    features = converter.build_lerobot_features(sample)
    assert features["action.qpos_gripper_actions"]["shape"] == (14,)
    assert features["observation.qpos_gripper_states"]["shape"] == (16,)
    assert features["observation.qpos_gripper_states"]["names"][0] == "piper-0/joint8"
    assert features["observation.qpos_gripper_states"]["names"][8] == "piper-1/joint7"
    assert (path.read_bytes(), path.with_suffix(".json").read_bytes()) == before


def test_common_task_contract_has_precedence_over_motion_planning(tmp_path):
    path, metadata, _, _, _ = _source(tmp_path)
    common = deepcopy(metadata["motion_planning"]["native_action_layout"])
    common[0]["state_joint_names"] = list(reversed(common[0]["state_joint_names"]))
    metadata["embodiment_sync"] = {"action_order": common, "control_freq": 20}
    metadata["motion_planning"]["control_freq"] = 30
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with h5py.File(path) as source:
        contract = converter.native_data_contract(source, 14)
    assert contract["action_order"] == common
    assert contract["control_freq"] == 20


def test_legacy_piper_needs_recorded_joint_order_without_rewriting_its_sidecar(tmp_path):
    path, _, _, _, _ = _source(tmp_path, layout=False)
    before = (path.read_bytes(), path.with_suffix(".json").read_bytes())
    with h5py.File(path) as source, pytest.raises(ValueError, match="explicit native action layout"):
        converter.load_trajectory_data(source, "traj_0", drop_terminal_frame=True)
    assert (path.read_bytes(), path.with_suffix(".json").read_bytes()) == before


@pytest.mark.parametrize("record_frequency", [False, True])
def test_original_panda_legacy_fallback_remains_available(tmp_path, record_frequency):
    path = tmp_path/"panda.h5"
    with h5py.File(path, "w") as source:
        source["traj_0/actions"] = np.zeros((2, 16), dtype=np.float32)
    if record_frequency:
        path.with_suffix(".json").write_text(json.dumps({"motion_planning": {
            "control_freq": 20, "action_order": ["panda_wristcam-0", "panda_wristcam-1"]}}))
    with h5py.File(path) as source:
        contract = converter.native_data_contract(source, 16)
    assert [item["key"] for item in contract["action_order"]] == ["panda_wristcam-0", "panda_wristcam-1"]
    assert [item["size"] for item in contract["action_order"]] == [8, 8]
    assert contract["control_freq"] == (20 if record_frequency else None)


def test_declared_non_panda_does_not_borrow_a_sixteen_action_fallback(tmp_path):
    path = tmp_path/"other.h5"
    with h5py.File(path, "w") as source:
        source["traj_0/actions"] = np.zeros((2, 16), dtype=np.float32)
    path.with_suffix(".json").write_text(json.dumps({"motion_planning": {"robot_uids": ["other", "other"]}}))
    with h5py.File(path) as source, pytest.raises(ValueError, match="legacy fallback"):
        converter.native_data_contract(source, 16)


def test_piper_wrong_conversion_fps_is_rejected_before_creating_a_writer(tmp_path, monkeypatch):
    from examples.baselines.lerobot_dataset import lerobot_dataset

    path, _, _, _, _ = _source(tmp_path)
    monkeypatch.setattr(lerobot_dataset.LeRobotDataset, "create",
        lambda *args, **kwargs: pytest.fail("writer created before frequency validation"))
    passed, _, message = converter.process_h5_file(str(path), str(tmp_path/"out"),
                                                  fps=30, drop_terminal_frame=True)
    assert not passed and "control frequency" in message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
