"""Native controller meanings survive the original-task bridge."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from imitator.embodiments.native import resolve_robot, robot_pair
from imitator.envs.original_task_profiles import get_task_profile
from imitator.envs.original_task_bridge import OriginalTaskAdapter, configure_task, legacy_to_native_tcp
from imitator.planning._tig_sync.trajectory_execution import native_arm_action
from imitator.planning._tig_sync.measurements import gripper_is_open


def test_aliases_resolve_to_one_existing_20_agent() -> None:
    assert robot_pair("dual_piper") == robot_pair("imsim_piper") == ("imsim_piper",) * 2
    assert resolve_robot("panda_wristcam") == "panda"
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_robot("xarm6_allegro")


def test_default_original_panda_has_no_native_profile() -> None:
    pair, profile = configure_task("TwoRobotPickPillToRegions-v1", ("panda_wristcam",) * 2, None)
    assert pair == ("panda_wristcam",) * 2 and profile is None
    assert len(get_task_profile("TwoRobotPickPillToRegions-v1", "imsim_arx_x5").mount_poses) == 2
    with pytest.raises(ValueError, match="no profile"):
        get_task_profile("TwoRobotPickAppleBananaToBaskets-v1", "ur5_wsg")


def test_original_task_adapter_keeps_lifecycle_boundary() -> None:
    adapter = OriginalTaskAdapter.from_args(
        "TwoRobotPickPillToRegions-v1", ("imsim_piper",) * 2, "piper")
    assert adapter.robot_uids == ("imsim_piper",) * 2
    assert adapter.profile is get_task_profile("TwoRobotPickPillToRegions-v1", "piper")
    assert len(adapter.load_poses()) == 2


@pytest.mark.parametrize("opened,closed", [(-1., 1.), (1., -1.), (0., .81)])
def test_native_scalar_gripper_uses_record_endpoints(opened: float, closed: float) -> None:
    agent = NS(record=NS(gripper_cmd_open=[opened], gripper_cmd_closed=[closed]))
    arm = NS(names=("joint1", "joint2"), agent=agent)
    assert native_arm_action(arm, [.2, -.3, 1.])[-1] == pytest.approx(opened)
    assert native_arm_action(arm, [.2, -.3, -1.])[-1] == pytest.approx(closed)
    assert native_arm_action(arm, [.2, -.3, 0.])[-1] == pytest.approx((opened+closed)/2)


def test_opposed_signed_ur5_fingers_are_open() -> None:
    q = np.array([0., -.055, .055])
    record = NS(name="ur5_wsg", home_qpos=lambda: {"left": -.055, "right": .055})
    agent = NS(record=record, robot=NS(get_qlimits=lambda: np.array([[[-3., 3.], [-.055, -.0027], [.0027, .055]]])))
    arm = NS(agent=agent, finger_indices=np.array([1, 2]), all_names=("joint1", "left", "right"), full_qpos=lambda: q)
    assert gripper_is_open(arm)
    q[1:] = [-.01, .01]
    assert not gripper_is_open(arm)


def test_unchanged_piper_tcp_does_not_shift_target() -> None:
    assert np.array_equal(legacy_to_native_tcp("piper"), np.eye(4))


def test_model_identity_ignores_generated_xml_comments(tmp_path) -> None:
    from imitator.embodiments.native import _file_hash

    first = tmp_path / "first.urdf"
    second = tmp_path / "second.urdf"
    first.write_text("<!-- /machine/a/source.urdf -->\n<robot>\n  <link name='base'/>\n</robot>\n")
    second.write_text("<!-- /machine/b/source.urdf -->\n<robot><link name='base'/></robot>\n")
    assert _file_hash(first) == _file_hash(second)


def test_public_make_dispatches_original_task_without_a_fabricated_contract(monkeypatch) -> None:
    from imitator import api
    from imitator.embodiments import native
    calls = []
    sentinel = object()
    monkeypatch.setattr(api, "choose_render", lambda *args: "sapien_gpu")
    monkeypatch.setattr(api.compat, "patch_render_backend", lambda: None)
    monkeypatch.setattr(native, "make_native_env", lambda *args, **kw: (calls.append((args, kw)) or sentinel))
    monkeypatch.setattr(api, "get_task", lambda *args: (_ for _ in ()).throw(AssertionError("legacy route requested TaskSpec")))
    assert api.make("TwoRobotPickPillToRegions-v1", "dual_piper", obs="rgb") is sentinel
    assert calls[0][0] == ("TwoRobotPickPillToRegions-v1", "dual_piper")
    assert calls[0][1]["obs_mode"] == "rgb" and calls[0][1]["render_backend"] == "cuda"
    with pytest.raises(ValueError, match="sim='cpu'"):
        api.make("TwoRobotPickPillToRegions-v1", "piper", sim="gpu")


def test_native_fingers_use_the_20_record_in_contact_labels() -> None:
    from imitator.planning._tig_sync.physical_observation import contact_identities
    entities = [object(), object()]
    links = [NS(name=name, _objs=[NS(entity=entity)]) for name, entity in zip(("link7", "link8"), entities)]
    agent = NS(record=NS(pad_links=("link7", "link8")), robot=NS(get_links=lambda: links))
    raw = NS(scene=NS(actors={}))
    labels, _ = contact_identities(raw, [agent], [], [])
    assert all(labels[entity]["is_grasp_finger"] for entity in entities)


def test_empty_world_root_keeps_only_fixed_base_support_links(monkeypatch) -> None:
    from imitator.planning._tig_sync.physical_observation import mount_support_links
    from imitator.embodiments import dual
    record = NS(urdf_path="fixture.urdf")
    agent = NS(record=record, robot=NS(root=NS(get_name=lambda: "world")))
    monkeypatch.setattr(dual, "base_links_on_table", lambda rec, base_z: ["base_link"])
    assert mount_support_links(agent) == {"world", "base_link"}
    assert "shoulder_link" not in mount_support_links(agent)
