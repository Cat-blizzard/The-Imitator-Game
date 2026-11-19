"""Archive versions select retreat checks without changing native actions."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest

from mani_skill.examples.motionplanning.dual.sync import replay
from mani_skill.examples.motionplanning.dual.sync.runtime_policy import DEFAULT_POLICY
from mani_skill.examples.motionplanning.dual.sync.solve import phase_sequence


def archive(robot="panda_wristcam", backend=None, episode_backend=None, features=None):
    width = 8 if robot == "panda_wristcam" else 7
    motion = {"action_order": [f"{robot}-{i}" for i in range(2)],
              "action_sizes": [width, width], "robot_uids": [robot, robot]}
    sync = {"robot_uids": [robot, robot]}
    if backend is not None:
        motion["planner_backend"] = backend
    if episode_backend is not None:
        sync["backend"] = episode_backend
    if features is not None:
        sync["execution_features"] = features
    return {"env_info": {"env_id": replay.TASK_ID, "env_kwargs": {}},
            "motion_planning": motion}, {"dual_sync": sync}


@pytest.mark.parametrize("robot,backend,episode_backend,features,expected", [
    ("panda_wristcam", "mplib", None, None, False),
    ("panda_wristcam", "curobo", "curobo", None, True),
    ("panda_wristcam", None, "curobo", None, True),
    ("panda_wristcam", "mplib", "curobo", None, False),
    ("panda_wristcam", "mplib", None, {"release_separation": True}, True),
    ("panda_wristcam", "curobo", "curobo", {"release_separation": False}, False),
    ("piper", "mplib", None, None, False),
    ("piper", "curobo", "curobo", None, False),
    ("piper", "mplib", None, {"release_separation": True}, False),
    ("xarm6_robotiq", "curobo", None, {"release_separation": True}, False),
])
def test_archive_selects_only_declared_supported_release_checks(
        robot, backend, episode_backend, features, expected):
    metadata, episode = archive(robot, backend, episode_backend, features)
    assert replay.release_separation_enabled(metadata, episode) is expected


def test_legacy_curobo_pair_is_identified_by_native_action_order():
    metadata, episode = archive(backend="curobo")
    metadata["motion_planning"].pop("robot_uids")
    episode["dual_sync"].pop("robot_uids")
    assert replay.release_separation_enabled(metadata, episode)
    metadata["env_info"]["env_kwargs"]["robot_uids"] = ["piper", "piper"]
    assert not replay.release_separation_enabled(metadata, episode)


def test_release_feature_does_not_enable_panda_gate_for_other_tasks():
    metadata, episode = archive(features={"release_separation": True})
    metadata["env_info"]["env_id"] = "TwoRobotPickAppleBananaToBaskets-v1"
    assert not replay.release_separation_enabled(metadata, episode)


@pytest.mark.parametrize("invalid", [1, "false", None])
def test_release_feature_requires_explicit_boolean(invalid):
    metadata, episode = archive(features={"release_separation": invalid})
    with pytest.raises(ValueError, match="must be boolean"):
        replay.release_separation_enabled(metadata, episode)


@pytest.mark.parametrize("preparation", [None, {}, []])
def test_empty_explicit_preparation_never_invents_a_phase_length(tmp_path, preparation):
    metadata, episode = archive()
    sync = episode["dual_sync"]
    sync.update(schema="dual-panda-sync-v1", policy=DEFAULT_POLICY.to_dict(),
                preparation=preparation, stages=[
                    {"name": name, "status": "passed", "environment_steps": 1}
                    for name in phase_sequence(DEFAULT_POLICY)])
    episode.update(episode_id=0)
    metadata["episodes"] = [episode]
    path = tmp_path / "empty-preparation.h5"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="Explicit preparation metadata"):
        replay.load_episode(path)


@pytest.mark.parametrize("robot", ["panda_wristcam", "piper", "xarm6_robotiq"])
def test_legacy_replay_skips_separation_gate_and_keeps_native_action(monkeypatch, robot):
    metadata, episode = archive(robot, "mplib")
    episode.update(episode_seed=401, reset_kwargs={})
    order, sizes = replay.action_layout(metadata)
    actions = np.linspace(-.4, .81, sum(sizes), dtype=np.float32)[None, :]
    events = []
    agents = [NS(control_mode="pd_joint_pos", tcp=NS(pose=NS(p=np.zeros((1, 3)))))
              for _ in range(2)]
    raw = NS(num_envs=1, control_timestep=.05, sim_timestep=.01,
             robot_uids=(robot, robot), agent=NS(agents=agents),
             elapsed_steps=np.array([0]), set_state_dict=lambda state: None,
             evaluate=lambda: {"success": np.array([True])})
    env = NS(unwrapped=raw, reset=lambda **kwargs: None,
             close=lambda: events.append("closed"))
    task = NS(targets=(object(), object()), official_predicate_keys=("left", "right"),
              region_predicates=lambda: (lambda actor: {"inside": True},) * 2)

    class Checker:
        # Old routes need no release-separation API at all.
        def set_measured_attachment_prediction(self, enabled):
            pass

        def set_target_contacts_allowed(self, allowed):
            pass

    class Capture:
        def __init__(self, *args):
            self.last_step = None

        def install(self):
            pass

        def restore(self):
            events.append("restored")

        def summary(self):
            return {"max_held_position_slip_m": [0., 0.],
                    "max_held_rotation_slip_rad": [0., 0.]}

    class Phase:
        def __init__(self, capture, checker, name, **kwargs):
            self.arms = [NS(key=key) for key in order]
            self.row = {"measurements": {key: {"finger_positions_tcp_m": [[0., 0., 0.]] * 2}
                                         for key in order},
                        "objects": [{"is_grasping": False}] * 2, "tracking_error_rad": 0.}
            self.report = {"initial_observation": self.row,
                           "initial_measurements": self.row["measurements"],
                           "trace": [], "status": "running"}

        def preflight(self):
            return True

        def tick(self, command, phase, *, native_actions=False):
            assert native_actions
            np.testing.assert_array_equal(np.concatenate([command[key] for key in order]), actions[0])
            events.append("native_action")
            raw.elapsed_steps += 1
            for agent in agents:
                agent.tcp.pose.p[0, 2] += .081
            return self.row

        def opened(self, *args):
            return True

    policy = replace(DEFAULT_POLICY, require_final_stability=False)
    monkeypatch.setattr(replay, "load_episode", lambda *args: (
        metadata, episode, policy,
        [("pregrasp", 0), ("retreat", 1), ("observe_task_outcome", 0)], actions, {}))
    monkeypatch.setattr(replay.gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(replay, "task_components", lambda task_id: (
        lambda raw: task, NS(from_solvers=lambda *args, **kwargs: Checker())))
    monkeypatch.setattr(replay, "pill_planners", lambda *args: (object(), object()))
    monkeypatch.setattr(replay, "ObservedEnvironment", Capture)
    monkeypatch.setattr(replay, "_Phase", Phase)
    result = replay.replay("unused.h5")
    assert result["execution"]["status"] == "passed", result
    assert result["executed_actions"] == 1
    assert events == ["native_action", "restored", "closed"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
