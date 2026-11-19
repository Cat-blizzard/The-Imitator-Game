"""Preparation waits for measured readiness without creating a physics scene."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mani_skill.examples.motionplanning.dual.sync import dual_arm_execution as execution
from mani_skill.examples.motionplanning.dual.sync.runtime_policy import DEFAULT_POLICY


class FakePreparation:
    """Script independent observations; commands never teleport the fixtures."""

    def __init__(self, fault: str | None = None, *, ongoing: bool = False):
        self.unwrapped = self
        self.control_timestep = .05
        self.elapsed_steps = np.array([0])
        self.fault, self.ongoing = fault, ongoing
        self.initial_q = (np.arange(7) * .1, -np.arange(7) * .1)
        self.q = [np.r_[value, .04, .04] for value in self.initial_q]
        self.v = [np.zeros(9), np.zeros(9)]
        self.matrices = [np.eye(4), np.eye(4)]
        self.actions: list[dict[str, np.ndarray]] = []
        self.validated: list[dict[str, Any]] = []
        self.targets = tuple(NS(pose=NS(to_transformation_matrix=
            lambda index=index: self.matrices[index][None].copy())) for index in (0, 1))
        limits = np.array([[[-3., 3.]] * 7 + [[0., .04]] * 2])
        self.arms = tuple(NS(key=key, indices=np.arange(7), all_names=tuple(f"q{i}" for i in range(9)),
            finger_indices=np.array([7, 8]),
            arm_qpos=lambda index=index: self.q[index][:7].copy(),
            full_qpos=lambda index=index: self.q[index].copy(),
            agent=NS(robot=NS(get_qvel=lambda index=index: self.v[index][None].copy(),
                             get_qlimits=lambda: limits.copy())))
            for index, key in enumerate(('left', 'right')))

    def step(self, action: dict[str, Any]) -> tuple[Any, ...]:
        # Both robots receive the same control tick and retain their original
        # seven-joint targets even if measured state is disturbed later.
        assert list(action) == ['left', 'right']
        for arm, initial in zip(self.arms, self.initial_q):
            np.testing.assert_array_equal(action[arm.key], np.r_[initial, 1.])
        self.actions.append({key: np.asarray(value).copy() for key, value in action.items()})
        self.elapsed_steps += 1
        tick = len(self.actions)
        self.q = [np.r_[value, .04, .04] for value in self.initial_q]
        self.v = [np.zeros(9), np.zeros(9)]
        self.matrices = [np.eye(4), np.eye(4)]
        disturbed = self.ongoing or tick == 3
        offset = tick if self.ongoing else int(tick >= 3)
        if self.fault == 'target_translation':
            self.matrices[0][0, 3] = .004 * offset
        elif self.fault == 'target_rotation':
            angle = .02 * offset
            self.matrices[1][:2, :2] = [[np.cos(angle), -np.sin(angle)],
                                       [np.sin(angle), np.cos(angle)]]
        elif self.fault == 'right_joint_position':
            self.q[1][0] += .004 * offset
        elif self.fault == 'right_joint_velocity' and disturbed:
            self.v[1][0] = .1
        elif self.fault == 'right_finger' and disturbed:
            self.q[1][-1] = .03
        return None, 0., np.array([False]), np.array([False]), {}

    def bind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def canonical(env: Any) -> Any:
            assert env is self
            return self.arms

        def validate(env: Any, arms: Any, action: Any) -> None:
            assert env is self and arms is self.arms
            assert set(action) == {'left', 'right'}
            assert all(np.asarray(value).shape == (8,) and np.isfinite(value).all()
                       for value in action.values())
            self.validated.append(action)

        monkeypatch.setattr(execution, '_canonical', canonical)
        monkeypatch.setattr(execution, '_actions_valid', validate)


def test_static_preparation_uses_three_real_paired_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakePreparation()
    env.bind(monkeypatch)
    report = execution.prepare_dual_open(env, env.targets)
    assert report['environment_steps'] == len(env.actions) == 3
    assert report['stable_steps_observed'] == 3 and report['early_completion'] is True
    assert len(env.validated) == 1 and env.elapsed_steps.tolist() == [3]


@pytest.mark.parametrize('fault', ['target_translation', 'target_rotation',
    'right_joint_position', 'right_joint_velocity', 'right_finger'])
def test_measured_disturbance_resets_readiness_counter(
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    env = FakePreparation(fault)
    env.bind(monkeypatch)
    report = execution.prepare_dual_open(env, env.targets)
    # Two ready ticks precede the disturbance; only ticks 4, 5 and 6 count
    # afterwards. This also catches one-sided or raw-velocity-only checks.
    assert report['environment_steps'] == len(env.actions) == 6
    assert report['stable_steps_observed'] == 3 and report['early_completion'] is True


def test_unready_preparation_keeps_original_ten_step_fallback(
        monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakePreparation('target_translation', ongoing=True)
    env.bind(monkeypatch)
    report = execution.prepare_dual_open(env, env.targets)
    assert report['environment_steps'] == len(env.actions) == 10
    assert report['stable_steps_observed'] == 0 and report['early_completion'] is False


def test_preparation_obeys_resolved_reach_window(monkeypatch: pytest.MonkeyPatch) -> None:
    env = FakePreparation()
    env.bind(monkeypatch)
    report = execution.prepare_dual_open(env, env.targets,
        policy=replace(DEFAULT_POLICY, reach_stable_steps=4))
    assert report['environment_steps'] == len(env.actions) == 4
    assert report['stable_steps_observed'] == report['stable_steps_required'] == 4


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
