"""pp2 execution worker: the retained controller with three pp2 adjustments.

1. Settle guard that separates grip slip from arm vibration. A loaded arm can keep a sub-2 mm limit
   cycle that never decays while the object does not move in the grip at all (measured on the 0.3 kg
   soup can). The relative-pose bounds stay unchanged; only the world-pose bound of an otherwise
   stable grip widens.
2. Robot start: the scene's `pp2_robot_init_qpos` (scenes.robot_start) replaces the retained fixed
   reset pose, so trajectories do not all leave from one configuration.
3. Final settle until rest: after the last release the retained 40-step hold is extended (10 steps at
   a time, bounded) while any object still moves, so a rolling ball is not recorded mid-roll.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest import mock

import numpy as np

WORLD_BOUND_M = .006
FINAL_SETTLE_STEPS = 40          # the retained solve's last hold
EXTRA_SETTLE_STEPS = 120         # at most this much longer (6 s at 20 Hz)
AT_REST_LINEAR_M_S, AT_REST_ANGULAR_RAD_S = .002, .05


def worker(root, case):
    from resimpair_runtime import configure
    configure()
    from mani_skill.utils.registration import REGISTERED_ENVS
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation
    from resimpair_pickplace import stability_motion
    from resimpair_tasks import production_integration_v4 as retained
    original = stability_motion.window_stability

    def grip_first(samples, arm, index):
        result = original(samples, arm, index)
        if (not result['stable'] and result.get('all_frames_grasped')
                and result['relative_translation_range_m'] <= result['translation_bound_m']
                and result['relative_rotation_range_deg'] <= result['rotation_bound_deg']
                and result['world_rotation_range_deg'] <= result['rotation_bound_deg']
                and result['world_translation_range_m'] <= WORLD_BOUND_M):
            return {**result, 'stable': True, 'stable_by': 'grip_stable_arm_vibration_allowed',
                    'world_translation_bound_m': WORLD_BOUND_M}
        return result

    register = simulation.register

    def register_with_start(scene, performance=None, *args, **kwargs):
        result = register(scene, performance, *args, **kwargs)
        cls = REGISTERED_ENVS['ReSimPairSharedTable-v4'].cls
        initialize = cls._initialize_episode

        def _initialize_episode(self, env_idx, options):
            initialize(self, env_idx, options)
            for agent, qpos in zip(self.agent.agents, self.scene_spec.get('pp2_robot_init_qpos') or ()):
                agent.reset(np.asarray(qpos, dtype=float))
        cls._initialize_episode = _initialize_episode
        return result

    hold = simulation.Recorder.hold

    def hold_until_rest(self, steps, grippers=(1, 1)):
        hold(self, steps, grippers)
        if self.stage != 'final_settle' or steps != FINAL_SETTLE_STEPS:
            return
        for _ in range(EXTRA_SETTLE_STEPS // 10):
            moving = self.measure()['object_states']  # per object: position, quaternion, linear, angular velocity
            if (np.linalg.norm(moving[:, 7:10], axis=1).max(initial=0.) < AT_REST_LINEAR_M_S
                    and np.linalg.norm(moving[:, 10:13], axis=1).max(initial=0.) < AT_REST_ANGULAR_RAD_S):
                return
            hold(self, 10, grippers)

    with mock.patch.object(stability_motion, 'window_stability', grip_first), \
            mock.patch.object(simulation, 'register', register_with_start), \
            mock.patch.object(simulation.Recorder, 'hold', hold_until_rest):
        return retained.worker(root, case)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('command', choices=['worker'])
    p.add_argument('--root', type=Path, required=True); p.add_argument('--case', required=True)
    a = p.parse_args(); print(worker(a.root, a.case))
