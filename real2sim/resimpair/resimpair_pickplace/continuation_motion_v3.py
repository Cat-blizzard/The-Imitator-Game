"""Opt-in native controller: bounded continuation lift, retained other skills.

Only an explicitly registered candidate worker may install this process-local
adapter. It reuses the complete existing grasp/carry/place/release loop and
unchanged acceptance. No per-asset translations, contact changes or retries.
"""
from contextlib import contextmanager
import argparse
from pathlib import Path

import numpy as np

from .transport_fallback_v2 import TransportFallbackMotion
from .shape_preserving_toppra_v2 import cartesian_path
from . import transport_fallback_v2
from resimpair_grasp.common import ROOT, binding, read, write
from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array

VERSION = 'bounded_continuation_motion_v3'


class ContinuationMotion(TransportFallbackMotion):
    def plan(self, pose, stage, current=None):
        if stage != 'lift':
            return super().plan(pose, stage, current)
        current = array(self.agent.robot.qpos).reshape(9).copy() if current is None else current.copy()
        if self.use_attach:
            self.refresh_attachment(stage)
        goal = np.asarray(pose.to_transformation_matrix())
        directory = self.directory / f'{self.counter:03d}_lift_continuation_v3'
        self.counter += 1
        directory.mkdir()
        write(directory / 'request.json', {'stage': stage, 'current': current.tolist(), 'goal': goal.tolist(),
            'frame': len(self.rec.rows), 'attached': self.use_attach, 'policy': VERSION})
        try:
            with self.rec.performance.section('motion_plan_bounded_continuation'):
                path = cartesian_path(self.planner, self.root, current, goal, self.rec.control_timestep)
            with self.rec.performance.section('dense_physical_collision_guard'):
                report, dense, poses = self.check(path, current, goal, stage)
            np.savez_compressed(directory / 'path.npz', positions=path['position'], current=current,
                goal=goal, velocity=path['velocity'], acceleration=path['acceleration'])
            write(directory / 'continuation_trace.json', path['continuation_trace'])
            if dense is not None:
                np.savez_compressed(directory / 'dense.npz', joints=dense, tcp=poses)
            write(directory / 'decision.json', {'accepted': report['passed'], 'report': report,
                'policy': VERSION, 'retiming_policy': path['retiming_policy'], 'duration_seconds': path['duration']})
            if report['passed']:
                path['guard_evidence'] = str(directory / 'decision.json')
                return path
        except Exception as exc:
            if not (directory / 'decision.json').exists():
                write(directory / 'decision.json', {'accepted': False, 'error': repr(exc), 'policy': VERSION})
        # These are additional planning proposals, never another physical grasp.
        # Preserve all rejected candidate evidence before the retained branches.
        return super().plan(pose, stage, current)


@contextmanager
def install_for_worker(root):
    """Explicitly bind the reused controller's only injected dependency."""
    from .runner import verify
    root = Path(root).resolve()
    spec = verify(root)
    if spec.get('candidate_motion_override') != VERSION or spec.get('motion_policy') != 'transport_fallback_v2':
        raise ValueError('explicit_candidate_campaign_required')
    for name in ('continuation_motion_v3.py', 'continuation_ik_v2.py', 'shape_preserving_toppra_v2.py'):
        key = 'resimpair_pickplace/' + name
        if spec['code'].get(key) != binding(ROOT / key):
            raise ValueError('candidate_source_binding_missing')
    if transport_fallback_v2.TransportFallbackMotion is not TransportFallbackMotion:
        raise ValueError('existing_process_local_controller_override')
    transport_fallback_v2.TransportFallbackMotion = ContinuationMotion
    try:
        yield
    finally:
        transport_fallback_v2.TransportFallbackMotion = TransportFallbackMotion


def worker(args):
    from .stability_runner import worker as retained_worker
    with install_for_worker(args.root):
        return retained_worker(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    args = parser.parse_args()
    print(worker(args))
