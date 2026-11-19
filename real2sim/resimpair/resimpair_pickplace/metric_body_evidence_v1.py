"""Read actual PhysX mass/COM/inertia alongside retained collision evidence."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from resimpair_grasp.common import binding, read, resolve, write


def record(rec, directory):
    rows = []
    for item, actor in zip(rec.scene_spec['objects'], rec.objects):
        if item['kind'] != 'movable':
            continue
        body = actor._bodies[0]
        pose = body.cmass_local_pose
        R = np.asarray(pose.to_transformation_matrix())[:3, :3]
        descriptor = read(resolve(item['asset_descriptor']))
        values = {'mass_kg': float(body.mass), 'center_of_mass_m': pose.p.tolist(),
                  'principal_inertia_kg_m2': body.inertia.tolist(),
                  'inertia_kg_m2': (R @ np.diag(body.inertia) @ R.T).tolist(),
                  'cmass_pose_pq': list(map(float, pose.p)) + list(map(float, pose.q))}
        if (not all(np.isfinite(np.asarray(v)).all() for v in values.values())
                or values['mass_kg'] <= 0 or min(values['principal_inertia_kg_m2']) <= 0):
            raise ValueError('nonfinite_or_nonpositive_live_mass_properties')
        rows.append({'object': item['id'], 'asset': item['asset_descriptor'],
                     'offline_native_properties': descriptor.get('native_properties'),
                     'native_readback': values})
    write(Path(directory) / 'runtime_mass_properties.json', {
        'schema': 'actual_physx_mass_properties_v1', 'implementation': binding(__file__),
        'rows': rows, 'actor_updates': 0, 'extra_physics_steps': 0,
        'evidence_scope': 'live dynamic body after actual runtime reset'})


@contextmanager
def install():
    from resimpair_grasp import runner
    original = runner.native_geometry
    def capture(rec, directory):
        result = original(rec, directory)
        record(rec, directory)
        return result
    runner.native_geometry = capture
    try:
        yield
    finally:
        runner.native_geometry = original
