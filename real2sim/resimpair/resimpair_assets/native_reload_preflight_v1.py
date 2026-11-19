"""Read final collider files through PhysX and the unchanged geometry evaluator.

This preflight does not move an actor, simulate a step, export a replacement
mesh, update a descriptor or change a cache. It diagnoses serialization/cooking
drift before a candidate is used for task generation; it is not a grasp test.
"""
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_grasp import evaluator


def check(entries, output):
    import sapien
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_native_reload_check')
    refs = [e['asset'] for e in entries]
    if len({digest(x) for x in refs}) != len(refs):
        raise ValueError('unique_descriptor_bindings_required')
    write(output / 'PROTOCOL.json', {'schema': 'native_reload_preflight_v1',
        'implementation': binding(__file__), 'retained_evaluator': binding(evaluator.__file__),
        'asset_descriptors': refs, 'maximum_native_actor_builds': len(refs),
        'simulation_steps': 0, 'new_collider_exports': 0, 'API_calls': 0, 'GPU_calls': 0,
        'scope': 'exact persisted mesh reload and original native geometry/mass requirements'})
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    rows = []; began = time.monotonic()
    for number, ref in enumerate(refs):
        d = read(resolve(ref)); folder = output / f'{number:04d}'; start = time.monotonic()
        entity = sapien.Entity(); body = sapien.physx.PhysxRigidDynamicComponent(); entity.add_component(body)
        added = False; row = {'source_id': d['source_id'], 'asset': ref, 'passed': False}
        try:
            for part in d['collisions']:
                shape = sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(resolve(part)), scale=[1., 1., 1.],
                    material=sapien.physx.PhysxMaterial(.6, .6, 0.))
                shape.density = float(np.float32(d['density_kg_m3'])); body.attach(shape)
            scene.add_entity(entity); added = True
            obj = {'id': 'asset', 'asset_descriptor': ref, 'kind': 'movable'}
            evidence = {'objects': [{'id': 'asset', 'descriptor': ref, 'mass': float(body.mass),
                'shapes': [{'vertices': np.asarray(s.vertices).tolist(), 'scale': np.asarray(s.scale).tolist(),
                    'pose': np.asarray(s.local_pose.to_transformation_matrix()).tolist(), 'density': float(s.density)}
                    for s in body.get_collision_shapes()]}]}
            write(folder / 'NATIVE_GEOMETRY.json', evidence)
            row['actual_native_geometry'] = binding(folder / 'NATIVE_GEOMETRY.json')
            row['original_geometry_evaluator'] = evaluator.geometry(folder / 'NATIVE_GEOMETRY.json', {'objects': [obj]})
            row['passed'] = True
        except Exception as error:
            row['error'] = type(error).__name__ + ':' + str(error)
        finally:
            if added:
                scene.remove_entity(entity)
        row['seconds'] = time.monotonic() - start
        write(folder / 'RESULT.json', row); rows.append(row)
    result = {'schema': 'native_reload_preflight_v1', 'protocol': binding(output / 'PROTOCOL.json'),
              'checked': len(rows), 'passed': sum(r['passed'] for r in rows), 'rows': rows,
              'seconds': time.monotonic() - began, 'simulation_steps': 0, 'API_calls': 0, 'GPU_calls': 0}
    write(output / 'RESULTS.json', result)
    return result
