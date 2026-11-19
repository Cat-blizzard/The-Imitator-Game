"""One geometry qualification rule for learned, optimized and hybrid poses.

Preserves every six-dimensional pose. This is an offline necessary condition,
not a dynamics label. Native collider qualification is recorded separately.
"""
import time
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_grasp.grasps import GRIPPER

VERSION = 'common_native_hand_6d_v1'


def tcp_to_hand_seed(poses, opening=.08):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError('Expected finite N x 4 x 4 native TCP poses')
    q = np.zeros((len(poses), 9))
    q[:, :3] = poses[:, :3, 3] - poses[:, :3, :3] @ np.array([0., 0., .1034])
    q[:, 3:7] = Rotation.from_matrix(poses[:, :3, :3]).as_quat(scalar_first=True)
    q[:, 7:] = opening / 2
    return q


def qualify(asset, raw_binding, output, geometry='visual', maximum=128):
    from resimpair_grasp.cartesian_clearance import VerticalClearance
    from resimpair_pickplace.closure_quality import ClosingSweep
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    output = Path(output)
    if (output / 'decisions.json').exists():
        report = read(output / 'decisions.json')
        if report['asset'] != asset or report['raw'] != raw_binding or report['geometry'] != geometry:
            raise ValueError('Qualification input changed')
        return report
    start = time.monotonic()
    descriptor = read(resolve(asset))
    if geometry == 'visual':
        mesh = trimesh.load(resolve(descriptor['visual']), force='scene', process=False).to_geometry()
    elif geometry == 'native':
        mesh = trimesh.util.concatenate([trimesh.load(resolve(b), force='mesh')
                                        for b in descriptor['collisions']])
    else:
        raise ValueError('Unknown geometry scope')
    ray, clearance, closure = RayMeshIntersector(mesh), VerticalClearance(mesh), ClosingSweep(mesh)
    raw = np.load(resolve(raw_binding), allow_pickle=False)
    poses = raw['poses']
    candidates, decisions, seen = [], [], set()
    for i, pose in enumerate(poses[:maximum]):
        row = {'rank': i, 'rejected': None}
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(pose[:3, :3]) < .999):
            row['rejected'] = 'invalid_pose'
        else:
            bucket = tuple(np.round(np.r_[pose[:3, 3] / .004,
                Rotation.from_matrix(pose[:3, :3]).as_rotvec() / .12]).astype(int))
            if bucket in seen:
                row['rejected'] = 'duplicate'
            seen.add(bucket)
        if row['rejected']:
            decisions.append(row); continue
        closing = pose[:3, 1]
        origins = np.array([pose[:3, 3] - .045 * closing, pose[:3, 3] + .045 * closing])
        locations, ids, faces = ray.intersects_location(origins, np.array([closing, -closing]), multiple_hits=False)
        if len(ids) != 2:
            row['rejected'] = 'no_two_closing_contacts'
            decisions.append(row); continue
        contacts = locations[np.argsort(ids)]
        width = float((contacts[1] - contacts[0]) @ closing)
        row['width_m'] = width
        if not .012 <= width <= .077:
            row['rejected'] = 'native_opening_range'
            decisions.append(row); continue
        ok, gap, reason = clearance.check(pose, float(mesh.bounds[0, 2]), .002)
        row['open_hand'] = {'passed': bool(ok), 'clearance_m': gap, 'reason': reason}
        if not ok:
            row['rejected'] = reason
            decisions.append(row); continue
        measured = closure.measure(pose, width)
        row['closing_sweep'] = measured
        if not measured['contact_pair_reachable']:
            row['rejected'] = 'native_first_contact_width_mismatch'
            decisions.append(row); continue
        candidates.append({'id': f'p{i:04d}', 'tcp_in_asset': pose.tolist(), 'width_m': width,
            'contacts_in_asset': contacts.tolist(), 'closing_sweep': measured,
            'proposal_rank': i, 'physical_status': 'unmeasured',
            'geometry_scope': geometry, 'raw_pose_unchanged': True})
        decisions.append(row)
    signature = {'descriptor': asset, 'gripper': GRIPPER, 'raw_proposals': raw_binding,
                 'qualifier': binding(__file__), 'rule_version': VERSION, 'geometry_scope': geometry}
    cache = {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature,
             'key': digest(signature), 'candidates': candidates, 'physical_qualification': 'unknown'}
    write(output / 'grasp_cache.json', cache)
    report = {'version': VERSION, 'asset': asset, 'raw': raw_binding, 'geometry': geometry,
              'proposed': len(poses), 'checked': len(decisions), 'retained': len(candidates),
              'rejections': dict(Counter(r['rejected'] for r in decisions if r['rejected'])),
              'rows': decisions, 'elapsed_seconds': time.monotonic() - start,
              'cache': binding(output / 'grasp_cache.json'), 'physical_success': 'unmeasured'}
    write(output / 'decisions.json', report)
    return report
