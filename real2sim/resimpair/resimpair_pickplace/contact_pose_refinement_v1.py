"""Bounded native-finger refinement of cached optimizer proposals, CPU only.

The original proposals remain the prefix of the derived proposal set. Local
translations align a TCP cross-section with an actual opposing finger contact
region. This is a candidate generator, never a change to the common qualifier
or a dynamic success label. No runtime optimizer or paid API is imported.
"""
import argparse
from pathlib import Path
import time

import numpy as np
import trimesh

from resimpair_grasp.common import ROOT, binding, read, resolve, write
from resimpair_tabletop import geometry as _geometry  # existing CPU dependencies
from resimpair_pickplace.closure_quality import ClosingSweep


VERSION = 'native_finger_local_translation_v1'
LOCAL_X = (0., -.006, .006)
LOCAL_Z = (0., -.004, .004, -.008, .008)


def centered_pose(pose, left_width, right_width):
    """Translate toward contact midpoint; preserve TCP rotation and frame."""
    pose = np.asarray(pose, float)
    widths = np.asarray([left_width, right_width], float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.isfinite(widths).all() or np.any(widths < 0)):
        raise ValueError('Finite pose and nonnegative opposing-contact widths required')
    shift = float((widths[0] - widths[1]) / 4.)
    if abs(shift) > .006:
        raise ValueError('Contact-centering translation exceeds the 6mm proposal budget')
    result = pose.copy()
    result[:3, 3] += pose[:3, 1] * shift
    return result, shift


def generate(asset, raw, output, maximum_seeds=24):
    from resimpair_benchmark.proposals import qualify
    if not 1 <= maximum_seeds <= 32:
        raise ValueError('Finite seed limit 1..32 required')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new candidate output; never overwrite old poses')
    start = time.monotonic()
    descriptor = read(resolve(asset))
    if descriptor['qualification']['geometry'] != 'pass':
        raise ValueError('Native collision geometry must qualify before refinement')
    with np.load(resolve(raw), allow_pickle=False) as archive:
        originals = np.asarray(archive['poses'], dtype=float).copy()
        optimizer_success = int(np.count_nonzero(archive['success'])) if 'success' in archive else None
    if (originals.ndim != 3 or originals.shape[1:] != (4, 4)
            or not 1 <= len(originals) <= 256 or not np.isfinite(originals).all()):
        raise ValueError('Expected 1..256 finite existing native TCP proposals')
    frozen_code = [binding(__file__), binding(ROOT/'resimpair_benchmark/proposals.py'),
                   binding(ROOT/'resimpair_pickplace/closure_quality.py'),
                   binding(ROOT/'resimpair_grasp/cartesian_clearance.py')]
    write(output/'protocol.json', {'schema': VERSION, 'asset': asset, 'raw': raw,
        'maximum_seeds': maximum_seeds, 'local_x_offsets_m': LOCAL_X,
        'local_z_offsets_m': LOCAL_Z, 'maximum_center_shift_m': .006,
        'originals_are_unchanged_prefix': True, 'code': frozen_code,
        'native_qualifier_unchanged': True, 'physical_status': 'unknown',
        'gpu_jobs': 0, 'api_calls': 0})
    baseline = qualify(asset, raw, output/'baseline_full', geometry='native', maximum=len(originals))
    mesh = trimesh.util.concatenate([trimesh.load(resolve(b), force='mesh')
                                    for b in descriptor['collisions']])
    sweep = ClosingSweep(mesh)
    # Deduplicate starting poses only for proposal construction. The retained
    # qualifier continues to apply its own unchanged deduplication to all poses.
    seeds, seen, seed_measurements = [], set(), []
    for rank, pose in enumerate(originals):
        bucket = tuple(np.round(pose[:3, :].reshape(-1) / .004).astype(int))
        if bucket in seen: continue
        seen.add(bucket)
        if len(seeds) >= maximum_seeds: break
        seeds.append((rank, pose))
    poses = [p.copy() for p in originals]
    origins = [{'parent_rank': i, 'original': True, 'delta_local_m': [0., 0., 0.]}
               for i in range(len(originals))]
    for rank, base in seeds:
        for dx in LOCAL_X:
            for dz in LOCAL_Z:
                pose = base.copy()
                pose[:3, 3] += base[:3, :3] @ [dx, 0., dz]
                # expected_width affects only the returned comparison label;
                # the two first-contact widths are measured independently.
                measured = sweep.measure(pose, .04)
                widths = [f['first_contact_width_m'] for f in measured['fingers']]
                row = {'parent_rank': rank, 'dx_m': dx, 'dz_m': dz,
                       'finger_first_contact_widths_m': widths}
                seed_measurements.append(row)
                if any(v is None for v in widths):
                    row['proposal_rejected'] = 'missing_native_finger_contact'; continue
                if not .012 <= sum(widths)/2 <= .077:
                    row['proposal_rejected'] = 'mean_contact_width_outside_existing_range'; continue
                try:
                    centered, shift = centered_pose(pose, *widths)
                except ValueError:
                    row['proposal_rejected'] = 'centering_budget'; continue
                # Both unchanged lateral centre and measured midpoint are
                # proposals. Only the retained common qualifier can admit one.
                for moved, dy in [(pose, 0.), (centered, shift)]:
                    if dx == dz == dy == 0: continue
                    poses.append(moved)
                    origins.append({'parent_rank': rank, 'original': False,
                                    'delta_local_m': [dx, dy, dz],
                                    'optimizer_success_inherited': False})
                row['center_shift_m'] = shift
    derived = output/'derived_proposals.npz'
    np.savez_compressed(derived, poses=np.asarray(poses),
                        parent_rank=np.asarray([r['parent_rank'] for r in origins]))
    write(output/'derivation.json', {'original': raw, 'asset': asset, 'origins': origins,
        'seed_measurements': seed_measurements, 'original_optimizer_success_count': optimizer_success,
        'derived_optimizer_success': 'not_evaluated_do_not_inherit',
        'new_pose_count': len(poses)-len(originals)})
    result = qualify(asset, binding(derived), output/'qualified', geometry='native', maximum=len(poses))
    if result['rows'][:len(originals)] != baseline['rows']:
        raise ValueError('Original-prefix qualification changed')
    cache = read(resolve(result['cache']))
    cache['candidate_generation'] = {'version': VERSION, 'parent_raw': raw,
        'derivation': binding(output/'derivation.json'), 'not_direct_cudex_output': True}
    candidate_path = output/'candidate_cache.json'; write(candidate_path, cache)
    summary = {'schema': VERSION, 'source_id': descriptor['source_id'],
        'baseline_retained': baseline['retained'], 'refined_retained': result['retained'],
        'new_qualified_candidates': result['retained']-baseline['retained'],
        'original_prefix_exactly_reproduced': True, 'seeds_checked': len(seeds),
        'new_proposals': len(poses)-len(originals), 'elapsed_seconds': time.monotonic()-start,
        'asset': asset, 'cache': binding(candidate_path),
        'baseline': binding(output/'baseline_full/decisions.json'),
        'qualification': binding(output/'qualified/decisions.json'),
        'physical_status': 'unknown', 'gpu_jobs': 0, 'api_calls': 0}
    for b in frozen_code+[asset, raw]: resolve(b)
    write(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--maximum-seeds', type=int, default=24)
    a = p.parse_args()
    print(generate(binding(a.asset), binding(a.raw), a.output, a.maximum_seeds))
