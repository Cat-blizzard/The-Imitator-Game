"""Repair saved VHACD output containing rank-deficient zero-volume fragments.

Only algebraically non-volumetric fragments are omitted. The retained native
compiler still checks the complete visual surface against the cooked compound,
mass/inertia, actual 1.5 s settling, and the original 3 mm surface tolerance.
This is a candidate collision builder, not a more permissive acceptance rule.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import trimesh

from resimpair_grasp.common import binding, digest, read, resolve, write
from . import metric_native_v5 as retained

VERSION = 'metric_nonvolumetric_fragment_repair_v7'


def fragment_geometry(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all()):
        raise ValueError('finite_fragment_vertex_array_required')
    singular = np.linalg.svd(vertices - vertices.mean(axis=0), compute_uv=False)
    tolerance = max(1e-12, float(singular[0]) * 1e-12)
    rank = int(np.count_nonzero(singular > tolerance))
    proof = {'singular_values_m': singular.tolist(), 'rank_tolerance_m': tolerance,
        'rank': rank, 'rank_rule': 'max_1picometer_or_relative_1e_minus12',
        'omitted': rank < 3}
    if rank < 3:
        return None, {**proof, 'reason': 'solver_fragment_has_no_3D_volume'}
    part = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not part.is_watertight or not part.is_winding_consistent or part.volume <= 0:
        raise ValueError('full_rank_non_solid_fragment_remains_unrepaired')
    return part, {**proof, 'volume_m3': float(part.volume)}


def repair_saved(diagnostic_result_path, output):
    """Replay a bound saved decomposition; do not invoke the solver again."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    diagnostic = read(diagnostic_result_path)
    failure = resolve(diagnostic['original_failure'])
    original_job = read(failure.parent / 'job.json')
    parent_result = failure.parent / 'native_0/first_native/result.json'
    if not parent_result.exists():
        raise ValueError('saved_primary_native_result_required')
    parent = read(parent_result)
    descriptor = read(resolve(parent['asset']))
    if descriptor['source_id'] != diagnostic['source_id'] or descriptor['source_id'] != original_job['source_id']:
        raise ValueError('diagnostic_native_source_binding_mismatch')
    output.mkdir(parents=True)
    parts = []
    decisions = []
    for index, row in enumerate(diagnostic['parts']):
        path = resolve(row['raw'])
        with np.load(path, allow_pickle=False) as raw:
            part, decision = fragment_geometry(raw['vertices'], raw['faces'])
        decisions.append({'raw': row['raw'], **decision})
        if part is not None:
            path = output / f'collision_{index:03d}.obj'
            part.export(path, digits=12)
            parts.append(binding(path))
    if not parts:
        raise ValueError('no_volumetric_collision_shapes_remain')
    if not any(row['omitted'] for row in decisions):
        raise ValueError('no_rank_deficient_fragment_repair_trigger')
    proof = {'schema': VERSION, 'diagnostic_result': binding(diagnostic_result_path),
        'parent_native_result': binding(parent_result), 'parent_descriptor': parent['asset'],
        'original_job': binding(failure.parent / 'job.json'),
        'decisions': decisions, 'parts_retained': len(parts),
        'source_surface_and_settle_rules_changed': False,
        'solver_rerun': False, 'implementation': binding(__file__),
        'native_validator': binding(retained.__file__), 'API_calls': 0, 'GPU_jobs': 0}
    write(output / 'fragment_repair.json', proof)
    candidate = copy.deepcopy(descriptor)
    candidate.update(collisions=parts, fragment_repair=binding(output / 'fragment_repair.json'),
        parent_collision_descriptor=parent['asset'],
        collision_policy='rank_deficient_VHACD_fragments_omitted_pending_unchanged_native_qualification')
    candidate['geometry_variant_key'] = digest({'parent': parent['asset'], 'repair': binding(output / 'fragment_repair.json')})
    candidate['variant_id'] = candidate['geometry_variant_key']
    candidate['id'] = candidate['variant_id'][:24]
    candidate['grasp_cache_invalidated'] = True
    candidate['requires_new_grasp_qualification'] = True
    path = output / 'candidate_descriptor.json'
    write(path, candidate)
    result = retained.compile_variant(path, output / 'native_requalification')
    value = {**result, 'schema': VERSION, 'source_error_repaired': 'rank_deficient_solver_fragment',
        'repair': binding(output / 'fragment_repair.json'), 'old_failure_preserved': True,
        'native_surface_and_support_rules_changed': False,
        'optimizer_jobs': 0, 'API_calls': 0}
    write(output / 'result.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(repair_saved(args.diagnostic_result, args.output))
