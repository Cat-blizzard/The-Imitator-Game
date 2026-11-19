"""Metric GSO/YCB container intake with a curved-floor cavity rule (pp2 container v1).

Reuses raw_metric_containers_v1.compile_one unchanged (source metric scale,
stable upright pose, native static mesh, texture preservation). Only the cavity
detector is swapped for a version that accepts curved bowl floors: the floor
disk may deviate up to 12 mm (was 3 mm) and upward normals down to 0.80 (was
0.90). The retained support check reads the tolerance from the cavity record.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re
import sqlite3
import time
from unittest import mock

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, write

VERSION = 'pp2_metric_container_intake_v3'
# v2: walls are measured from 15 mm above the lowest point, so hemispherical bowls qualify.
# v3: walls only over the lower 60% of the depth (flared or carved rims), 60/64 rays may count as enclosed
#     (small gaps in modelled meshes), and shallow plates (>= 8 mm) are measured from 5 mm.
FLOOR_TOLERANCE_M, FLOOR_NORMAL_MIN, MIN_RADIUS_M, WALL_START_M = .020, .75, .045, .015
MIN_DEPTH_M, WALL_FRACTION, MIN_RAY_HITS = .008, .60, 60
NAME = re.compile(r'bowl|plate|tray|basket|pan\b|_pan|pot\b|_pot|bin\b|_bin|box|dish|crate|bucket|tub\b|caddy|'
                  r'organizer|planter|saucer|platter|colander|strainer|skillet|container|ramekin', re.I)


def relaxed_cavity(mesh):
    """Same ray construction as resimpair_pickplace.containers.cavity, with curved-floor tolerances."""
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    ray = RayMeshIntersector(mesh)
    lower, upper = mesh.bounds
    center = (lower + upper) / 2
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    directions = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(64)])
    best = None
    for dx, dy in [(0, 0), (-.015, 0), (.015, 0), (0, -.015), (0, .015)]:
        xy = center[:2] + [dx, dy]
        hit, _, face = ray.intersects_location([[*xy, upper[2] + .02]], [[0, 0, -1]], multiple_hits=False)
        if not len(hit):
            continue
        floor = float(hit[0, 2]); depth = float(upper[2] - floor)
        if not MIN_DEPTH_M <= depth <= .16 or abs(mesh.face_normals[face[0], 2]) < FLOOR_NORMAL_MIN:
            continue
        radii = []
        start = min(WALL_START_M, .5 * depth)
        top = max(start + .002, min(floor + .06, floor + WALL_FRACTION * depth, upper[2] - .003) - floor)
        for z in floor + np.linspace(start, top, 5):
            loc, ids, _ = ray.intersects_location(np.tile([*xy, z], (64, 1)), directions, multiple_hits=False)
            if len(set(ids)) < MIN_RAY_HITS:
                break
            radii.append(float(np.min(np.linalg.norm(loc[:, :2] - xy, axis=1))))
        if len(radii) != 5:
            continue
        radius = min(radii) - .006
        if radius < MIN_RADIUS_M:
            continue
        for _ in range(10):
            pts = np.vstack([xy, xy + directions[:, :2] * radius, xy + directions[:, :2] * (radius / 2)])
            starts = np.column_stack([pts, np.full(len(pts), upper[2] + .02)])
            loc, ids, faces = ray.intersects_location(starts, np.tile([0, 0, -1], (len(pts), 1)), multiple_hits=False)
            if (len(set(ids)) == len(pts) and np.max(np.abs(loc[:, 2] - floor)) <= FLOOR_TOLERANCE_M
                    and mesh.face_normals[faces, 2].min() >= FLOOR_NORMAL_MIN - .05):
                break
            radius *= .88
        else:
            continue
        if radius < .04:
            continue
        value = {'center_xy_m': xy.tolist(), 'floor_z_m': floor, 'flat_floor_radius_m': radius,
                 'minimum_wall_radius_m': min(radii), 'rim_z_m': float(upper[2]), 'depth_m': depth,
                 'rays_per_level': 64, 'wall_levels': 5, 'floor_tolerance_m': FLOOR_TOLERANCE_M,
                 'qualification': 'sampled_mesh_geometry_only; pp2 curved-floor rule'}
        if best is None or radius > best['flat_floor_radius_m']:
            best = value
    if best is None:
        raise ValueError('no_supported_open_cavity')
    return best


def compile_job(job_path):
    from resimpair_assets import raw_metric_containers_v1 as base
    from resimpair_pickplace import containers
    started = time.monotonic()
    try:
        with mock.patch.object(containers, 'cavity', relaxed_cavity):
            result = base.compile_one(job_path)
    except Exception as error:
        result = {'status': 'failed', 'reason': f'{type(error).__name__}:{error}'[:200]}
    result['seconds'] = time.monotonic() - started
    write(Path(job_path).parent / 'result.json', result)
    return result


def abo_dependencies(row):
    """ABO meshes are self-contained GLB in meters (glTF convention, checked against product metadata)."""
    source = Path(row['path'])
    if source.suffix.lower() != '.glb' or not source.is_file() or source.stat().st_size > 64 * 2**20:
        raise ValueError('bounded_abo_glb_required')
    ref = binding(source)
    if ref['sha256'] != row['source_sha']:
        raise ValueError('raw_source_hash_changed')
    return {'source': ref, 'resolver': {}}


def intake(output, workers=16, sources=('google_scanned_objects_original_fuel', 'ycb_official')):
    from resimpair_assets.raw_metric_containers_v1 import raw_dependencies
    output = Path(output).resolve()
    if (output / 'FINAL.json').exists():
        raise ValueError('preserve_existing_intake')
    database = ROOT / 'assets/expanded_library/raw_roles_v4.sqlite'
    with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(r) for r in connection.execute(
            'SELECT * FROM assets WHERE source IN (%s)' % ','.join('?' * len(sources)), tuple(sources))]
    rows = [r for r in rows if NAME.search(r['name'] or '')]
    jobs, skipped = [], []
    for row in rows:
        folder = output / 'jobs' / digest(row['id'])[:24]
        try:
            deps = abo_dependencies(row) if row['source'] == 'abo' else raw_dependencies(row)
        except (OSError, ValueError) as error:
            skipped.append({'source_id': row['id'], 'reason': str(error)}); continue
        write(folder / 'job.json', {'schema': VERSION, 'source_id': row['id'], 'source_type': row['source'],
              'name': row['name'], 'partition': row['partition'], 'source_record': json.loads(row['record']),
              'dependencies': deps, 'source_aliases': [{'id': row['id'], 'partition': row['partition']}],
              'alias_split_consistent': True, 'source_scale_factor': 1.0,
              'scale_evidence': 'original_source_metric_convention_not_per_object_measurement', 'output': str(folder)})
        jobs.append(folder / 'job.json')
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(compile_job, map(str, jobs)))
    ready = [r for r in results if r.get('status') == 'candidate_geometry_ready']
    write(output / 'FINAL.json', {'schema': VERSION, 'selected': len(rows), 'jobs': len(jobs), 'skipped': skipped,
          'ready': len(ready), 'assets': [r['asset'] for r in ready],
          'failures': {k: sum(1 for r in results if r.get('reason', r.get('status')) == k)
                       for k in {r.get('reason', r.get('status')) for r in results if r not in ready}},
          'rules': {'wall_start_above_floor_m': WALL_START_M, 'min_depth_m': MIN_DEPTH_M, 'wall_fraction': WALL_FRACTION,
                    'min_ray_hits_of_64': MIN_RAY_HITS, 'floor_tolerance_m': FLOOR_TOLERANCE_M, 'floor_normal_min': FLOOR_NORMAL_MIN, 'min_radius_m': MIN_RADIUS_M,
                    'scale': 'source metric convention, factor 1'},
          'implementation': binding(Path(__file__))})
    return {'selected': len(rows), 'jobs': len(jobs), 'ready': len(ready)}



def intake_files(rows, output, workers=16, license_name=None):
    """Containers from already-downloaded metric GLB files: rows of {id, name, path, partition}."""
    output = Path(output).resolve()
    if (output / 'FINAL.json').exists():
        raise ValueError('preserve_existing_intake')
    jobs = []
    for row in rows:
        folder = output / 'jobs' / digest(row['id'])[:24]
        write(folder / 'job.json', {'schema': VERSION, 'source_id': row['id'], 'source_type': row['id'].split(':')[0],
              'name': row['name'], 'partition': row['partition'], 'source_record': {'license': license_name},
              'dependencies': {'source': binding(row['path']), 'resolver': {}},
              'source_aliases': [{'id': row['id'], 'partition': row['partition']}], 'alias_split_consistent': True,
              'source_scale_factor': 1.0, 'scale_evidence': 'source_authored_metric_scale_not_per_object_measurement',
              'output': str(folder)})
        jobs.append(folder / 'job.json')
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(compile_job, map(str, jobs)))
    ready = [r for r in results if r.get('status') == 'candidate_geometry_ready']
    write(output / 'FINAL.json', {'schema': VERSION, 'selected': len(rows), 'jobs': len(jobs), 'ready': len(ready),
          'assets': [r['asset'] for r in ready], 'license': license_name,
          'failures': {k: sum(1 for r in results if r.get('reason', r.get('status')) == k)
                       for k in {r.get('reason', r.get('status')) for r in results if r not in ready}},
          'implementation': binding(Path(__file__))})
    return {'selected': len(rows), 'ready': len(ready)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--sources', nargs='+', default=['google_scanned_objects_original_fuel', 'ycb_official'])
    a = p.parse_args(); print(json.dumps(intake(a.output, a.workers, tuple(a.sources))))
