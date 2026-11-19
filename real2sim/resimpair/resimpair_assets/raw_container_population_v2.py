"""Direct raw container population, preserving cavities and explicit source size.

The retained cavity and oriented-support functions remain the only geometric
acceptance rules. Diagnostic tracing observes their actual executed branches;
it never supplies an alternative score or modifies a mesh. CPU/static native
mesh cooking only: no dynamic scene, renderer, GPU, API or default promotion.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import struct
import subprocess
import sys
import time
import traceback
from urllib.parse import unquote, urlsplit

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from .raw_metric_containers_v1 import campaign_lock, identity, sufficient_disk

VERSION = 'raw_container_population_v2'
MAX_SOURCES, WORKERS, ASSET_SECONDS, WALL_SECONDS = 2859, 4, 120, 4 * 3600
QUALIFICATION_COUNT, SEED = 12, 20261031
INDEX = ROOT / 'runs/metric_assets_20261030_v7/inventory_r1/source_state_index.json'
DATABASE = ROOT / 'assets/expanded_library/raw_roles_v4.sqlite'
METADATA = ROOT / 'runs/metric_assets_20261030_v7/inventory_r1/abo_metric_evidence.json'
OLD_CATALOGS = [ROOT / 'runs/metric_assets_20261030_v7/container_union_v6/catalog.json',
                ROOT / 'runs/pickplace_containers_20261026_catalog_v2.json']
OLD_37 = ROOT / 'reports/asset_release_rethink_20261031/container_intake/SOURCE_SELECTION.json'
RULE_FILES = ['resimpair_pickplace/containers.py', 'resimpair_pickplace/support_surface.py',
    'resimpair_assets/metric_visual_precision_v3.py', 'resimpair_assets/raw_metric_containers_v1.py',
    'resimpair_assets/raw_size_policy_v1.py', 'resimpair_assets/metric_inventory_v7.py',
    'resimpair_assets/raw_mesh_io_v1.py',
    'resimpair_assets/metric_semantics_v1.py', 'configs/metric_population_v7.json',
    'resimpair_tabletop/geometry.py', 'resimpair_grasp/common.py', 'resimpair_runtime.py',
    'resimpair_tasks/task_factory_v1.py']
PRIORITIES = [r'\b(trays?|platters?|basins?|washbasins?|baking sheets?|roasting pans?)\b',
              r'\b(storage|organizer|tote|crates?|open boxes?|bins?)\b',
              r'\b(baskets?|bowls?|pans?)\b']


def priority(row):
    text = re.sub(r'[_/\-]+', ' ', row.get('name') or '').lower()
    return next((i for i, pattern in enumerate(PRIORITIES) if re.search(pattern, text)), len(PRIORITIES))


def select_sources(index=INDEX, old_catalogs=OLD_CATALOGS, old_37=OLD_37):
    """The same explicit source-level remainder as the read-only funnel audit."""
    seen = set(read(old_37)['source_ids'])
    for path in old_catalogs:
        seen.update(row['source_id'] for row in read(path)['rows'])
    rows = [row for row in read(index)['entries'] if row['status'] == 'raw_or_offline_pending_native'
            and row['role'] == 'container' and row['source_id'] not in seen]
    by_source = {}
    for row in rows:
        by_source.setdefault(row['source'], []).append(row)
    key = lambda row: (priority(row), digest([SEED, row['source_id']]))
    initial = []
    for source in sorted(by_source):
        initial.extend(sorted(by_source[source], key=key)[:4])
    initial = initial[:QUALIFICATION_COUNT]
    selected = {row['source_id'] for row in initial}
    rows = initial + sorted((r for r in rows if r['source_id'] not in selected), key=key)
    if len({row['source_id'] for row in rows}) != len(rows):
        raise ValueError('raw_source_inventory_contains_duplicate_source_ids')
    return {'schema': VERSION, 'source_ids': [row['source_id'] for row in rows],
        'counts_by_source': dict(Counter(row['source'] for row in rows)),
        'retrieval_priority_counts': dict(Counter(priority(row) for row in rows)),
        'qualification_source_ids': [row['source_id'] for row in initial],
        'order': 'first four per source family by metadata priority/hash, then remaining priority/hash; before geometry outcomes',
        'exclusions': [binding(path) for path in [*old_catalogs, old_37]], 'index': binding(index),
        'priority_patterns': PRIORITIES, 'seed': SEED, 'semantic_identity_verified': False}


def _gltf_document(path):
    if path.suffix.lower() == '.gltf':
        if path.stat().st_size > 8 * 2**20: raise ValueError('bounded_GLTF_JSON_byte_limit')
        return json.loads(path.read_text())
    with path.open('rb') as stream:
        header = stream.read(20)
        if len(header) != 20: raise ValueError('truncated_GLB_header')
        magic, version, total, length, kind = struct.unpack('<4sIIII', header)
        if magic != b'glTF' or version != 2 or kind != 0x4e4f534a or total != path.stat().st_size:
            raise ValueError('invalid_GLB_v2_header')
        if length > 8 * 2**20: raise ValueError('bounded_GLB_JSON_byte_limit')
        return json.loads(stream.read(length).rstrip(b' \x00'))


def raw_dependencies(row):
    """Explicit local resources only; no network, directory scan or gray fallback."""
    source = Path(row['path']).resolve()
    if not source.is_file(): raise ValueError('raw_source_file_missing')
    if source.stat().st_size > 256 * 2**20: raise ValueError('bounded_raw_source_byte_limit')
    source_ref = binding(source)
    if row.get('source_sha') and source_ref['sha256'] != row['source_sha']:
        raise ValueError('raw_source_hash_changed')
    suffix = source.suffix.lower()
    if suffix == '.obj':
        from .raw_metric_containers_v1 import raw_dependencies as retained_obj
        return retained_obj({**row, 'source_sha': source_ref['sha256']})
    if suffix not in {'.glb', '.gltf', '.ply', '.stl'}:
        raise ValueError('unsupported_raw_container_format')
    resolver = {}
    if suffix in {'.glb', '.gltf'}:
        document = _gltf_document(source)
        resources = [row['uri'] for key in ('buffers', 'images') for row in document.get(key, []) if row.get('uri')]
        for uri in resources:
            if uri.startswith('data:'): continue
            split = urlsplit(uri)
            if split.scheme or split.netloc or split.query or split.fragment:
                raise ValueError('nonlocal_GLTF_resource_forbidden')
            target = (source.parent / unquote(split.path)).resolve()
            if not target.is_relative_to(source.parent) or not target.is_file():
                raise ValueError('GLTF_resource_missing_or_outside_package')
            if target.stat().st_size > 128 * 2**20: raise ValueError('bounded_GLTF_resource_byte_limit')
            resolver[uri] = binding(target)
    if len(resolver) > 64 or sum(resolve(r).stat().st_size for r in resolver.values()) > 512 * 2**20:
        raise ValueError('bounded_source_material_resource_budget')
    return {'source': source_ref, 'resolver': resolver, 'raw_files_modified': False,
            'raw_catalog_hash_present': bool(row.get('source_sha'))}


def material_summary(scene):
    import numpy as np
    rows = []
    for mesh in scene.geometry.values():
        material = getattr(mesh.visual, 'material', None); textures = {}
        for channel in ('baseColorTexture', 'image', 'metallicRoughnessTexture', 'normalTexture',
                        'occlusionTexture', 'emissiveTexture'):
            image = getattr(material, channel, None)
            if image is not None:
                pixels = np.asarray(image.convert('RGBA'))
                textures[channel] = {'pixels_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                                     'shape': list(pixels.shape)}
        uv = getattr(mesh.visual, 'uv', None)
        rows.append({'material_class': type(material).__name__, 'textures': textures,
            'vertices': len(mesh.vertices), 'faces': len(mesh.faces), 'has_UV': uv is not None,
            'UV_sha256': hashlib.sha256(np.asarray(uv, '<f8').tobytes()).hexdigest() if uv is not None else None})
    return rows


def verify_materials(before, after):
    """All supported texture channels and actual UV values; no rendering claim."""
    import numpy as np
    first, second = list(before.geometry.values()), list(after.geometry.values())
    if len(first) != len(second): raise ValueError('source_material_instance_count_changed')
    a, b = material_summary(before), material_summary(after)
    # OBJ SimpleMaterial may become a PBR baseColorTexture, preserving pixels.
    canon = lambda rows: Counter((('baseColorTexture' if channel == 'image' else channel),
        texture['pixels_sha256'], tuple(texture['shape'])) for row in rows for channel, texture in row['textures'].items())
    if canon(a) != canon(b): raise ValueError('source_texture_channel_pixels_not_preserved')
    errors = []
    for x, y in zip(first, second):
        u, v = getattr(x.visual, 'uv', None), getattr(y.visual, 'uv', None)
        if (u is None) != (v is None): raise ValueError('source_texture_UV_presence_changed')
        if u is not None:
            if np.shape(u) != np.shape(v): raise ValueError('source_texture_UV_shape_changed')
            error = float(np.max(np.abs(np.asarray(u) - np.asarray(v))))
            if not np.isfinite(error) or error > 1e-6: raise ValueError('source_texture_UV_values_changed')
            errors.append(error)
    return {'source': a, 'reloaded': b, 'all_supported_texture_channel_pixel_multiset_equal': True,
        'UV_coordinates_preserved_within_1e-6': True, 'maximum_UV_coordinate_error': max(errors, default=0.),
        'shader_and_render_equivalence': 'unknown; no rendering or lighting comparison was run'}


def cavity_with_diagnostics(mesh):
    """Observe retained cavity decisions without another ray pass or rule copy.

    The line trace is restricted to the frozen function's code object and is
    restored in finally. In production each source has its own single-threaded
    process. Returned geometry and ValueError are those of the original call.
    """
    import numpy as np
    from resimpair_pickplace.containers import cavity
    lines, start = inspect.getsourcelines(cavity)
    tree = ast.parse(''.join(lines))
    continues = sorted(node.lineno + start - 1 for node in ast.walk(tree) if isinstance(node, ast.Continue))
    reasons = ['missing_floor_hit', 'depth_or_floor_slope', 'incomplete_enclosure_wall_rays',
               'initial_conservative_radius_below_45mm', 'no_supported_flat_disk_after_ten_shrinks',
               'final_supported_radius_below_40mm']
    if len(continues) != len(reasons): raise RuntimeError('retained_cavity_branch_shape_changed')
    branch = dict(zip(continues, reasons))
    begin = next(start + i for i, line in enumerate(lines) if 'hit, _, face = ray.intersects_location' in line)
    observations = []; current = None
    def local_trace(frame, event, arg):
        nonlocal current
        if event != 'line': return local_trace
        values = frame.f_locals
        if frame.f_lineno == begin:
            current = {'center_xy_m': np.asarray(values['xy']).tolist(), 'outcome': 'retained_cavity_candidate'}
            observations.append(current)
        if current is not None and frame.f_lineno in branch:
            reason = branch[frame.f_lineno]; current['outcome'] = reason
            if reason == 'depth_or_floor_slope':
                depth = float(values['depth']); normal = abs(float(mesh.face_normals[values['face'][0], 2]))
                current.update(depth_m=depth, absolute_floor_normal_z=normal,
                    failed_conditions=[name for name, fails in [('depth_below_8mm', depth < .008),
                        ('depth_above_150mm', depth > .15), ('floor_normal_z_below_0p90', normal < .90)] if fails])
            if reason not in {'missing_floor_hit', 'depth_or_floor_slope'}:
                current['depth_m'] = float(values['depth'])
                current['completed_wall_levels'] = len(values.get('radii', []))
                if values.get('radii'): current['minimum_wall_radius_m'] = min(values['radii'])
                if reason != 'incomplete_enclosure_wall_rays': current['tested_radius_m'] = float(values['radius'])
                if 'ids' in values: current['last_ray_hit_ids'] = len(set(values['ids']))
                if reason == 'no_supported_flat_disk_after_ten_shrinks' and len(values.get('loc', [])):
                    current['last_floor_deviation_m'] = float(np.max(np.abs(values['loc'][:,2] - values['floor'])))
        return local_trace
    def tracer(frame, event, arg):
        return local_trace if frame.f_code is cavity.__code__ else None
    previous = sys.gettrace()
    try:
        sys.settrace(tracer)
        try: result, error = cavity(mesh), None
        except ValueError as caught: result, error = None, str(caught)
    finally: sys.settrace(previous)
    counts = Counter(row['outcome'] for row in observations)
    return result, {'retained_error': error, 'center_attempts': observations, 'branch_counts': dict(counts),
        'method': 'actual retained function line events; no alternate geometry test or threshold',
        'semantic_or_physical_failure_inferred': False}


def prepare(root, *, source_selection=None, database=DATABASE, metadata=METADATA):
    root = Path(root).resolve()
    if root.exists(): raise FileExistsError('preserve_existing_container_population_root')
    free = sufficient_disk(root.parent)
    selection = read(source_selection) if source_selection else select_sources()
    ids = selection['source_ids']
    if not 1 <= len(ids) <= MAX_SOURCES or len(set(ids)) != len(ids):
        raise ValueError('bounded_unique_source_selection_required')
    root.mkdir(parents=True); write(root / 'SOURCE_SELECTION.json', selection)
    metadata_ref = binding(metadata); selected = []; jobs = []; seen = {}
    with sqlite3.connect(f'file:{Path(database).resolve()}?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for source_id in ids:
            raw = connection.execute('SELECT * FROM assets WHERE id=?', (source_id,)).fetchone()
            if raw is None: raise ValueError('selected_source_not_in_raw_database')
            row = dict(raw); selected.append(row)
            source = Path(row['path']).resolve(); expected = row.get('source_sha')
            if not expected and source.is_file(): expected = binding(source)['sha256']
            aliases = ([dict(r) for r in connection.execute('SELECT id,partition FROM assets WHERE source_sha=?', (row['source_sha'],))]
                       if row.get('source_sha') else [{'id': source_id, 'partition': row['partition']}])
            duplicate = seen.get(expected) if expected else None
            if expected: seen.setdefault(expected, source_id)
            output = root / 'jobs' / digest(source_id)[:24]
            record = json.loads(row['record'])
            write(output / 'job.json', {'schema': VERSION, 'source_id': source_id, 'raw_row': row,
                'source_record': record, 'source_categories': record.get('categories', []),
                'expected_source': {'path': str(source), 'sha256': expected} if expected else None,
                'source_aliases': aliases, 'alias_split_consistent': len({a['partition'] for a in aliases}) == 1,
                'duplicate_selected_source': duplicate, 'metadata': metadata_ref,
                'output': str(output), 'retrieval_priority': priority(row)})
            jobs.append(binding(output / 'job.json'))
    protocol = {'schema': VERSION, 'jobs': jobs, 'source_count': len(jobs),
        'source_selection': binding(root / 'SOURCE_SELECTION.json'), 'database': binding(database),
        'metadata': metadata_ref, 'maximum_sources': MAX_SOURCES, 'maximum_CPU_workers': WORKERS,
        'per_source_timeout_seconds': ASSET_SECONDS, 'maximum_wall_seconds_from_first_run': WALL_SECONDS,
        'qualification_sources': min(QUALIFICATION_COUNT, len(jobs)), 'per_source_attempts': 1, 'retries': 0,
        'implementation': binding(__file__), 'rule_bindings': [binding(ROOT / path) for path in RULE_FILES],
        'resources': {'minimum_free_bytes': 50 * 2**30, 'free_bytes_at_prepare': free, 'worker_threads': 1,
            'CUDA_VISIBLE_DEVICES': '', 'source_byte_limit': 256 * 2**20, 'material_resource_byte_limit': 512 * 2**20},
        'rules': {'stable_pose_candidates': 10, 'stable_pose_probability_threshold': .005,
            'stable_hull_face_limit': 400, 'source_mesh_face_limit': 300000,
            'cavity_floor_tolerance_m': .003, 'native_triangle_vertex_error_limit_m': 1e-5,
            'export_bounds_rtol': 1e-6, 'export_bounds_atol_m': 1e-7, 'same_original_cavity_and_support_rules': True},
        'source_type_counts': dict(Counter(r['source'] for r in selected)),
        'API_calls': 0, 'GPU_jobs': 0, 'physics_steps': 0, 'default_catalog_modified': False,
        'unknown_reservations_never_resubmitted': True, 'source_seed': SEED}
    write(root / 'PROTOCOL.json', protocol)
    return {'root': str(root), 'registered': len(jobs), 'counts': {'not_submitted': len(jobs)}}


def validate(root):
    protocol = read(Path(root) / 'PROTOCOL.json')
    if tuple(protocol.get(key) for key in ('maximum_sources', 'maximum_CPU_workers', 'per_source_timeout_seconds',
        'maximum_wall_seconds_from_first_run', 'per_source_attempts', 'retries')) != (MAX_SOURCES, WORKERS, ASSET_SECONDS, WALL_SECONDS, 1, 0):
        raise ValueError('frozen_finite_budget_changed')
    if len(protocol['jobs']) != protocol['source_count'] or not 1 <= len(protocol['jobs']) <= MAX_SOURCES:
        raise ValueError('registered_source_count_changed')
    for ref in [protocol['implementation'], protocol['source_selection'], protocol['metadata'], *protocol['rule_bindings']]: resolve(ref)
    return protocol


def compile_one(job_path):
    import numpy as np
    import trimesh
    import sapien
    from scipy.spatial import cKDTree
    from .raw_size_policy_v1 import decide_raw
    from .raw_mesh_io_v1 import load_raw_scene, export_visual
    from .metric_visual_precision_v3 import bake_visual_world
    from resimpair_pickplace.support_surface import check_support
    job = read(job_path); output = Path(job['output']); started = time.monotonic(); row = job['raw_row']
    if not job['expected_source']: raise ValueError('source_identity_unavailable')
    resolve(job['expected_source'])
    dependencies = raw_dependencies(row); write(output / 'SOURCE_DEPENDENCIES.json', dependencies)
    source = resolve(dependencies['source'])
    resolver = {name: resolve(ref).read_bytes() for name, ref in dependencies['resolver'].items()}
    if source.suffix.lower() == '.gltf' or (source.suffix.lower() == '.glb' and resolver):
        scene = trimesh.load(source, force='scene', process=False, resolver=resolver)
        # The bounded local GLTF extension resolves only hash-bound resources.
        scene = bake_visual_world(scene); original = scene.to_geometry()
        load_receipt = {'implementation': binding(__file__), 'dependencies': dependencies,
            'extension': 'explicit_local_GLTF_resources', 'raw_nodes_baked_before_metric_scale': True}
    else:
        scene, original, load_receipt = load_raw_scene({**row, 'source_sha': dependencies['source']['sha256']})
    write(output / 'RAW_LOAD.json', load_receipt)
    if not isinstance(original, trimesh.Trimesh) or not len(original.faces): raise ValueError('no_triangle_geometry')
    if len(original.faces) > 300000: raise ValueError('container_face_budget')
    if not np.isfinite(original.vertices).all(): raise ValueError('nonfinite_container')
    original_obb = original.bounding_box_oriented.primitive.extents
    metadata = read(resolve(job['metadata']))['matches']
    decision = decide_raw({**row, 'source_categories': job['source_categories']}, original_obb,
        metadata_records=metadata.get(row['id'].split(':', 1)[1], []) if row['source'] == 'abo' else [],
        seed=SEED, role_override='container')
    write(output / 'SIZE_DECISION.json', decision)
    if decision['role'] != 'container': raise ValueError('size_policy_did_not_admit_container_role')
    scale = float(decision['source_unit_to_m']); scaled = original.copy(); scaled.apply_scale(scale)
    write(output / 'SOURCE_GEOMETRY.json', {'source': dependencies['source'],
        'raw_bounds_in_source_units': original.bounds.tolist(), 'raw_obb_in_source_units': original_obb.tolist(),
        'source_unit_to_m': scale, 'metric_bounds_m': scaled.bounds.tolist(), 'metric_dimensions_m': scaled.extents.tolist(),
        'mesh_vertices': len(original.vertices), 'mesh_faces': len(original.faces),
        'node_transforms_baked_before_uniform_unit_conversion': True, 'size_decision': binding(output / 'SIZE_DECISION.json')})
    if max(scaled.extents) > 2 or min(scaled.extents) <= 0:
        raise ValueError('outside_retained_static_container_numeric_bounds')
    hull = scaled.convex_hull
    if len(hull.faces) > 400: hull = hull.simplify_quadric_decimation(face_count=400).convex_hull
    poses, probabilities = trimesh.poses.compute_stable_poses(hull, n_samples=1, sigma=0, threshold=.005)
    candidates = []; searches = []
    for index, (pose, probability) in enumerate(zip(poses[:10], probabilities[:10])):
        placed = scaled.copy(); placed.apply_transform(pose)
        center = placed.bounds.mean(axis=0); center[2] = placed.bounds[0, 2]
        placed.apply_translation(-center); translation = np.eye(4); translation[:3, 3] = -center
        transform = translation @ pose @ np.diag([scale, scale, scale, 1.])
        opening, diagnostics = cavity_with_diagnostics(placed)
        searches.append({'index': index, 'stable_pose_probability': float(probability),
            'status': 'geometry_candidate' if opening else 'retained_cavity_rejected',
            'cavity': opening, 'diagnostics': diagnostics})
        if opening: candidates.append((opening['flat_floor_radius_m'], placed, transform, opening, float(probability), index))
    search = {'maximum_stable_candidates': 10, 'source_unit_to_m': scale, 'rows': searches,
        'branch_counts': dict(sum((Counter(r['diagnostics']['branch_counts']) for r in searches), Counter()))}
    write(output / 'orientation_search.json', search)
    if not candidates: raise ValueError('retained_cavity_rules_rejected_all_recorded_orientations')
    _, mesh, transform, opening, probability, pose_index = max(candidates, key=lambda item: item[0])
    singular = np.linalg.svd(transform[:3, :3], compute_uv=False)
    if not np.allclose(singular, scale, rtol=1e-10, atol=1e-10) or np.linalg.det(transform[:3, :3]) <= 0:
        raise ValueError('source_to_asset_is_not_selected_positive_uniform_scale_and_rigid_pose')
    visual = scene.copy(); visual.apply_transform(transform); visual = bake_visual_world(visual)
    visual_path = output / 'visual.glb'
    export_report = export_visual(visual, visual_path, mesh)
    reloaded_scene = trimesh.load(visual_path, force='scene', process=False)
    if not np.allclose(reloaded_scene.to_geometry().bounds, mesh.bounds, rtol=1e-6, atol=1e-7):
        raise ValueError('container_visual_export_changed_scale_or_origin')
    write(output / 'MATERIAL_EXPORT.json', {**verify_materials(visual, reloaded_scene),
        'shared_export_check': export_report})
    collision_path = output / 'collision.obj'; mesh.export(collision_path, digits=12)
    native = sapien.physx.PhysxCollisionShapeTriangleMesh(filename=str(collision_path), scale=[1,1,1],
        material=sapien.physx.PhysxMaterial(.6,.6,0.))
    vertices = np.asarray(native.vertices)
    error = max(cKDTree(mesh.vertices).query(vertices)[0].max(), cKDTree(vertices).query(mesh.vertices)[0].max())
    if not np.isfinite(error) or error > 1e-5: raise ValueError('native_triangle_geometry_changed')
    saved_collision = trimesh.load(collision_path, force='mesh', process=False)
    support = check_support(saved_collision, opening)
    write(output / 'NATIVE_GEOMETRY_AND_SUPPORT.json', {'native_triangle_vertex_error_m': float(error),
        'native_triangle_vertex_count': len(vertices), 'source_to_asset_singular_values': singular.tolist(),
        'native_oriented_support': support, 'scene_created': False, 'physics_steps': 0, 'GPU_jobs': 0})
    variant = digest({'source': dependencies, 'source_to_asset': transform.tolist(),
        'size_decision': binding(output / 'SIZE_DECISION.json'), 'generator': binding(__file__)})
    descriptor = {'schema': 'stationary_mesh_container_v1', 'id': variant[:24], 'variant_id': variant,
        'source_id': job['source_id'], 'name': row['name'], 'partition': row['partition'],
        'source': dependencies['source'], 'source_dependencies': dependencies['resolver'],
        'license': job['source_record'].get('license'), 'source_aliases': job['source_aliases'],
        'source_alias_split_consistent': job['alias_split_consistent'],
        'visual': binding(visual_path), 'collision': binding(collision_path), 'source_to_asset': transform.tolist(),
        'bounds_m': mesh.bounds.tolist(), 'dimensions_m': mesh.extents.tolist(), 'cavity': opening,
        'stable_pose_index': pose_index, 'geometric_support_probability': probability,
        'native_triangle_vertex_error_m': float(error), 'native_policy': 'static_triangle_mesh_only; cavity preserved',
        'native_support_check': binding(output / 'NATIVE_GEOMETRY_AND_SUPPORT.json'),
        'roles': {'stationary_container': bool(support['passed']), 'graspable_container': False},
        'metric_size': {'mode': decision['size_confidence'], 'source_unit_to_m': scale,
            'additional_scale_in_container_stage': 1., 'size_decision': binding(output / 'SIZE_DECISION.json'),
            'per_object_metric_truth': decision['size_evidence'].get('source_metric_truth', 'unknown')},
        'semantic_status': 'retrieved role and sampled geometry only; identity, empty interior and payload fit unverified',
        'physical_status': 'unmeasured', 'relative_scale': 1., 'generator': binding(__file__)}
    write(output / 'descriptor.json', descriptor)
    state = 'candidate_geometry_ready' if support['passed'] else 'native_support_failed'
    if not job['alias_split_consistent']: state = 'candidate_geometry_split_conflict'
    return {'source_id': job['source_id'], 'status': state, 'asset': binding(output / 'descriptor.json'),
        'dimensions_m': mesh.extents.tolist(), 'cavity': opening, 'support': support,
        'size_confidence': decision['size_confidence'], 'size_category': decision['category'],
        'native_triangle_vertex_error_m': float(error), 'elapsed_seconds': time.monotonic() - started,
        'source_unit_to_m': scale, 'physical_success_inferred': False, 'default_admitted': False,
        'API_calls': 0, 'GPU_jobs': 0, 'physics_steps': 0}


def worker(root, job_path):
    root, job_path = Path(root).resolve(), Path(job_path).resolve(); protocol = validate(root)
    matches = [ref for ref in protocol['jobs'] if (ROOT / ref['path']).resolve() == job_path]
    if len(matches) != 1: raise ValueError('worker_job_not_in_frozen_source_scope')
    job_path = resolve(matches[0]); job = read(job_path); output = Path(job['output']).resolve()
    if output != job_path.parent or output.parent != root / 'jobs' or job_path.name != 'job.json':
        raise ValueError('worker_output_must_equal_registered_job_folder')
    if not (output / 'reservation.json').is_file(): raise ValueError('worker_requires_prior_registered_reservation')
    reservation = read(output / 'reservation.json')
    if (reservation.get('job'), reservation.get('source_id'), reservation.get('attempt')) != (matches[0], job['source_id'], 1):
        raise ValueError('worker_reservation_does_not_match_frozen_job')
    if (output / 'worker_started.json').exists(): raise ValueError('existing_attempt_requires_recovery_not_retry')
    write(output / 'worker_started.json', {'identity': identity(os.getpid()), 'job': matches[0], 'unix': time.time()})
    started = time.monotonic()
    try: result = compile_one(job_path)
    except Exception as error:
        from scipy.spatial import QhullError
        # Unexpected implementation/resource errors must halt admission. They
        # are not evidence that thousands of remaining meshes are unsuitable.
        known = isinstance(error, (ValueError, QhullError))
        result = {'source_id': job['source_id'], 'status': 'geometry_or_input_failed' if known else 'unknown_system_failure',
            'reason': type(error).__name__ + ':' + str(error), 'trace': traceback.format_exc(),
            'elapsed_seconds': time.monotonic() - started, 'physical_success_inferred': False,
            'API_calls': 0, 'GPU_jobs': 0, 'physics_steps': 0}
        if (output / 'orientation_search.json').is_file():
            result['diagnostics'] = binding(output / 'orientation_search.json')
    write(output / 'result.json', result)
    write(output / 'worker_terminal.json', {'result': binding(output / 'result.json'),
        'identity': identity(os.getpid()), 'unix': time.time()})
    return result


def status(root):
    root = Path(root).resolve(); protocol = read(root / 'PROTOCOL.json'); rows = []
    for ref in protocol['jobs']:
        path = resolve(ref); job = read(path); folder = path.parent
        if (folder / 'terminal.json').is_file(): row = read(folder / 'terminal.json')
        elif (folder / 'worker_terminal.json').is_file():
            receipt = read(folder / 'worker_terminal.json')
            row = {'source_id': job['source_id'], 'status': 'worker_result_complete_awaiting_supervisor_receipt',
                   'result_status': read(resolve(receipt['result']))['status']}
        elif (folder / 'reservation.json').is_file():
            process = read(folder / 'process.json') if (folder / 'process.json').is_file() else {}
            old = process.get('identity'); live = bool(old and identity(old['pid']) == old)
            row = {'source_id': job['source_id'], 'status': 'running' if live else 'unknown_reserved_no_terminal'}
        else: row = {'source_id': job['source_id'], 'status': 'not_submitted'}
        rows.append(row)
    return {'schema': VERSION, 'root': str(root), 'registered': len(rows),
        'counts': dict(Counter(row['status'] for row in rows)), 'rows': rows,
        'API_calls': 0, 'GPU_jobs': 0, 'physics_steps': 0, 'new_default_assets': 0}


def publish(root, phase):
    root = Path(root); summary = status(root); assets = []
    for row in summary['rows']:
        if row['status'] == 'candidate_geometry_ready': assets.append(read(resolve(row['result']))['asset'])
    path = root / ('FINAL' if not summary['counts'].get('not_submitted') and not summary['counts'].get('running')
                   and not summary['counts'].get('unknown_reserved_no_terminal') else f'snapshots/{time.time_ns()}')
    write(path / 'GEOMETRY_CATALOG.json', {'schema': 'stationary_container_catalog_v1', 'assets': assets,
        'scope': 'new candidate geometry only; no default or physical admission', 'protocol': binding(root / 'PROTOCOL.json')})
    # Invoke the real consumer's unchanged semantic/split checks, with no active
    # grasp assets. Payload fit and active-source partition remain task checks.
    write(path / 'EMPTY_ACTIVE_INDEX.json', {'entries': []})
    from resimpair_tasks.task_factory_v1 import load_pool
    _, usable, _, rejected = load_pool(path / 'EMPTY_ACTIVE_INDEX.json', path / 'GEOMETRY_CATALOG.json')
    write(path / 'SAMPLER_CATALOG.json', {'schema': 'stationary_container_catalog_v1',
        'assets': [row['binding'] for row in usable], 'scope': 'opt-in candidate catalog; native static support plus unchanged name gate',
        'physical_status': 'unmeasured', 'parent': binding(path / 'GEOMETRY_CATALOG.json'), 'rejected': rejected,
        'actual_factory_consumer': binding(ROOT / 'resimpair_tasks/task_factory_v1.py')})
    summary.update(phase=phase, geometric_candidates=len(assets), consumer_candidates=len(usable),
        sampler_catalog=binding(path / 'SAMPLER_CATALOG.json'), generated_unix=time.time())
    write(path / 'SUMMARY.json', summary)
    if phase == 'qualification' and not (root / 'QUALIFICATION.json').exists():
        write(root / 'QUALIFICATION.json', {'summary': binding(path / 'SUMMARY.json'),
            'protocol': binding(root / 'PROTOCOL.json'), 'qualifying_first_sources': min(QUALIFICATION_COUNT, summary['registered']),
            'continuation_requires_review': True})
    return summary


def run(root, *, phase='qualification', qualification_gate=None):
    root = Path(root).resolve(); protocol = validate(root)
    if phase not in {'qualification', 'population'}: raise ValueError('explicit_qualification_or_population_phase_required')
    if phase == 'population':
        if qualification_gate is None: raise ValueError('population_requires_recorded_qualification_gate')
        gate = read(qualification_gate)
        if (gate.get('protocol') != binding(root / 'PROTOCOL.json')
            or gate.get('qualification') != binding(root / 'QUALIFICATION.json')
            or gate.get('decision') != 'allow_same_frozen_population'):
            raise ValueError('qualification_gate_does_not_bind_current_population')
    with campaign_lock(root):
        sufficient_disk(root)
        if not (root / 'START.json').exists():
            now = time.time(); write(root / 'START.json', {'identity': identity(os.getpid()), 'unix': now, 'deadline_unix': now + WALL_SECONDS})
        deadline = read(root / 'START.json')['deadline_unix']; work = []
        refs = protocol['jobs'][:protocol['qualification_sources']] if phase == 'qualification' else protocol['jobs']
        for ref in protocol['jobs']:
            path = resolve(ref); job = read(path); folder = path.parent
            if (folder / 'terminal.json').exists(): continue
            if (folder / 'worker_terminal.json').exists():
                receipt = read(folder / 'worker_terminal.json'); result = read(resolve(receipt['result']))
                write(folder / 'terminal.json', {'source_id': job['source_id'], 'status': result['status'],
                    'result': receipt['result'], 'recovered_from_complete_worker_receipt': True, 'resubmitted': False})
                continue
            if (folder / 'reservation.json').exists():
                raise RuntimeError('unknown_or_live_reservation_recover_before_admission:' + job['source_id'])
            if ref in refs: work.append((path, job))
        def one(item):
            path, job = item; folder = path.parent
            if time.time() >= deadline: return None
            sufficient_disk(root)
            write(folder / 'reservation.json', {'source_id': job['source_id'], 'job': binding(path), 'unix': time.time(), 'attempt': 1})
            if job['duplicate_selected_source']:
                row = {'source_id': job['source_id'], 'status': 'duplicate_selected_source', 'canonical_source': job['duplicate_selected_source']}
                write(folder / 'terminal.json', row); return row
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            started = time.monotonic()
            with (folder / 'worker.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-m', 'resimpair_assets.raw_container_population_v2', 'worker',
                    '--root', str(root), '--job', str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=environment, start_new_session=True)
                write(folder / 'process.json', {'identity': identity(process.pid), 'unix': time.time()})
                try: code = process.wait(timeout=min(ASSET_SECONDS, max(.001, deadline-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                    code = 'timeout'
            if (folder / 'worker_terminal.json').exists():
                receipt = read(folder / 'worker_terminal.json'); state = read(resolve(receipt['result']))['status']
            else: receipt = {}; state = 'source_timeout' if code == 'timeout' else 'unknown_system_failure'
            terminal = {'source_id': job['source_id'], 'status': state, 'returncode': code,
                'result': receipt.get('result'), 'elapsed_seconds': time.monotonic()-started, 'retries': 0}
            write(folder / 'terminal.json', terminal); print(job['source_id'], state, flush=True)
            return terminal
        pending = iter(work)
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            active = {executor.submit(one, item) for item in [next(pending, None) for _ in range(WORKERS)] if item is not None}
            halt = False
            while active:
                future = next(as_completed(active)); active.remove(future); terminal = future.result()
                halt = halt or bool(terminal and terminal['status'] == 'unknown_system_failure')
                item = next(pending, None) if not halt and time.time() < deadline else None
                if item is not None: active.add(executor.submit(one, item))
        return publish(root, phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'status', 'worker'])
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--job', type=Path)
    parser.add_argument('--phase', choices=['qualification', 'population'], default='qualification')
    parser.add_argument('--qualification-gate', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.root)
    elif args.command == 'run': result = run(args.root, phase=args.phase, qualification_gate=args.qualification_gate)
    elif args.command == 'worker': result = worker(args.root, args.job)
    else: result = status(args.root)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}, ensure_ascii=False))
