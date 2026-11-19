"""Bake visual graph transforms before metric export, retaining frozen checks.

Some original GSO GLBs contain large local vertex coordinates canceled by node
translations. Exporting a scaled version quantizes those coordinates to float32
before cancellation, losing sub-micrometer precision in the final metric frame.
This adapter bakes the parent visual into its existing asset frame in float64
before scaling. It preserves texture/material instances and never changes the
source-to-asset transform, collision geometry, size/mass policy or tolerances.

The scoped loader hook exists only inside this process. Use this entry in a
per-source subprocess, not concurrently with another materializer thread.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import time

import numpy as np
import trimesh

from resimpair_grasp.common import binding, read, resolve, write
from . import metric_variants_v1 as geometry
from . import metric_variants_v2 as bounded

VERSION = 'world_baked_metric_visual_v3'


def bake_visual_world(scene):
    """Copy every visible mesh instance into world coordinates with its material."""
    baked = trimesh.Scene(metadata=copy.deepcopy(scene.metadata))
    for index, node in enumerate(scene.graph.nodes_geometry):
        transform, key = scene.graph[node]
        source = scene.geometry[key]
        if not isinstance(source, trimesh.Trimesh):
            raise ValueError('visual_graph_contains_non_triangle_geometry')
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError('visual_graph_transform_must_be_finite')
        mesh = source.copy()
        mesh.apply_transform(transform)
        baked.add_geometry(mesh, node_name=f'instance_{index}', geom_name=f'geometry_{index}')
    if not baked.geometry:
        raise ValueError('visual_graph_has_no_geometry')
    # The original metric check below remains stricter than a display-only
    # approximation; this step also prevents silently dropping graph instances.
    original = scene.to_geometry()
    result = baked.to_geometry()
    if len(original.faces) != len(result.faces) or len(original.vertices) != len(result.vertices):
        raise ValueError('baked_visual_topology_count_changed')
    if not np.allclose(original.bounds, result.bounds, rtol=1e-12, atol=1e-12):
        raise ValueError('baked_visual_world_bounds_changed')
    return baked


@contextmanager
def _baked_parent_visual(reference):
    original = geometry.load_geometry
    path = resolve(reference)

    def load(value):
        scene, mesh = original(value)
        if resolve(value) == path:
            return bake_visual_world(scene), mesh
        return scene, mesh

    geometry.load_geometry = load
    try:
        yield
    finally:
        geometry.load_geometry = original


def materialize(descriptor_path, policy_path, output, bounds_path=bounded.DEFAULT_BOUNDS):
    output = Path(output)
    if output.exists():
        raise ValueError('preserve_existing_visual_precision_output')
    parent = read(descriptor_path)
    recipe = {'schema': VERSION, 'parent_descriptor': binding(descriptor_path),
        'policy': binding(policy_path), 'numeric_bounds': binding(bounds_path),
        'implementation': binding(__file__), 'retained_geometry': binding(geometry.__file__),
        'retained_numeric_checks': binding(bounded.__file__),
        'operation': 'bake_parent_visual_node_world_transforms_before_uniform_metric_scale',
        'collision_geometry_changed_by_this_adapter': False,
        'original_export_tolerance': {'rtol': 1e-6, 'atol_m': 1e-7},
        'tolerances_changed': False, 'API_calls': 0, 'GPU_jobs': 0,
        'native_physics_executions': 0, 'started_unix': time.time()}
    write(output / 'recipe.json', recipe)
    start = time.monotonic()
    try:
        with _baked_parent_visual(parent['visual']):
            value = bounded.materialize(descriptor_path, policy_path, output / 'retained', bounds_path)
        entries = []
        for row in value['variants']:
            descriptor = read(resolve(row['descriptor']))
            descriptor['visual_precision_recipe'] = binding(output / 'recipe.json')
            descriptor['generator'] = binding(__file__)
            destination = output / row['variant_id'] / 'descriptor.json'
            write(destination, descriptor)
            # Independent reload uses the unhooked, original v2 auditor.
            audit = bounded.audit_variant(destination, bounds_path)
            audit_path = destination.parent / 'independent_numeric_reload.json'
            write(audit_path, audit)
            entries.append({**row, 'descriptor': binding(destination),
                'visual_precision_recipe': binding(output / 'recipe.json'),
                'independent_numeric_reload': binding(audit_path)})
        terminal = {**value, 'schema': VERSION, 'variants': entries,
            'elapsed_seconds': time.monotonic() - start, 'recipe': binding(output / 'recipe.json'),
            'native_qualification': 'not_performed', 'grasp_qualification': 'not_performed'}
        write(output / 'catalog.json', {'schema': 'resimpair_mesh_catalog_v1',
            'assets': [row['descriptor'] for row in entries], 'qualification': terminal['status'], 'failures': []})
        write(output / 'terminal.json', terminal)
        return terminal
    except Exception as error:
        write(output / 'terminal.json', {'schema': VERSION, 'status': 'failed_preserve_evidence',
            'reason': str(error), 'elapsed_seconds': time.monotonic() - start,
            'recipe': binding(output / 'recipe.json'), 'API_calls': 0, 'GPU_jobs': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bounds', type=Path, default=bounded.DEFAULT_BOUNDS)
    args = parser.parse_args()
    print(materialize(args.descriptor, args.policy, args.output, args.bounds)['status'])
