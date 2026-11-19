"""Preserve affine scene edges exactly and serialize identity-graph mesh instances.

Trimesh's SceneGraph.get may repair a near-rigid scale, and its graph copy does
not retain repair_rigid=None. Read the stored edge matrices directly, then bake
each visible instance in float64. The normal mesh/file bounds checks stay intact.
The retained-loader hooks are restricted to one isolated source worker process.
"""
from contextlib import contextmanager
import copy

import numpy as np
import trimesh

from resimpair_grasp.common import binding, resolve

VERSION = 'exact_scene_transforms_v1'


def world_transform(scene, node):
    """Compose every stored edge, without rigid repair or identity tolerances."""
    graph = scene.graph
    if node == graph.base_frame:
        return np.eye(4)
    path = graph.transforms.shortest_path(graph.base_frame, node)
    result = np.eye(4)
    for parent, child in zip(path[:-1], path[1:]):
        direct = graph.transforms.edge_data.get((parent, child))
        if direct is None:
            matrix = np.linalg.inv(graph.transforms.edge_data[(child, parent)]['matrix'])
        else:
            matrix = direct['matrix']
        matrix = np.asarray(matrix, dtype=np.float64)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.array_equal(matrix[3], [0., 0., 0., 1.])):
            raise ValueError('finite_affine_scene_edge_required')
        result = result @ matrix
    return result


def transformed_mesh(source, transform):
    """Keep topology/materials and avoid apply_transform's near-identity skip."""
    mesh = source.copy()
    expected = np.asarray(source.vertices, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]
    mesh.apply_transform(transform)
    if not np.array_equal(mesh.vertices, expected):
        mesh.vertices = expected
    return mesh


class ExactScene(trimesh.Scene):
    """A material-preserving scene whose transforms are applied to its vertices."""

    def copy(self):
        return bake_world(self)

    def apply_transform(self, transform):
        transformed = bake_world(self, transform)
        self.geometry = transformed.geometry
        self.graph = transformed.graph
        self._cache.clear()
        return self


def bake_world(scene, transform=None):
    transform = np.eye(4) if transform is None else np.asarray(transform, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.array_equal(transform[3], [0., 0., 0., 1.])):
        raise ValueError('finite_affine_bake_transform_required')
    baked = ExactScene(metadata=copy.deepcopy(scene.metadata))
    expected_vertices = []
    expected_faces = 0
    for number, node in enumerate(scene.graph.nodes_geometry):
        key = scene.graph.transforms.node_data[node]['geometry']
        source = scene.geometry[key]
        if not isinstance(source, trimesh.Trimesh):
            raise ValueError('visual_graph_contains_non_triangle_geometry')
        matrix = transform @ world_transform(scene, node)
        mesh = transformed_mesh(source, matrix)
        expected_vertices.append(np.asarray(source.vertices) @ matrix[:3, :3].T + matrix[:3, 3])
        expected_faces += len(source.faces)
        baked.add_geometry(mesh, node_name=f'instance_{number}', geom_name=f'geometry_{number}')
    if not expected_vertices:
        raise ValueError('visual_graph_has_no_geometry')
    expected = np.concatenate(expected_vertices)
    actual = baked.to_geometry()
    if len(actual.vertices) != len(expected) or len(actual.faces) != expected_faces:
        raise ValueError('exact_bake_changed_instance_or_topology_count')
    if not np.array_equal(actual.vertices, expected):
        raise ValueError('exact_bake_changed_transformed_vertices')
    return baked


def load_geometry(reference):
    scene = bake_world(trimesh.load(resolve(reference), force='scene', process=False))
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('triangle_mesh_required')
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
        raise ValueError('finite_three_dimensional_geometry_required')
    return scene, mesh


@contextmanager
def retained_io_scope():
    """Only replace geometry transport; preserve every retained qualification."""
    from . import raw_mesh_io_v1, metric_variants_v1
    old_bake = raw_mesh_io_v1.bake_visual_world
    old_load = metric_variants_v1.load_geometry
    raw_mesh_io_v1.bake_visual_world = bake_world
    metric_variants_v1.load_geometry = load_geometry
    try:
        yield
    finally:
        raw_mesh_io_v1.bake_visual_world = old_bake
        metric_variants_v1.load_geometry = old_load


def receipt():
    return {'implementation': binding(__file__), 'operation': 'compose_stored_scene_edges_then_bake_float64_vertices',
        'graph_rigid_repair_used': False, 'scene_identity_skip_used': False,
        'output_visible_instances_have_identity_graph_transforms': True,
        'bounds_tolerances_changed': False, 'size_or_mass_rules_changed': False}
