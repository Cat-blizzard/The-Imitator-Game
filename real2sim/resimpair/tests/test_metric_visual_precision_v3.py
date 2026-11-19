import numpy as np
import pytest
import trimesh

from resimpair_assets.metric_visual_precision_v3 import bake_visual_world, _baked_parent_visual
from resimpair_assets import metric_variants_v1 as retained


def test_world_baking_preserves_instances_faces_material_and_input():
    mesh = trimesh.creation.box([.08, .06, .1])
    mesh.apply_translation([1000., -2000., 3000.])
    mesh.visual.vertex_colors = [90, 40, 10, 255]
    scene = trimesh.Scene()
    first = np.eye(4)
    first[:3, 3] = [-1000., 2000., -3000.]
    second = first.copy()
    second[:3, 3] += [.12, .05, .01]
    scene.add_geometry(mesh, node_name='first', geom_name='shared', transform=first)
    scene.graph.update(frame_to='second', matrix=second, geometry='shared')
    before = mesh.vertices.copy()
    output = bake_visual_world(scene)
    assert len(output.geometry) == 2
    assert len(output.to_geometry().faces) == 2 * len(mesh.faces)
    assert np.array_equal(mesh.vertices, before)
    assert np.allclose(output.bounds, scene.bounds, rtol=1e-12, atol=1e-12)
    assert all(np.array_equal(m.visual.vertex_colors[0], [90, 40, 10, 255]) for m in output.geometry.values())
    assert all(np.allclose(output.graph[node][0], np.eye(4)) for node in output.graph.nodes_geometry)


def test_empty_visual_is_not_accepted():
    with pytest.raises(ValueError, match='no_geometry'):
        bake_visual_world(trimesh.Scene())


def test_scoped_loader_restored_after_failure(tmp_path):
    from resimpair_grasp.common import binding
    mesh_path = tmp_path / 'visual.obj'
    trimesh.creation.box([.1, .1, .1]).export(mesh_path)
    original = retained.load_geometry
    with pytest.raises(RuntimeError, match='diagnostic'):
        with _baked_parent_visual(binding(mesh_path)):
            assert retained.load_geometry is not original
            retained.load_geometry(binding(mesh_path))
            raise RuntimeError('diagnostic')
    assert retained.load_geometry is original
