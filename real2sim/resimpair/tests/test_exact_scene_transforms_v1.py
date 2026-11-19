import numpy as np
import pytest
import trimesh

from resimpair_assets import exact_scene_transforms_v1 as exact
from resimpair_assets import raw_mesh_io_v1, metric_variants_v1
from resimpair_grasp.common import binding


def test_bakes_stored_near_one_scale_even_after_repaired_graph_cache():
    scene = trimesh.Scene()
    mesh = trimesh.creation.box([.204, .054, .054])
    factor = 1.0000034868511176
    matrix = np.diag([factor, factor, factor, 1.])
    scene.add_geometry(mesh, node_name='mesh', geom_name='mesh', transform=matrix)
    repaired, _ = scene.graph['mesh']
    assert np.allclose(repaired, np.eye(4), rtol=0, atol=1e-12)
    baked = exact.bake_world(scene)
    assert np.array_equal(baked.to_geometry().vertices, mesh.vertices * factor)
    assert np.array_equal(exact.world_transform(baked, 'instance_0'), np.eye(4))


def test_nested_sub_identity_edges_and_instances_are_not_discarded():
    mesh = trimesh.creation.box([.2, .1, .04])
    scene = trimesh.Scene()
    near_identity = np.diag([1. + 5e-9, 1. + 5e-9, 1. + 5e-9, 1.])
    scene.graph.update(frame_to='parent', matrix=near_identity)
    scene.add_geometry(mesh, node_name='first', geom_name='mesh', parent_node_name='parent')
    other = np.eye(4); other[0, 3] = .3
    scene.graph.update(frame_to='second', frame_from='world', matrix=other, geometry='mesh')
    baked = exact.bake_world(scene)
    expected = np.vstack([mesh.vertices * (1 + 5e-9), mesh.vertices + [.3, 0, 0]])
    assert np.array_equal(baked.to_geometry().vertices, expected)
    assert len(baked.to_geometry().faces) == 2 * len(mesh.faces)


def test_copy_scale_export_reloads_with_original_bounds_tolerance(tmp_path):
    from PIL import Image
    mesh = trimesh.creation.box([.204, .054, .054])
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)),
        image=Image.new('RGB', (2, 2), (117, 59, 31)))
    original = exact.bake_world(trimesh.Scene(mesh))
    copied = original.copy()
    factor = 1.0000034868511176
    copied.apply_scale(factor)
    assert isinstance(copied, exact.ExactScene)
    assert np.array_equal(copied.to_geometry().vertices, original.to_geometry().vertices * factor)
    with exact.retained_io_scope():
        receipt = raw_mesh_io_v1.export_visual(copied, tmp_path / 'visual.glb', copied.to_geometry())
    ordinary = trimesh.load(tmp_path / 'visual.glb', force='scene', process=False)
    assert np.allclose(ordinary.bounds, original.bounds * factor, rtol=1e-6, atol=1e-7)
    assert receipt['texture_pixels_preserved'] and receipt['UV_presence_preserved']
    assert np.array_equal(original.to_geometry().vertices, mesh.vertices)


def test_scoped_loader_changes_are_restored_on_exception():
    old_bake, old_load = raw_mesh_io_v1.bake_visual_world, metric_variants_v1.load_geometry
    with pytest.raises(RuntimeError):
        with exact.retained_io_scope():
            assert raw_mesh_io_v1.bake_visual_world is exact.bake_world
            assert metric_variants_v1.load_geometry is exact.load_geometry
            raise RuntimeError('sentinel')
    assert raw_mesh_io_v1.bake_visual_world is old_bake
    assert metric_variants_v1.load_geometry is old_load
