import numpy as np
import pytest
import trimesh

from resimpair_assets.metric_collision_fragments_v7 import fragment_geometry


def test_full_rank_thin_solid_is_kept():
    mesh = trimesh.creation.box([.1, .1, 1e-6])
    result, proof = fragment_geometry(mesh.vertices, mesh.faces)
    assert result is not None
    assert proof['rank'] == 3
    assert not proof['omitted']


def test_algebraically_planar_fragment_is_omitted():
    mesh = trimesh.creation.box([.1, .1, .1])
    vertices = mesh.vertices.copy()
    vertices[:, 2] = 2 * vertices[:, 0] + vertices[:, 1]
    result, proof = fragment_geometry(vertices, mesh.faces)
    assert result is None
    assert proof['rank'] == 2
    assert proof['omitted']


def test_nonfinite_data_is_rejected_not_omitted():
    mesh = trimesh.creation.box()
    vertices = mesh.vertices.copy()
    vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite_fragment'):
        fragment_geometry(vertices, mesh.faces)


def test_full_rank_open_surface_is_rejected():
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError, match='full_rank_non_solid'):
        fragment_geometry(mesh.vertices, mesh.faces[:-1])
