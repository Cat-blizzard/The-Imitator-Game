import numpy as np
import pytest
import trimesh
from resimpair_grasp.cudex_seed_initialization_v1 import (
    surface_samples,validate_surface_pool,validate_seed_tensor,SeedInitializationError)


def test_box_vertex_inflation_counterexample_and_face_clearance():
    mesh=trimesh.creation.box(extents=[.04,.05,.03]);inflated=mesh.copy()
    inflated.vertices+=.1*inflated.vertex_normals
    # Offset vertices at a corner is not a 10cm offset of its supporting face.
    assert np.min(np.ptp(inflated.bounds,axis=0)-np.ptp(mesh.bounds,axis=0))<.18
    points,normals=surface_samples(mesh,256,seed=7)
    origins=mesh.vertices[mesh.faces[:,0]];n=mesh.face_normals
    plane=np.einsum('pfi,fi->pf',points[:,None,:3]-origins[None],n).max(axis=1)
    assert np.all(plane>=.1-1e-9)
    assert points.shape==(256,4) and np.allclose(points[:,3],.09)
    assert np.allclose(np.linalg.norm(normals,axis=1),1)


@pytest.mark.parametrize('lower,upper',[(np.array([]),np.array([])),([0],[0]),([0,4],[4,4])])
def test_empty_environment_never_becomes_successful_empty_cache(lower,upper):
    with pytest.raises(SeedInitializationError):validate_surface_pool(np.zeros((4,3)),lower,upper,2)


def test_finite_seed_shape_and_environment_pool():
    assert validate_surface_pool(np.zeros((9,3)),[0,4],[4,9],2)==[4,5]
    assert validate_seed_tensor(np.zeros((256,1,8)),1,256,8).shape==(256,1,8)
    with pytest.raises(SeedInitializationError):validate_seed_tensor(np.empty((0,1,8)),1,256,8)
    with pytest.raises(SeedInitializationError):validate_seed_tensor(np.full((256,1,8),np.nan),1,256,8)
