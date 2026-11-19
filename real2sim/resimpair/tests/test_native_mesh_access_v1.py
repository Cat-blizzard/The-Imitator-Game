from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from resimpair_pickplace.native_mesh_access_v1 import assemble_collision_meshes


class NativeArticulation:
    def __init__(self): self.links = [SimpleNamespace(index=0), SimpleNamespace(index=1)]
    @property
    def pose(self): return np.eye(4)
    @pose.setter
    def pose(self, value): raise AssertionError('mesh reading must not assign root pose')


def test_exact_native_link_geometry_without_pose_self_assignment():
    transforms = [np.eye(4), np.eye(4)]
    transforms[0][:3, 3] = [.6, .2, .1]; transforms[1][:3, 3] = [-.1, .5, .3]
    native = NativeArticulation()
    links = [SimpleNamespace(pose=[SimpleNamespace(sp=SimpleNamespace(to_transformation_matrix=lambda T=T: T))])
             for T in transforms]
    wrapper = SimpleNamespace(merged=False, scene=SimpleNamespace(gpu_sim_enabled=False), _objs=[native], links=links)
    parts = [trimesh.creation.box([.03, .04, .05]), trimesh.creation.icosphere(subdivisions=1, radius=.02)]
    def components(link): return [parts[link.index].copy()]
    def merge(meshes): return trimesh.util.concatenate(meshes) if meshes else None
    actual = assemble_collision_meshes(wrapper, first_only=True, component_meshes=components, merge=merge)
    expected_parts = [m.copy().apply_transform(T) for m, T in zip(parts, transforms)]
    expected = trimesh.util.concatenate(expected_parts)
    np.testing.assert_array_equal(actual.vertices, expected.vertices)
    np.testing.assert_array_equal(actual.faces, expected.faces)
    wrapper.scene.gpu_sim_enabled = True
    with pytest.raises(ValueError, match='CPU_only'):
        assemble_collision_meshes(wrapper, component_meshes=components, merge=merge)
