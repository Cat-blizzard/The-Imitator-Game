"""Read native link collision geometry without reassigning an articulation pose.

The retained ManiSkill CPU getter self-assigns the root pose before reading.
On this installation that changes fixed-base z by a float32 quantum. The
assembly below preserves its link/face/transform logic and removes that write.
CPU simulation has already stepped before planning; no state reset is used.
"""
from contextlib import contextmanager


def assemble_collision_meshes(articulation, to_world_frame=True, first_only=False,
                              component_meshes=None, merge=None):
    if articulation.merged:
        raise ValueError('merged_articulation_geometry_not_supported')
    if articulation.scene.gpu_sim_enabled:
        raise ValueError('read_only_native_mesh_candidate_is_CPU_only')
    if component_meshes is None or merge is None:
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes
        component_meshes, merge = get_component_meshes, merge_meshes
    meshes = []
    for i, art in enumerate(articulation._objs):
        art_meshes = []
        for link in art.links:
            mesh = merge(component_meshes(link))
            if mesh is not None:
                if to_world_frame:
                    pose = articulation.links[link.index].pose[i]
                    mesh.apply_transform(pose.sp.to_transformation_matrix())
                art_meshes.append(mesh)
        mesh = merge(art_meshes)
        if mesh is not None: meshes.append(mesh)
        if first_only: break
    if not meshes: return []
    return meshes[0] if first_only else meshes


@contextmanager
def install():
    from mani_skill.utils.structs.articulation import Articulation
    original = Articulation.get_collision_meshes
    Articulation.get_collision_meshes = assemble_collision_meshes
    try:
        yield
    finally:
        Articulation.get_collision_meshes = original
