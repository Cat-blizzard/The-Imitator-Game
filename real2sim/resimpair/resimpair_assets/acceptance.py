"""Static eligibility rules, bound before import; independent of the preparer.

A conservative box is allowed only for noninteractive static clutter. It does
not certify a manipulable mesh, a container, articulation or visual semantics.
"""
import numpy as np
import trimesh
from resimpair_grasp.common import binding, read, resolve

VERSION = "static_conservative_proxy_v1"


def evaluate(path):
    import sapien
    d = read(path)
    if d["collision_policy"] != "static_only_conservative_AABB": raise ValueError("Unexpected static policy")
    if d["roles"]["rigid_manipulation_candidate"] or d["roles"]["container_support"]: raise ValueError("Static proxy cannot acquire manipulation roles")
    scene = trimesh.load(resolve(d["visual"]), force="scene", process=False)
    mesh = scene.to_geometry()
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces): raise ValueError("Invalid visual mesh")
    if len(mesh.faces) > 160000: raise ValueError("Visual face budget exceeded")
    extent = mesh.extents
    if extent.min() < .002 or extent.max() > .20 or extent.max() / extent.min() > 35:
        raise ValueError("Static tabletop size/aspect limit")
    if not np.allclose(mesh.bounds, d["bounds_m"], atol=1e-6): raise ValueError("Visual bounds differ from declaration")
    if not np.allclose(extent, d["dimensions_m"], atol=1e-6) or not np.allclose(extent / 2, d["half_size"], atol=1e-6):
        raise ValueError("Dimension mismatch")
    if not np.allclose(mesh.bounds.mean(0), 0., atol=1e-6): raise ValueError("Origin not at declared center")
    if abs(extent.max() - d["design_longest_m"]) > 1e-6: raise ValueError("Engineering size changed")
    if len(d["collisions"]) != 1: raise ValueError("Static proxy must have one declared box")
    collision_path = resolve(d["collisions"][0]); hull = trimesh.load(collision_path, force="mesh", process=False)
    if len(hull.vertices) != 8 or not np.isclose(hull.volume, np.prod(extent), rtol=1e-4):
        raise ValueError("Declared proxy is not a solid bounding box")
    if not np.allclose(hull.bounds, mesh.bounds, atol=1e-6): raise ValueError("Conservative proxy bounds changed")
    # A box must cover the full visual extent, not just a convenient sample.
    if not np.all((mesh.vertices >= hull.bounds[0] - 1e-6) & (mesh.vertices <= hull.bounds[1] + 1e-6)):
        raise ValueError("Proxy fails to enclose visual geometry")
    material = sapien.physx.PhysxMaterial(.5, .5, 0.)
    native = sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(collision_path), scale=[1., 1., 1.], material=material)
    vertices = np.asarray(native.vertices)
    distances = np.linalg.norm(hull.vertices[:, None] - vertices[None], axis=-1)
    error = float(max(distances.min(0).max(), distances.min(1).max()))
    if error >= 1e-5: raise ValueError("Fresh-process native collision mismatch")
    return {"version": VERSION, "rule": binding(__file__), "passed": True,
            "native_vertex_error_m": error, "visual_faces": len(mesh.faces), "visual_meshes": len(scene.geometry),
            "scope": "eligible static clutter; no contact affordance or visual-semantic confirmation"}
