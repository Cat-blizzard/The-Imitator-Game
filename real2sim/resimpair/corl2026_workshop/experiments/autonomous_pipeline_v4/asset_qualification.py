"""Frozen additional geometry gate for active meshes; no task threshold changes.

This does not promote the static catalog or replace measured grasp/support gates.
"""
import numpy as np
import trimesh

from .core import checked, read, require
from .active_assets import validate_object


def qualify(bundle):
    objects = [o for o in bundle["scene"]["objects"] if o["shape"] == "convex_mesh"]
    if not objects:
        return None
    evidence = read(checked(bundle["native_asset_evidence"]))
    rows = evidence["records"]
    require(len(rows) == len(objects) and {r["instance_id"] for r in rows} == {o["id"] for o in objects},
            "active_mesh_native_identity_mismatch")
    checks = []
    for obj in objects:
        spec = validate_object(obj)
        row = next(r for r in rows if r["instance_id"] == obj["id"])
        require(row["descriptor"] == obj["asset_descriptor"], "active_mesh_descriptor_binding_changed")
        require(len(row["shapes"]) == 1, "active_mesh_collision_shape_count_changed")
        shape = row["shapes"][0]
        require(shape["type"] == "PhysxCollisionShapeConvexMesh", "active_mesh_replaced_by_proxy")
        mesh = trimesh.load(checked(spec["collision"]), force="mesh", process=False)
        expected = np.asarray(mesh.vertices) + spec["mesh_local_translation"]
        matrix = np.asarray(shape["local_transform"])
        actual = (np.asarray(shape["vertices"]) * shape["scale"]) @ matrix[:3, :3].T + matrix[:3, 3]
        distances = np.linalg.norm(actual[:, None] - expected[None, :], axis=-1)
        error = float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
        expected_com = mesh.center_mass + spec["mesh_local_translation"]
        com_error = float(np.linalg.norm(np.asarray(row["center_of_mass_local"]) - expected_com))
        expected_inertia = np.sort(np.linalg.eigvalsh(mesh.moment_inertia * spec["density_kg_m3"]))
        inertia_error = float(np.max(np.abs(np.sort(row["inertia_kg_m2"]) - expected_inertia) / expected_inertia))
        density_error = abs(shape["density_kg_m3"] - spec["density_kg_m3"]) / spec["density_kg_m3"]
        passed = (error < 1e-5 and abs(row["mass_kg"] - spec["mass_kg"]) < 1e-4
                  and com_error < 1e-4 and inertia_error < .02 and density_error < 1e-4
                  and np.isfinite(row["inertia_kg_m2"]).all() and (np.asarray(row["inertia_kg_m2"]) > 0).all())
        checks.append({"instance_id": obj["id"], "passed": bool(passed), "cooked_mesh_vertex_error_m": error,
                       "mass_kg": row["mass_kg"], "center_of_mass_error_m": com_error,
                       "principal_inertia_relative_error": inertia_error, "density_relative_error": density_error,
                       "descriptor": row["descriptor"]})
    return {"passed": all(r["passed"] for r in checks), "checks": checks,
            "native_evidence": bundle["native_asset_evidence"],
            "scope": "declared convex collision geometry and engineering dynamics; task/contact acceptance is separately required"}
