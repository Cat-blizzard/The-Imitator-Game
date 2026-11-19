"""Opt-in mesh manipulation candidates, separate from qualified production assets.

Reuses the retained catalog/file selector. No category-name match promotes an
asset, no AABB proxy qualifies as active contact geometry, no real mass is inferred.
"""
from pathlib import Path
import numpy as np
import trimesh

from .core import bind, checked, read, require, write

VERSION = "resimpair_active_mesh_candidate_v1"
COOKED_VERSION = "resimpair_active_mesh_candidate_v2"


def descriptor(catalog, category, *, mass_kg):
    from data_generation_methods.asset_library.task_asset_selection import load_static_catalog, normalized
    rows = [a for a in load_static_catalog(catalog) if normalized(a["category"]) == normalized(category)]
    require(len(rows) == 1, "active_candidate_requires_one_exact_catalog_match")
    a = rows[0]
    require(a["collision_policy"] == "whole_object_convex_hull", "active_contact_rejects_AABB_proxy")
    mesh = trimesh.load(a["collision"], force="mesh", process=False)
    bounds = mesh.bounds
    extents = np.ptp(bounds, axis=0)
    require(mesh.is_convex and mesh.is_watertight and 4 <= len(mesh.vertices) <= 255 and mesh.volume > 0,
            "active_candidate_requires_small_watertight_convex_mesh")
    require(np.allclose(bounds, a["bounds_rotated_m"], atol=1e-6, rtol=0), "catalog_collision_bounds_changed")
    require(extents[2] >= .04, "vertical_pinch_insufficient_table_clearance")
    require(min(extents[:2]) <= .077, "active_candidate_exceeds_gripper_opening")
    require(np.isfinite(mass_kg) and .01 <= mass_kg <= .5, "explicit_bounded_engineering_mass_required")
    return {"schema": VERSION, "source_catalog": bind(catalog), "source_asset_uid": a["uid"],
            "category": a["category"], "source_intended_use": a["intended_use"],
            "visual": {"path": a["visual"], "sha256": a["files"][a["visual"]]},
            "collision": {"path": a["collision"], "sha256": a["files"][a["collision"]]},
            "collision_policy": a["collision_policy"], "bounds_m": bounds.tolist(),
            "half_size": (extents / 2).tolist(), "mesh_local_translation": (-bounds.mean(axis=0)).tolist(),
            "runtime_scale": [1., 1., 1.], "collision_volume_m3": float(mesh.volume),
            "mass_kg": float(mass_kg), "density_kg_m3": float(mass_kg / mesh.volume),
            "mass_provenance": "declared engineering assumption; not measured real-world mass",
            "metric_scale_provenance": "retained metric design frame; not measured real-world scale",
            "grasp_policy": "vertical_pinch_below_top", "grasp_depth_m": min(.04, float(extents[2] / 2)),
            "mechanism_scope": "rigid_pick_lift_place_release_only_no_opening_or_deformation",
            "status": "candidate_requires_native_qualification", "physics_qualified": False, "training_admitted": False}


def validate_descriptor(binding):
    value = read(checked(binding))
    require(value.get("schema") in {VERSION, COOKED_VERSION}, "unsupported_active_asset_descriptor")
    expected = descriptor(checked(value["source_catalog"]), value["category"], mass_kg=value["mass_kg"])
    if value["schema"] == COOKED_VERSION:
        expected = cooked_descriptor(expected, value["cooking_report"])
    require(value == expected, "active_descriptor_differs_from_catalog_geometry_or_policy")
    checked(value["visual"])
    checked(value["collision"])
    return value


def cooked_descriptor(original, report_binding):
    """Declare a stable engine-prepared collider; retain source approximation error.

    This changes the input collider, not the native geometry acceptance tolerance.
    The original visual, source hull and failed executions remain immutable.
    """
    report = read(checked(report_binding))
    require(report["source_collision"] == original["collision"], "cooking_source_geometry_changed")
    require(1 <= len(report["passes"]) <= 3 and report["stable"] is True, "bounded_stable_cooking_required")
    previous = original["collision"]
    for row in report["passes"]:
        require(row["input"] == previous, "cooking_chain_disconnected")
        checked(row["input"])
        checked(row["output"])
        previous = row["output"]
    require(previous == report["declared_collision"], "cooking_result_binding_mismatch")
    mesh = trimesh.load(checked(previous), force="mesh", process=False)
    source = trimesh.load(checked(original["collision"]), force="mesh", process=False)
    require(mesh.is_watertight and mesh.is_convex and mesh.volume > 0, "cooked_collision_not_convex")
    # Measure geometric surfaces, not nearest vertices, for source approximation.
    _, forward, _ = trimesh.proximity.closest_point_naive(mesh, source.vertices)
    _, backward, _ = trimesh.proximity.closest_point_naive(source, mesh.vertices)
    surface_error = float(max(forward.max(), backward.max()))
    volume_error = abs(float(mesh.volume / source.volume - 1))
    require(surface_error <= .001 and volume_error <= .01, "cooking_changed_source_shape_too_much")
    bounds = mesh.bounds
    return {**original, "schema": COOKED_VERSION, "source_collision": original["collision"],
            "collision": previous, "cooking_report": report_binding,
            "runtime_collision_policy": "prebaked_stable_PhysX_convex_hull",
            "source_collision_surface_error_m": surface_error, "source_collision_relative_volume_error": volume_error,
            "bounds_m": bounds.tolist(), "half_size": (np.ptp(bounds, axis=0) / 2).tolist(),
            "mesh_local_translation": (-bounds.mean(axis=0)).tolist(),
            "collision_volume_m3": float(mesh.volume), "density_kg_m3": float(original["mass_kg"] / mesh.volume)}


def prepare_cooked(catalog, category, *, mass_kg, output):
    import sapien
    output = Path(output)
    require(not output.exists(), "cooked_candidate_already_exists_no_overwrite")
    original = descriptor(catalog, category, mass_kg=mass_kg)
    output.mkdir(parents=True)
    material = sapien.physx.PhysxMaterial(.5, .5, 0.)
    previous, passes = original["collision"], []
    stable = False
    for index in range(3):
        shape = sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(checked(previous)),
                                                         scale=[1., 1., 1.], material=material)
        mesh = trimesh.Trimesh(vertices=np.asarray(shape.vertices), faces=np.asarray(shape.triangles), process=False)
        source = trimesh.load(checked(previous), force="mesh", process=False)
        distances = np.linalg.norm(source.vertices[:, None] - mesh.vertices[None, :], axis=-1)
        error = float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))
        target = output / ("cooked_%d.obj" % index)
        mesh.export(target)
        current = bind(target)
        passes.append({"input": previous, "output": current, "vertex_error_m": error,
                       "input_vertices": len(source.vertices), "output_vertices": len(mesh.vertices)})
        previous = current
        if error < 1e-6:
            stable = True
            break
    report = {"schema": "resimpair_bounded_mesh_cooking_v1", "source_collision": original["collision"],
              "declared_collision": previous, "passes": passes, "stable": stable,
              "max_passes": 3, "physics_steps": 0, "scope": "CPU collision cooking, no environment or motion execution"}
    report_path = output / "cooking_report.json"
    write(report_path, report)
    result = cooked_descriptor(original, bind(report_path))
    write(output / "descriptor.json", result)
    return bind(output / "descriptor.json")


def object_spec(binding, instance_id, xy, yaw=0.):
    d = validate_descriptor(binding)
    return {"id": instance_id, "kind": "movable", "shape": "convex_mesh", "asset_descriptor": binding,
            "half_size": d["half_size"], "mass_kg": d["mass_kg"], "yaw": float(yaw),
            "position": [float(xy[0]), float(xy[1]), d["half_size"][2] + .0005],
            "rgba": [.5, .5, .5, 1.], "color_name": "original asset material",
            "semantic_description": "the " + d["category"].replace("_", " ") + " with its original mesh and material"}


def validate_object(item):
    d = validate_descriptor(item["asset_descriptor"])
    require(item["shape"] == "convex_mesh" and item["kind"] == "movable", "active_descriptor_requires_dynamic_mesh")
    require(item["half_size"] == d["half_size"] and item["mass_kg"] == d["mass_kg"], "active_object_geometry_or_mass_changed")
    return d


def export_native(rec, path):
    """Export actual cooked PhysX shapes, not just the requested file paths."""
    records = []
    for actor, item in zip(rec.unwrapped.objects, rec.unwrapped.scene_spec["objects"]):
        if item["shape"] != "convex_mesh":
            continue
        body = actor._bodies[0]
        shapes = []
        for shape in body.get_collision_shapes():
            shapes.append({"type": type(shape).__name__, "vertices": np.asarray(shape.vertices).tolist(),
                           "scale": np.asarray(shape.scale).tolist(),
                           "local_transform": np.asarray(shape.local_pose.to_transformation_matrix()).tolist(),
                           "density_kg_m3": float(shape.density)})
        records.append({"instance_id": item["id"], "descriptor": item["asset_descriptor"],
                        "mass_kg": float(body.mass), "inertia_kg_m2": np.asarray(body.inertia).tolist(),
                        "center_of_mass_local": np.asarray(body.cmass_local_pose.p).tolist(), "shapes": shapes})
    if records:
        write(path, {"schema": "resimpair_native_active_mesh_v1", "records": records})
        return bind(path)
    return None
