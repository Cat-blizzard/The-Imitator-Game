"""Portable geometry catalog. Raw asset search and qualification are separate."""
from pathlib import Path
import re
import time
import numpy as np
import trimesh

from .common import ROOT, binding, read, resolve, sha, write


def key(value):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)[:72] + "_" + __import__("hashlib").sha256(value.encode()).hexdigest()[:8]


def index_raw(source, output):
    import sqlite3
    output = Path(output)
    if output.exists():
        raise ValueError("Index exists; choose a new version")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with sqlite3.connect(output) as db:
        db.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, source TEXT, name TEXT, categories TEXT, license TEXT, path TEXT, exists_now INTEGER, physics_qualified INTEGER)")
        for line in Path(source).read_text().splitlines():
            r = __import__("json").loads(line)
            path = r.get("primary_path")
            db.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?)", (r["asset_id"], r["source"], r.get("name"), str(r.get("categories")), r.get("license"), path, int(bool(path and Path(path).is_file())), 0))
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        db.execute("CREATE INDEX source_idx ON assets(source)")
    write(output.with_suffix(".json"), {"source": binding(source), "counts": counts, "physics_qualified_inferred": False, "runtime_dependency": False})
    return counts


def _stable_cook(mesh, directory, number):
    """Bounded CPU PhysX cooking; cache the geometry the engine actually uses."""
    import sapien
    material = sapien.physx.PhysxMaterial(.5, .5, 0.)
    passes = []
    path = directory / f"hull_{number}_source.obj"
    mesh.export(path)
    for iteration in range(3):
        shape = sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(path), scale=[1., 1., 1.], material=material)
        cooked = trimesh.Trimesh(vertices=np.asarray(shape.vertices), faces=np.asarray(shape.triangles), process=False)
        distance = np.linalg.norm(mesh.vertices[:, None] - cooked.vertices[None, :], axis=-1)
        error = float(max(distance.min(0).max(), distance.min(1).max()))
        new = directory / f"hull_{number}_cooked_{iteration}.obj"
        cooked.export(new)
        passes.append({"input": binding(path), "output": binding(new), "vertex_error_m": error})
        path, mesh = new, cooked
        if error < 1e-6:
            return binding(path), passes
    raise ValueError("PhysX cooking failed to converge within 3 passes")


def _prepare(row, selection, directory):
    start = time.monotonic()
    _, category, size, orientation, partition = selection
    source = Path(row["primary_path"])
    scene = trimesh.load(source, force="scene", process=False)
    if not len(scene.geometry):
        raise ValueError("Empty geometry")
    mesh = scene.to_geometry()
    if not np.isfinite(mesh.vertices).all() or min(mesh.extents) < 1e-9:
        raise ValueError("Degenerate or nonfinite geometry")
    transform = np.eye(4)
    stable = []
    if orientation == "stable":
        poses, probabilities = trimesh.poses.compute_stable_poses(mesh.convex_hull, n_samples=1, sigma=0., threshold=.01)
        if not len(poses):
            raise ValueError("No stable support pose")
        transform = poses[int(np.argmax(probabilities))]
        stable = [{"source_pose": p.tolist(), "geometric_probability": float(q)} for p, q in zip(poses, probabilities)]
    elif orientation != "z_up":
        raise ValueError("Explicit support orientation required")
    mesh.apply_transform(transform)
    scale = size / float(mesh.extents.max())
    metric = np.diag([scale, scale, scale, 1.])
    transform = metric @ transform
    mesh.apply_scale(scale)
    shift = -mesh.bounds.mean(0)
    transform[:3, 3] += shift
    scene.apply_transform(transform)
    mesh = scene.to_geometry()
    # Preserve original textures and scene transforms in a self-contained GLB.
    visual = directory / "visual.glb"
    scene.export(visual)
    visual_check = trimesh.load(visual, force="scene", process=False).to_geometry()
    if not np.allclose(visual_check.bounds, mesh.bounds, atol=1e-6):
        raise ValueError("Visual export changed metric bounds")
    # Collision approximation is explicit; cavities are not silently filled for
    # manipulation. Near-convex solids retain a hull with measured surface error.
    hull = mesh.convex_hull
    pts, _ = trimesh.sample.sample_surface(mesh, 1600, seed=17)
    _, distance, _ = trimesh.proximity.closest_point(hull, pts)
    hull_error = float(np.max(distance))
    if hull_error <= .002:
        parts = [hull]
        collision_policy = "convex_hull_surface_error_le_2mm"
    else:
        import coacd
        coacd.set_log_level("error")
        collision_input = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
        if len(collision_input.faces) > 6000:
            collision_input = collision_input.simplify_quadric_decimation(face_count=6000)
        parts = [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in coacd.run_coacd(
            coacd.Mesh(collision_input.vertices, collision_input.faces), threshold=.04,
            max_convex_hull=16, resolution=1500, mcts_nodes=10, mcts_iterations=30,
            mcts_max_depth=3, max_ch_vertex=128, seed=0)]
        collision_policy = "coacd_1.0.7_compound_convex_candidate"
    if not parts or len(parts) > 16:
        raise ValueError("Collision part budget exceeded")
    collisions, cooking = [], []
    for i, part in enumerate(parts):
        b, passes = _stable_cook(part, directory, i)
        collisions.append(b); cooking.append(passes)
    cooked = [trimesh.load(resolve(x), force="mesh", process=False) for x in collisions]
    union = trimesh.util.concatenate(cooked)
    # Measure visual->collider and collider->visual, not just box extents.
    visual_points, _ = trimesh.sample.sample_surface(mesh, 1800, seed=71)
    distances = np.min([trimesh.proximity.closest_point(p, visual_points)[1] for p in cooked], axis=0)
    native_points, _ = trimesh.sample.sample_surface(union, 1800, seed=73)
    back = trimesh.proximity.closest_point(mesh, native_points)[1]
    surface = {"sample_count_each_direction": 1800,
               "visual_to_collision_p95_m": float(np.quantile(distances, .95)),
               "visual_to_collision_max_m": float(distances.max()),
               "collision_to_visual_p95_m": float(np.quantile(back, .95)),
               "collision_to_visual_max_m": float(back.max())}
    geometry_ok = max(surface["visual_to_collision_p95_m"], surface["collision_to_visual_p95_m"]) <= .003
    volume = float(sum(p.volume for p in cooked))
    if volume <= 0:
        raise ValueError("Nonpositive convex volume")
    value = {"schema": "resimpair_mesh_asset_v1", "id": key(row["asset_id"]), "source_id": row["asset_id"],
        "name": row["name"], "category": category, "partition": partition,
        "provenance": {"source": row["source"], "original_sha256": sha(source), "license": row.get("license"),
                       "original_location_for_provenance_only": str(source)},
        "visual": binding(visual), "collisions": collisions, "cooking": cooking,
        "source_to_asset": transform.tolist(), "stable_pose_candidates": stable,
        "support_pose": "baked asset frame +Z normal; actor origin is visual AABB center",
        "dimensions_m": mesh.extents.tolist(), "half_size": (mesh.extents / 2).tolist(),
        "bounds_m": mesh.bounds.tolist(), "collision_bounds_m": union.bounds.tolist(),
        "size_provenance": "explicit tabletop engineering scale, not measured metric ground truth",
        "design_longest_m": size, "mass_kg": .08, "density_kg_m3": .08 / volume,
        "mass_provenance": "explicit engineering assumption, no real mass inferred",
        "collision_policy": collision_policy, "surface_comparison": surface,
        "roles": {"static_distractor": True, "rigid_manipulation_candidate": geometry_ok,
                  "container_support": False, "articulated": False},
        "qualification": {"geometry": "pass" if geometry_ok else "fail", "grasp_proposals": "unknown",
                          "physical_execution": "unknown", "training_admitted": False},
        "elapsed_seconds": time.monotonic() - start}
    write(directory / "descriptor.json", value)
    return binding(directory / "descriptor.json")


def prepare_one(index_path, selection, output):
    rows = {r["asset_id"]: r for r in map(__import__("json").loads, Path(index_path).read_text().splitlines())}
    source_id = selection[0]
    if source_id not in rows:
        raise ValueError("Asset not found: " + source_id)
    directory = Path(output) / key(source_id)
    if (directory / "descriptor.json").exists():
        return binding(directory / "descriptor.json")
    if directory.exists():
        raise ValueError("Incomplete import preserved; use a new output version")
    directory.mkdir(parents=True)
    write(directory / "request.json", {"selection": selection, "source_index": binding(index_path)})
    return _prepare(rows[source_id], selection, directory)


def validate(item):
    d = read(resolve(item["asset_descriptor"]))
    if d["schema"] != "resimpair_mesh_asset_v1" or item["half_size"] != d["half_size"]:
        raise ValueError("Descriptor geometry mismatch")
    if item["kind"] == "movable" and not d["roles"]["rigid_manipulation_candidate"]:
        raise ValueError("Asset is distractor-only")
    resolve(d["visual"])
    for x in d["collisions"]:
        resolve(x)
    return d


def add_geometry(builder, item):
    d = validate(item)
    for collider in d["collisions"]:
        builder.add_convex_collision_from_file(str(resolve(collider)), density=d["density_kg_m3"])
    builder.add_visual_from_file(str(resolve(d["visual"])))


def select(catalog, *, role="static_distractor", category=None, max_dimension=None, exclude=()):
    rows = []
    for entry in read(catalog)["assets"]:
        d = read(resolve(entry))
        if d["id"] in exclude or not d["roles"].get(role, False):
            continue
        if category and d["category"] not in category:
            continue
        if max_dimension and max(d["dimensions_m"]) > max_dimension:
            continue
        rows.append((entry, d))
    return rows


def refine_catalog(catalog, output, include_household=True):
    """Re-evaluate union surfaces and fresh-process cooking without overwrites.

    Interior faces between convex parts are not the object's exterior. Keeping
    them in a Hausdorff-like surface statistic falsely penalizes decomposition.
    The 3mm exterior p95 bound and 10um native-vertex bound do not change.
    """
    import sapien
    output = Path(output)
    if output.exists(): raise ValueError("Refinement version already exists")
    output.mkdir(parents=True)
    records = []; changes = []
    material = sapien.physx.PhysxMaterial(.5, .5, 0.)
    for b in read(catalog)["assets"]:
        d = read(resolve(b)); parts = [trimesh.load(resolve(c), force="mesh", process=True) for c in d["collisions"]]
        native_errors = []
        for c, mesh in zip(d["collisions"], parts):
            native = sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(resolve(c)), scale=[1, 1, 1], material=material)
            distance = np.linalg.norm(mesh.vertices[:, None] - np.asarray(native.vertices)[None], axis=-1)
            native_errors.append(float(max(distance.min(0).max(), distance.min(1).max())))
        points = []
        for i, part in enumerate(parts):
            p, face = trimesh.sample.sample_surface(part, 400, seed=71 + i)
            outward = p + part.face_normals[face] * .0001
            keep = np.ones(len(p), bool)
            for j, other in enumerate(parts):
                if i != j: keep &= ~other.contains(outward)
            points.extend(p[keep])
        visual = trimesh.load(resolve(d["visual"]), force="scene", process=True).to_geometry()
        _, distance, _ = trimesh.proximity.closest_point(visual, np.asarray(points))
        exterior_p95 = float(np.quantile(distance, .95))
        native_ok = max(native_errors) < 1e-5
        geometry_ok = native_ok and max(d["surface_comparison"]["visual_to_collision_p95_m"], exterior_p95) <= .003
        v = {**d, "previous_descriptor": b,
            "surface_metric_version": "external_boundary_of_convex_union_v2",
            "surface_comparison": {**d["surface_comparison"], "union_external_p95_m": exterior_p95,
                                   "union_external_samples": len(points)},
            "fresh_process_native_vertex_errors_m": native_errors,
            "roles": {**d["roles"], "static_distractor": native_ok, "rigid_manipulation_candidate": geometry_ok},
            "qualification": {**d["qualification"], "geometry": "pass" if geometry_ok else "fail",
                              "native_collision_replay": "pass" if native_ok else "fail"}}
        path = output / (d["id"] + ".json"); write(path, v); records.append(binding(path))
        changes.append({"id": d["id"], "previous_geometry": d["qualification"]["geometry"],
                        "geometry": v["qualification"]["geometry"], "native_replay": native_ok})
    if include_household:
        from resimpair_runtime import prepare_asset_catalog
        old = read(prepare_asset_catalog())
        for d in old["assets"]:
            folder = output / ("household_" + d["category"]); folder.mkdir()
            scene = trimesh.load(d["visual"], force="scene", process=False)
            shift = -scene.bounds.mean(0)
            scene.apply_translation(shift); visual = folder / "visual.glb"; scene.export(visual)
            mesh = trimesh.load(d["collision"], force="mesh", process=False); mesh.apply_translation(shift)
            try: collision, cooking = _stable_cook(mesh, folder, 0)
            except Exception as exc:
                changes.append({"id": d["category"], "legacy_import_failure": repr(exc)}); continue
            part = trimesh.load(resolve(collision), force="mesh", process=False)
            native = sapien.physx.PhysxCollisionShapeConvexMesh(filename=str(resolve(collision)), scale=[1, 1, 1], material=material)
            distance = np.linalg.norm(part.vertices[:, None] - np.asarray(native.vertices)[None], axis=-1)
            native_ok = max(distance.min(0).max(), distance.min(1).max()) < 1e-5
            value = {"schema": "resimpair_mesh_asset_v1", "id": "household_" + d["category"],
                "source_id": "objaverse:" + d["source_uid"], "name": d["source_name"], "category": d["category"],
                "partition": "shared_distractor", "provenance": {"source": "objaverse", "license": d["license"],
                    "attribution": d.get("attribution"), "original_descriptor": binding(Path(d["visual"]).parent / "metadata.json") if (Path(d["visual"]).parent / "metadata.json").exists() else {"source_record": binding(d["source_record"]["path"])}},
                "visual": binding(visual), "collisions": [collision], "cooking": [cooking],
                "bounds_m": scene.bounds.tolist(), "half_size": (scene.extents / 2).tolist(), "dimensions_m": scene.extents.tolist(),
                "mass_kg": .08, "density_kg_m3": .08 / part.volume,
                "mass_provenance": "unused engineering mass for static distractor",
                "size_provenance": "retained explicitly engineered household sizes",
                "collision_policy": d["collision_policy"], "roles": {"static_distractor": bool(native_ok), "rigid_manipulation_candidate": False, "container_support": False},
                "qualification": {"geometry": "static_only", "native_collision_replay": "pass" if native_ok else "fail",
                                  "physical_execution": "unknown", "training_admitted": False}}
            path = folder / "descriptor.json"; write(path, value); records.append(binding(path))
    result = {"schema": "resimpair_mesh_catalog_v1", "assets": records, "prior_catalog": binding(catalog),
        "prior_import_failures": read(catalog)["failures"], "refinement_decisions": changes,
        "physical_execution_qualified_count": 0, "api_calls": 0,
        "metric_correction": "Compare exterior of collider union, preserve internal-face statistic and original failures"}
    write(output / "catalog.json", result)
    return {"catalog": str(output / "catalog.json"), "assets": len(records)}
