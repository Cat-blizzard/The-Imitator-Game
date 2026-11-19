"""MJCF per-author-part convex-hull semantics, explicit v2 CPU intake.

Retains authored metric units, component transforms and all original meshes.
Convexification is a recorded geometry interpretation, never a surface-gate
relaxation. Native v6 and its physical/geometry rules remain frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import trimesh

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write
from . import raw_robocasa_xml_v1 as xml
from . import raw_mesh_io_v1 as visual_io
from . import raw_size_policy_v1 as retained_policy
from . import metric_variants_v2 as variants
from . import metric_native_v6 as native
from . import metric_admission_v1 as admission
from .metric_semantics_v1 import profile
from .raw_manipulation_v1 import completed_settle_steps
from .raw_metric_containers_v1 import identity, campaign_lock, sufficient_disk

from . import raw_robocasa_native_v1 as base
from .raw_robocasa_native_v1 import authored_size_decision, freeze_source_inputs

VERSION = "MJCF_author_parts_convex_hull_native_v2"
DEFAULT_REGISTRY = ROOT / "runs/raw_robocasa_native_registry_v1"


def _probes(mesh, limit=20000):
    """Deterministic diagnostics only; never simplify collision input vertices."""
    points = np.vstack([mesh.vertices, mesh.triangles.mean(1), mesh.vertices[mesh.edges_unique].mean(1)])
    if len(points) > limit:
        return points[np.linspace(0, len(points) - 1, limit, dtype=int)], len(points)
    return points, len(points)


def load_author_convex_collision_parts(receipt):
    """Each MJCF collision mesh remains one convex shape over its own vertices.

    Original triangles/vertices and transforms stay bound. The explicit hull is
    the engine's collision interpretation, not a substitute visual model. Its
    difference from the source triangulation is measured and recorded; only the
    unchanged native qualification decides whether the complete object passes.
    """
    declarations = [g for g in receipt["geoms"] if g["kind"] == "collision"]
    if not 1 <= len(declarations) <= 128:
        raise ValueError("one_to_128_explicit_author_collision_parts_required")
    parts, rows = [], []
    for index, item in enumerate(declarations):
        original = trimesh.load(resolve(item["mesh_file"]), force="mesh", process=True,
                                skip_materials=True, resolver={})
        if (not isinstance(original, trimesh.Trimesh) or len(original.vertices) < 4
                or not len(original.faces) or not np.isfinite(original.vertices).all()):
            raise ValueError("finite_authored_collision_mesh_vertices_and_triangles_required")
        transformed = original.copy(); transformed.apply_transform(np.asarray(item["mesh_to_world"], float))
        hull = transformed.convex_hull
        if (not hull.is_watertight or not hull.is_winding_consistent
                or not np.isfinite(hull.vertices).all() or hull.volume <= 0):
            raise ValueError("positive_closed_author_part_convex_hull_required")
        points, full_count = _probes(transformed)
        _, distances, _ = trimesh.proximity.closest_point(hull, points)
        reverse, full_reverse_count = _probes(hull)
        _, reverse_distances, _ = trimesh.proximity.closest_point(transformed, reverse)
        bounds_delta = float(np.max(abs(hull.bounds - transformed.bounds)))
        if bounds_delta > 1e-10:
            raise ValueError("per_author_part_convex_hull_changed_original_bounds")
        rows.append({"index": index, "source_mesh": item["mesh_file"],
            "source_mesh_to_XML_world": item["mesh_to_world"],
            "source_declaration": item["mesh_declaration"], "geom_declaration": item["geom_attributes"],
            "original_vertices": len(original.vertices), "original_faces": len(original.faces),
            "original_watertight": bool(original.is_watertight),
            "original_winding_consistent": bool(original.is_winding_consistent),
            "original_convex_boolean_diagnostic_only": bool(original.is_convex),
            "original_transformed_signed_volume_m3": float(transformed.volume),
            "volume_m3": float(hull.volume), "vertices": len(hull.vertices), "faces": len(hull.faces),
            "convexification_delta": {"bounds_max_abs_m": bounds_delta,
                "volume_difference_m3": float(hull.volume - transformed.volume),
                "source_to_hull_max_probe_distance_m": float(max(distances)),
                "source_to_hull_p95_probe_distance_m": float(np.percentile(distances, 95)),
                "hull_to_source_max_probe_distance_m": float(max(reverse_distances)),
                "hull_to_source_p95_probe_distance_m": float(np.percentile(reverse_distances, 95)),
                "forward_probe_count": len(points), "forward_possible_points": full_count,
                "reverse_probe_count": len(reverse), "reverse_possible_points": full_reverse_count,
                "probe_definition": "vertices, triangle centres and unique edge midpoints; deterministic linspace cap 20000",
                "complete_Hausdorff_proof": False, "native_surface_gate_still_required": True},
            "processing": "one explicit convex hull per original author collision mesh; no merging of author groups, no visual hull, no size normalization"})
        parts.append(hull)
    return parts, {"geometry_policy": VERSION, "parts": rows,
        "compound_properties": variants.retained.compound_properties(parts),
        "mass_volume_model": "sum of original author-group convex-hull volumes; overlaps not union-corrected",
        "author_collision_grouping_preserved": True, "original_meshes_remain_bound": True,
        "new_decomposition": False, "explicit_MJCF_convex_interpretation": True,
        "native_acceptance_changed": False}

def compile_raw(raw_row, output, *, seed=20261031, license=None, source_aliases=(), prior_source_job=None):
    """One CPU native attempt in a fresh output, preserving author scale."""
    began = time.monotonic(); output = Path(output).resolve()
    if output.exists():
        raise ValueError("fresh_RoboCasa_native_output_required_no_retry")
    output.mkdir(parents=True)
    write(output / "INPUT.json", {"raw_row": raw_row, "seed": seed, "source_aliases": list(source_aliases),
        "prior_source_job": prior_source_job, "implementation": binding(__file__)})
    scene, mesh, receipt = xml.load_raw_scene(raw_row)
    write(output / "raw_input.json", receipt)
    original_obb = mesh.convex_hull.bounding_box_oriented.primitive.extents
    decision = authored_size_decision(raw_row, original_obb, receipt, seed)
    write(output / "size_decision.json", decision)
    if decision["role"] != "manipulation":
        return {"source_id": raw_row["id"], "status": "source_role_requires_other_use",
                "decision": binding(output / "size_decision.json"), "robot_physical_executions": 0,
                "elapsed_seconds": time.monotonic() - began, "recorded_settle_steps": 0}
    parts, collision_receipt = load_author_convex_collision_parts(receipt)
    write(output / "author_collision_import.json", collision_receipt)
    properties = collision_receipt["compound_properties"]
    hull = trimesh.util.concatenate(parts).convex_hull
    if len(hull.faces) > 400:
        support_hull = hull.simplify_quadric_decimation(face_count=400).convex_hull
    else:
        support_hull = hull
    poses, probabilities = trimesh.poses.compute_stable_poses(support_hull,
        center_mass=np.asarray(properties["center_of_mass_m"]), n_samples=1, sigma=0., threshold=.005)
    if not len(poses):
        raise ValueError("no_author_collision_COM_geometric_support_pose")
    transform = poses[int(np.argmax(probabilities))].copy()
    visual = scene.copy(); visual.apply_transform(transform)
    shift = -visual.bounds.mean(0); transform[:3, 3] += shift; visual.apply_translation(shift)
    canonical_parts = [part.copy() for part in parts]
    for part in canonical_parts:
        part.apply_transform(transform)
    actual = visual.to_geometry()
    exported = visual_io.export_visual(visual, output / "authored_metric/visual.glb", actual)
    write(output / "authored_metric/material_export.json", exported)
    refs = []
    for index, part in enumerate(canonical_parts):
        path = output / "authored_metric" / f"collision_{index:03d}.obj"
        part.export(path, digits=12); refs.append(binding(path))
    properties = variants.retained.compound_properties(canonical_parts)
    frame = {"source_frame": "composed_XML_world_in_author_metres",
        "source_to_initial_asset": transform.tolist(), "uniform_scale_after_XML_composition": 1.,
        "initial_orientation": "maximum geometric stable support probability using original collision COM",
        "geometric_support_proposal_hull_only": {"original_faces": len(hull.faces), "proposal_faces": len(support_hull.faces),
            "actual_collision_parts_simplified": False},
        "source_components": [{"kind": g["kind"], "source_mesh": g["mesh_file"],
            "component_to_XML_world": g["mesh_to_world"],
            "component_to_initial_asset": (transform @ np.asarray(g["mesh_to_world"])).tolist()} for g in receipt["geoms"]],
        "semantic_upright_axis": "unknown; geometric support orientation is recorded separately",
        "native_canonicalization_may_add_a_rigid_transform": True}
    write(output / "raw_frame.json", frame)
    bounds = np.array([np.min([p.bounds[0] for p in canonical_parts], axis=0),
                       np.max([p.bounds[1] for p in canonical_parts], axis=0)])
    mass = float(properties["collision_shape_volume_sum_m3"] * decision["effective_density_kg_m3"])
    parent = {"schema": "resimpair_mesh_asset_v1", "id": digest([VERSION, raw_row["id"], receipt, decision])[:24],
        "source_id": raw_row["id"], "name": raw_row.get("name", raw_row["id"]),
        "category": decision["category"], "partition": raw_row["partition"],
        "provenance": {"source": "robocasa", "original_sha256": raw_row["source_sha"],
            "raw_input": binding(output / "raw_input.json"), "raw_size_decision": binding(output / "size_decision.json"),
            "source_frame": binding(output / "raw_frame.json"), "original_collision_import": binding(output / "author_collision_import.json"),
            "license": license, "source_aliases": list(source_aliases), "prior_source_job": prior_source_job,
            "ingestion_group": "raw_robocasa_authored_scale_convex_semantics_v2"},
        "visual": binding(output / "authored_metric/visual.glb"), "collisions": refs,
        "source_to_asset": transform.tolist(), "dimensions_m": actual.extents.tolist(),
        "bounds_m": actual.bounds.tolist(), "half_size": (actual.extents / 2).tolist(),
        "collision_bounds_m": bounds.tolist(), "semantic_role": "manipulation", "semantic_category": decision["category"],
        "qualification": {"geometry": "pending", "physical_execution": "unmeasured"},
        "roles": {"rigid_manipulation_candidate": False, "static_distractor": False},
        "mass_kg": mass, "density_kg_m3": decision["effective_density_kg_m3"], "new_grasp_required": True}
    write(output / "authored_metric/parent.json", parent)
    policy = retained_policy.metric_policy(decision, binding(output / "size_decision.json"))
    write(output / "metric_policy.json", policy)
    prepared = variants.materialize(output / "authored_metric/parent.json", output / "metric_policy.json", output / "metric_variant")
    variant = prepared["variants"][0]
    if len(prepared["variants"]) != 1 or variant["relative_scale"] != 1. or not np.isclose(variant["parent_scale_multiplier"], 1., atol=1e-9):
        raise ValueError("authored_dimensions_were_rescaled")
    native_result = native.compile_variant(resolve(variant["descriptor"]), output / "native")
    final_descriptor = read(resolve(native_result["asset"]))
    if not np.isclose(variants.retained.source_scale(final_descriptor["source_to_asset"]), 1., atol=1e-8):
        raise ValueError("native_source_transform_changed_author_scale")
    state = {"source_id": raw_row["id"], "role": "manipulation", "profile": decision["category"],
        "partition": raw_row["partition"], "semantic_admission": "engineered_role_and_scale_candidate",
        "base_longest_m": decision["target_longest_m"], "size_confidence": decision["size_confidence"],
        "policy": binding(output / "metric_policy.json"), "raw_size_decision": binding(output / "size_decision.json")}
    write(output / "source_state.json", state)
    entry = {**native_result, "role": "manipulation", "source_aliases": list(source_aliases),
        "ingestion_group": "raw_robocasa_authored_scale_convex_semantics_v2", "source_state": binding(output / "source_state.json")}
    entry["semantic_scale_admission"] = admission.inspect(entry, state) if native_result["geometry_qualified"] else {
        "passed": False, "reasons": ["native_geometry_or_support_not_qualified"], "current_policy": state["policy"]}
    write(output / "native_entry.json", entry)
    write(output / "final_source_frame.json", {"source_frame": "composed_XML_world_in_author_metres",
        "source_to_final_asset": final_descriptor["source_to_asset"], "original_components": binding(output / "raw_frame.json"),
        "final_asset": native_result["asset"], "uniform_scale_after_XML_composition": 1.,
        "source_provider": "robocasa", "original_author_part_convex_hulls_used_first": True, "geometry_policy": VERSION,
        "retained_native_v6_collision_refinement_triggered": native_result["metric_decomposition_triggered"]})
    return {"source_id": raw_row["id"], "status": "native_ready" if native_result["geometry_qualified"] and entry["semantic_scale_admission"]["passed"] else "native_qualification_failed",
        "entry": binding(output / "native_entry.json"), "state": binding(output / "source_state.json"),
        "size_decision": binding(output / "size_decision.json"), "original_author_collision_groups_imported": len(parts), "geometry_policy": VERSION,
        "author_collision_volume_sum_m3": properties["collision_shape_volume_sum_m3"], "initial_effective_mass_kg": mass,
        "metric_decomposition_triggered": native_result["metric_decomposition_triggered"],
        "recorded_settle_steps": completed_settle_steps(output), "elapsed_seconds": time.monotonic() - began,
        "settle_scope": "asset-only CPU rigid-body support; no robot execution",
        "robot_physical_executions": 0, "GPU_jobs": 0, "API_calls": 0,
        "grasp_qualification": "not_run", "physical_success": "unmeasured"}


def implementation_bindings():
    return [*base.implementation_bindings(), binding(__file__)]


def prepare(root, sources, *, maximum_new_sources, workers=2, source_timeout_seconds=240,
            wall_seconds=1800, registry=DEFAULT_REGISTRY, method_comparisons=()):
    """Freeze v2 inputs; reuse v1 outcomes unless an explicit comparison exists.

    A full 706-source continuation reuses the five untouched v1 outcomes and
    the explicit mug v2 comparison. It allocates only the remaining 700 sources.
    The v1 mug attempt is retained in historical accounting even though the v2
    candidate is selected. A new directory never erases either allocation.
    """
    root, registry = Path(root).resolve(), Path(registry).resolve()
    if root.exists(): raise ValueError("existing_XML_native_campaign_must_resume_not_reset")
    if (not 1 <= len(sources) <= 706 or type(maximum_new_sources) is not int or not 0 <= maximum_new_sources <= 706
            or type(workers) is not int or not 1 <= workers <= 2
            or type(source_timeout_seconds) is not int or not 1 <= source_timeout_seconds <= 240
            or type(wall_seconds) is not int or not 1 <= wall_seconds <= 7 * 86400):
        raise ValueError("explicit_bounded_source_CPU_and_wall_budget_required")
    source_ids = [s["raw_row"]["id"] for s in sources]
    if len(set(source_ids)) != len(source_ids): raise ValueError("duplicate_source_in_XML_campaign")
    comparisons = {c["source_id"]: c for c in method_comparisons}
    if (len(comparisons) != len(method_comparisons) or set(comparisons) - set(source_ids)
            or any(not str(c.get("reason", "")).strip() for c in comparisons.values())):
        raise ValueError("explicit_source_and_reason_required_for_method_comparison")
    registry.mkdir(parents=True, exist_ok=True)
    with campaign_lock(registry):
        planned, historical, new_count = [], {}, 0
        for source in sources:
            row = source["raw_row"]
            if row.get("source") != "robocasa" or binding(row["path"])["sha256"] != row["source_sha"]:
                raise ValueError("bound_original_RoboCasa_source_required")
            key = digest([VERSION, row["id"], row["source_sha"]])[:32]
            legacy_key = digest([base.VERSION, row["id"], row["source_sha"]])[:32]
            claim_path = registry / "claims" / (key + ".json")
            legacy_path = registry / "claims" / (legacy_key + ".json")
            current = read(claim_path) if claim_path.exists() else None
            legacy = read(legacy_path) if legacy_path.exists() else None
            for prior in [legacy, current]:
                if prior:
                    historical[digest(prior["job"])] = prior
            comparison = comparisons.get(row["id"])
            if comparison:
                if not legacy or comparison.get("prior_job") != legacy["job"]:
                    raise ValueError("method_comparison_requires_bound_prior_v1_allocation")
                if not (resolve(legacy["job"]).parent / "terminal.json").exists():
                    raise ValueError("prior_method_must_be_terminal_before_comparison")
            # Existing current-method claims always win and are never repeated.
            prior = current or (legacy if not comparison else None)
            frozen_inputs = freeze_source_inputs(row)
            if prior:
                prior_job = read(resolve(prior["job"]))
                if prior_job["frozen_inputs"] != frozen_inputs:
                    raise ValueError("prior_source_dependencies_changed_no_silent_resubmission")
                if not (Path(prior_job["campaign"]) / "PROTOCOL.json").is_file():
                    raise ValueError("prior_source_claim_requires_recovery_before_new_queue")
            else: new_count += 1
            planned.append((source, key, claim_path, prior, frozen_inputs, comparison))
        if new_count > maximum_new_sources:
            raise ValueError("new_sources_exceed_declared_cumulative_allocation")
        root.mkdir(parents=True); jobs, new_claims = [], []
        for source, key, claim_path, prior, frozen_inputs, comparison in planned:
            if prior:
                prior_job = read(resolve(prior["job"]))
                jobs.append({"job": prior["job"], "owned": False, "source_id": source["raw_row"]["id"],
                    "selected_geometry_policy": prior_job.get("geometry_policy", base.VERSION)})
                continue
            path = root / "jobs" / key / "job.json"
            write(path, {**source, "campaign": str(root), "registry": str(registry), "geometry_policy": VERSION,
                "key": key, "frozen_inputs": frozen_inputs, "method_comparison": comparison,
                "output": str(path.parent / "artifacts"), "seed": source.get("seed", 20261031)})
            jobs.append({"job": binding(path), "owned": True, "source_id": source["raw_row"]["id"],
                         "selected_geometry_policy": VERSION})
            new_claims.append((claim_path, {"job": binding(path), "source_id": source["raw_row"]["id"], "campaign": str(root)}))
        protocol = {"schema": VERSION, "root": str(root), "registry": str(registry), "jobs": jobs,
            "maximum_new_sources": maximum_new_sources, "newly_allocated_sources": new_count,
            "reused_source_claims": len(jobs) - new_count, "maximum_CPU_workers": workers,
            "per_source_timeout_seconds": source_timeout_seconds, "maximum_wall_seconds": wall_seconds,
            "implementation_bindings": implementation_bindings(), "source_retries": 0,
            "explicit_method_comparisons": list(comparisons.values()),
            "historical_allocations_for_selected_sources": list(historical.values()),
            "historical_allocation_count": len(historical),
            "cumulative_allocations_for_selected_sources_including_comparisons": len(historical) + new_count,
            "independent_review_required_before_submission": new_count > 1,
            "authored_metric_scale": 1., "geometry_policy": VERSION,
            "author_collision_grouping_preserved": True,
            "native_v6_bounded_refinement_is_not_source_resubmission": True,
            "new_robot_executions": 0, "GPU_jobs": 0, "API_calls": 0, "default_promoted": False}
        write(root / "PROTOCOL.json", protocol)
        for path, claim in new_claims: write(path, {**claim, "protocol": binding(root / "PROTOCOL.json")})
    return {"protocol": binding(root / "PROTOCOL.json"), "registered": len(jobs), "new_allocations": new_count,
        "reused_prior_allocations": len(jobs) - new_count, "historical_allocation_count": len(historical),
        "cumulative_allocations_for_selected_sources_including_comparisons": len(historical) + new_count}

def _validate(root):
    root = Path(root).resolve(); protocol = read(root / "PROTOCOL.json")
    if protocol["schema"] != VERSION or protocol["root"] != str(root):
        raise ValueError("bound_XML_native_protocol_required")
    if (not 1 <= protocol["maximum_CPU_workers"] <= 2
            or not 1 <= protocol["per_source_timeout_seconds"] <= 240
            or not 1 <= protocol["maximum_wall_seconds"] <= 7 * 86400
            or sum(bool(item["owned"]) for item in protocol["jobs"]) != protocol["newly_allocated_sources"]
            or protocol["newly_allocated_sources"] > protocol["maximum_new_sources"]):
        raise ValueError("frozen_XML_native_budget_inconsistent")
    for ref in protocol["implementation_bindings"]: resolve(ref)
    for item in protocol["jobs"]:
        path = resolve(item["job"]); job = read(path)
        claim = read(Path(protocol["registry"]) / "claims" / (job["key"] + ".json"))
        if (claim["job"] != item["job"] or job["raw_row"]["id"] != item["source_id"]
                or claim["protocol"] != binding(Path(job["campaign"]) / "PROTOCOL.json")
                or (item["owned"] and job["campaign"] != str(root))):
            raise ValueError("cross_directory_source_allocation_changed")
    return protocol


def worker(root, job_path):
    root, job_path = Path(root).resolve(), Path(job_path).resolve()
    protocol = _validate(root); ref = binding(job_path)
    _release_gate(root, protocol)
    if not any(x["job"] == ref and x["owned"] for x in protocol["jobs"]):
        raise ValueError("worker_requires_owned_frozen_source_allocation")
    folder = job_path.parent; job = read(job_path)
    reservation = read(folder / "reservation.json")
    if (reservation["job"] != ref or reservation["attempt"] != 1
            or reservation["protocol"] != binding(root / "PROTOCOL.json")
            or (folder / "worker_started.json").exists()):
        raise ValueError("one_reserved_source_attempt_required_no_retry")
    write(folder / "worker_started.json", {"job": ref, "identity": identity(os.getpid()), "unix": time.time()})
    began = time.monotonic()
    try:
        for source_ref in job["frozen_inputs"]["files"]:
            resolve(source_ref)
        if freeze_source_inputs(job["raw_row"]) != job["frozen_inputs"]:
            raise ValueError("frozen_XML_source_dependencies_changed")
        result = compile_raw(job["raw_row"], job["output"], seed=job["seed"], license=job.get("license"),
            source_aliases=job.get("source_aliases", []), prior_source_job=job.get("prior_source_job"))
    except Exception as error:
        from scipy.spatial import QhullError
        result = {"source_id": job["raw_row"]["id"],
            "status": "geometry_or_policy_failed" if isinstance(error, (ValueError, QhullError, xml.ET.ParseError)) else "unknown_system_failure",
            "reason": type(error).__name__ + ":" + str(error), "traceback": traceback.format_exc(),
            "recorded_settle_steps": completed_settle_steps(job["output"]), "elapsed_seconds": time.monotonic() - began,
            "robot_physical_executions": 0, "GPU_jobs": 0, "API_calls": 0}
    write(folder / "result.json", result)
    write(folder / "worker_terminal.json", {"job": ref, "result": binding(folder / "result.json"), "unix": time.time()})
    return result


def status(root):
    root = Path(root).resolve(); protocol = read(root / "PROTOCOL.json"); rows = []
    for item in protocol["jobs"]:
        path = resolve(item["job"]); folder = path.parent
        row = {"source_id": item["source_id"], "owned": item["owned"], "job": item["job"], "selected_geometry_policy": item["selected_geometry_policy"]}
        if (folder / "terminal.json").exists(): row.update(read(folder / "terminal.json"))
        elif (folder / "worker_terminal.json").exists():
            receipt = read(folder / "worker_terminal.json"); result = read(resolve(receipt["result"]))
            row.update(status="completed_awaiting_supervisor_receipt", result_status=result["status"], result=receipt["result"])
        elif (folder / "reservation.json").exists():
            process = read(folder / "process.json") if (folder / "process.json").exists() else {}
            old = process.get("identity")
            row["status"] = "running" if old and identity(old["pid"]) == old else "unknown_reserved_no_terminal"
        else: row["status"] = "not_submitted" if item["owned"] else "prior_campaign_not_submitted"
        rows.append(row)
    return {"schema": VERSION, "protocol": binding(root / "PROTOCOL.json"), "rows": rows,
        "counts": dict(Counter(row["status"] for row in rows)), "registered": len(rows),
        "source_retries": 0, "robot_physical_executions": 0, "GPU_jobs": 0, "API_calls": 0}


def _release_gate(root, protocol, qualification_gate=None):
    if not protocol["independent_review_required_before_submission"]:
        return None
    recorded = Path(root) / "RELEASE_GATE.json"
    if recorded.exists():
        ref = read(recorded)["independent_gate"]
        if qualification_gate is not None and binding(qualification_gate) != ref:
            raise ValueError("independent_release_gate_cannot_change_on_resume")
    elif qualification_gate is None:
        raise ValueError("independent_review_required_before_population_submission")
    else:
        ref = binding(qualification_gate)
    gate = read(resolve(ref))
    if (gate.get("schema") != "raw_robocasa_native_v2_independent_release"
            or gate.get("protocol") != binding(Path(root) / "PROTOCOL.json")
            or gate.get("passed") is not True or gate.get("reviewer_role") != "independent"
            or not gate.get("evidence")):
        raise ValueError("bound_independent_population_review_gate_required")
    for evidence in gate["evidence"]:
        resolve(evidence)
    return ref


def run(root, *, qualification_gate=None):
    root = Path(root).resolve(); protocol = _validate(root)
    # The registry-wide lock bounds concurrent XML native supervisors as well
    # as per-campaign workers; unrelated GLB queues are not touched.
    with campaign_lock(Path(protocol["registry"])), campaign_lock(root):
        gate = _release_gate(root, protocol, qualification_gate)
        if gate is not None:
            write(root / "RELEASE_GATE.json", {"independent_gate": gate, "protocol": binding(root / "PROTOCOL.json")})
        pending = []
        for item in protocol["jobs"]:
            if not item["owned"]: continue
            path = resolve(item["job"]); folder = path.parent
            if (folder / "terminal.json").exists(): continue
            if (folder / "worker_terminal.json").exists():
                receipt = read(folder / "worker_terminal.json")
                if receipt["job"] != item["job"]: raise ValueError("completed_worker_job_binding_mismatch")
                write(folder / "terminal.json", {"source_id": item["source_id"], "status": read(resolve(receipt["result"]))["status"],
                    "result": receipt["result"], "recovered_without_resubmission": True}); continue
            if (folder / "reservation.json").exists():
                raise RuntimeError("reserved_unknown_or_live_source_requires_recovery_no_retry:" + item["source_id"])
            pending.append((item, path))
        # Recovery/status do not launch native work or need a large-output disk
        # allocation. Check resources before the first actual new submission.
        if pending:
            sufficient_disk(root)
            if not (root / "START.json").exists():
                now = time.time(); write(root / "START.json", {"unix": now, "deadline_unix": now + protocol["maximum_wall_seconds"]})
        deadline = read(root / "START.json")["deadline_unix"] if (root / "START.json").exists() else time.time()
        def one(pair):
            item, path = pair; folder = path.parent
            if time.time() >= deadline: return None
            write(folder / "reservation.json", {"job": item["job"], "attempt": 1, "unix": time.time(), "protocol": binding(root / "PROTOCOL.json")})
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            began = time.monotonic()
            with (folder / "worker.log").open("x") as log:
                process = subprocess.Popen([sys.executable, "-m", "resimpair_assets.raw_robocasa_native_v2", "worker",
                    "--root", str(root), "--job", str(path)], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
                write(folder / "process.json", {"identity": identity(process.pid), "unix": time.time()})
                try: code = process.wait(timeout=min(protocol["per_source_timeout_seconds"], max(.001, deadline - time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                    code = "timeout"
            receipt = read(folder / "worker_terminal.json") if (folder / "worker_terminal.json").exists() else {}
            outcome = read(resolve(receipt["result"]))["status"] if receipt else "source_timeout" if code == "timeout" else "unknown_system_failure"
            terminal = {"source_id": item["source_id"], "status": outcome, "returncode": code,
                "result": receipt.get("result"), "elapsed_seconds": time.monotonic() - began, "retries": 0}
            write(folder / "terminal.json", terminal); print(item["source_id"], outcome, flush=True)
            return terminal
        iterator = iter(pending)
        with ThreadPoolExecutor(max_workers=protocol["maximum_CPU_workers"]) as executor:
            active = {executor.submit(one, pair) for pair in [next(iterator, None) for _ in range(protocol["maximum_CPU_workers"])] if pair}
            halted = False
            while active:
                future = next(as_completed(active)); active.remove(future); terminal = future.result()
                halted |= bool(terminal and terminal["status"] == "unknown_system_failure")
                pair = next(iterator, None) if not halted and time.time() < deadline else None
                if pair: active.add(executor.submit(one, pair))
        summary = status(root); entries = []
        for row in summary["rows"]:
            if row["status"] == "native_ready": entries.append(read(resolve(read(resolve(row["result"]))["entry"])))
        folder = root / "snapshots" / str(time.time_ns())
        write(folder / "native_candidates.json", {"schema": VERSION, "entries": entries,
            "protocol": binding(root / "PROTOCOL.json"), "grasp_qualification": "not_run", "default_promoted": False})
        summary["native_candidates"] = binding(folder / "native_candidates.json")
        write(folder / "SUMMARY.json", summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "status", "worker"])
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--input", type=Path)
    parser.add_argument("--job", type=Path); parser.add_argument("--maximum-new-sources", type=int)
    parser.add_argument("--method-comparisons", type=Path)
    parser.add_argument("--qualification-gate", type=Path)
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--source-timeout-seconds", type=int, default=240)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.root, read(args.input)["sources"], maximum_new_sources=args.maximum_new_sources,
            workers=args.workers, source_timeout_seconds=args.source_timeout_seconds, wall_seconds=args.wall_seconds,
            method_comparisons=read(args.method_comparisons)["comparisons"] if args.method_comparisons else ()))
    elif args.command == "run": print(run(args.root, qualification_gate=args.qualification_gate))
    elif args.command == "status": print(status(args.root))
    else: print(worker(args.root, args.job))
