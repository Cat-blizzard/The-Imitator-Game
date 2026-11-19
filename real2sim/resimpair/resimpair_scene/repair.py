"""Bounded, evidence-driven repair proposals for existing native executions.

No simulator steps occur here. Reordering keeps exact targets; a target change
is explicitly a new related-layout task, never a retrospective success label.
Every geometric proposal uses the retained native collision checker.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write

VERSION = "native_scene_repair_proposals_v1"


def collision_pairs(value):
    found = set()
    def visit(item):
        if isinstance(item, dict):
            if isinstance(item.get("link1"), str) and isinstance(item.get("link2"), str):
                found.add((item["link1"], item["link2"]))
            for v in item.values():
                visit(v)
        elif isinstance(item, list):
            for v in item:
                visit(v)
    visit(value)
    return sorted(found)


def source_case(execution):
    execution = Path(execution).resolve()
    cohort = execution.parent.parent
    cases = read(cohort / "cohort.json")["cases"]
    rows = [c for c in cases if c.get("execution_id", c["id"]) == execution.name]
    if len(rows) != 1:
        raise ValueError("one_bound_original_case_required")
    case = rows[0]
    scene, task = read(resolve(case["scene"])), read(resolve(case["task"]))
    return case, scene, task


def failed_release(execution, operation=None):
    """Find a recorded live rejection, excluding geometric preview failures."""
    execution = Path(execution).resolve()
    terminal = read(execution / "terminal.json")
    worker = read(execution / "worker_result.json")
    summary = read(execution / "rolling_execution_summary.json")
    failure = worker.get("failure") or {}
    if (type(terminal.get("returncode")) is not int
            or failure.get("message") != "independent_guard_rejected:release_above_rim"):
        raise ValueError("no_unrecovered_terminal_release_collision; do_not_repair_recovered_path_rejections")
    completed = set(summary["completed_operation_ids"])
    rows = []
    motions = sorted(execution.glob("transfer_*_motion"))
    if len(motions) > 8:
        raise ValueError("bounded_eight_operation_scope")
    for motion in motions:
        oid = motion.name.removesuffix("_motion")
        if oid in completed or (operation is not None and oid != operation):
            continue
        decisions = sorted(motion.glob("*release_above_rim*/decision.json"))
        if len(decisions) > 64:
            raise ValueError("too_many_release_proposals")
        for decision in decisions:
            request = decision.with_name("request.json")
            if not request.exists():
                continue
            q, d = read(request), read(decision)
            pairs = collision_pairs(d)
            if (q.get("stage") == "release_above_rim" and q.get("attached") is True
                    and not q.get("preview", False) and d.get("accepted") is False and pairs):
                rows.append({"operation": oid, "request": binding(request),
                             "decision": binding(decision), "pairs": pairs,
                             "frame": q["frame"]})
    if not rows:
        raise ValueError("no_recorded_live_release_collision; do_not_guess_repair")
    return max(rows, key=lambda r: (r["frame"], r["decision"]["path"]))


def reordered_before(task, operation, blocker_operation):
    if task.get("incidental_controller_order_is_required", False):
        raise ValueError("task_order_is_required")
    ids = [op["id"] for op in task["operations"]]
    if operation not in ids or blocker_operation not in ids or operation == blocker_operation:
        raise ValueError("distinct_existing_operations_required")
    preferred = [i for i in ids if i != operation]
    preferred.insert(preferred.index(blocker_operation), operation)
    edges = task.get("required_operation_precedence", [])
    result = []
    while len(result) < len(ids):
        ready = [i for i in preferred if i not in result
                 and all(a in result for a, b in edges if b == i)]
        if not ready:
            raise ValueError("cyclic_operation_precedence")
        result.append(ready[0])
    if result.index(operation) >= result.index(blocker_operation):
        raise ValueError("required_precedence_forbids_repair_order")
    return result


def target_geometry(scene, task, operation, offset_world_xy):
    """Exact cached vertices against the original qualified floor disk.

    Compiled offsets are world XY relative to support origin, whereas source
    request offsets are local XY relative to the measured cavity center.
    """
    import trimesh
    from resimpair_pickplace.requested_tasks import floor_fit, rotation
    op = next(o for o in task["operations"] if o["id"] == operation)
    obj = next(o for o in scene["objects"] if o["id"] == op["object"])
    support = next(s for s in scene["supports"] if s["id"] == op["target"]["support"])
    descriptor = read(resolve(support["container_descriptor"]))
    body = read(resolve(obj["asset_descriptor"]))
    mesh = trimesh.load(resolve(body["visual"]), force="scene", process=False).to_geometry()
    offset = np.asarray(offset_world_xy, dtype=float)
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("finite_world_offset_required")
    local = rotation(support["yaw"]).T @ offset - np.asarray(descriptor["cavity"]["center_xy_m"])
    floor = floor_fit(mesh.vertices, descriptor["cavity"], op["target"]["yaw"], support["yaw"], local)
    xy = np.asarray(support["position"][:2]) + offset
    objects = {o["id"]: o for o in scene["objects"]}
    supports = {s["id"]: s for s in scene["supports"]}
    radius = float(np.linalg.norm(obj["half_size"][:2]))
    separations = []
    for other in task["operations"]:
        if other["id"] == operation:
            continue
        goal = other["target"]
        origin = np.zeros(2) if goal["support"] == "table" else np.asarray(supports[goal["support"]]["position"][:2])
        point = origin + np.asarray(goal["offset_xy"])
        gap = float(np.linalg.norm(xy - point) - radius - np.linalg.norm(objects[other["object"]]["half_size"][:2]))
        separations.append({"other_operation": other["id"], "gap_m": gap, "passed": gap > .015})
    return {"passed": floor["passed"] and all(s["passed"] for s in separations),
            "support_instance": support["id"], "support_descriptor": support["container_descriptor"],
            "floor": floor, "goal_separations": separations, "source_local_offset_xy_m": local.tolist(),
            "compiled_world_offset_xy_m": offset.tolist(),
            "scope": "unchanged floor and payload separation; hand clearance requires native path check"}


def _restore(execution, failure):
    from mplib.pymp.collision_detection import fcl
    from resimpair_pickplace.cartesian_transfer import restore_unattached_guard
    request = resolve(failure["request"]); q = read(request); motion = request.parent.parent
    guard = restore_unattached_guard(execution, motion, {**q, "attached": False})
    matches = [(p, read(p)) for p in motion.glob("attachment_*release_above_rim.json")
               if read(p)["frame"] == q["frame"]]
    if len(matches) != 1:
        raise ValueError("one_exact_measured_attachment_required")
    p, measured = matches[0]
    guard.payload_to_tcp = np.asarray(measured["measured_object_in_tcp"], dtype=float)
    guard.use_attach = True
    geometry = fcl.load_mesh_as_BVH(str(motion / "selected_physical.obj"), np.ones(3))
    guard.payload_collision = fcl.CollisionObject(geometry)
    guard.objects.append((geometry, guard.payload_collision))
    return guard, q, binding(p)


def propose_repairs(execution, output, *, operation=None, maximum_candidates=24, maximum_seconds=90):
    """Propose one evidence-based order edit and bounded alternative goals.

    These are offline counterfactuals. Nothing changes a running scene or its
    accepted goal, and no proposal is counted as a physical success.
    """
    if type(maximum_candidates) is not int or not 1 <= maximum_candidates <= 64:
        raise ValueError("candidate_limit_must_be_1_to_64")
    if not isinstance(maximum_seconds, (int, float)) or not 1 <= maximum_seconds <= 300:
        raise ValueError("CPU_search_limit_must_be_1_to_300_seconds")
    execution, output = Path(execution).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("preserve_existing_repair_proposals")
    case, scene, task = source_case(execution)
    failure = failed_release(execution, operation)
    op = next(o for o in task["operations"] if o["id"] == failure["operation"])
    output.mkdir(parents=True)
    snapshot = output / "repair_implementation.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    protocol = {"schema": VERSION, "implementation": binding(snapshot),
        "source_execution": binding(execution / "execution_bundle.json"),
        "source_scene": case["scene"], "source_task": case["task"], "failure": failure,
        "maximum_candidates": maximum_candidates, "maximum_seconds": maximum_seconds,
        "physical_steps": 0, "API_calls": 0, "GPU_jobs": 0,
        "rules_changed": False, "physical_success": "unmeasured"}
    write(output / "protocol.json", protocol)
    proposals = []; rejections = []
    for a, b in failure["pairs"]:
        blocker = b if b.startswith("object_") else a if a.startswith("object_") else None
        matching = [o for o in task["operations"] if o["object"] == blocker]
        if not matching:
            continue
        try:
            order = reordered_before(task, op["id"], matching[0]["id"])
            edit = {"type": "reorder", "operation_ids": order}
            proposals.append({"id": "order_" + digest(edit)[:12], "edit": edit,
                "scope": "same_exact_goal_task", "reason": "perform observed blocked placement before its movable blocker",
                "collision_evidence": failure["decision"], "native_path_qualified": "unmeasured",
                "physical_success": "unmeasured"})
        except ValueError as error:
            rejections.append({"type": "reorder", "reason": str(error)})
    from resimpair_pickplace.requested_tasks import rotation
    from resimpair_pickplace.whole_path_motion_v1 import geometric_transport
    from resimpair_pickplace.payload_descent_v1 import payload_corridor
    from data_generation_methods.human_to_robot_pipeline.strict_motion import world_fk
    support = next(s for s in scene["supports"] if s["id"] == op["target"]["support"])
    desc = read(resolve(support["container_descriptor"]))
    old_offset = np.asarray(op["target"]["offset_xy"])
    center = rotation(support["yaw"]) @ np.asarray(desc["cavity"]["center_xy_m"])
    # Same finite disk proposals for all assets; no case names or tuned offsets.
    raw = [old_offset]
    for fraction in (.25, .5, .75, 1.):
        raw.append(old_offset * (1 - fraction) + center * fraction)
    for radius in (.015, .030, .045, .060):
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            raw.append(old_offset + radius * np.asarray([np.cos(angle), np.sin(angle)]))
    unique = {tuple(np.round(p, 9)): p for p in raw}
    ordered = sorted(unique.values(), key=lambda p: (float(np.linalg.norm(p-old_offset)), float(np.linalg.norm(p-center))))
    guard, request, attachment = _restore(execution, failure)
    start = time.monotonic(); checked = 0
    current = np.asarray(request["current"]); original_goal = np.asarray(request["goal"])
    for proposed in ordered:
        if checked >= maximum_candidates or time.monotonic() - start >= maximum_seconds:
            break
        layout = target_geometry(scene, task, op["id"], proposed)
        if not layout["passed"]:
            rejections.append({"offset_world_xy": proposed.tolist(), "layout": layout})
            continue
        number = checked; checked += 1
        folder = output / "native_candidates" / f"{number:03d}"
        folder.mkdir(parents=True)
        delta = proposed - old_offset
        row = {"offset_world_xy": proposed.tolist(), "delta_world_xy_m": delta.tolist(),
               "layout": layout, "attachment": attachment, "native_path_qualified": False}
        began = time.monotonic()
        try:
            goal = original_goal.copy(); goal[:2, 3] += delta
            above = world_fk(current[None, :7], guard.root)[0]
            above[:2, 3] = goal[:2, 3]
            above[:3, :3] = goal[:3, :3]
            route = []; q = current.copy()
            # First move above the new goal, then descend. Both are checked
            # against the same actual native scene and measured attachment.
            for stage, endpoint in [("above_bowl", above), ("release_above_rim", goal)]:
                path = geometric_transport(guard.planner, guard.root, q, endpoint, .05)
                report, dense, poses = guard.check(path, q, endpoint, stage)
                corridor = payload_corridor(poses, guard.payload_to_tcp, endpoint) if stage == "release_above_rim" and poses is not None else None
                np.savez_compressed(folder / (stage + ".npz"), positions=path["position"], current=q, goal=endpoint)
                route.append({"stage": stage, "guard": report, "payload_corridor": corridor})
                if not report["passed"] or (corridor is not None and not corridor["passed"]):
                    break
                q[:7] = path["position"][-1]
            row["route"] = route
            row["native_path_qualified"] = (len(route) == 2 and all(r["guard"]["passed"] for r in route)
                and route[-1]["payload_corridor"] is not None and route[-1]["payload_corridor"]["passed"])
        except Exception as error:
            row["error"] = repr(error)
        row["seconds"] = time.monotonic() - began
        write(folder / "result.json", row)
        if row["native_path_qualified"] and np.linalg.norm(delta) > 1e-9:
            edit = {"type": "target_offset", "operation": op["id"], "offset_world_xy_m": proposed.tolist()}
            proposals.append({"id": "layout_" + digest(edit)[:12], "edit": edit,
                "scope": "new_related_layout_task", "reason": "native hand and held-payload clearance at a new qualified floor position",
                "native_path_qualified": True, "physical_success": "unmeasured",
                "evidence": binding(folder / "result.json"), "goal_displacement_m": float(np.linalg.norm(delta))})
            # Keep a small ranked set; no need to explore an unbounded grid.
            if sum(p["scope"] == "new_related_layout_task" for p in proposals) >= 3:
                break
    result = {"schema": VERSION, "protocol": binding(output / "protocol.json"),
        "source_case": case, "failure": failure, "proposals": proposals,
        "static_rejections": rejections, "native_candidates_checked": checked,
        "seconds": time.monotonic()-start, "search_exhausted": checked >= maximum_candidates,
        "physical_success": "unmeasured", "API_calls": 0, "GPU_jobs": 0, "physical_steps": 0}
    write(output / "proposals.json", result)
    return result
