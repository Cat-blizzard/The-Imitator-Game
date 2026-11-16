#!/usr/bin/env python3
"""Isolated pill-only direct-release transport recipe and minimal runner patch.

The frozen runner has a post-lift phase_hook but no transport-stage selector.
This candidate therefore does not patch or replace that runner at import time.
It supplies a pure route selector, the explicit optional interface/CLI patch for
review, and a CPU self-test. The patch is never applied by this module.

XYZ arrays are finite world-frame [2,3] metres. The selector preserves the exact
task-adapter release positions and the existing runner supplies unchanged carry
quaternions. It returns only the existing carry_lower phase, whose unchanged
planner, full-pair checker, held executor, settle gate and future-tail budget are
then used once. All grasp/lift, native detach, release, retreat and placement code
stays in the shared runner. No environment, joint, object or collision state is
written by this module. No cached or fabricated grasp evidence is produced.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import difflib
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "infrastructure/development/tig-dual-arm").is_dir())
DEV = ROOT / "infrastructure/development/tig-dual-arm"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))
PILL = "TwoRobotPickPillToRegions-v1"
RECIPE = "pill-direct-release-v1"
EVIDENCE = ROOT / "infrastructure/validation/dual-arm/curobo-complete-20261116"
RUNNER = DEV / "mani_skill/examples/motionplanning/dual/validate_curobo_complete.py"
ARCHIVE = EVIDENCE / "extensions-v1/curobo-pill-s401-dev01/source"
ARCHIVED_RUNNER = ARCHIVE / "validate_curobo_complete.py"
ARCHIVED_RUNNER_SHA256 = "cfc73a4d1e6aa13b8591e451e055aaa477897ef875a2fbdafb56cafda099cc45"
SIGNATURE_OLD = """def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any], *,
               phase_hook: Any = None) -> None:"""
SIGNATURE_NEW = """def native_run(args: argparse.Namespace, output: Path, report: dict[str, Any], *,
               phase_hook: Any = None, transport_recipe: Any = None) -> None:"""
LOOP_OLD = """        for name,positions in (('carry_orient_and_transport',route['above_baskets']),
                               ('carry_lower',route['release'])):
"""
LOOP_NEW = """        transport_stages = (('carry_orient_and_transport',route['above_baskets']),
                            ('carry_lower',route['release']))
        if transport_recipe is not None:
            transport_stages = transport_recipe(args.task,route,report['source_departure_after_lift'])
        report['transport_recipe_selection'] = {'policy':transport_policy,
            'selected_phases':[name for name,_ in transport_stages],
            'goal_positions_world_m':[np.asarray(xyz).tolist() for _,xyz in transport_stages],
            'carry_quaternions':'original task adapter carry_q',
            'acceptance_evidence':'original planning, stages and continuous100Hz held audit'}
        for name,positions in transport_stages:
"""


def _finite(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {label}{shape}")
    return result


def _true_pair(value: Any, label: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 2 or any(v is not True for v in value):
        raise ValueError(f"Both measured {label} must be boolean True")


def pill_direct_transport_recipe(task_id: str, route: dict[str, Any],
                                 source_departure: dict[str, Any]) -> tuple[tuple[str, np.ndarray], ...]:
    """Select one finite world-frame [2,3] release target, without side effects.

    The shared caller has already completed its physical grasp, >=10cm lift,
    geometric source departure and100Hz physical departure barriers. This
    selector additionally rejects absent/nonboolean measured departure. It
    cannot alter quaternions, limits, collision permissions or the physical tail.
    """
    if task_id != PILL:
        raise ValueError("Direct-release transport is confined to the pill task")
    if not isinstance(route, dict) or not isinstance(source_departure, dict):
        raise ValueError("Actual route and measured source-departure mappings required")
    _true_pair(source_departure.get("effective_measured_departure"), "geometric source departure")
    release = _finite(route.get("release"), (2, 3), "original release XYZ")
    return (("carry_lower", release.copy()),)


def _validated_saved_fixture(*, args: argparse.Namespace, route: dict[str, Any],
                             report: dict[str, Any], substeps: list[dict[str, Any]]) -> dict[str, Any]:
    """CPU evidence check for the existing runner barrier, not a runtime substitute."""
    if getattr(args, "task", None) != PILL or report.get("task") != PILL:
        raise ValueError("Direct-release transport is confined to the pill task")
    configuration = report.get("execution_configuration", {})
    expected = {"joint_vel_limit_rad_s": 1.5, "joint_acc_limit_rad_s2": 2.6,
        "approach_joint_vel_limit_rad_s": 1., "approach_joint_acc_limit_rad_s2": 1.3,
        "max_episode_steps": 200, "final_common_placement_steps": 20,
        "retreat_command_m": .081, "minimum_actual_retreat_m": .08}
    if any(configuration.get(k) != v for k, v in expected.items()):
        raise ValueError("The fixed pill speed/budget/retreat/placement configuration changed")
    if not np.array_equal(_finite(configuration.get("min_actual_height_gain_m"), (2,), "height limits"), [.1, .1]):
        raise ValueError("The original >=10cm lift requirement must remain")
    stages = report.get("stages")
    if not isinstance(stages, list) or not stages or stages[-1].get("name") != "lift_and_hold":
        raise ValueError("Direct transport must begin immediately after the original lift stage")
    closed = next((s for s in stages if s.get("name") == "close_and_confirm"), {})
    lift = stages[-1]
    if closed.get("status") != "passed" or lift.get("status") != "passed":
        raise ValueError("The original close and lift executors must both pass")
    required = closed.get('stable_steps_required')
    observed = closed.get('stable_steps_observed')
    if (type(required) is not int or required < 5 or not isinstance(observed,list)
            or len(observed)!=2 or any(type(n) is not int or n < required for n in observed)):
        raise ValueError('The original five-tick dual close barrier is incomplete')
    required = lift.get('stable_steps_required')
    observed = lift.get('stable_steps_observed')
    if type(required) is not int or required < 3 or type(observed) is not int or observed < required:
        raise ValueError('The original three-tick common lift endpoint barrier is incomplete')
    departure = report.get("source_departure_after_lift", {})
    _true_pair(departure.get("effective_measured_departure"), "geometric source departure")
    if not substeps or substeps[-1].get("phase") != "lift_and_hold":
        raise ValueError("Current100Hz post-lift source evidence is required")
    last_physics = substeps[-1]
    _true_pair(last_physics.get("physical_support_released_snapshot"), "physical source departure")
    trace = lift.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("Measured final lift observations are required")
    last = trace[-1]
    step = last.get("environment_step")
    if type(step) is not int or step < 1 or last_physics.get("control_step") != step:
        raise ValueError("Geometric/physical observations refer to different control steps")
    objects = last.get("objects")
    if not isinstance(objects, list) or len(objects) != 2:
        raise ValueError("Both measured held objects must be present")
    gains = _finite([obj.get("object_height_gain_m") for obj in objects], (2,), "measured lift gains")
    if np.any(gains < .1) or any(obj.get("is_grasping") is not True for obj in objects):
        raise ValueError("Both objects must actually gain >=10cm and remain grasped")
    release = _finite(route.get("release"), (2, 3), "original release XYZ")
    original = _finite(report.get("route", {}).get("release"), (2, 3), "recorded adapter release XYZ")
    if not np.array_equal(release, original):
        raise ValueError("Direct recipe cannot change the task adapter's original release positions")
    selected = pill_direct_transport_recipe(args.task, route, departure)
    assert selected[0][0] == "carry_lower" and np.array_equal(selected[0][1], release)
    return {"version": RECIPE, "task_id": PILL,
        "selected_phases": ["carry_lower"], "skipped_waypoint": "above_baskets",
        "skipped_independent_phase": "carry_orient_and_transport",
        "semantics": "one held move from confirmed lift directly to the unchanged release pose",
        "release_xyz_m": release.tolist(), "carry_quaternions": "unchanged shared runner value",
        "barrier_control_step": step, "measured_object_height_gain_m": gains.tolist(),
        "source_support_exception_during_transport": False,
        "executor": "unchanged execute_dual_held_move", "final_endpoint_settle_gate": "unchanged",
        "native_detach_release_retreat_final20": "unchanged shared runner tail"}


def _patch_edits() -> tuple[tuple[str, str], ...]:
    """Every proposed edit is explicit; default phase limits remain byte-identical."""
    selected_function = '''def selected_transport_recipe(args: argparse.Namespace) -> str:
    """Old callers retain the original two-stage transport recipe."""
    policy = getattr(args,'transport_recipe','standard')
    if policy not in TRANSPORT_RECIPES:
        raise ValueError('Unknown fixed transport recipe')
    if policy == 'pill-direct-release-v1' and getattr(args,'task',None) != TASKS[1]:
        raise ValueError('The direct-release recipe is confined to the pill task')
    return policy


'''
    native_prefix = '''    transport_policy = selected_transport_recipe(args)
    if report['execution_configuration'].get('transport_recipe','standard') != transport_policy:
        raise ValueError('Transport selection differs from the common execution configuration')
    if transport_policy == 'pill-direct-release-v1':
        from .validate_curobo_pill_direct import pill_direct_transport_recipe
        if transport_recipe is None:
            transport_recipe = pill_direct_transport_recipe
        elif transport_recipe is not pill_direct_transport_recipe:
            raise ValueError('The fixed pill recipe requires its reviewed route selector')
    elif transport_recipe is not None:
        raise ValueError('A transport callback requires an explicit fixed recipe')
    report['transport_recipe'] = transport_policy
'''
    shared_test = '''    assert selected_transport_recipe(argparse.Namespace()) == 'standard'
    assert selected_transport_recipe(argparse.Namespace(task=TASKS[1],
        transport_recipe='pill-direct-release-v1')) == 'pill-direct-release-v1'
    direct = execution_configuration(TASKS[1],transport_recipe='pill-direct-release-v1')
    standard = execution_configuration(TASKS[1])
    assert direct['route'] == 'merge-lift-direct-release'
    assert direct['transport_phase_sequence'] == ['carry_lower']
    recipe_fields = {'route','transport_recipe','transport_phase_sequence'}
    assert {k:v for k,v in direct.items() if k not in recipe_fields} == {
        k:v for k,v in standard.items() if k not in recipe_fields}
    for task,policy in ((TASKS[0],'pill-direct-release-v1'),
                        (TASKS[2],'pill-direct-release-v1'),(TASKS[1],'unknown')):
        try:
            selected_transport_recipe(argparse.Namespace(task=task,transport_recipe=policy))
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid task-specific transport recipe accepted')
'''
    return (
        ("PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1')\n",
         "PAYLOAD_REFINEMENTS = ('baseline', 'fixed-task-v1')\nTRANSPORT_RECIPES = ('standard', 'pill-direct-release-v1')\n"),
        ("                            payload_refinement: str = 'baseline') -> dict[str, Any]:",
         "                            payload_refinement: str = 'baseline',\n                            transport_recipe: str = 'standard') -> dict[str, Any]:"),
        ("    return {'joint_vel_limit_rad_s': parameters['joint_vel_limit'],\n",
         "    if transport_recipe not in TRANSPORT_RECIPES:\n        raise ValueError('Expected an explicit fixed transport recipe')\n    if transport_recipe == 'pill-direct-release-v1' and task_id != TASKS[1]:\n        raise ValueError('The direct-release recipe is confined to the pill task')\n    return {'joint_vel_limit_rad_s': parameters['joint_vel_limit'],\n"),
        ("        'route': 'merge-lift', 'execution_profile': 'execution',\n",
         "        'route': 'merge-lift-direct-release' if transport_recipe == 'pill-direct-release-v1' else 'merge-lift',\n        'transport_recipe': transport_recipe,\n        'transport_phase_sequence': ['carry_lower'] if transport_recipe == 'pill-direct-release-v1' else [\n            'carry_orient_and_transport','carry_lower'],\n        'execution_profile': 'execution',\n"),
        ("def configured_payload_snapshot(snapshot: dict[str,Any], policy: str) -> dict[str,Any]:\n",
         selected_function + "def configured_payload_snapshot(snapshot: dict[str,Any], policy: str) -> dict[str,Any]:\n"),
        (SIGNATURE_OLD, SIGNATURE_NEW),
        ("    payload_refinement = selected_payload_refinement(args)\n",
         native_prefix + "    payload_refinement = selected_payload_refinement(args)\n"),
        (LOOP_OLD, LOOP_NEW),
        ("    assert config['payload_refinement'] == 'baseline'\n",
         "    assert config['payload_refinement'] == 'baseline'\n" + shared_test),
        ("    parser.add_argument('--payload-refinement',choices=PAYLOAD_REFINEMENTS,default='baseline')\n",
         "    parser.add_argument('--payload-refinement',choices=PAYLOAD_REFINEMENTS,default='baseline')\n    parser.add_argument('--transport-recipe',choices=TRANSPORT_RECIPES,default='standard')\n"),
        ("    faulthandler.enable()\n",
         "    if args.transport_recipe == 'pill-direct-release-v1' and args.task != TASKS[1]:\n        parser.error('--transport-recipe pill-direct-release-v1 requires the pill task')\n    faulthandler.enable()\n"),
        ("        'payload_refinement':args.payload_refinement,\n",
         "        'payload_refinement':args.payload_refinement,\n        'transport_recipe':args.transport_recipe,\n"),
        ("            args.task,approach_recipe=args.approach_recipe,payload_refinement=args.payload_refinement),\n",
         "            args.task,approach_recipe=args.approach_recipe,payload_refinement=args.payload_refinement,\n            transport_recipe=args.transport_recipe),\n"),
    )


def archived_runner_source() -> str:
    """Use immutable v1 evidence so generation/self-tests survive promotion."""
    value = ARCHIVED_RUNNER.read_bytes()
    if hashlib.sha256(value).hexdigest() != ARCHIVED_RUNNER_SHA256:
        raise ValueError("Archived v1 runner SHA identity changed")
    return value.decode()


def proposed_runner_source(source: str) -> str:
    """Return the reviewed optional recipe/CLI patch in memory; never edit runtime."""
    result = source
    for before, after in _patch_edits():
        if result.count(before) != 1:
            raise ValueError("Frozen runner anchors changed; review the interface patch again")
        result = result.replace(before, after, 1)
    ast.parse(result)
    return result


def interface_patch(source: str) -> str:
    path = RUNNER.relative_to(ROOT).as_posix()
    return "".join(difflib.unified_diff(source.splitlines(keepends=True),
        proposed_runner_source(source).splitlines(keepends=True), fromfile="a/" + path, tofile="b/" + path))


def _ast_contract(source: str, proposed: str) -> dict[str, Any]:
    """Prove every original runner statement is retained outside the selector."""
    old_tree, new_tree = ast.parse(source), ast.parse(proposed)
    old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_run")
    new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_run")
    assert new.args.kwonlyargs[-1].arg == "transport_recipe"
    assert isinstance(new.args.kw_defaults[-1], ast.Constant) and new.args.kw_defaults[-1].value is None
    new.args.kwonlyargs.pop()
    new.args.kw_defaults.pop()
    # Recipe validation and lazy import occur before the original environment
    # imports; after removing those additions the native physical body is exact.
    first_original = next(i for i,n in enumerate(new.body) if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'payload_refinement')
    assert first_original > 0
    del new.body[:first_original]
    old_try = next(n for n in old.body if isinstance(n, ast.Try))
    new_try = next(n for n in new.body if isinstance(n, ast.Try))
    index = next(i for i, n in enumerate(new_try.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "transport_stages")
    assignment, selection, metadata, loop = new_try.body[index:index + 4]
    assert isinstance(selection, ast.If) and isinstance(loop, ast.For)
    assert isinstance(metadata,ast.Assign) and 'transport_recipe_selection' in ast.unparse(metadata.targets[0])
    assert isinstance(loop.iter, ast.Name) and loop.iter.id == "transport_stages"
    old_loop = old_try.body[index]
    assert isinstance(old_loop, ast.For) and ast.dump(assignment.value) == ast.dump(old_loop.iter)
    assert ast.dump(loop.target) == ast.dump(old_loop.target)
    assert [ast.dump(n) for n in loop.body] == [ast.dump(n) for n in old_loop.body]
    assert "source_departure_after_lift" in ast.unparse(new_try.body[index - 2])
    loop.iter = deepcopy(assignment.value)
    del new_try.body[index:index + 3]
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    # All remaining differences are the explicitly listed config/CLI additions.
    restored = proposed
    for before,after in reversed(_patch_edits()):
        assert restored.count(after) == 1
        restored = restored.replace(after,before,1)
    assert restored == source
    return {"all_original_statements_preserved": True, "default_two_stage_sequence_unchanged": True,
        "original_transport_plan_executor_budget_body_unchanged": True,
        "source_departure_guard_precedes_selector": True,
        "native_detach_release_retreat_final20_tail_unchanged": True,
        "explicit_cli_default_standard": True,"all_edits_explicit_and_reversible": True}


def self_test() -> dict[str, Any]:
    from mani_skill.examples.motionplanning.dual import validate_curobo_complete as common
    assert common.FUTURE_MINIMUM_TICKS['carry_lower'] == 26
    rejected = []

    def reject(label: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except (ValueError, RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError(f"Unsafe recipe accepted: {label}")

    fixture_rows = []
    for backend in ("curobo", "mplib"):
        directory = EVIDENCE / f"extensions-v1/{backend}-pill-s401-dev01"
        report = json.loads((directory / "report.json").read_text())
        lift_index = next(i for i, row in enumerate(report["stages"]) if row["name"] == "lift_and_hold")
        report["stages"] = report["stages"][:lift_index + 1]
        physics = []
        with (directory / "physics-substeps.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["phase"] == "lift_and_hold":
                    physics.append(row)
        route = deepcopy(report["route"])
        original_report, original_route = deepcopy(report), deepcopy(route)
        args = argparse.Namespace(task=PILL)
        metadata = _validated_saved_fixture(args=args, route=route, report=report, substeps=physics)
        selected = pill_direct_transport_recipe(args.task,route,report['source_departure_after_lift'])
        assert selected[0][0] == "carry_lower" and len(selected) == 1
        assert np.array_equal(selected[0][1], route["release"])
        assert route == original_route
        assert report == original_report
        selected[0][1][0, 0] += 1.
        assert route == original_route
        fixture_rows.append({"backend": backend, "input_directory": str(directory),
            "barrier_control_step": metadata["barrier_control_step"],
            "release_positions_unchanged": True, "all_other_report_values_unchanged": True,
            "selected_phases": metadata["selected_phases"],
            "measured_height_gains_m": metadata["measured_object_height_gain_m"]})
        if backend == "curobo":
            fixture = (args, route, report, physics)
    args, route, report, physics = fixture

    def changed(label: str, edit: Callable[[dict[str, Any]], None]) -> None:
        value = deepcopy(report)
        edit(value)
        reject(label, lambda: _validated_saved_fixture(args=args, route=route, report=value, substeps=physics))

    reject("foreign_task", lambda: pill_direct_transport_recipe(common.TASKS[0],route,report['source_departure_after_lift']))
    reject("foreign_fruits_task", lambda: pill_direct_transport_recipe(common.TASKS[2],route,report['source_departure_after_lift']))
    reject("missing_selector_departure", lambda: pill_direct_transport_recipe(PILL,route,{}))
    reject("nonboolean_selector_departure", lambda: pill_direct_transport_recipe(PILL,route,{'effective_measured_departure':[1,True]}))
    reject("false_selector_departure", lambda: pill_direct_transport_recipe(PILL,route,{'effective_measured_departure':[True,False]}))
    reject("malformed_selector_route", lambda: pill_direct_transport_recipe(PILL,None,report['source_departure_after_lift']))
    changed("foreign_report_task", lambda r: r.update(task=common.TASKS[2]))
    changed("lift_not_last", lambda r: r["stages"].append({"name": "other", "status": "passed"}))
    changed("failed_lift", lambda r: r["stages"][-1].update(status="failed"))
    changed("missing_geometric_departure", lambda r: r.pop("source_departure_after_lift"))
    changed("one_source_not_departed", lambda r: r["source_departure_after_lift"].update(effective_measured_departure=[True, False]))
    changed("nonboolean_departure", lambda r: r["source_departure_after_lift"].update(effective_measured_departure=[1, 1]))
    changed("insufficient_close_stability", lambda r: next(s for s in r["stages"] if s["name"] == "close_and_confirm").update(stable_steps_observed=[4, 5]))
    changed("insufficient_lift_stability", lambda r: r["stages"][-1].update(stable_steps_observed=2))
    changed("height_below10cm", lambda r: r["stages"][-1]["trace"][-1]["objects"][0].update(object_height_gain_m=.099999))
    changed("nonfinite_height", lambda r: r["stages"][-1]["trace"][-1]["objects"][0].update(object_height_gain_m=float("nan")))
    changed("lost_grasp", lambda r: r["stages"][-1]["trace"][-1]["objects"][1].update(is_grasping=False))
    for key, value in (("final_common_placement_steps", 19), ("minimum_actual_retreat_m", .079),
                       ("max_episode_steps", 400), ("approach_joint_vel_limit_rad_s", 1.5)):
        changed("changed_" + key, lambda r, key=key, value=value: r["execution_configuration"].update({key: value}))
    for label, change in (("physical_source_not_departed", {"physical_support_released_snapshot": [False, True]}),
                          ("stale_physical_observation", {"control_step": 1}),
                          ("wrong_physical_phase", {"phase": "carry_lower"})):
        invalid = deepcopy(physics)
        invalid[-1].update(change)
        reject(label, lambda invalid=invalid: _validated_saved_fixture(args=args, route=route, report=report, substeps=invalid))
    for label, release in (("bad_release_shape", [[0., 0., 0.]]),
                           ("nonfinite_release", [[float("nan"), 0., 0.], [0., 0., 0.]]),
                           ("changed_original_release", (np.asarray(route["release"]) + .001).tolist())):
        invalid_route = {**route, "release": release}
        reject(label, lambda invalid_route=invalid_route: _validated_saved_fixture(args=args, route=invalid_route, report=report, substeps=physics))
    for label,release in (('selector_bad_release_shape',[[0.,0.,0.]]),
                          ('selector_nonfinite_release',[[float('inf'),0.,0.],[0.,0.,0.]])):
        reject(label,lambda release=release: pill_direct_transport_recipe(PILL,{'release':release},report['source_departure_after_lift']))
    # Representative synthetic zero-valued [2,3] target and immutable receipt.
    zeros = {'release':[[0.,0.,0.],[0.,0.,0.]]}
    evidence = {'effective_measured_departure':[True,True]}
    copied_zeros,copied_evidence = deepcopy(zeros),deepcopy(evidence)
    selected = pill_direct_transport_recipe(PILL,zeros,evidence)
    assert len(selected)==1 and selected[0][0]=='carry_lower' and selected[0][1].shape==(2,3)
    assert not selected[0][1].any() and zeros==copied_zeros and evidence==copied_evidence
    source = archived_runner_source()
    proposed = proposed_runner_source(source)
    ast_result = _ast_contract(source, proposed)
    reject("changed_runner_anchor", lambda: proposed_runner_source(source.replace(SIGNATURE_OLD, SIGNATURE_NEW)))
    # Evaluate only the proposed CPU configuration/selector functions and literal
    # constants. No proposed native_run/main, environment or physics is executed.
    tree=ast.parse(proposed)
    wanted={'TASKS','APPROACH_RECIPES','PAYLOAD_REFINEMENTS','TRANSPORT_RECIPES','FINAL_STABLE_STEPS'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in wanted for t in n.targets)) or (
        isinstance(n,ast.FunctionDef) and n.name in ('execution_configuration','selected_transport_recipe'))]
    namespace={'__name__':'cpu_recipe_configuration','__package__':common.__package__,
        'argparse':argparse,'Any':Any,'SOLVER_LIMIT_FRACTION':common.SOLVER_LIMIT_FRACTION}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<CPU configuration-only recipe candidate>','exec'),namespace)
    config=namespace['execution_configuration']
    select=namespace['selected_transport_recipe']
    assert select(argparse.Namespace())=='standard'
    assert select(argparse.Namespace(task=PILL,transport_recipe=RECIPE))==RECIPE
    for task in common.TASKS:
        # Compare with the immutable archived standard config, whether the
        # installed runtime has subsequently adopted this optional policy or not.
        old_namespace=dict(namespace)
        old_tree=ast.parse(source)
        old_function=next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name=='execution_configuration')
        exec(compile(ast.Module(body=[old_function],type_ignores=[]),'<archived CPU configuration>','exec'),old_namespace)
        before=old_namespace['execution_configuration'](task)
        after=config(task)
        assert {k:v for k,v in after.items() if k not in ('transport_recipe','transport_phase_sequence')}==before
        assert after['transport_recipe']=='standard'
        assert after['transport_phase_sequence']==['carry_orient_and_transport','carry_lower']
    direct=config(PILL,transport_recipe=RECIPE)
    standard=config(PILL)
    recipe_fields={'route','transport_recipe','transport_phase_sequence'}
    assert {k:v for k,v in direct.items() if k not in recipe_fields}=={
        k:v for k,v in standard.items() if k not in recipe_fields}
    assert direct['route']=='merge-lift-direct-release' and direct['transport_phase_sequence']==['carry_lower']
    for task in (common.TASKS[0],common.TASKS[2]):
        reject('foreign_task_cli_configuration_'+task,lambda task=task: config(task,transport_recipe=RECIPE))
        reject('foreign_task_early_selector_'+task,lambda task=task: select(argparse.Namespace(task=task,transport_recipe=RECIPE)))
    for invalid in ('unknown',True,None):
        reject('unknown_cli_configuration_'+str(invalid),lambda invalid=invalid: config(PILL,transport_recipe=invalid))
        reject('unknown_early_selector_'+str(invalid),lambda invalid=invalid: select(argparse.Namespace(task=PILL,transport_recipe=invalid)))
    protocol = json.loads((EVIDENCE / "task-extension-protocol-v1.json").read_text())
    expected = protocol["frozen_expectations"]["source_sha256"]
    archive_differences=[name for name,sha in expected.items() if not (ARCHIVE/Path(name).name).is_file()
        or hashlib.sha256((ARCHIVE/Path(name).name).read_bytes()).hexdigest()!=sha]
    assert len(expected)==75 and not archive_differences
    differences = [name for name, sha in expected.items()
                   if not Path(name).exists() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha]
    torch = sys.modules.get("torch")
    assert torch is None or not torch.cuda.is_initialized()
    return {"status": "passed", "recipe": RECIPE, "task_id": PILL,
        "selected_target_shape": [2, 3], "new_native_runs": 0, "environment_steps": 0,
        "actual_saved_lift_fixtures": fixture_rows, "negative_tests_rejected": len(rejected),
        "negative_tests": rejected, "interface_patch_contract": ast_result,
        "CPU_configuration_contract":{"default_standard_all_legacy_fields_unchanged":True,
            "direct_only_route_recipe_fields_changed":True,"both_backends_use_same_configuration":True,
            "explicit_cli_policy":RECIPE,"nonpill_rejected_before_native_run":True},
        "runner_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "proposed_runner_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "archived_frozen_source_count":len(expected),"archived_source_differences":archive_differences,
        "current_legacy_source_differences_from_v1":differences,
        "current_runner_sha256":hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        "runtime_integration":"this self-test never modifies runtime or executes native_run",
        "GPU_validation": "NOT_RUN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--emit-patch", action="store_true")
    args = parser.parse_args()
    if args.emit_patch:
        print(interface_patch(archived_runner_source()), end="")
    else:
        print(json.dumps(self_test(), allow_nan=False, indent=2))
