"""Opt-in release-aware selection and one related receiver-only layout revision.

All source/variant selection, initial layout, static clutter and final compilation
remain the existing APIs. A failed fixed revision is retained, never rerolled.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, check_separation, table_fit
from . import metric_selection_factory_v1 as coordinator
from . import source_first_task_selection_v2 as selection
from .source_first_factory_adapter_v1 import propose, _check_compiled

VERSION = 'source_first_factory_adapter_v2_release_geometry'


def check_compiled_release(cohort):
    rows = []; scene_cache = {}
    for case in read(cohort)['cases']:
        scene_ref = case['scene']; key = digest(scene_ref)
        scene_cache.setdefault(key, read(resolve(scene_ref)))
        scene = scene_cache[key]; task = read(resolve(case['task']))
        checks = [selection.check_operation(scene, operation) for operation in task['operations']]
        rows.append({'case_id': case['id'], 'scene': scene_ref, 'task': case['task'],
            'operations': checks, 'passed': all(row['passed'] for row in checks)})
    return {'schema': VERSION, 'cohort': binding(cohort), 'rows': rows,
        'passed': bool(rows) and all(row['passed'] for row in rows),
        'criterion': 'same first 12 cache poses and original 3cm/1cm/15cm release geometry',
        'physical_success': 'unmeasured', 'IK_calls': 0, 'physics_steps': 0, 'GPU_jobs': 0, 'API_calls': 0}


def revise_receivers(source_case, output, *, task_ids=('select_one_0', 'distribute_all')):
    """One compile of a related request; preserve every selected payload and pose.

    Candidate receivers are tested before selection for release, original fixed
    offsets and the footprint at the original receiver position. No changed
    seed, resized object, new cache or new initial-pose draw is permitted.
    """
    source_case, output = Path(source_case).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('preserve_existing_related_layout_no_retry')
    request_ref = binding(source_case/'requests.json'); original = read(resolve(request_ref))
    if len(original['scenes']) != 1: raise ValueError('one_bound_source_request_required')
    source = original['scenes'][0]; cohort = read(source_case/'factory/cohort.json')
    by_task = {row['requested_task_id']: row for row in cohort['cases']}
    if len(set(task_ids)) != len(task_ids) or any(name not in by_task for name in task_ids):
        raise ValueError('existing_unique_related_task_subset_required')
    original_scene_ref = by_task[task_ids[0]]['scene']; original_scene = read(resolve(original_scene_ref))
    if any(by_task[name]['scene'] != original_scene_ref for name in task_ids):
        raise ValueError('related_tasks_must_share_one_original_scene')
    factory_protocol = read(source_case/'factory/protocol.json')
    bundle = resolve(factory_protocol['bundle']); seed = factory_protocol['seed']
    b, entries, population, bins = coordinator.source.admitted_pool(bundle)
    selected_entries = coordinator.factory.select_variants(entries,
        [o['source_id'] for o in source['objects']], seed, source['variant_choices'])
    group = [(entry, read(resolve(entry['asset']))) for entry in selected_entries]
    by_object = {row['id']: item for row, item in zip(source['objects'], group)}
    write(output/'PROTOCOL.json', {'schema': VERSION, 'mode': 'posthoc_related_receiver_only_layout',
        'original_request': request_ref, 'original_scene': original_scene_ref,
        'original_tasks': [by_task[name]['task'] for name in task_ids], 'bundle': binding(bundle),
        'implementation': binding(__file__), 'selection': binding(selection.__file__),
        'retained_factory': binding(coordinator.factory.__file__), 'seed': seed,
        'maximum_compile_calls': 1, 'payload_reselection': False, 'payload_rescaling': False,
        'initial_pose_resampling': False, 'grasp_cache_replacement': False,
        'receiver_positions_preserved': True, 'release_policy': selection.RELEASE_POLICY,
        'IK_calls': 0, 'physics_steps': 0, 'GPU_jobs': 0, 'API_calls': 0})
    try:
        base_tasks = [read(resolve(by_task[name]['task'])) for name in task_ids]
        before = [{'task': by_task[name]['task'], 'operations': [selection.check_operation(original_scene, op)
            for op in task['operations']]} for name, task in zip(task_ids, base_tasks)]
        write(output/'ORIGINAL_RELEASE_PREFLIGHT.json', {'rows': before})
        incompatible = {row['support'] for task in before for row in task['operations'] if not row['passed']}
        if not incompatible or 'table' in incompatible: raise ValueError('incompatible_container_receivers_required')
        revised = deepcopy(source); checker = selection.ReleaseCompatibility(); decisions = []; changes = []
        source_ids = {d['source_id'] for _, d in group}
        old_supports = {row['id']: row for row in original_scene['supports']}
        for receiver in revised['containers']:
            rid = receiver['id']
            if rid not in incompatible: continue
            operations = [op for task in base_tasks for op in task['operations'] if op['target']['support'] == rid]
            member_ids = sorted({op['object'] for op in operations})
            original_placements = [p for task in source['tasks'] if task['id'] in task_ids
                for p in task['placements'] if p['target'].get('container') == rid]
            candidates = [(ref,d) for ref,d in bins if d['partition'] == original_scene['partition']
                and d['source_id'] not in source_ids and d.get('roles',{}).get('stationary_container')]
            candidates.sort(key=lambda pair:(float(np.linalg.norm(np.asarray(pair[1]['dimensions_m'][:2])/2)),
                digest([seed,'fixed_receiver_revision',rid,pair[1]['source_id'],pair[0]['sha256']])))
            chosen = None
            for ref, descriptor in candidates:
                row = {'receiver': rid, 'candidate': ref, 'source_id': descriptor['source_id'], 'passed': False}
                try:
                    radius = float(np.linalg.norm(np.asarray(descriptor['dimensions_m'][:2])/2))
                    margins = [descriptor['cavity']['flat_floor_radius_m']
                        - np.linalg.norm(p['target'].get('offset_xy_m',[0.,0.]))
                        - selection.sizes.footprint(by_object[p['object']]) for p in original_placements]
                    if min(margins) < .012: raise Rejected('fixed_original_goal_offsets_do_not_fit', margins_m=margins)
                    checks = [checker.check(by_object[oid],ref,descriptor) for oid in member_ids]
                    row['release_checks'] = checks
                    if not all(check['passed'] for check in checks): raise Rejected('receiver_release_geometry_not_qualified')
                    xy = np.asarray(receiver['xy']); table_fit(xy,radius)
                    occupied = [(o['id'],np.asarray(o['position'][:2]),float(np.linalg.norm(o['half_size'][:2])))
                        for o in original_scene['objects']]
                    occupied.extend((s['id'],np.asarray(s['position'][:2]),float(np.linalg.norm(s['half_size'][:2])))
                        for s in old_supports.values() if s['id'] != rid)
                    check_separation(xy,radius,occupied)
                    # A source can bind only one exact offline receiver variant.
                    existing = revised['container_variant_choices'].get(descriptor['source_id'])
                    if existing is not None and existing != ref and descriptor['source_id'] != receiver['source_id']:
                        raise Rejected('replacement_receiver_variant_conflicts_with_other_instance')
                    row.update(passed=True, original_xy_preserved=receiver['xy'], outside_radius_m=radius,
                        fixed_goal_boundary_margins_m=[float(m) for m in margins])
                    chosen = (ref,descriptor); decisions.append(row); break
                except Rejected as error:
                    row.update(reason=error.code,details=error.details);decisions.append(row)
            if chosen is None:
                write(output/'RECEIVER_SELECTION.json',{'rows':decisions,'passed':False})
                raise Rejected('no_release_and_footprint_receiver_for_fixed_scene',receiver=rid)
            ref, descriptor = chosen; old_source = receiver['source_id']; old_ref = revised['container_variant_choices'][old_source]
            receiver['source_id'] = descriptor['source_id']
            revised['container_variant_choices'][descriptor['source_id']] = ref
            if not any(r['source_id']==old_source for r in revised['containers']):
                revised['container_variant_choices'].pop(old_source)
            changes.append({'receiver_id':rid,'original_source_id':old_source,'original_descriptor':old_ref,
                'replacement_source_id':descriptor['source_id'],'replacement_descriptor':ref,
                'xy':receiver['xy'],'yaw':receiver['yaw'],'target_container_change_explicit':True})
        revised['id']='related_release_'+digest([request_ref,changes,VERSION])[:32]
        revised['tasks']=[deepcopy(next(t for t in source['tasks']if t['id']==name))for name in task_ids]
        assert revised['objects']==source['objects'] and revised['variant_choices']==source['variant_choices']
        write(output/'RECEIVER_SELECTION.json',{'rows':decisions,'passed':True,'changes':changes,
            'release_preflight':checker.receipt()})
        write(output/'requests.json',{'schema':original['schema'],'scenes':[revised]})
        summary=coordinator.factory.compile_requests(bundle,output/'requests.json',output/'factory',seed)
        if summary['compiled_scenes']!=1:
            raise Rejected('related_layout_original_compiler_rejected',scene_results=read(output/'factory/scenes.json'))
        scene_ref=_check_compiled(group,output/'factory/cohort.json'); scene=read(resolve(scene_ref))
        old_objects={o['id']:o for o in original_scene['objects']};new_objects={o['id']:o for o in scene['objects']}
        if old_objects != new_objects or original_scene['grasp_caches'] != scene['grasp_caches']:
            raise Rejected('related_layout_changed_original_objects_or_clutter',changed_ids=[key for key in set(old_objects)|set(new_objects)
                if old_objects.get(key)!=new_objects.get(key)])
        for key in ('table_top_z','robot_roots','cameras','appearance'):
            if original_scene[key]!=scene[key]:raise Rejected('related_layout_changed_other_scene_conditions',field=key)
        checks=check_compiled_release(output/'factory/cohort.json');write(output/'RELEASE_PREFLIGHT.json',checks)
        if not checks['passed']:raise Rejected('related_layout_release_geometry_not_qualified')
        new_cases=read(output/'factory/cohort.json')['cases']
        for name,old in zip(task_ids,base_tasks):
            new=read(resolve(next(c['task']for c in new_cases if c['requested_task_id']==name)))
            if [(o['id'],o['object'])for o in old['operations']]!=[(o['id'],o['object'])for o in new['operations']]:
                raise ValueError('related_task_object_order_changed')
        result={'schema':VERSION,'prepared':True,'mode':'posthoc_related_receiver_only_layout',
            'original_case':str(source_case),'cohort':binding(output/'factory/cohort.json'),'scene':scene_ref,
            'tasks':[row['task']for row in new_cases],'changes':changes,'requested_task_ids':list(task_ids),
            'all_payload_source_size_cache_initial_pose_preserved':True,'all_original_clutter_preserved':True,
            'physical_success':'unmeasured','physics_execution_authorized_by_this_module':False}
    except Rejected as error:
        result={'schema':VERSION,'prepared':False,'reason':getattr(error,'code',str(error)),
            'details':getattr(error,'details',{}),'original_case':str(source_case),'no_source_resampling':True,
            'physical_success':'unmeasured'}
    except Exception as error:
        write(output/'ERROR.json',{'schema':VERSION,'status':'unknown_system_failure',
            'error_type':type(error).__name__,'message':str(error),'new_admission_stopped':True})
        raise
    result.update(IK_calls=0,physics_steps=0,GPU_jobs=0,API_calls=0)
    write(output/'RESULT.json',result);return result


def build(bundle, output, *, schedule, selection_config=None, task_capacity_contract=None):
    """At most 12 fixed selections and 12 original factory calls, without retry."""
    schedule = coordinator.schedule_rows(schedule)
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("preserve_existing_source_first_factory_no_resample")
    config = deepcopy(selection_config or {"mode": "source_uniform"})
    contract = selection.task_contract(task_capacity_contract)
    b, entries, population, bins = coordinator.source.admitted_pool(bundle)
    before = digest([population, bins])
    protocol = {"schema": VERSION, "bundle": binding(bundle), "schedule": schedule,
        "selection_config": config, "task_capacity_contract": contract,
        "implementation": binding(__file__), "selection": binding(selection.__file__),
        "retained_coordinator": binding(coordinator.__file__),
        "retained_factory": binding(coordinator.factory.__file__),
        "retained_compiler": binding(coordinator.factory.retained.__file__),
        "maximum_selections": len(schedule), "maximum_compilations": len(schedule),
        "same_attempt_object_reselection": False, "scene_count_reduction": False,
        "new_IK": 0, "new_physics": 0, "new_GPU": 0, "API_calls": 0,
        "default_promoted": False, "execution_allocation_registered": False}
    write(output / "PROTOCOL.json", protocol)
    static = coordinator.factory.static_source_catalog(resolve(b["static_catalog"]), output / "static_inputs")
    rows = []
    for i, setting in enumerate(schedule):
        folder = output / "cases" / f"{i:02d}"
        row = {"index": i, **setting, "source_selected": False, "targets_planned": False,
               "layout_proposed": False, "factory_called": False, "compiled": False,
               "compiled_tasks": 0, "physical_success": "unmeasured"}
        began = time.monotonic()
        try:
            pool = {s: xs for s, xs in population.items() if xs[0][1]["partition"] == setting["partition"]}
            stage_start = time.monotonic()
            group, receipt = selection.choose_objects(pool, setting["N"], setting["partition"], setting["seed"], **config)
            row["selection_seconds"] = time.monotonic() - stage_start
            write(folder / "selection.json", receipt)
            row.update(source_selected=True, selection=binding(folder / "selection.json"))
            stage_start = time.monotonic()
            plan = selection.plan_targets(group, bins, setting["partition"], setting["seed"], contract=contract)
            row["target_planning_seconds"] = time.monotonic() - stage_start
            write(folder / "target_plan.json", plan)
            row.update(targets_planned=True, target_plan=binding(folder / "target_plan.json"))
            identity = "source_first_" + digest([protocol, i, receipt])[:32]
            stage_start = time.monotonic()
            request, allocation = propose(identity, group, plan, setting["seed"], static)
            row["layout_seconds"] = time.monotonic() - stage_start
            write(folder / "allocation.json", allocation)
            write(folder / "requests.json", {"schema": coordinator.factory.VERSION, "scenes": [request]})
            row.update(layout_proposed=True, factory_called=True)
            stage_start = time.monotonic()
            summary = coordinator.factory.compile_requests(bundle, folder / "requests.json", folder / "factory", setting["seed"])
            row["compilation_seconds"] = time.monotonic() - stage_start
            row["factory_summary"] = summary
            if summary["compiled_scenes"] == 1:
                row.update(scene=_check_compiled(group, folder / "factory/cohort.json"),
                    compiled=True, compiled_tasks=summary["compiled_tasks"],
                    source_size_cache_partition_preserved=True, cohort=binding(folder / "factory/cohort.json"))
                release_report = check_compiled_release(folder / "factory/cohort.json")
                write(folder / "RELEASE_PREFLIGHT.json", release_report)
                row["release_preflight"] = binding(folder / "RELEASE_PREFLIGHT.json")
                if not release_report["passed"]:
                    raise Rejected("compiled_release_geometry_not_qualified", report=row["release_preflight"])
            else:
                row["factory_rejections"] = read(folder / "factory/scenes.json")["scenes"]
        except Rejected as error:
            row.update(compiled=False, reason=getattr(error, "code", type(error).__name__),
                       details=getattr(error, "details", {"message": str(error)}))
        except Exception as error:
            row.update(compiled=False, reason="unknown_system_failure", error_type=type(error).__name__,
                       details={"message": str(error)}, new_admission_stopped=True,
                       seconds=time.monotonic()-began)
            write(folder / "RESULT.json", row)
            raise
        row["seconds"] = time.monotonic() - began
        write(folder / "RESULT.json", row); rows.append(row)
        print({"case": i, "compiled": row["compiled"], "reason": row.get("reason")}, flush=True)
    if digest([population, bins]) != before:
        raise ValueError("source_first_adapter_mutated_original_population")
    result = {"schema": VERSION, "protocol": binding(output / "PROTOCOL.json"), "rows": rows,
        "requested_scenes": len(rows), "source_selections": sum(r["source_selected"] for r in rows),
        "target_plans": sum(r["targets_planned"] for r in rows),
        "layout_proposals": sum(r["layout_proposed"] for r in rows),
        "actual_compile_calls": sum(r["factory_called"] for r in rows),
        "compiled_scenes": sum(r["compiled"] for r in rows),
        "compiled_tasks": sum(r["compiled_tasks"] for r in rows if r["compiled"]),
        "all_attempts_retained": True, "source_size_cache_partition_preserved": True,
        "new_IK": 0, "new_physics": 0, "new_GPU": 0, "API_calls": 0,
        "default_promoted": False, "execution_allocation_registered": False,
        "physical_success": "unmeasured"}
    write(output / "RESULTS.json", result)
    return result
