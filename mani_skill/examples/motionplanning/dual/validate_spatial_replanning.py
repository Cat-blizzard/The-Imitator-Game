"""Thin held-blocker/spatial-detour acceptance using the existing M1 prefix.

Native runs must be scheduled by the root after the120 paired M1 allocations.
This file itself does not launch a batch or reserve validation seeds. Development
uses the design's finite world-goal grid; frozen validation uses one recipe with
no per-seed geometry selection. Setup is a real checked right-arm movement;
nominal/wait/replan queries then share the unchanged real post-setup checkpoint.
Successful controlled held transport is never reported as basket placement.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import faulthandler
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import resource
import signal
import time
import traceback
from typing import Any

import numpy as np

from .combined_collision import validate_schedule
from .dual_arm_execution import execute_dual_held_move
from .multitask_adapters import task_execution_parameters
from .spatial_replanning import (DT,RESERVE_STEPS,ReadOnlyQueryGuard,controller_context,native_snapshot,
    plan_stationary_peer_route,stationary_pair,route_summary,save_schedule,schedule_fk_xyz,
    spatial_change_certificate,validate_frozen_blocker,validate_occupied_endpoint,blocker_witnesses,write_json)
from .spatial_replanning import CAPABILITY_INTERFACE_VERSION, generic_hook_call_keywords
from .trajectory_execution import as_numpy
from .trajectory_schedule import ArmTrajectory,build_hold_schedule
from .validate_grasp import json_value,save_report
from .validate_sync import held_substep_audit,hold_joint_substep_audit
from .validate_sync_compact import compact_configuration
from .validate_multitask import native_run as generic_native_run, termination_handler

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[7]
EVIDENCE=ROOT/'infrastructure/validation/dual-arm/m1-multitask-20261116'
DESIGN=EVIDENCE/'capability-acceptance-design-v1.json'
TASK='TwoRobotPickAppleBananaToBaskets-v1'
RECIPE_SCHEMA='m1-spatial-held-blocker-world-recipe-v1'


def validate_geometry_recipe(value: Any) -> dict[str,Any]:
    if not isinstance(value,dict) or value.get('schema')!=RECIPE_SCHEMA: raise ValueError('Frozen spatial world-recipe schema required')
    geometry={}
    for key in ('blocker_xyz_m','nominal_middle_xyz_m','goal_xyz_m','detour_middle_xyz_m'):
        point=np.asarray(value.get(key),dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all(): raise ValueError(f'Finite world XYZ required:{key}')
        geometry[key]=point.tolist()
    if value.get('orientation_policy')!='measured_post_lift_TCP_WXYZ_frozen_for_entire_case':
        raise ValueError('No orientation search or mid-case goal orientation change allowed')
    if value.get('right_occupancy_policy')!='physical_setup_then_stationary_CLOSED': raise ValueError('Fixed real held-blocker policy required')
    return {'schema':RECIPE_SCHEMA,**geometry,'orientation_policy':value['orientation_policy'],
            'right_occupancy_policy':value['right_occupancy_policy']}


def candidate_recipes(design: dict[str,Any],fixed: dict[str,Any] | None=None) -> list[dict[str,Any]]:
    if fixed is not None: return [validate_geometry_recipe(fixed)]
    grid=design['spatial_replanning']['finite_development_search']
    blockers=sorted(tuple(v) for v in grid['blocker_candidates_xyz_m'])
    goals=sorted(tuple(v) for v in grid['goal_candidates_xyz_m'])
    detours=sorted(tuple(v) for v in grid['detour_middle_candidates_xyz_m'])
    return [validate_geometry_recipe({'schema':RECIPE_SCHEMA,'blocker_xyz_m':blocker,
        'nominal_middle_xyz_m':grid['nominal_required_middle_xyz_m'],'goal_xyz_m':goal,'detour_middle_xyz_m':middle,
        'orientation_policy':'measured_post_lift_TCP_WXYZ_frozen_for_entire_case',
        'right_occupancy_policy':'physical_setup_then_stationary_CLOSED'})
        for blocker,goal,middle in itertools.product(blockers,goals,detours)]


def _world_waypoints(recipe: dict[str,Any],orientations: np.ndarray,which: str) -> np.ndarray:
    middle=recipe['nominal_middle_xyz_m'] if which=='nominal' else recipe['detour_middle_xyz_m']
    if which=='blocker': return np.asarray([np.r_[recipe['blocker_xyz_m'],orientations[1]]])
    if which=='negative': return np.asarray([np.r_[recipe['nominal_middle_xyz_m'],orientations[0]]])
    return np.asarray([np.r_[middle,orientations[0]],np.r_[recipe['goal_xyz_m'],orientations[0]]])


def _query_validation(schedule: Any,context: dict[str,Any],checker: Any,guard: Any) -> dict[str,Any]:
    with guard:
        result=validate_schedule(schedule,context['starts'][0],context['starts'][1],checker,max_joint_step_rad=.02,max_samples=20000)
    if guard.output is not None: write_json(guard.output/f'{guard.label}-validation.json',result)
    return result


def _safe_hold(context: dict[str,Any]) -> Any:
    sources=tuple(ArmTrajectory(key,names,[0.],[start],-1.) for key,names,start in zip(context['keys'],context['names'],context['starts']))
    return build_hold_schedule(*sources,control_timestep_s=DT)


def qualify_recipe(context: dict[str,Any],*,recipes: list[dict[str,Any]],orientations: np.ndarray,
                   solvers: Any,raw: Any,checker: Any,output: Path,report: dict[str,Any],begin: Any) -> tuple[dict | None,Any]:
    """Finite private queries choose one setup; no physical attempt is retried."""
    attempts=[];selected=None;setup_selected=None
    def guard(label: str) -> ReadOnlyQueryGuard:
        return ReadOnlyQueryGuard(lambda:native_snapshot(solvers,raw,checker),output,label)
    for number,recipe in enumerate(recipes):
        label=f'qualify-{number:02d}';item={'index':number,'recipe':recipe,'status':'running'};attempts.append(item)
        def reject(reason: str) -> None:
            item.update(status='rejected',reason=reason)
            write_json(output/f'{label}-qualification.json',item)
        begin('spatial_'+label+'_nominal')
        nominal,nominal_receipt=plan_stationary_peer_route(solvers,raw,checker,moving_arm=0,
            world_waypoints=_world_waypoints(recipe,orientations,'nominal'),configuration=report['execution_configuration'],
            output=output,label=label+'-nominal',report=report)
        item['nominal_native_plan']=nominal_receipt
        if nominal is None:
            reject('native_nominal_no_solution');continue
        clear=_query_validation(nominal,context,checker,guard(label+'-originally-clear'))
        item['nominal_before_blocker_validation']=clear
        if clear['status']!='passed':
            reject('nominal_not_initially_clear');continue
        begin('spatial_'+label+'_blocker_setup')
        setup,setup_receipt=plan_stationary_peer_route(solvers,raw,checker,moving_arm=1,
            world_waypoints=_world_waypoints(recipe,orientations,'blocker'),configuration=report['execution_configuration'],
            output=output,label=label+'-blocker-setup',report=report)
        item['setup_native_plan']=setup_receipt
        if setup is None:
            reject('native_blocker_setup_no_solution');continue
        item['setup_validation']=_query_validation(setup,context,checker,guard(label+'-setup-clear'))
        if item['setup_validation']['status']!='passed':
            reject('blocker_setup_path_not_clear');continue
        private_starts=np.asarray(context['starts']).copy();private_starts[1]=setup.trajectories[1].positions_rad[-1]
        blocked=stationary_pair(nominal.trajectories[0],context,0,paired_starts=private_starts)
        item['static_blocker']=validate_frozen_blocker(blocked,checker,guard_factory=guard,label=label+'-all17-waits')
        write_json(output/f'{label}-static-blocker-proof.json',item['static_blocker'])
        if item['static_blocker']['status']!='passed':
            reject('nominal_static_blocker_not_qualified');continue
        qmid=np.asarray(nominal_receipt['native_waypoint_controller_qpos_rad'][0])
        with guard(label+'-occupied-endpoint-witness'):
            endpoint=checker.check(qmid,private_starts[1])
        item['occupied_endpoint_witness']=endpoint
        write_json(output/f'{label}-occupied-endpoint-witness.json',{'collisions':endpoint})
        if not blocker_witnesses(endpoint,checker):
            reject('nominal_middle_not_a_named_blocked_endpoint');continue
        begin('spatial_'+label+'_detour')
        detour,detour_receipt=plan_stationary_peer_route(solvers,raw,checker,moving_arm=0,
            world_waypoints=_world_waypoints(recipe,orientations,'detour'),configuration=report['execution_configuration'],
            output=output,label=label+'-detour',report=report,paired_starts=private_starts)
        item['detour_native_plan']=detour_receipt
        if detour is None:
            reject('native_detour_no_solution');continue
        with guard(label+'-detour-clear-and-spatial-change'):
            validation=validate_schedule(detour,*private_starts,checker,max_joint_step_rad=.02,max_samples=20000)
            nominal_xyz=schedule_fk_xyz(blocked,checker,context)
            detour_xyz=schedule_fk_xyz(detour,checker,context)
            final_goal=_world_waypoints(recipe,orientations,'nominal')[-1]
            changed=spatial_change_certificate(nominal_xyz,detour_xyz,np.asarray(detour_receipt['native_waypoint_fk_xyz_wxyz'][0])[:3],final_goal,final_goal)
        np.savez(output/f'{label}-sampled-native-FK.npz',nominal_xyz=nominal_xyz,detour_xyz=detour_xyz)
        item.update(detour_validation=validation,spatial_change=changed)
        write_json(output/f'{label}-detour-geometry-proof.json',{'validation':validation,'spatial_change':changed})
        if validation['status']!='passed' or changed['status']!='passed':
            reject('native_detour_geometry_or_spatial_change_failed');continue
        needed=setup.num_steps+5+detour.num_steps+RESERVE_STEPS
        remaining=200-int(as_numpy(raw.elapsed_steps)[0])
        item['budget']={'remaining_control_ticks':remaining,'setup_scheduled_ticks':setup.num_steps,
            'setup_settle_reserve_ticks':5,'detour_scheduled_ticks':detour.num_steps,'final_settle_hover_margin_ticks':RESERVE_STEPS,
            'total_needed_ticks':needed}
        if needed>remaining:
            reject('qualified_native_routes_exceed_frozen200_budget');continue
        item.update(status='selected',reason=None)
        write_json(output/f'{label}-qualification.json',item)
        selected=recipe;setup_selected=setup;break
    write_json(output/'development-qualification.json',{'status':'selected' if selected else 'no_qualified_recipe',
        'candidate_count':len(attempts),'declared_recipe_count':len(recipes),'candidates':attempts,
        'physical_attempts_in_this_search':0,'environment_steps':0,'physics_steps':0,
        'fixed_recipe_validation':len(recipes)==1})
    if selected is not None: write_json(output/'selected-recipe.json',selected)
    return selected,setup_selected


def capability_hook(*,args: Any,output: Path,report: dict,raw: Any,capture: Any,solvers: Any,
                    checker: Any,adapter: Any,plan: Any,begin: Any,finish: Any,substeps: list,
                    ticks: list,references: Any,route: Any,carry_q: Any,original_q: Any) -> None:
    design=json.loads(Path(args.design_file).read_text())
    fixed=None if args.develop else validate_geometry_recipe(json.loads(Path(args.recipe).read_text()))
    recipes=candidate_recipes(design,fixed)
    context=controller_context(solvers,raw,checker)
    prefix_step=int(as_numpy(raw.elapsed_steps)[0])
    prefix_tcp=np.stack([as_numpy(agent.tcp.pose.raw_pose)[0].copy() for agent in raw.agent.agents])
    prefix_object=np.stack([as_numpy(actor.pose.raw_pose)[0].copy() for actor in adapter.targets])
    orientations=np.stack([as_numpy(a.tcp.pose.q)[0].copy() for a in raw.agent.agents])
    report['spatial_capability']={'status':'running','mode':args.case,'prefix_completed_control_step':prefix_step,
        'case_orientations_wxyz':orientations.tolist(),'attachment_capture_count':report['attachment_capture']['count'],
        'native_default_episode_steps':200,'physical_capability_budget':200,
        'basket_placement_success_claimed':False,'declared_candidate_count':len(recipes)}
    recipe,setup=qualify_recipe(context,recipes=recipes,orientations=orientations,solvers=solvers,raw=raw,checker=checker,
        output=output,report=report,begin=begin)
    if recipe is None:
        report.update(status='failed',reason='no_qualified_static_blocker_spatial_recipe');return
    begin('carry_blocker_setup')
    setup_result=execute_dual_held_move(capture,setup,checker,adapter.targets,settle_max_steps=5,stable_steps=3)
    if not finish('carry_blocker_setup',setup_result): return
    checker.refresh();checkpoint=controller_context(solvers,raw,checker)
    fault_step=int(as_numpy(raw.elapsed_steps)[0])
    np.savez(output/'fault-checkpoint.npz',**native_snapshot(solvers,raw,checker)['arrays'])
    report['spatial_capability']['fault_checkpoint_control_step']=fault_step
    report['spatial_capability']['physical_blocker_setup']={
        'environment_steps':fault_step-prefix_step,'right_tcp_start_xyz_wxyz':prefix_tcp[1].tolist(),
        'right_tcp_end_xyz_wxyz':as_numpy(raw.right_agent.tcp.pose.raw_pose)[0].tolist(),
        'right_payload_start_xyz_wxyz':prefix_object[1].tolist(),
        'right_payload_end_xyz_wxyz':as_numpy(adapter.targets[1].pose.raw_pose)[0].tolist(),
        'right_joint_max_change_rad':float(np.max(np.abs(checkpoint['starts'][1]-context['starts'][1])))}
    def guard(label: str) -> ReadOnlyQueryGuard:
        return ReadOnlyQueryGuard(lambda:native_snapshot(solvers,raw,checker),output,label)
    begin('spatial_actual_nominal_source')
    nominal,nominal_receipt=plan_stationary_peer_route(solvers,raw,checker,moving_arm=0,
        world_waypoints=_world_waypoints(recipe,orientations,'nominal'),configuration=report['execution_configuration'],
        output=output,label='actual-nominal',report=report)
    if nominal is None:
        report.update(status='failed',reason='native_nominal_regeneration_failed_after_real_setup');return
    counterfactual_starts=np.asarray(checkpoint['starts']).copy()
    counterfactual_starts[1]=context['starts'][1]
    same_source_before_blocker=stationary_pair(nominal.trajectories[0],checkpoint,0,paired_starts=counterfactual_starts)
    with guard('actual-source-counterfactual-original-right-occupancy'):
        counterfactual=validate_schedule(same_source_before_blocker,*counterfactual_starts,checker,max_joint_step_rad=.02,max_samples=20000)
    write_json(output/'actual-source-counterfactual-original-occupancy.json',{'validation':counterfactual,
        'same_left_source':route_summary(nominal)['sources'][0],
        'pre_setup_right_qpos_rad':context['starts'][1].tolist(),'actual_post_setup_right_qpos_rad':checkpoint['starts'][1].tolist(),
        'physical_environment_steps':0,'scope':'private counterfactual occupancy check; no physical rewind'})
    report['spatial_capability']['same_actual_nominal_source_clear_at_original_right_occupancy']=counterfactual
    if counterfactual['status']!='passed':
        report.update(status='failed',reason='actual_nominal_not_clear_at_original_right_occupancy');return
    static=validate_frozen_blocker(nominal,checker,guard_factory=guard,label='actual-all17-waits')
    write_json(output/'actual-static-blocker-proof.json',static)
    report['spatial_capability']['actual_static_blocker_proof']=static
    if static['status']!='passed':
        report.update(status='failed',reason='actual_stationary_blocker_failed_qualification');return
    before_rejection=int(as_numpy(raw.elapsed_steps)[0])
    with guard('actual-nominal-public-executor-zero-step-rejection'):
        nominal_rejection=execute_dual_held_move(capture,nominal,checker,adapter.targets,settle_max_steps=5)
    rejection_steps=int(as_numpy(raw.elapsed_steps)[0])-before_rejection
    write_json(output/'actual-nominal-public-executor-rejection.json',{'result':nominal_rejection,'environment_steps':rejection_steps})
    report['spatial_capability']['nominal_public_executor_rejection']=nominal_rejection
    if rejection_steps or nominal_rejection.get('reason')!='candidate_collision':
        report.update(status='failed',reason='nominal_conflict_not_rejected_before_execution');return
    if args.case=='endpoint-occupied':
        begin('spatial_occupied_endpoint_negative')
        negative,negative_receipt=plan_stationary_peer_route(solvers,raw,checker,moving_arm=0,
            world_waypoints=_world_waypoints(recipe,orientations,'negative'),configuration=report['execution_configuration'],
            output=output,label='actual-occupied-endpoint',report=report)
        if negative is None:
            report.update(status='failed',reason='negative_native_source_missing_not_geometric_proof');return
        with guard('actual-occupied-endpoint-all17-waits'):
            rejected=validate_occupied_endpoint(negative,checker)
        if rejected['status']!='passed':
            report.update(status='failed',reason='negative_endpoint_not_actually_occupied');return
        before=int(as_numpy(raw.elapsed_steps)[0])
        with guard('actual-occupied-endpoint-public-executor-zero-step-rejection'):
            unsafe_result=execute_dual_held_move(capture,negative,checker,adapter.targets,settle_max_steps=5)
        unsafe_steps=int(as_numpy(raw.elapsed_steps)[0])-before
        rejected.update(public_executor_rejection=unsafe_result,unsafe_route_environment_steps=unsafe_steps)
        write_json(output/'occupied-endpoint-negative-proof.json',rejected)
        if unsafe_steps or unsafe_result.get('reason')!='candidate_collision':
            report.update(status='failed',reason='occupied_endpoint_not_rejected_before_execution');return
        begin('carry_negative_safe_hold')
        held=execute_dual_held_move(capture,_safe_hold(controller_context(solvers,raw,checker)),checker,adapter.targets,
            hold_steps=20,settle_max_steps=5,stable_steps=3)
        if not finish('carry_negative_safe_hold',held): return
        report['spatial_capability']['negative']=rejected
        outcome=held
    else:
        begin('spatial_actual_detour_replan')
        detour,detour_receipt=plan_stationary_peer_route(solvers,raw,checker,moving_arm=0,
            world_waypoints=_world_waypoints(recipe,orientations,'detour'),configuration=report['execution_configuration'],
            output=output,label='actual-detour-replan',report=report)
        if detour is None:
            report.update(status='failed',reason='actual_native_spatial_replan_failed');return
        with guard('actual-detour-complete-qualification'):
            valid=validate_schedule(detour,*checkpoint['starts'],checker,max_joint_step_rad=.02,max_samples=20000)
            nominal_xyz=schedule_fk_xyz(nominal,checker,checkpoint)
            detour_xyz=schedule_fk_xyz(detour,checker,checkpoint)
            goal=_world_waypoints(recipe,orientations,'nominal')[-1]
            change=spatial_change_certificate(nominal_xyz,detour_xyz,np.asarray(detour_receipt['native_waypoint_fk_xyz_wxyz'][0])[:3],goal,goal)
        np.savez(output/'actual-spatial-native-FK.npz',nominal_xyz=nominal_xyz,detour_xyz=detour_xyz)
        report['spatial_capability'].update(actual_detour_validation=valid,actual_spatial_change=change)
        if valid['status']!='passed' or change['status']!='passed':
            report.update(status='failed',reason='actual_native_detour_safety_or_spatial_change_failed');return
        remaining=200-int(as_numpy(raw.elapsed_steps)[0])
        if detour.num_steps+RESERVE_STEPS>remaining:
            report.update(status='failed',reason='actual_detour_exceeds_remaining_frozen200_budget');return
        begin('carry_spatial_detour')
        outcome=execute_dual_held_move(capture,detour,checker,adapter.targets,hold_steps=20,settle_max_steps=5,stable_steps=3)
        if not finish('carry_spatial_detour',outcome): return
    report['held_physics_substep_audit']=held_substep_audit(substeps,references)
    hover_steps=[int(outcome['initial_elapsed_steps'][0])+row['step'] for row in outcome['trace'] if row['phase']=='hover']
    report['physics_hold_joint_audit']=hold_joint_substep_audit(substeps,hover_steps)
    after_lift=[row for row in substeps if row['control_step']>prefix_step]
    original_z=np.asarray(report['attachment_capture']['initial_actor_origin_z_m'])
    height_gains=np.asarray([[row['object_poses'][i][2]-original_z[i] for i in range(2)] for row in after_lift])
    min_gains=height_gains.min(axis=0) if len(height_gains) else np.full(2,-np.inf)
    right_hold=[row for row in substeps if row['control_step']>fault_step]
    right_error=max((float(np.max(np.abs(np.asarray(row['right_qpos'])[:7]-checkpoint['starts'][1]))) for row in right_hold),default=0.)
    same_attachments=all(np.array_equal(a,b) for a,b in zip(checker.attachments,references))
    passed=(report['held_physics_substep_audit']['status']=='passed' and report['physics_hold_joint_audit']['status']=='passed'
        and len(hover_steps)==20 and np.all(min_gains>=.10) and right_error<=.03 and same_attachments
        and int(as_numpy(raw.elapsed_steps)[0])<200)
    report['spatial_capability'].update(status='passed' if passed else 'failed',minimum_actual_height_gain_m=min_gains.tolist(),
        original_attachment_unchanged=same_attachments,right_max_measured_hold_error_rad=right_error,hover_steps=hover_steps,
        goal_recipe=recipe,controlled_held_transport_success=passed,native_official_evaluation=json_value(raw.evaluate()))
    report['spatial_capability']['final_measured_TCP_xyz_wxyz']=[as_numpy(agent.tcp.pose.raw_pose)[0].tolist() for agent in raw.agent.agents]
    report.update(status='passed' if passed else 'failed',controlled_capability_outcome='passed' if passed else 'failed',
        physical_task_outcome='not_claimed_controlled_held_transport',official_200_step_complete=False,stage='finished')
    report['attachment_final_coverage']=json_value(checker.coverage)


def self_test() -> dict[str,Any]:
    import inspect
    from .spatial_replanning import self_test as core_test
    design=json.loads(DESIGN.read_text());recipes=candidate_recipes(design)
    assert len(recipes)==36 and recipes==sorted(recipes,key=lambda r:(r['blocker_xyz_m'],r['goal_xyz_m'],r['detour_middle_xyz_m']))
    assert candidate_recipes(design,recipes[0])==[recipes[0]]
    for key in ('blocker_xyz_m','goal_xyz_m'):
        invalid={**recipes[0],key:[float('nan'),0,0]}
        try: validate_geometry_recipe(invalid)
        except ValueError: pass
        else: raise AssertionError('Invalid world recipe accepted')
    q=np.array([[1.,0,0,0],[1.,0,0,0]])
    assert _world_waypoints(recipes[0],q,'nominal').shape==(2,7)
    assert _world_waypoints(recipes[0],q,'blocker').shape==(1,7)
    assert np.array_equal(_world_waypoints(recipes[0],q,'nominal')[-1],_world_waypoints(recipes[0],q,'detour')[-1])
    context={'keys':('left','right'),'names':(tuple(f'j{i}' for i in range(7)),)*2,'starts':np.zeros((2,7))}
    assert _safe_hold(context).num_steps==0
    actual_keywords=generic_hook_call_keywords('phase_hook')
    inspect.signature(capability_hook).bind(**{key:object() for key in actual_keywords})
    try:
        inspect.signature(capability_hook).bind(**{key:object() for key in actual_keywords if key!='references'})
    except TypeError: pass
    else: raise AssertionError('A missing real generic hook argument was accepted')
    return {'status':'passed','declared_finite_development_recipes':len(recipes),'boundary_cases':7,
        'capability_interface_version':CAPABILITY_INTERFACE_VERSION,'actual_generic_phase_hook_keywords':list(actual_keywords),
        'actual_generic_signature_bind':'passed',
        'core':core_test(),'environment_creations':0,'physics_steps':0,'native_status':'NOT_RUN'}


def main() -> int:
    faulthandler.enable();resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--output-dir',type=Path)
    choice=parser.add_mutually_exclusive_group();choice.add_argument('--develop',action='store_true');choice.add_argument('--recipe',type=Path)
    parser.add_argument('--case',choices=('positive','endpoint-occupied'),default='positive');parser.add_argument('--design-file',type=Path,default=DESIGN)
    args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2));return 0
    signal.signal(signal.SIGTERM,termination_handler)
    if args.output_dir is None or not (args.develop or args.recipe is not None): parser.error('Unused output and explicit --develop or --recipe required')
    if args.develop and args.seed not in range(5): parser.error('Development seed pool is0–4; validation allocations are separate')
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]+',os.environ.get('CUDA_VISIBLE_DEVICES','')): parser.error('GPU UUID required')
    if any(os.environ.get(k)=='1' for k in ('MANI_SKILL_L1','MANI_SKILL_L2','MANI_SKILL_L3')): parser.error('L0 only')
    if args.recipe is not None: validate_geometry_recipe(json.loads(args.recipe.read_text()))
    args.task_id=TASK;args.route='merge-lift';args.execution_profile='execution';args.mode='positive'
    np.random.seed(args.seed)
    for key,value in task_execution_parameters(TASK).items(): setattr(args,key,value)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    baseline=json.loads((EVIDENCE/'prechange-guard.json').read_text())
    files=[Path(p) for p in baseline['baseline_source_sha256']]+[Path(__file__),HERE/'spatial_replanning.py',HERE/'validate_multitask.py',
        HERE/'multitask_adapters.py',HERE/'multitask_recording.py',HERE/'support_surface_collision.py',HERE.parent/'planner.py',HERE.parent/'panda/motionplanner.py']
    paths={str(path):path for path in files};sources={path.name:path.read_bytes() for path in paths.values()}
    if len(sources)!=len(paths): raise ValueError('Source snapshot basename collision')
    snapshot=output/'source';snapshot.mkdir()
    for name,data in sources.items(): (snapshot/name).write_bytes(data)
    (output/'acceptance-design.json').write_bytes(args.design_file.read_bytes())
    if args.recipe is not None: (output/'input-recipe.json').write_bytes(args.recipe.read_bytes())
    configuration=compact_configuration(args)
    report={'status':'running','native_status':'running','started_at':datetime.now(timezone.utc).isoformat(),'task':TASK,'seed':args.seed,
        'capability_interface_version':CAPABILITY_INTERFACE_VERSION,
        'method':'spatial-capability','mode':'positive','capability_case':args.case,'development':args.develop,'execution_configuration':configuration,
        'source_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in sources.items()},'source_original_paths':{p.name:str(p) for p in paths.values()},
        'source_snapshot':str(snapshot),'acceptance_design_sha256':hashlib.sha256(args.design_file.read_bytes()).hexdigest(),
        'input_recipe_sha256':None if args.recipe is None else hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        'official_default_max_episode_steps':200,'max_episode_steps':200,'official_200_step_complete':False,
        'physical_pose_assignment':False,'physical_attachment_created':False,'collision_masks_modified':False,
        'phase_budget':'prefix+actual_blocker_setup+native_detour+settle+20hover <200; no overrides',
        'success_scope':'controlled held transport or correct occupied-endpoint rejection; no basket closure claim',
        'randomization':{'numpy_seed':args.seed,'native_IK':'original backend multistart/random-configuration policy unmodified; all returned solutions retained'},
        'middle_semantics':'nominal middle is replaceable route geometry, not a mandatory task subgoal',
        'native_launch_order':'root schedules only after all120 formal M1 paired allocations terminate'}
    save_report(output/'report.json',report);started=time.monotonic()
    try: generic_native_run(args,output,report,phase_hook=capability_hook)
    except Exception as exc: report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    report.update(native_status='completed',completed_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started)
    if any(hashlib.sha256(Path(report['source_original_paths'][name]).read_bytes()).hexdigest()!=digest for name,digest in report['source_sha256'].items()):
        report.update(status='failed',source_identity_violation=True)
    if hashlib.sha256(args.design_file.read_bytes()).hexdigest()!=report['acceptance_design_sha256'] or (
            args.recipe is not None and hashlib.sha256(args.recipe.read_bytes()).hexdigest()!=report['input_recipe_sha256']):
        report.update(status='failed',protocol_or_recipe_identity_violation=True)
    save_report(output/'report.json',report);print(json.dumps({'status':report['status'],'report':str(output/'report.json')}),flush=True)
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
