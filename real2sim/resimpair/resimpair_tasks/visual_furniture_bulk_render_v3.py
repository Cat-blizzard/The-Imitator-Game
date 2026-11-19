"""Complete the frozen independent-validator interface before any GPU attempt.

The unexecuted v2 preparation is retained. Its single 36-job allocation is
transferred once, with a registry tombstone that prevents the old protocol from
ever submitting. No historical allowance or extra render job is introduced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from pathlib import Path
import shutil
import time

from resimpair_grasp.common import ROOT,binding,digest,read,resolve,write
from . import visual_furniture_bulk_render_v2 as batch
from . import scene_quality_render_v3 as render
from resimpair_pickplace import render_lineage_validator_v3,render_lineage_validator_v4

VERSION='bulk_background_validator_interface_repair_v3'
RULES=ROOT/'validation/pickplace_automation_20261029/independent/scene_domain_v2/quality_rules_v4.json'
REGISTRY=ROOT/'runtime/visual_furniture_bulk_render_v3'


def repair_preparation(previous,output):
    previous=Path(previous).resolve();output=Path(output).resolve()
    if output.exists(): raise ValueError('new_preparation_repair_directory_required')
    old,limits=render.verify(previous)
    if limits['allocation']!={'screen':24,'video':12}:raise ValueError('existing_current_allocation_required')
    with (previous/'.supervisor.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if list((previous/'budget/reservations').glob('*.json')) or list((previous/'cases').glob('*/reservation.json')):
            raise ValueError('cannot_transfer_any_submitted_or_unknown_allocation')
        if list((previous/'cases').glob('*/terminal.json')):raise ValueError('preparation_must_have_zero_attempts')
        REGISTRY.mkdir(parents=True,exist_ok=True)
        migration=REGISTRY/(digest(binding(previous/'budget/limits.json'))+'.json')
        with (REGISTRY/'.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if migration.exists():raise ValueError('allocation_already_transferred_recover_registered_target')
            independent=deepcopy(read(resolve(old['independent_protocol'])))
            sources=deepcopy(old['sources']);cases=deepcopy(old['cases'])
            for source in sources:
                screen=next(c for c in cases if c['source_id']==source['id'] and c['stage']=='screen')
                source['screen_state_indices']=screen['state_indices']
                source['source_artifacts']=source['artifacts']
            source_map={s['id']:s for s in sources}
            for case in cases:
                prior_folder=previous/'cases'/case['id'];folder=output/'cases'/case['id']
                shutil.copytree(prior_folder,folder)
                case['seed']=source_map[case['source_id']]['source_spec']['seed']
                repair=folder/'source_metadata_repair.json'
                if repair.exists():
                    shutil.copy2(repair,folder/'metadata_repair.json')
                    case['metadata_repair']=binding(folder/'metadata_repair.json')
                # Scene references deliberately remain bound to unchanged v2
                # preparation files, proving no appearance was retuned here.
            evaluation={str(Path(m.__file__).relative_to(ROOT)):binding(m.__file__)
                for m in (render_lineage_validator_v3,render_lineage_validator_v4)}
            evaluation[str(RULES.relative_to(ROOT))]=binding(RULES)
            independent.update(schema=VERSION,sources=sources,cases=cases,
                frozen_original_evaluation_dependencies=evaluation,
                original_unexecuted_preparation=binding(previous/'protocol.json'),
                independent_rule_revision='unchanged_v4',
                interface_repair_only=True,visual_scene_files_unchanged=True,
                multi_operation_readiness_not_inferred_from_three_screen_frames=True)
            write(output/'selection_protocol.json',independent)
            repaired_limits={**limits,'allocation_transfer':{
                'schema':VERSION,'original_allocation':binding(previous/'budget/limits.json'),
                'original_registration':old['budget_registration'],
                'original_root':str(previous),'old_actual_attempts':0,'new_render_allowance_added':0,
                'same_authorization_transferred_once':True}}
            write(output/'budget/limits.json',repaired_limits)
            new={**old,'bulk_background_interface_adapter':binding(__file__),
                'independent_protocol':binding(output/'selection_protocol.json'),
                'independent_validator':binding(render_lineage_validator_v4.__file__),
                'independent_quality_rules':binding(RULES),'sources':sources,'cases':cases,
                'budget':binding(output/'budget/limits.json'),
                'code':{**old['code'],str(Path(__file__).relative_to(ROOT)):binding(__file__)},
                'original_unexecuted_preparation':binding(previous/'protocol.json')}
            write(output/'protocol.json',new)
            refs={'protocol':binding(output/'protocol.json')}
            for ref in render.bindings_in(new):resolve(ref);refs['dependency_'+digest(ref)]=ref
            for scene in (read(resolve(c['scene'])) for c in cases):
                for ref in render.bindings_in(scene):resolve(ref);refs['resource_'+digest(ref)]=ref
            write(output/'frozen_inputs.json',refs)
            render.SCOPE_REGISTRY.mkdir(parents=True,exist_ok=True)
            old_scope=render.SCOPE_REGISTRY/(digest(old['independent_protocol'])+'.json')
            with (render.SCOPE_REGISTRY/'.lock').open('a+') as scope_lock:
                fcntl.flock(scope_lock,fcntl.LOCK_EX)
                if old_scope.exists():raise ValueError('old_scope_registered_cannot_transfer')
                write(old_scope,{'schema':VERSION,'status':'cancelled_before_first_attempt',
                    'original_scope':old['independent_protocol'],'replacement_protocol':binding(output/'protocol.json'),
                    'reason':'independent_validator_metadata_missing_no_render_submitted'})
            write(migration,{'schema':VERSION,'allocation_id':limits['allocation_id'],
                'old_protocol':binding(previous/'protocol.json'),'new_protocol':binding(output/'protocol.json'),
                'budget':binding(output/'budget/limits.json'),'old_scope_submission_tombstone':binding(old_scope),
                'new_render_allowance_added':0,'historical_actual_render_jobs':limits['historical_actual_render_jobs'],
                'maximum_lineage_render_jobs':limits['maximum_lineage_render_jobs_including_history'],'created_unix':time.time()})
            write(previous/'UNEXECUTED_PREPARATION_SUPERSEDED.json',{'migration':binding(migration),
                'reason':'required independent-validator interface fields absent','actual_render_attempts':0,
                'original_protocol_files_preserved':True})
            result={**read(previous/'prepared.json'),'schema':VERSION,'interface_repair':binding(migration),
                'actual_render_jobs':0,'new_allowance_added':0}
            write(output/'prepared.json',result)
            render.verify(output)
            return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('repair-preparation');p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('screen','video'),required=True)
    p.add_argument('--limit',type=int,required=True);p.add_argument('--gate',type=Path)
    p=sub.add_parser('gallery');p.add_argument('--output',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='repair-preparation':print(repair_preparation(args.previous,args.output))
    elif args.command=='run':print(batch.run(args.output,stage=args.stage,limit=args.limit,gate=args.gate))
    else:print(batch.gallery(args.output,args.destination))
