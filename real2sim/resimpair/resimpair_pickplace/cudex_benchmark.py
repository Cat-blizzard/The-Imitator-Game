"""Frozen-source paired actual-cuDex/geometry comparison, zero generation API."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import ROOT, binding, read, resolve, write

BASE = ROOT/'runs/cudex_physical_20261026'
SELECTION = ROOT/'validation/cudex_contact_quality_20261026/selection.json'


def scene_for(entry, index):
    from .scenes import container_choices
    from resimpair_grasp.assets import select
    from resimpair_grasp.scenes import item
    from corl2026_workshop.experiments.autonomous_pipeline_v4.library import make_scene
    descriptor = read(resolve(entry['asset']))
    seed = 2026102600 + index
    rng = np.random.default_rng(seed)
    audit = read(ROOT/'runs/support_surface_audit_20261026_v1.json')
    permitted = {r['descriptor']['sha256'] for r in audit['rows'] if r['passed']}
    choices = [(b, c) for b, c in container_choices(ROOT/'runs/pickplace_containers_20261026_catalog_v2.json',
        descriptor['partition'], descriptor['half_size']) if b['sha256'] in permitted]
    choices.sort(key=lambda x: (x[1]['dimensions_m'][2], x[1]['id']))
    if not choices: raise ValueError('no_partition_and_size_compatible_upward_support')
    cb, container = choices[0]
    sign = -1 if index % 2 == 0 else 1
    xy, goal = [-.22, sign*.47], [-.12, sign*.10]
    yaw, cyaw = map(float, rng.uniform(-np.pi, np.pi, 2))
    scene = make_scene(f'cudex_pair_{index:02d}', seed, nuisance=True, layout=False)
    scene.update(schema='general_pickplace_scene_v1', scene_instance_id=f'cudex_pair_{index:02d}',
        split_group=descriptor['source_id'], partition=descriptor['partition'], source_coverage='synthetic_only')
    scene['objects'] = [item(entry['asset'], descriptor, 'target_object', xy, yaw, True)]
    scene['grasp_caches'] = {'target_object': entry['cache']}
    scene['supports'] = [{'id':'container_0', 'shape':'stationary_mesh_container',
        'container_descriptor':cb, 'position':[*goal, 0.], 'yaw':cyaw,
        'half_size':(np.asarray(container['dimensions_m'])/2).tolist(),
        'top_z':container['cavity']['floor_z_m'], 'semantic_description':f"the {container['name']}",
        'color_name':'original material', 'rgba':[.5,.5,.5,1.]}]
    clutter = [(b, d) for b, d in select(ROOT/'assets/expanded_library/scene_catalog_v3.json',
        exclude={descriptor['id']}, max_dimension=.16) if d['partition'] in {descriptor['partition'], 'shared_distractor'}]
    rng.shuffle(clutter)
    if len(clutter)<3: raise ValueError('insufficient_partition_compatible_clutter')
    for i, ((b, d), pos) in enumerate(zip(clutter, [[.32,-.60],[.34,0],[.32,.60]])):
        scene['objects'].append(item(b, d, f'distractor_{i}', pos, rng.uniform(-np.pi,np.pi)))
    scene['randomization'] = {'seed':seed, 'source_and_goal':'controlled paired positions, alternating side',
        'object_and_container_yaw':'independent uniform yaw', 'distractor_count':3,
        'initial_placement':'disjoint bounding disks; no success-driven resampling',
        'distractor_source_ids':[read(resolve(o['asset_descriptor']))['source_id'] for o in scene['objects'][1:]]}
    offset=(Rotation.from_euler('z',cyaw).as_matrix()@np.r_[container['cavity']['center_xy_m'],0])[:2].tolist()
    task={'schema':'resimpair_task_intent_v4', 'id':scene['id']+'_place', 'family':'relocate',
        'required_skills':['pick_place'], 'object_ids':['target_object'],
        'goal':f"place {descriptor['name']} in {container['name']}",
        'roles':{'movable':'target_object','destination':'container_0',
                 'distractors':[o['id'] for o in scene['objects'][1:]]},
        'operations':[{'id':'transfer_0','primitive':'pick_place','object':'target_object','arm':None,
            'target':{'support':'container_0','offset_xy':offset,'yaw':yaw}}],
        'required_operation_precedence':[], 'equivalence_groups':[],
        'incidental_controller_order_is_required':False, 'human_hand_policy':'preserve_robot_body_side',
        'non_target_policy':'preserve_identity_and_pose','release_precondition_control':True}
    return scene, task


def prepare_scenes():
    for i, entry in enumerate(read(SELECTION)['entries']):
        scene, task = scene_for(entry, i)
        write(BASE/'scenes'/f'{i:02d}.json', scene)
        write(BASE/'scenes'/f'{i:02d}_intent.json', task)
    write(BASE/'scene_manifest.json', {'selection':binding(SELECTION),
        'frozen_before_physical_results':True, 'scenes':[binding(BASE/'scenes'/f'{i:02d}.json') for i in range(8)]})


def register(root):
    from .cartesian_reachability import bind_selection
    from .runner import code
    from . import evaluator
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import core, resources
    if root.exists(): raise ValueError('Preserve existing campaign')
    root.mkdir(parents=True)
    snapshot=resources.gpu_snapshot(2)
    write(root/'experiment.json', {'schema':'general_pickplace_experiment_v1',
        'budget':str(ROOT/'runs/cudex_physical_budget_20261026'), 'gpu':2,'gpu_uuid':snapshot['uuid'],
        'timeout_seconds':360,'code':code(),'legacy_code':core.implementation(),
        'seed':2026102600,'api_limit':0, 'source_selection':binding(SELECTION),
        'protocol':binding(ROOT/'validation/cudex_physical_20261026/protocol.json'),
        'addendum':binding(ROOT/'validation/cudex_physical_20261026/proposal_protocol_addendum.json'),
        'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    write(root/'rules.json', {'version':evaluator.VERSION,'files':evaluator.bindings(),
        'frozen_before_attempts':True,'retained_rules_unchanged':True})
    cases=[]
    for i, entry in enumerate(read(SELECTION)['entries']):
        descriptor=read(resolve(entry['asset']))
        base_scene=read(BASE/'scenes'/f'{i:02d}.json');intent=read(BASE/'scenes'/f'{i:02d}_intent.json')
        key='dev_00' if i==0 else f'asset_{i:02d}'
        for method in ['geometry','cudex']:
            identity=f'asset_{i:02d}_{method}'
            folder=root/'inputs'/identity
            scene, task=copy.deepcopy(base_scene),copy.deepcopy(intent)
            task['id']=identity
            cache=entry['cache'] if method=='geometry' else binding(BASE/'proposal_caches'/key/'grasp_cache.json')
            scene['grasp_caches']['target_object']=cache
            write(folder/'role_intent.json',task)
            selected,planned=bind_selection(scene,task,folder)
            case={'id':identity,'execution_id':identity,'asset':entry['asset'],
                'source_id':descriptor['source_id'],'partition':descriptor['partition'],
                'experiment_split':entry['split'],'split_group':descriptor['source_id'],
                'shared_scene':binding(BASE/'scenes'/f'{i:02d}.json'), 'method':method,
                'source_coverage':'synthetic_only','layout':0,
                'arm_selection':binding(folder/'arm_selection.json'),'eligible':selected is not None}
            if selected is not None:
                write(folder/'scene.json',selected);write(folder/'task.json',planned)
                case.update(scene=core.bind(folder/'scene.json'),task=core.bind(folder/'task.json'))
            else:
                case['preflight_reason']='no_native_candidate_or_single_arm_endpoint_solution'
            cases.append(case)
            print(identity,case['eligible'],flush=True)
    write(root/'cohort.json', {'cases':cases,'failures_retained':True,
        'same_physical_scenes_assets_initial_seeds_and_controller':True,
        'common_selector':'native Cartesian world-Z sweep + retained endpoint IK',
        'new_heldout_sources_not_used_for_tuning':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scenes','register'])
    p.add_argument('--root',type=Path,default=BASE/'paired_physics')
    a=p.parse_args()
    if a.mode=='scenes':prepare_scenes()
    else:register(a.root)
