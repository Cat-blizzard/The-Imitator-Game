"""Task semantics are selected objects, goal relations and event precedence.

Object names do not grant new contact skills. This first factory composes the
existing rigid pick/place adapter; other mechanisms require their own state,
controller and independent event checks before registration.
"""
from copy import deepcopy
import numpy as np
from resimpair_grasp.common import read,resolve,digest,write
from resimpair_grasp.scenes import sample,item

FAMILIES={'select','clear','arrange','ordered'}
CAPABILITIES={
    'pick_place':{'supported':True,'controller':'resimpair_grasp.motion.solve',
                  'acceptance':'generic_mesh_acceptance_v1','requires':['rigid_manipulation_candidate','panda_grasp_cache','supported_goal_surface']},
    'pour':{'supported':False,'missing':['functional_container_geometry','fill_or_liquid_state','pour_controller','independent_flow_event_check']},
    'insert':{'supported':False,'missing':['mating_interface_parameters','asset_specific_insertion_adapter_registration']},
    'articulate':{'supported':False,'missing':['joint_and_handle_semantics','articulated_scene_adapter_registration']},
    'stack':{'supported':False,'missing':['load_bearing_support_qualification','stack_stability_guard']}
}


def validate_task(scene,task):
    if any(not CAPABILITIES.get(s,{}).get('supported') for s in task['required_skills']):
        raise ValueError('Unsupported capability; mesh category alone is not a task adapter')
    objects={o['id']:o for o in scene['objects']}
    if len(task['operations'])!=len({o['object'] for o in task['operations']}):
        raise ValueError('Repeated transfers need explicit operation windows; not supported by this factory')
    if set(task['object_ids'])!={o['object'] for o in task['operations']}:raise ValueError('Role/program mismatch')
    for op in task['operations']:
        obj=objects[op['object']];d=read(resolve(obj['asset_descriptor']))
        if obj['kind']!='movable' or not d['roles']['rigid_manipulation_candidate']:
            raise ValueError('Static-only asset cannot fill manipulated role')
        if op['arm'] not in {0,1} or op['primitive']!='pick_place':raise ValueError('Unsupported effector/operator')
        if op['target']['support'] not in {'table',*[x['id'] for x in scene['supports']]}:
            raise ValueError('Goal support not registered')
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import planning_protocols,simulation
    ids=list(objects);supports=ids+[x['id'] for x in scene['supports']]+['table']
    return planning_protocols.pick_place(scene,task,ids,supports,simulation.targets(scene,task))


def compose(scene,roles,families):
    unknown=set(families)-FAMILIES
    if unknown:raise ValueError('Unregistered task family: '+','.join(sorted(unknown)))
    if len(roles)!=2:raise ValueError('Current composite scene supports two manipulated assets')
    base_ops=[]
    for r in roles:
        base_ops.append({'id':'move_'+r['object'],'primitive':'pick_place','object':r['object'],'arm':r['arm'],
                        'target':{'support':r['tray'],'offset_xy':[0.,0.],'yaw':r['yaw']}})
    programs=[]
    def add(name,ops,family,precedence=()):
        task={'schema':'resimpair_task_intent_v4','id':scene['id']+'_'+name,'family':family,
              'composition_template':name,'required_skills':['pick_place'],'object_ids':[o['object'] for o in ops],
              'goal':name.replace('_',' '),'operations':deepcopy(ops),'required_operation_precedence':list(precedence),
              'equivalence_groups':[],'incidental_controller_order_is_required':False,
              'human_hand_policy':'preserve_robot_body_side','non_target_policy':'preserve_identity_and_pose',
              'release_precondition_control':True,'source_coverage':'synthetic_only',
              'scope':'semantic compositions of rigid pick_place; not new contact mechanisms'}
        validate_task(scene,task);programs.append(task)
    if 'select' in families:
        for i,op in enumerate(base_ops):add(f'select_{i}',[op],'relocate')
    if 'clear' in families:add('clear_pair',base_ops,'sort')
    if 'arrange' in families:
        ops=deepcopy(base_ops)
        for op in ops:
            sign=-1 if op['arm']==0 else 1
            op['target'].update(support='table',offset_xy=[-.18,sign*.14])
        add('arrange_pair',ops,'arrange')
    if 'ordered' in families:
        ops=list(reversed(base_ops))
        add('ordered_pair',ops,'sort',[(ops[0]['id'],ops[1]['id'])])
    return programs


def create_scene(catalog,entries,seed,number,output,families):
    a,b=entries;da,db=read(resolve(a['asset'])),read(resolve(b['asset']))
    if da['source_id']==db['source_id'] or da['partition']!=db['partition']:
        raise ValueError('Distinct active assets with one partition are required')
    scene_id=f'scene_{number:03d}_'+digest([da['source_id'],db['source_id'],seed])[:10]
    # Remove both target identities from the static sampling pool, including
    # static-role descriptors of an otherwise identical asset.
    all_assets=read(catalog)['assets']
    pool=[x for x in all_assets if read(resolve(x))['source_id'] not in {da['source_id'],db['source_id']}]
    pool_path=output/(scene_id+'_static_pool.json');write(pool_path,{'assets':pool})
    layout=number%2
    scene,_=sample(pool_path,a['asset'],a['cache'],seed,layout)
    scene.update(id=scene_id,scene_instance_id=scene_id,
                 split_group='asset_group_'+digest(sorted([da['source_id'],db['source_id']]))[:16])
    first=scene['objects'][0];first['id']='object_a'
    first_tray=scene['supports'][0];first_tray['id']='tray_a'
    rng=np.random.default_rng(seed+7919);arm=1-layout;sign=-1 if arm==0 else 1
    second=item(b['asset'],db,'object_b',[rng.uniform(-.27,-.13),sign*rng.uniform(.24,.36)],rng.uniform(-np.pi,np.pi),True)
    scene['objects'].insert(1,second)
    scene['grasp_caches']={'object_a':a['cache'],'object_b':b['cache']}
    scene['supports'].append({'id':'tray_b','shape':'tray','color_name':'purple','rgba':[.46,.23,.64,1.],
        'position':[float(rng.uniform(.015,.10)),sign*float(rng.uniform(.24,.36)),0.],
        'half_size':[.105,.105,.006],'top_z':.012,'wall_height':.025,
        'semantic_description':'the purple open shallow destination tray'})
    roles=[{'object':'object_a','arm':layout,'tray':'tray_a','yaw':first['yaw']},
           {'object':'object_b','arm':arm,'tray':'tray_b','yaw':second['yaw']}]
    scene['asset_roles']=roles
    scene['active_source_ids']=[da['source_id'],db['source_id']]
    scene['randomization']['second_source_xy']=second['position'][:2]
    scene['randomization']['independent_trajectories_not_equal_to_asset_or_task_counts']=True
    # All possible goals are reserved, not only the default task's destination.
    # Existing sampler puts clutter in outer strips; assert the actual footprints.
    placed=[];proposals=0
    for o in scene['objects'][2:]:
        radius=float(np.linalg.norm(o['half_size'][:2]))
        for _ in range(120):
            proposals+=1
            if rng.random()<.65:
                p=np.array([rng.uniform(.205+radius,.46-radius),rng.uniform(-.9+radius,.9-radius)]) if radius<.127 else None
            else:
                p=np.array([rng.uniform(-.35,.30),rng.choice([-1,1])*rng.uniform(.53+radius,.90-radius)]) if radius<.185 else None
            if p is None:continue
            if any(np.linalg.norm(p-q)<radius+r+.025 for q,r in placed):continue
            if any(np.linalg.norm(p-np.array(m['position'][:2]))<radius+np.linalg.norm(m['half_size'][:2])+.025 for m in scene['objects'][:2]):continue
            o['position'][:2]=p.tolist();placed.append((p,radius));break
        else:raise ValueError('Cannot place clutter outside all task workspaces within finite sampling budget')
    scene['randomization']['clutter_position_proposals']=proposals
    scene['randomization']['shared_work_area']={'x_max':.20,'absolute_y_max':.52}
    return scene,compose(scene,roles,families)
