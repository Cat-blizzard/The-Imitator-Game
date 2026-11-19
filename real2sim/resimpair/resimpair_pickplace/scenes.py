"""Role-based task variants over broad layouts, reusing one pick/place skill."""
import re
import numpy as np
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import read,resolve
from resimpair_grasp.scenes import item
from resimpair_grasp.assets import select
from corl2026_workshop.experiments.autonomous_pipeline_v4.library import make_scene


def container_fits(descriptor, half_size):
    # A conservative disk encloses the whole payload, for arbitrary yaw.
    radius=float(np.linalg.norm(half_size[:2]))
    return radius+.008 <= descriptor['cavity']['flat_floor_radius_m']


def container_choices(catalog, partition, half_size):
    values=[]
    for b in read(catalog)['assets']:
        d=read(resolve(b))
        if d['partition'] != partition or not container_fits(d,half_size):continue
        # A geometrically flat soup surface is not an empty receptacle.
        if re.search(r'\b(soup|filled|water|cereal|fruit|food)\b',d['name'],re.I):continue
        values.append((b,d))
    return values


def sample(entry, containers, clutter_catalog, seed, layout):
    d=read(resolve(entry['asset']));rng=np.random.default_rng(seed)
    identity=d['id']+f'_general_L{layout}'
    scene=make_scene(identity,seed,nuisance=True,layout=False)
    scene.update(schema='general_pickplace_scene_v1',scene_instance_id=identity,
        split_group=d['source_id'],partition=d['partition'],source_coverage='synthetic_only')
    # Broad independent positions. Geometry-only rejection prevents initial
    # overlap; IK failures are kept, not resampled until a rollout succeeds.
    source=np.array([rng.uniform(-.34,.16),rng.uniform(-.54,.54)])
    yaw=float(rng.uniform(-np.pi,np.pi))
    scene['objects']=[item(entry['asset'],d,'target_object',source,yaw,True)]
    scene['grasp_caches']={'target_object':entry['cache']}
    scene['supports']=[];occupied=[(source,float(np.linalg.norm(d['half_size'][:2])))]
    for i,(b,c) in enumerate(containers):
        radius=float(np.linalg.norm(np.asarray(c['dimensions_m'])[:2]/2))
        for trial in range(100):
            xy=np.array([rng.uniform(-.30,.18),rng.uniform(-.54,.54)])
            if all(np.linalg.norm(xy-p)>radius+r+.025 for p,r in occupied):break
        else:raise ValueError('no_nonoverlapping_container_layout')
        cyaw=float(rng.uniform(-np.pi,np.pi));cavity=c['cavity']
        scene['supports'].append({'id':f'container_{i}','shape':'stationary_mesh_container',
            'container_descriptor':b,'position':[*xy,0.],'yaw':cyaw,
            'half_size':(np.asarray(c['dimensions_m'])/2).tolist(),'top_z':cavity['floor_z_m'],
            'semantic_description':f"the {c['name']}",'color_name':'original material','rgba':[.5,.5,.5,1.]})
        occupied.append((xy,radius))
    # Reserve a separate clear goal region for the same-scene table task.
    for _ in range(100):
        table_goal=np.array([rng.uniform(-.30,.18),rng.uniform(-.54,.54)])
        radius=float(np.linalg.norm(d['half_size'][:2]))
        if all(np.linalg.norm(table_goal-p)>radius+r+.035 for p,r in occupied):break
    else:raise ValueError('no_nonoverlapping_table_goal')
    occupied.append((table_goal,radius))
    pool=[(b,c) for b,c in select(clutter_catalog,exclude={d['id']},max_dimension=.18)
          if c['partition'] in {d['partition'],'shared_distractor'}]
    rng.shuffle(pool);categories=set();distinct=[];repeated=[]
    for pair in pool:
        (repeated if pair[1]['category'] in categories else distinct).append(pair)
        categories.add(pair[1]['category'])
    pool=distinct+repeated;categories=set()
    desired=int(rng.integers(3,9));placed=[]
    for b,c in pool:
        radius=float(np.linalg.norm(c['half_size'][:2]))
        for _ in range(100):
            if rng.random()<.65:p=np.array([rng.uniform(.29,.38),rng.uniform(-.73,.73)])
            else:p=np.array([rng.uniform(-.36,.28),rng.choice([-1,1])*rng.uniform(.65,.79)])
            if p[0]+radius>.46 or abs(p[1])+radius>.90:continue
            if any(np.linalg.norm(p-q)<=radius+r+.025 for q,r in occupied):continue
            break
        else:continue
        scene['objects'].append(item(b,c,f'distractor_{len(placed)}',p,rng.uniform(-np.pi,np.pi)))
        occupied.append((p,radius));placed.append(c['source_id']);categories.add(c['category'])
        if len(placed)>=desired:break
    if len(placed)<3:raise ValueError('insufficient_clutter_after_geometry_sampling')
    scene['randomization']={'seed':seed,'source_xy_m':[[-.34,.16],[-.54,.54]],
        'destination_xy_m':[[-.30,.18],[-.54,.54]],'object_yaw_rad':[-float(np.pi),float(np.pi)],
        'container_yaw_rad':[-float(np.pi),float(np.pi)],'distractor_count':len(placed),
        'distractor_source_ids':placed,'distractor_categories':sorted(categories),
        'initial_placement':'nonoverlapping bounding disks; no success-driven position resampling',
        'arm_policy':'joint source/destination IK then complete online path guard',
        'grasp_pose_policy':'asset-local cache transformed to the sampled world pose'}
    goals=[]
    for support,(_,c) in zip(scene['supports'],containers):
        local=np.r_[c['cavity']['center_xy_m'],0.]
        xy=(Rotation.from_euler('z',support['yaw']).as_matrix()@local)[:2]
        goals.append((support['id'],support['id'],xy.tolist()))
    goals.append(('table_relocate','table',table_goal.tolist()))
    tasks=[]
    for name,support,offset in goals:
        tasks.append({'schema':'resimpair_task_intent_v4','id':identity+'_'+name,'family':'relocate',
            'required_skills':['pick_place'],'object_ids':['target_object'],
            'goal':f"move {d['name']} to {support}",
            'roles':{'movable':'target_object','destination':support,'distractors':[o['id'] for o in scene['objects'][1:]]},
            'operations':[{'id':'transfer_0','primitive':'pick_place','object':'target_object','arm':None,
                           'target':{'support':support,'offset_xy':offset,'yaw':yaw}}],
            'required_operation_precedence':[],'equivalence_groups':[],
            'incidental_controller_order_is_required':False,'human_hand_policy':'preserve_robot_body_side',
            'non_target_policy':'preserve_identity_and_pose','release_precondition_control':True,
            'scope':'rigid pick/place from tabletop; no handover, insertion or grasping the container'})
    return scene,tasks
