"""Same-camera visual candidate: role-aware surfaces and metric CC0 furniture.

The v2 sampler and its physical invariants are reused unchanged. All edits stay
inside scene_domain; meshes have no collision or action/state slots.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from resimpair_grasp.common import binding, digest, read, resolve
from . import scene_domain_v2 as base

VERSION='scene_quality_v3'
ENV_ID='ReSimPairSceneQuality-v3'
FURNITURE=Path('assets/visual_furniture_v1/repair_ledger_v2/metric_repair_v3/catalog.json')
MATERIALS=Path('assets/appearance_v3/material_policy_v3/index.json')
CONDITIONS=('baseline_v2','material_only','furniture_only','combined')


def apply_furniture(scene,catalog_path=FURNITURE):
    result=deepcopy(scene);d=result['scene_domain'];g=d['geometry']
    if g is None:raise ValueError('furniture_requires_workspace_geometry')
    catalog=read(catalog_path);entries={e['id']:e for e in catalog['entries']}
    model_id='steel_frame_shelves_01' if d['category'] in ('workshop','storage') else 'wooden_display_shelves_01'
    model=entries[model_id];resolve(model['mesh']);bounds=np.asarray(model['bounds_m'])
    # Native breadth faces the room; keep exact publisher scale and Z-up height.
    yaw=float(np.pi/2 if np.ptp(bounds,axis=0)[0]>np.ptp(bounds,axis=0)[1] else 0.)
    rotation=Rotation.from_euler('z',yaw).as_matrix();local=base.corners(bounds)@rotation.T
    b=np.stack([local.min(axis=0),local.max(axis=0)]);size=np.ptp(b,axis=0)
    protected=np.asarray(g['protected_bounds_m']);floor=next(x for x in g['boxes'] if x['id']=='floor')
    floor_z=floor['center_m'][2]+floor['size_m'][2]/2
    center=np.array([protected[0,0]-.09-b[1,0],-.55,floor_z-b[0,2]])
    world=b+center
    if np.all(world[1]>protected[0]-.06) and np.all(world[0]<protected[1]+.06):
        raise ValueError('furniture_intrudes_into_original_motion_envelope')
    def architecture(box):
        name=box['id']
        return (name in ('floor','ceiling','rear_wall','left_wall','right_wall','front_far_wall','ceiling_luminaire')
                or name.startswith('window_'))
    retained=[deepcopy(x) for x in g['boxes'] if architecture(x)]
    removed=[x['id'] for x in g['boxes'] if not architecture(x)]
    back=[x for x in retained if x['id']=='rear_wall' or x['id'].startswith('window_')]
    rear_inner=min(x['center_m'][0]+x['size_m'][0]/2 for x in back)
    extension=max(0.,rear_inner+.04-world[0,0])
    if extension>.35:raise ValueError('metric_furniture_requires_too_large_room_extension')
    if extension:
        for box in retained:
            if box['id']=='rear_wall' or box['id'].startswith('window_'):box['center_m'][0]-=extension
            elif box['id'] in ('floor','ceiling','left_wall','right_wall'):
                box['center_m'][0]-=extension/2;box['size_m'][0]+=extension
    for box in retained:
        lo=np.array(box['center_m'])-np.array(box['size_m'])/2;hi=np.array(box['center_m'])+np.array(box['size_m'])/2
        if np.all(hi>protected[0]-.06) and np.all(lo<protected[1]+.06):raise ValueError('room_extension_enters_motion_envelope')
    left=next(x for x in retained if x['id']=='left_wall');right=next(x for x in retained if x['id']=='right_wall')
    if world[0,1]<left['center_m'][1]+left['size_m'][1]/2+.02 or world[1,1]>right['center_m'][1]-right['size_m'][1]/2-.02:
        raise ValueError('furniture_does_not_fit_room_width')
    g['boxes']=retained
    g['furniture']=[{'id':'metric_furniture_0','source_id':model_id,'source':model['source'],'mesh':model['mesh'],
        'catalog':binding(catalog_path),'position_m':center.tolist(),'yaw_rad':yaw,'scale':[1.,1.,1.],
        'world_bounds_m':world.tolist(),'source_size_m':model['size_m'],'collision_shapes_added':0}]
    g['structural_signature']=digest({'boxes':retained,'furniture':g['furniture']})
    d['furniture_policy']={'implementation':binding(__file__),'catalog':binding(catalog_path),
        'removed_procedural_furniture':removed,'native_metric_scale_preserved':True,
        'rear_architecture_extension_m':extension,'lights_and_camera_retained_for_ablation':True,
        'selection_basis':'scene category only; no task object identity or color',
        'primary_model':model_id,'new_physical_actors':0}
    base.validate_scene(result)
    return result


def apply(baseline,condition,seed,*,material_index=MATERIALS,furniture_catalog=FURNITURE):
    if condition not in CONDITIONS:raise ValueError('unknown_visual_quality_condition')
    result=deepcopy(baseline)
    if condition in ('material_only','combined'):
        from .scene_material_policy_v3 import apply_policy
        result=apply_policy(result,seed,index_path=material_index,mode='semantic_rgb',asset_partition='development')
    if condition in ('furniture_only','combined'):result=apply_furniture(result,furniture_catalog)
    if base.physical_scene(result)!=base.physical_scene(baseline):raise ValueError('visual_candidate_changed_physics')
    if result['cameras']!=baseline['cameras'] or result['scene_domain']['lighting']!=baseline['scene_domain']['lighting']:
        raise ValueError('controlled_visual_candidate_changed_camera_or_lighting')
    return result


def _install_furniture(env):
    import sapien
    d=base.validate_scene(env.scene_spec);items=(d.get('geometry') or {}).get('furniture',[])
    if not items:return
    def inventory():
        return [[base._physical_entity_snapshot(e) for e in s.entities if any(isinstance(c,sapien.physx.PhysxRigidBaseComponent) for c in e.components)] for s in env.scene.sub_scenes]
    before=inventory();evidence=env.scene_domain_runtime_evidence
    for scene_index,subscene in enumerate(env.scene.sub_scenes):
        for item in items:
            shape=sapien.render.RenderShapeTriangleMesh(str(resolve(item['mesh'])),scale=np.asarray(item['scale'],dtype=np.float32))
            body=sapien.render.RenderBodyComponent();body.attach(shape)
            entity=sapien.Entity();entity.name='scene_domain/'+item['id'];entity.add_component(body)
            xyzw=Rotation.from_euler('z',item['yaw_rad']).as_quat()
            entity.set_pose(sapien.Pose(item['position_m'],np.r_[xyzw[3],xyzw[:3]]));subscene.add_entity(entity)
            evidence['render_only_entities'].append({'scene_index':scene_index,'name':entity.name,
                'component_types':[type(c).__name__ for c in entity.components],'mesh':item['mesh'],'source_id':item['source_id']})
    after=inventory()
    if digest(before)!=digest(after):raise RuntimeError('furniture_changed_physical_entities')
    evidence['metric_furniture']={'count':len(items),'before_sha256':digest(before),'after_sha256':digest(after),
                                 'new_physical_entities':0,'implementation':binding(__file__)}


def register(scene,performance=None,contact_policy='native_default'):
    from mani_skill.utils.registration import REGISTERED_ENVS,register_env
    base.register(scene,performance,contact_policy);parent=REGISTERED_ENVS[base.ENV_ID].cls
    @register_env(ENV_ID,max_episode_steps=4000)
    class SceneQualityEnv(parent):
        def _load_scene(self,options):
            super()._load_scene(options);_install_furniture(self)
    return ENV_ID
