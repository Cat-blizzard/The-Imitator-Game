"""Record complete actual initial PhysX geometry without advancing physics.

The canonical scene contains every registered/raw nonrobot physical entity and
every link/shape of both physical robots, including shapeless links. Geometry
is shape-local; poses are world XYZ/WXYZ, lengths metres. Native primitive
parameters and mesh vertices/triangles/scale remain exact readbacks. Auxiliary
world representations reuse the existing exporter and explicitly retain its
proxy policy; an infinite plane is also recorded as its true halfspace.

No planner, sphere model, environment step, pose assignment or mask change is
performed. Both the original state digest and two complete scene reads must
agree at elapsed step zero. Hashes exclude timestamps and planner identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .curobo_scene import (_array, _finite, _kind, _matrix, _mesh_arrays,
                           _pose, _positive, _put_world_shape, _single)
from .scene_collision import scene_entities

SCHEMA = 'tig-complete-actual-initial-scene-v1'


def _bytes(value: Any) -> bytes:
    """Deterministic UTF-8 representation; nonfinite numbers are forbidden."""
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def scene_content_sha256(scene: dict[str, Any]) -> str:
    """Hash the JSON scene subtree only; independent readers can use this algorithm."""
    if (not isinstance(scene,dict) or not isinstance(scene.get('entities'),dict)
            or not isinstance(scene.get('robots'),list) or len(scene['robots']) != 2
            or not isinstance(scene.get('physical_system'),dict)):
        raise ValueError('Complete canonical scene subtree required')
    return _hash(scene)


def _raw_pose(pose: Any) -> list[float]:
    _matrix(pose)
    return np.r_[_single(pose.p,3,'raw pose XYZ'),
                  _single(pose.q,4,'raw pose WXYZ')].tolist()


def _properties(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Read available native properties without calling a setter."""
    result: dict[str,Any] = {}
    for name in names:
        if not hasattr(obj,name):
            continue
        value = getattr(obj,name)
        if isinstance(value,(bool,np.bool_)):
            result[name] = bool(value)
        elif isinstance(value,(int,np.integer)):
            result[name] = int(value)
        elif hasattr(value,'p') and hasattr(value,'q'):
            result[name] = _raw_pose(value)
        else:
            array = _array(value)
            if not np.isfinite(array).all():
                raise ValueError(f'Nonfinite actual physical property: {name}')
            result[name] = float(array) if array.ndim == 0 else array.tolist()
    return result


def _native_geometry(shape: Any) -> dict[str, Any]:
    kind = _kind(shape)
    if kind == 'PhysxCollisionShapeBox':
        return {'half_size_m':_finite(_positive(shape.half_size,'half size'),(3,),'half size').tolist()}
    if kind in ('PhysxCollisionShapeSphere','PhysxCollisionShapeCapsule','PhysxCollisionShapeCylinder'):
        result = {'radius_m':float(_positive(shape.radius,'radius'))}
        if kind != 'PhysxCollisionShapeSphere':
            result.update(half_length_m=float(_positive(shape.half_length,'half length',zero=True)),
                          longitudinal_local_axis='X')
        return result
    if kind == 'PhysxCollisionShapePlane':
        return {'normal_shape_local':[1.,0.,0.], 'offset_shape_local_m':0.,
                'solid_halfspace':'normal dot point <= offset', 'infinite':True}
    vertices,faces = _mesh_arrays(shape.vertices,np.asarray(shape.get_triangles()).reshape(-1,3))
    scale = _finite(_positive(shape.scale,'native mesh scale'),(3,),'native mesh scale')
    return {'vertices_shape_local_unscaled_m':vertices.tolist(),'triangles':faces.tolist(),
            'native_scale_xyz':scale.tolist(),
            'scaled_vertices_sha256':hashlib.sha256(
                np.ascontiguousarray(vertices*scale,dtype=np.float64).tobytes()).hexdigest()}


def _shape_record(name: str, shape: Any, entity_world: np.ndarray) -> dict[str, Any]:
    groups = list(shape.get_collision_groups())
    if (len(groups) != 4 or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer))
                              or not 0 <= int(x) <= 0xffffffff for x in groups)):
        raise ValueError('Actual PhysX collision groups must be four uint32 words')
    world: dict[str,Any] = {'cuboid':{},'mesh':{}}
    auxiliary = _put_world_shape(world,name,shape,entity_world)
    kind = next(key for key,values in world.items() if name in values)
    local = shape.get_local_pose()
    material = shape.get_physical_material()
    result: dict[str,Any] = {'name':name,'type':_kind(shape),
        'shape_local_pose_raw_xyz_wxyz':_raw_pose(local),
        'shape_world_pose_xyz_wxyz':_pose(entity_world@_matrix(local)),
        'native_local_geometry':_native_geometry(shape),
        'collision_groups_uint32':[int(x) for x in groups],
        'native_shape_properties':_properties(shape,('contact_offset','rest_offset','density')),
        'physical_material':_properties(material,('static_friction','dynamic_friction','restitution')),
        'auxiliary_world_representation':{'kind':kind, 'geometry':world[kind][name],
            'geometry_policy':auxiliary['geometry_policy'],
            'planner_model_created':False}}
    if result['type'] == 'PhysxCollisionShapePlane':
        transform = entity_world@_matrix(local)
        normal = transform[:3,0]
        result['infinite_world_halfspace'] = {'normal':normal.tolist(),
            'offset_m':float(normal@transform[:3,3]),'solid_relation':'normal dot world_point <= offset_m'}
    result['native_local_geometry_sha256'] = _hash(result['native_local_geometry'])
    return result


def _body_properties(body: Any) -> dict[str, Any]:
    return {'type':type(body).__name__, **_properties(body,('kinematic','mass','inertia',
        'cmass_local_pose','linear_velocity','angular_velocity','disable_gravity','linear_damping',
        'angular_damping','solver_position_iterations','solver_velocity_iterations'))}


def _scene_content(raw: Any) -> tuple[dict[str,Any],dict[str,Any]]:
    """Return canonical physical scene and exact native-inventory coverage."""
    import sapien.physx as physx
    agents = tuple(raw.agent.agents)
    if len(agents) != 2:
        raise ValueError('Exactly two physical robot articulations required')
    if hasattr(raw,'left_agent') and (agents[0] is not raw.left_agent or agents[1] is not raw.right_agent):
        raise ValueError('Canonical physical left/right robot order changed')
    accepted: set[Any] = set()
    for agent in agents:
        instances = list(agent.robot._objs)
        if len(instances) != 1 or instances[0] in accepted:
            raise ValueError('Each physical robot must have one distinct native articulation')
        accepted.add(instances[0])
    for registered in raw.scene.articulations.values():
        if len(registered._objs) != 1 or registered._objs[0] not in accepted:
            raise ValueError('External or batched articulation cannot be omitted from initial scene')
    native_links = set(raw.scene.px.get_articulation_link_components())
    if any(body.articulation not in accepted for body in native_links):
        raise ValueError('Unregistered external articulation cannot be omitted from initial scene')
    entities = scene_entities(raw)
    content: dict[str,Any] = {'entities':{},'robots':[],
        'physical_system':_properties(raw.scene.px,('gravity',))}
    covered_nonrobot: set[Any] = set()
    scene_shape_count = 0
    for key,entity in sorted(entities.items()):
        bodies = [c for c in entity.components if isinstance(c,physx.PhysxRigidBaseComponent)]
        if len(bodies)>1 or (bodies and isinstance(bodies[0],physx.PhysxArticulationLinkComponent)):
            raise ValueError('Ambiguous or robot body registered as nonrobot scene entity')
        T = _matrix(entity.pose)
        shapes = list(bodies[0].get_collision_shapes()) if bodies else []
        row: dict[str,Any] = {'entity_name':str(entity.name),'registered_scene_key':key,
            'raw_physics_entity':key not in raw.scene.actors,
            'entity_world_pose_raw_xyz_wxyz':_raw_pose(entity.pose),
            'body':_body_properties(bodies[0]) if bodies else None,
            'shapes':[_shape_record(f'scene::{key}::shape_{i}',shape,T) for i,shape in enumerate(shapes)]}
        content['entities'][key] = row
        scene_shape_count += len(shapes)
        if bodies:
            covered_nonrobot.add(bodies[0])
    native_nonrobot = {body for body in (*raw.scene.px.get_rigid_static_components(),
        *raw.scene.px.get_rigid_dynamic_components())
        if not isinstance(body,physx.PhysxArticulationLinkComponent)}
    if covered_nonrobot != native_nonrobot:
        raise ValueError('Initial scene omits or duplicates an actual nonrobot PhysX body')
    covered_links: set[Any] = set()
    robot_shape_count = 0
    for index,agent in enumerate(agents):
        robot = agent.robot
        links: dict[str,Any] = {}
        for link in robot.get_links():
            name = str(link.get_name())
            instances = list(link._objs)
            if not name or name in links or len(instances) != 1:
                raise ValueError('Robot links require distinct canonical names and one actual body')
            body = instances[0]
            if (not isinstance(body,physx.PhysxArticulationLinkComponent)
                    or body.articulation not in accepted or body in covered_links):
                raise ValueError('Robot link body is absent, duplicated or belongs to another articulation')
            covered_links.add(body)
            T = _matrix(link.pose)
            if not np.allclose(T,_matrix(body.entity.pose),rtol=0,atol=1e-7):
                raise ValueError('Robot wrapper and actual entity poses differ')
            shapes = list(body.get_collision_shapes())
            links[name] = {'entity_name':str(body.entity.name),
                'entity_world_pose_raw_xyz_wxyz':_raw_pose(body.entity.pose),
                'body':_body_properties(body),
                'shapes':[_shape_record(f'robot::{index}::{name}::shape_{i}',shape,T)
                          for i,shape in enumerate(shapes)]}
            robot_shape_count += len(shapes)
        joints = [str(joint.get_name()) for joint in robot.get_active_joints()]
        qpos = _single(robot.get_qpos(),len(joints),'actual robot joint positions')
        qvel = _single(robot.get_qvel(),len(joints),'actual robot joint velocities')
        if len(set(joints)) != len(joints):
            raise ValueError('Actual robot joint names must be unique')
        content['robots'].append({'arm_index':index,'root_world_pose_raw_xyz_wxyz':_raw_pose(robot.pose),
            'joint_names':joints,'joint_positions_rad':qpos.tolist(),'joint_velocities_rad_s':qvel.tolist(),
            'links':dict(sorted(links.items()))})
    if covered_links != native_links:
        raise ValueError('Initial scene does not cover every native articulation link')
    coverage = {'all_covered':True,'nonrobot_entity_count':len(entities),
        'nonrobot_physical_body_count':len(native_nonrobot),'nonrobot_shape_count':scene_shape_count,
        'raw_physics_entity_names':sorted(k for k,v in content['entities'].items() if v['raw_physics_entity']),
        'shapeless_scene_entity_names':sorted(k for k,v in content['entities'].items() if not v['shapes']),
        'robot_count':2,'robot_link_counts':[len(r['links']) for r in content['robots']],
        'robot_shape_counts':[sum(len(x['shapes']) for x in r['links'].values()) for r in content['robots']],
        'native_articulation_link_count':len(native_links),'total_shape_count':scene_shape_count+robot_shape_count,
        'native_body_and_link_inventories_exactly_matched':True,
        'all_collision_shapes_included_without_mask_filtering':True}
    return content,coverage


def export_initial_scene_manifest(raw: Any) -> dict[str, Any]:
    """Read a complete, repeatable initial snapshot; input elapsed_steps must be [0]."""
    from .runtime_io import state_digest
    elapsed_before = _single(raw.elapsed_steps,1,'initial elapsed step')
    if elapsed_before[0] != 0:
        raise ValueError('Complete initial scene must be recorded before any environment step')
    before = state_digest(raw)
    scene,coverage = _scene_content(raw)
    repeated,repeated_coverage = _scene_content(raw)
    after = state_digest(raw)
    elapsed_after = _single(raw.elapsed_steps,1,'final initial elapsed step')
    if (before != after or not np.array_equal(elapsed_before,elapsed_after)
            or scene_content_sha256(scene) != scene_content_sha256(repeated) or coverage != repeated_coverage):
        raise RuntimeError('Reading initial scene changed actual state, geometry, masks or elapsed steps')
    sources = (Path(__file__),Path(__file__).with_name('curobo_scene.py'),
               Path(__file__).with_name('scene_collision.py'),
               Path(__file__).with_name('validate_curobo_payload.py'))
    return {'schema':SCHEMA,'frame':'world','units':'metres; radians; seconds; XYZ/WXYZ',
        'scene':scene,'scene_sha256':scene_content_sha256(scene),'coverage':coverage,
        'read_only_proof':{'state_digest_before':before,'state_digest_after':after,
            'elapsed_steps_before':elapsed_before.astype(int).tolist(),
            'elapsed_steps_after':elapsed_after.astype(int).tolist(),
            'state_and_elapsed_unchanged':True,'full_scene_repeated_read_consistent':True,
            'environment_steps_during_export':0,'planner_or_sphere_model_created':False},
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def save_initial_scene_manifest(output: Path, raw: Any) -> dict[str, Any]:
    """Write initial-scene.json once; return path, file/content hashes and coverage."""
    data = export_initial_scene_manifest(raw)
    path = Path(output)/'initial-scene.json'
    encoded = _bytes(data)
    with path.open('xb') as stream:
        stream.write(encoded)
    return {'schema':SCHEMA,'path':str(path.resolve()),'sha256':hashlib.sha256(encoded).hexdigest(),
        'scene_sha256':data['scene_sha256'],'coverage':data['coverage'],
        'read_only_proof':data['read_only_proof']}


def self_test() -> dict[str, Any]:
    """Actual CPU PhysX geometry, raw ground/wall, both articulations; no step."""
    import tempfile
    from types import SimpleNamespace
    import sapien
    import sapien.physx as physx
    import torch
    assert not torch.cuda.is_initialized()
    native = sapien.Scene([physx.PhysxCpuSystem()])
    material = physx.PhysxMaterial(.5,.4,.1)
    actors: dict[str,Any] = {}
    def entity(name: str, shapes: list[Any], *, register: bool = True) -> Any:
        item = sapien.Entity(); item.name = name
        if shapes:
            body = physx.PhysxRigidStaticComponent()
            for shape in shapes: body.attach(shape)
            item.add_component(body)
        native.add_entity(item)
        if register: actors[name] = SimpleNamespace(_objs=[item])
        return item
    table = entity('table',[physx.PhysxCollisionShapeBox([.2,.3,.02],material) for _ in range(5)])
    table.pose = sapien.Pose([.2,.3,.4])
    entity('ground',[physx.PhysxCollisionShapePlane(material)],register=False)
    wall = entity('wall',[physx.PhysxCollisionShapeBox([.1,.5,.7],material)],register=False)
    entity('visual',[])
    vertices = np.array([[x,y,z] for x in (-.01,.01) for y in (-.02,.02) for z in (-.03,.03)],dtype=np.float32)
    entity('primitives',[physx.PhysxCollisionShapeSphere(.1,material),
        physx.PhysxCollisionShapeCapsule(.1,.2,material),physx.PhysxCollisionShapeCylinder(.1,.2,material),
        physx.PhysxCollisionShapeConvexMesh(vertices,[1.,2.,1.],material),
        physx.PhysxCollisionShapeTriangleMesh(np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=np.float32),
            np.array([[0,1,2],[0,2,3]],dtype=np.uint32),[.1,.1,.1],material)])
    agents = []
    for i in range(2):
        root = physx.PhysxArticulationLinkComponent()
        root.attach(physx.PhysxCollisionShapeBox([.01,.02,.03],material))
        root_entity = sapien.Entity(); root_entity.name = f'robot{i}_root'; root_entity.add_component(root)
        native.add_entity(root_entity)
        child = physx.PhysxArticulationLinkComponent(root)
        child_entity = sapien.Entity(); child_entity.name = f'robot{i}_empty'; child_entity.add_component(child)
        native.add_entity(child_entity)
        links = [SimpleNamespace(_objs=[b],pose=e.pose,get_name=lambda n=n:n)
                 for b,e,n in ((root,root_entity,'root'),(child,child_entity,'empty'))]
        robot = SimpleNamespace(_objs=[root.articulation],pose=root_entity.pose,
            get_links=lambda links=links:links,get_active_joints=lambda:[],
            get_qpos=lambda:np.zeros((1,0)),get_qvel=lambda:np.zeros((1,0)))
        agents.append(SimpleNamespace(robot=robot))
    wrapper = SimpleNamespace(sub_scenes=[native],px=native.physx_system,gpu_sim_enabled=False,
        actors=actors,articulations={})
    raw = SimpleNamespace(scene=wrapper,num_envs=1,agent=SimpleNamespace(agents=agents),
        elapsed_steps=np.array([0],dtype=np.int64))
    raw.get_state_dict = lambda:{'actors':{k:np.r_[e._objs[0].pose.p,e._objs[0].pose.q] for k,e in actors.items()}}
    raw.step = lambda *_args,**_kwargs:(_ for _ in ()).throw(AssertionError('Environment stepping forbidden'))
    data = export_initial_scene_manifest(raw)
    assert data['coverage']['all_covered'] and data['coverage']['robot_link_counts'] == [2,2]
    assert data['coverage']['raw_physics_entity_names'] == ['physics::ground','physics::wall']
    assert data['coverage']['total_shape_count'] == 14
    assert len(data['scene']['entities']['table']['shapes']) == 5
    ground = data['scene']['entities']['physics::ground']['shapes'][0]
    assert ground['native_local_geometry']['infinite'] and ground['infinite_world_halfspace']['normal'] == [1.,0.,0.]
    assert data['scene_sha256'] == export_initial_scene_manifest(raw)['scene_sha256']
    rejected = 0
    with tempfile.TemporaryDirectory() as directory:
        receipt = save_initial_scene_manifest(Path(directory),raw)
        file = Path(receipt['path'])
        assert receipt['sha256'] == hashlib.sha256(file.read_bytes()).hexdigest()
        try: save_initial_scene_manifest(Path(directory),raw)
        except FileExistsError: rejected += 1
        else: raise AssertionError('Existing initial scene evidence overwritten')
    wall.pose = sapien.Pose([1.,0.,0.])
    assert export_initial_scene_manifest(raw)['scene_sha256'] != data['scene_sha256']
    shape = table.components[0].get_collision_shapes()[0]
    shape.set_collision_groups([3,7,0,0])
    assert export_initial_scene_manifest(raw)['scene']['entities']['table']['shapes'][0]['collision_groups_uint32'] == [3,7,0,0]
    raw.elapsed_steps[0] = 1
    try: export_initial_scene_manifest(raw)
    except ValueError: rejected += 1
    else: raise AssertionError('Noninitial state accepted')
    raw.elapsed_steps[0] = 0
    try: _native_geometry(SimpleNamespace())
    except ValueError: rejected += 1
    else: raise AssertionError('Unsupported collision shape accepted')
    external = physx.PhysxArticulationLinkComponent()
    external_entity = sapien.Entity(); external_entity.add_component(external); native.add_entity(external_entity)
    try: export_initial_scene_manifest(raw)
    except ValueError: rejected += 1
    else: raise AssertionError('Unregistered external articulation omitted')
    native.remove_entity(external_entity)
    calls = []
    def changing_state() -> dict[str,np.ndarray]:
        calls.append(True)
        return {'changing':np.array([len(calls)],dtype=np.int64)}
    raw.get_state_dict = changing_state
    try: export_initial_scene_manifest(raw)
    except RuntimeError: rejected += 1
    else: raise AssertionError('Changing actual state passed read-only proof')
    assert raw.elapsed_steps.tolist() == [0] and not torch.cuda.is_initialized()
    return {'status':'passed','environment_steps':0,'GPU_initialized':False,
        'covered_shape_count':data['coverage']['total_shape_count'],'robot_link_counts':[2,2],
        'negative_cases_rejected':rejected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(self_test(),allow_nan=False))
