"""Add stationary triangle-mesh containers to the retained two-camera runtime."""
import numpy as np
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import read,resolve,write

ENV_ID='ReSimPairMeshContainers-v1'


def register(scene,performance,contact_policy='native_default'):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import simulation
    from mani_skill.utils.registration import REGISTERED_ENVS,register_env
    import sapien
    simulation.register(scene,performance)
    parent=REGISTERED_ENVS['ReSimPairSharedTable-v4'].cls

    @register_env(ENV_ID,max_episode_steps=4000)
    class ContainerEnv(parent):
        def _load_scene(self,options):
            actual=self.scene_spec
            self.scene_spec={**actual,'supports':[]}
            try: super()._load_scene(options)
            finally: self.scene_spec=actual
            self.support_actors=[]
            for item in actual['supports']:
                d=read(resolve(item['container_descriptor']))
                builder=self.scene.create_actor_builder()
                builder.add_nonconvex_collision_from_file(str(resolve(d['collision'])),scale=[1,1,1])
                builder.add_visual_from_file(str(resolve(d['visual'])),scale=[1,1,1])
                builder.initial_pose=sapien.Pose(item['position'],Rotation.from_euler('z',item.get('yaw',0)).as_quat(scalar_first=True))
                self.support_actors.append(builder.build_static(item['id']))
            self.support_ids=self.object_ids+[s['id'] for s in actual['supports']]+['table']
            self.support_bodies={o._bodies[0].entity:i for i,o in enumerate(self.objects+self.support_actors+[self.table_scene.table])}
            self.support_impulses=np.zeros((len(self.objects),len(self.support_ids)))
            from .contact_policy import apply
            self.contact_policy_evidence=apply(self,contact_policy)


def native_evidence(rec,directory):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import array
    rows=[]
    for item,actor in zip(rec.scene_spec['supports'],rec.support_actors):
        body=actor._bodies[0];shapes=[]
        for shape in body.get_collision_shapes():
            shapes.append({'type':type(shape).__name__,'vertices':np.asarray(shape.vertices).tolist(),
                           'triangles':np.asarray(shape.get_triangles()).tolist(),
                           'scale':np.asarray(shape.scale).tolist(),
                           'local_pose':np.asarray(shape.local_pose.to_transformation_matrix()).tolist()})
        rows.append({'id':item['id'],'descriptor':item['container_descriptor'],
                     'world_pose':array(actor.pose.to_transformation_matrix()).reshape(4,4).tolist(),
                     'shapes':shapes,'static':True})
    write(directory/'native_containers.json',{'schema':'native_stationary_containers_v1','containers':rows})
    write(directory/'native_contact_policy.json',rec.unwrapped.contact_policy_evidence)
