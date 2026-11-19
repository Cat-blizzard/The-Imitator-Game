"""Candidate task-load ordering with retained source/destination Panda IK.

IK is a necessary condition only. The retained mesh controller still validates
every complete path against the scene, table, payload and inactive arm.
"""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from resimpair_grasp.common import ROOT, binding, read, resolve, write, digest
from resimpair_tabletop.geometry import Clearance

INITIAL = np.array([0., -.4, 0., -2., 0., 1.8, np.pi/4, .04, .04])


def selection_key(candidate, travel, policy):
    """Optional proposal priority; unscored retained caches keep old behavior."""
    if policy is None:
        return (travel,)
    if policy != 'task_load_proxy_then_joint_travel_v1':
        raise ValueError('unknown_grasp_selection_policy')
    from .contact_quality import quality_key, VERSION
    q = candidate.get('task_load_quality', {})
    if q.get('version') != VERSION or not q.get('model_valid') or q.get('worst_load_capacity') is None:
        raise ValueError('missing_or_invalid_task_load_proposal_quality')
    return (*quality_key(candidate)[:2], travel, candidate['id'])


def pose(position, yaw=0.):
    value = np.eye(4)
    value[:3,:3] = Rotation.from_euler('z',yaw).as_matrix()
    value[:3,3] = position
    return value


def pose7(T):
    return np.r_[T[:3,3], Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)]


class ArmSelector:
    def __init__(self, robot_roots, seed=0):
        import mplib
        self.roots = [pose(p) for p in robot_roots]
        self.planners = []
        self.seed = seed
        if hasattr(mplib, 'set_global_seed'):
            mplib.set_global_seed(seed)
        urdf = ROOT/'PolicyGen/mani_skill/assets/robots/panda/panda_v3.urdf'
        self.bindings = [binding(urdf), binding(urdf.with_suffix('.srdf'))]
        for _ in robot_roots:
            self.planners.append(mplib.Planner(urdf=str(urdf), srdf=str(urdf.with_suffix('.srdf')),
                move_group='panda_hand_tcp', joint_vel_limits=np.ones(7)*.8,
                joint_acc_limits=np.ones(7)*.8))

    def ik(self, arm, T, seed_qpos):
        planner = self.planners[arm]
        local = np.linalg.inv(self.roots[arm])@T
        if np.linalg.norm(local[:3,3]) > .95:
            return None, 'outside_conservative_reach_radius'
        status, values = planner.IK(pose7(local), seed_qpos.copy(),
            mask=[False]*7+[True,True], n_init_qpos=8, threshold=.001)
        if status != 'Success':
            return None, status
        q = min(values, key=lambda v:np.linalg.norm(v[:7]-seed_qpos[:7]))
        return np.asarray(q), status

    def select(self, scene, operation, container=None):
        from corl2026_workshop.experiments.autonomous_pipeline_v4.simulation import targets
        item = next(x for x in scene['objects'] if x['id']==operation['object'])
        cache = read(resolve(scene['grasp_caches'][item['id']]))
        policy = cache.get('selection_policy')
        # Reject incomplete candidate metadata before planning. This selection
        # score never changes collision checking or final physical acceptance.
        for c in cache['candidates']:
            selection_key(c, 0., policy)
        start = pose(item['position'],item['yaw'])
        destination = targets(scene, {'operations':[operation]})[item['id']]
        end = pose(np.asarray(destination)+[0,0,.002], operation['target']['yaw'])
        container_check = None
        if container is not None:
            import trimesh
            descriptor=read(resolve(container['container_descriptor']))
            container_check=Clearance(trimesh.load(resolve(descriptor['collision']),force='mesh'))
            container_T=pose(container['position'],container.get('yaw',0))
        decisions=[]; arm_choices=[]
        # Every selected grasp must admit approach, grasp, transport height and
        # release on the SAME arm. A handedness change is never substituted.
        for arm in range(len(self.planners)):
            accepted=[]
            for candidate in cache['candidates'][:12]:
                local=np.asarray(candidate['tcp_in_asset'])
                grasp=start@local; release=end@local
                row={'arm':arm,'candidate':candidate['id'],'accepted':False,'checks':[]}
                if container_check is not None:
                    ok,clearance,reason=container_check.check(np.linalg.inv(container_T)@release,
                        -container['position'][2],.002)
                    row['container_hand_clearance']={'passed':bool(ok),'reason':reason,'table_clearance_m':clearance}
                    if not ok:
                        decisions.append(row);continue
                stages=[('pregrasp',grasp.copy()),('grasp',grasp.copy()),
                        ('above_goal',release.copy()),('release',release.copy())]
                stages[0][1][2,3]+=.17; stages[2][1][2,3]+=.18
                q=INITIAL.copy();cost=0.; configurations=[]
                for name,T in stages:
                    next_q,status=self.ik(arm,T,q)
                    row['checks'].append({'stage':name,'world_tcp':T.tolist(),'status':status,
                                          'qpos':None if next_q is None else next_q.tolist()})
                    if next_q is None:break
                    cost+=float(np.linalg.norm(next_q[:7]-q[:7]));q=next_q;configurations.append(q.tolist())
                else:
                    row['accepted']=True;row['joint_travel_proxy']=cost
                    accepted.append((cost,candidate))
                decisions.append(row)
                if len(accepted)>=3:break
            if accepted:
                accepted.sort(key=lambda x:selection_key(x[1],x[0],policy))
                arm_choices.append((accepted[0][0],arm,[x[1] for x in accepted]))
        if not arm_choices:
            return {'passed':False,'reason':'no_single_arm_source_and_destination_grasp',
                    'decisions':decisions,'scope':'endpoint IK and open-hand cavity clearance; no path certificate',
                    'handover':'requires separate explicit skill; never silently switch arms'}
        arm_choices.sort(key=lambda x:selection_key(x[2][0],x[0],policy))
        cost,arm,candidates=arm_choices[0]
        return {'passed':True,'arm':arm,'candidates':candidates,'decisions':decisions,
                'joint_travel_proxy':cost,'robot_bindings':self.bindings,'seed':self.seed,
                'selection_policy':policy or 'retained_joint_travel',
                'scope':'endpoint IK and open-hand cavity clearance; complete path guard still mandatory'}


def bind_selection(scene, task, directory):
    """Produce derived task/cache files; preserve the original role-level intent."""
    import copy
    out_scene=copy.deepcopy(scene);out_task=copy.deepcopy(task)
    selector=ArmSelector(scene['robot_roots'],scene['seed'])
    reports=[]
    for op in out_task['operations']:
        support=next((x for x in scene['supports'] if x['id']==op['target']['support']),None)
        report=selector.select(scene,op,support if support and 'container_descriptor' in support else None)
        reports.append({'operation':op['id'],**report})
        if not report['passed']:
            write(directory/'arm_selection.json',{'passed':False,'operations':reports})
            return None,None
        op['arm']=report['arm']
        original=scene['grasp_caches'][op['object']];cache=read(resolve(original))
        cache={**cache,'signature':{**cache['signature'],'joint_endpoint_selector':binding(__file__),
                                  'parent_cache':original,'operation':op},'candidates':report['candidates']}
        cache['key']=digest(cache['signature']);cp=directory/(op['object']+'_grasp_cache.json');write(cp,cache)
        out_scene['grasp_caches'][op['object']]=binding(cp)
    write(directory/'arm_selection.json',{'passed':True,'operations':reports})
    return out_scene,out_task


def prepare(scene_binding, task_binding, index, output):
    """Explicit candidate entry; original scene, intent and caches are preserved."""
    import copy
    if output.exists(): raise ValueError('Preserve existing preflight evidence')
    scene=copy.deepcopy(read(resolve(scene_binding)));task=read(resolve(task_binding))
    candidates=read(index)['entries'];output.mkdir(parents=True)
    for op in task['operations']:
        item=next(x for x in scene['objects'] if x['id']==op['object'])
        matched=[e for e in candidates if e['asset']==item['asset_descriptor']]
        if not matched:
            report={'passed':False,'reason':'no_task_load_candidate_for_asset',
                    'object':op['object'],'physical_executions':0}
            write(output/'preflight.json',report);return report
        cache=read(resolve(matched[0]['cache']))
        if cache.get('selection_policy')!='task_load_proxy_then_joint_travel_v1':
            raise ValueError('Explicit quality-ranked cache required')
        scene['grasp_caches'][op['object']]=matched[0]['cache']
    selected,planned=bind_selection(scene,task,output)
    if selected is not None:
        write(output/'scene.json',selected);write(output/'task.json',planned)
    report={'passed':selected is not None,'source_scene':scene_binding,'source_task':task_binding,
            'candidate_index':binding(index),'physical_executions':0,'api_calls':0,
            'scope':'endpoint IK and open-hand container clearance only; full paths and dynamics unmeasured',
            'default_selector_unchanged':True}
    write(output/'preflight.json',report);return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--task',type=Path,required=True)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(prepare(binding(a.scene),binding(a.task),a.index,a.output))
