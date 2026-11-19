"""Robot actions in the current end-effector frame, in metres / rotation radians.

Every horizon target uses the SAME query frame: dp=R_t.T(p_target-p_t),
dr=Log(R_t.T R_target). These are pose offsets, not velocities or Euler differences.
Panda uses the simulator's exact URDF TCP. AgiBot uses its published flange,
with a separate embodiment adapter because the physical tool offset is unknown.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

URDF = Path(__file__).resolve().parents[3]/'mani_skill/assets/robots/panda/panda_v2.urdf'


@lru_cache(None)
def panda_chain():
    root=ET.parse(URDF).getroot()
    by_child={j.find('child').get('link'):j for j in root.findall('joint')}
    chain=[];link='panda_hand_tcp'
    while link in by_child:
        j=by_child[link];chain.append(j);link=j.find('parent').get('link')
    result=[]
    for j in reversed(chain):
        origin=j.find('origin');T=np.eye(4)
        if origin is not None:
            T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
            T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        active=j.get('type')=='revolute'
        if active:
            if j.find('axis').get('xyz')!='0 0 1':raise ValueError('Unexpected Panda joint axis')
            lim=j.find('limit');bounds=(float(lim.get('lower')),float(lim.get('upper')))
        else:bounds=None
        result.append((T,active,bounds))
    if sum(x[1] for x in result)!=7:raise ValueError('Wrong Panda TCP chain')
    return result


def panda_fk(q, jacobian=False):
    q=np.asarray(q,dtype=np.float64)
    if q.shape[-1]!=7:raise ValueError('Expected seven Panda joints')
    T=np.broadcast_to(np.eye(4),q.shape[:-1]+(4,4)).copy();origins=[];axes=[];i=0
    for origin,active,_ in panda_chain():
        T=T@origin
        if active:
            if jacobian:origins.append(T[...,:3,3].copy());axes.append(T[...,:3,2].copy())
            c=np.cos(q[...,i]);s=np.sin(q[...,i]);i+=1
            R=np.broadcast_to(np.eye(4),T.shape).copy()
            R[...,0,0]=c;R[...,0,1]=-s;R[...,1,0]=s;R[...,1,1]=c
            T=T@R
    if not jacobian:return T
    o=np.stack(origins,-2);z=np.stack(axes,-2)
    J=np.concatenate([np.cross(z,T[...,:3,3][...,None,:]-o),z],-1).swapaxes(-1,-2)
    return T,J


def pose_offsets(current,target):
    Rt=np.swapaxes(current[...,:3,:3],-1,-2)
    dp=(Rt@(target[...,:3,3]-current[...,:3,3])[...,None])[...,0]
    R=Rt@target[...,:3,:3]
    dr=Rotation.from_matrix(R.reshape(-1,3,3)).as_rotvec().reshape(R.shape[:-2]+(3,))
    return np.concatenate([dp,dr],-1).astype(np.float32)


def apply_offsets(current,offset):
    shape=np.broadcast_shapes(current.shape[:-2],offset.shape[:-1]);T=np.broadcast_to(current,shape+(4,4)).copy()
    T[...,:3,3]+=(current[...,:3,:3]@offset[...,:3,None])[...,0]
    R=Rotation.from_rotvec(np.asarray(offset[...,3:6]).reshape(-1,3)).as_matrix().reshape(offset.shape[:-1]+(3,3))
    T[...,:3,:3]=current[...,:3,:3]@R
    return T


def eef_actions(row,arrays,t,horizon):
    offset=int(row.get('target_start_offset',0 if row['source']=='imitator_sim' else 1))
    end=min(len(arrays['poses']),t+offset+horizon);idx=slice(t+offset,end)
    d=pose_offsets(arrays['poses'][t],arrays['target_poses'][idx])
    a=np.concatenate([d,arrays['openness'][idx,...,None]],-1).reshape(-1,14)
    m=arrays['eef_mask'][idx].copy()
    current_valid=arrays.get('pose_valid')
    if current_valid is not None:m &= np.repeat(current_valid[t],7)[None]
    return a,arrays['states'][t].copy(),m


def specs_eef():
    common=dict(representation='eef_relative_v3',action_dim=14,fps='per_episode',
        layout='[L dx,dy,dz,rx,ry,rz,open; R dx,dy,dz,rx,ry,rz,open]',
        coordinate_frame='Each arm current local end-effector frame: R(t)^T delta_p and Log(R(t)^T R(target)).',
        units='translation m; rotation vector rad; openness 0=closed,1=open',
        time_reference='query_state',chunk_convention='All targets relative to query t, never integrated.',
        state_layout='Embodiment-specific current joint proprioception; NOT action targets.')
    return {
        'panda_eef14_v3':dict(**common,state_dim=18,tcp='panda_hand_tcp from simulator panda_v2.urdf',
            target_start_offset=0,supervision='FK of native commanded joint targets relative to measured TCP(t).'),
        'franka_single_eef14_v3':dict(**common,state_dim=18,tcp='Canonical Panda URDF TCP computed from measured q; single arm in first slot, second masked.',
            target_start_offset=1,supervision='Future measured TCP motion. Not independently calibrated native controller commands.'),
        'agibot_flange14_v3':dict(**common,state_dim=16,tcp='Published flange pose, NOT a calibrated fingertip TCP.',
            target_start_offset=1,supervision='Future measured flange motion + released gripper signal; auxiliary motion labels.'),
    }


def panda_ik(target,seed,iterations=80):
    """Batched damped least squares, bounded joints, with an explicit failure mask.

    IK cannot restore the original redundant joint configuration from a TCP pose.
    Initialize from the observed configuration to prefer a nearby solution.
    """
    target=np.asarray(target,dtype=np.float64);q=np.broadcast_to(seed,target.shape[:-2]+(7,)).copy()
    bounds=np.array([b for _,active,b in panda_chain() if active]);q=np.clip(q,bounds[:,0],bounds[:,1])
    for _ in range(iterations):
        pose,J=panda_fk(q,True)
        dr=Rotation.from_matrix((target[...,:3,:3]@pose[...,:3,:3].swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(q.shape[:-1]+(3,))
        dp=target[...,:3,3]-pose[...,:3,3];error=np.concatenate([dp,dr],-1)
        if np.max(np.linalg.norm(dp,axis=-1))<2e-5 and np.max(np.linalg.norm(dr,axis=-1))<2e-4:break
        JT=J.swapaxes(-1,-2)
        dq=(JT@np.linalg.solve(J@JT+np.eye(6)*1e-4,error[...,None]))[...,0]
        scale=np.maximum(1.,np.max(np.abs(dq),axis=-1,keepdims=True)/.15)
        q=np.clip(q+dq/scale,bounds[:,0],bounds[:,1])
    pose=panda_fk(q);error=pose_offsets(pose,target)
    ok=(np.linalg.norm(error[...,:3],axis=-1)<.001)&(np.linalg.norm(error[...,3:],axis=-1)<.01)
    return q,ok,error


def panda_eef_to_native(actions,raw_state):
    """Restore native joint commands BEFORE temporal aggregation. Failed IK holds q."""
    import torch
    tensor=isinstance(actions,torch.Tensor)
    a=actions.detach().cpu().numpy() if tensor else np.asarray(actions)
    state=raw_state.detach().cpu().numpy() if tensor else np.asarray(raw_state)
    arm=np.stack([state[...,:7],state[...,9:16]],-2)
    current=panda_fk(arm)[...,None,:,:,:]
    chunks=a.reshape(*a.shape[:-1],2,7)
    targets=apply_offsets(current,chunks[...,:6])
    q,ok,error=panda_ik(targets,arm[...,None,:,:])
    q=np.where(ok[...,None],q,arm[...,None,:,:])
    grip=2*np.clip(chunks[...,6:7],0,1)-1
    result=np.concatenate([q,grip],-1).reshape(*a.shape[:-1],16).astype(np.float32)
    stats=dict(targets=int(ok.size),failed=int((~ok).sum()),max_position_error_m=float(np.max(np.linalg.norm(error[...,:3],axis=-1))))
    return (torch.as_tensor(result,device=actions.device,dtype=actions.dtype) if tensor else result),stats
