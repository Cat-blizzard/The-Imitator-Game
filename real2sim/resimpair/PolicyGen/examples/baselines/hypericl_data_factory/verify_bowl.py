"""Independent raw-state/cavity/contact verification for real-bowl skills."""
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .bootstrap import sha
from .verify_rich import recording_integrity
from .collect_dataset import verify_native
from ..hypericlzq.eef import panda_fk,eef_actions,apply_offsets
from ..hypericl_task_expansion.registry import PROGRAMS


def outcome(r,a,contacts,destination=None):
    index,target=PROGRAMS[r['program']][0]
    if destination is not None:target=destination
    points=np.asarray(trimesh.load(r['asset_specs'][index]['collision'],force='mesh').vertices)
    state=a['final_object_states'][index]
    rotation=Rotation.from_quat(np.r_[state[4:7],state[3]]).as_matrix()
    world=points@rotation.T+state[:3];center=a['container_centers_world'][target]
    profile=r['receptacle']['floor_profile'];floor=profile['center_inner_floor_z_m'];rim=profile['rim_z_m']
    c=dict(inside_opening=bool(np.linalg.norm(world[:,:2]-center[:2],axis=1).max()<.075),
        below_rim=bool(world[:,2].min()<rim-.01),not_below_floor=bool(world[:,2].min()>floor-.008),
        stable=bool(np.linalg.norm(state[7:10])<.015 and np.linalg.norm(state[10:13])<.3),
        released=not bool(a['grasp'][-1,:,index].any()),
        persistent=bool(np.linalg.norm(np.ptp(a['object_states'][-10:,index,:3],axis=0))<.008),
        grasped=bool(a['grasp'][:,:,index].any()),
        lifted=bool(a['object_states'][:,index,2].max()-a['object_states'][0,index,2]>.08),
        bowl_support_contact=bool((contacts['support_force_N'][-20:,index]>.02).sum()>=15),
        no_hard_robot_bowl_contact=bool(contacts['robot_bowl_force_N'].max()<4))
    other=1-index;c['unselected_untouched']=bool(np.linalg.norm(a['object_states'][:,other,:3]-a['object_states'][0,other,:3],axis=1).max()<.015)
    return all(c.values()),c


def verify(path):
    path=Path(path);r=json.loads(path.read_text())
    for name,digest in r['sha256'].items():
        if sha(path.parent/name)!=digest:raise ValueError('payload hash mismatch')
    for spec in [*r['asset_specs'],r['receptacle']]:
        for filename,digest in spec['files'].items():
            if sha(filename)!=digest:raise ValueError('asset changed')
    if sha(r['contact_data'])!=r['contact_sha256']:raise ValueError('contact payload changed')
    with np.load(r['arrays'],allow_pickle=False)as f:a={k:f[k]for k in f.files}
    with np.load(r['contact_data'],allow_pickle=False)as f:contacts={k:f[k]for k in f.files}
    recording_integrity(r,a);n=r['length']
    if r['randomization'].get('transfer_to_unique_empty_bowl'):
        selected,destination=PROGRAMS[r['program']][0]
        if destination!=1 or r['asset_specs'][selected]['name']!='apple':raise ValueError('transfer goal/source object mismatch')
        mesh=np.asarray(trimesh.load(r['asset_specs'][selected]['collision'],force='mesh').vertices)
        state=a['object_states'][0,selected]
        rot=Rotation.from_quat(np.r_[state[4:7],state[3]]).as_matrix();world=mesh@rot.T+state[:3]
        source=a['container_centers_world'][0,:2];target=a['container_centers_world'][1,:2]
        if np.linalg.norm(source-target)<.22:raise ValueError('transfer bowls insufficiently distinct')
        if np.linalg.norm(world[:,:2]-source,axis=1).max()>=.075:raise ValueError('apple did not start inside source bowl')
        if world[:,2].min()>=r['receptacle']['floor_profile']['rim_z_m']-.01:raise ValueError('apple initially above cavity')
        other=1-selected
        if np.linalg.norm(a['object_states'][0,other,:2]-target)<.1:raise ValueError('destination initially occupied')
    for key,shape in [('states',(n,18)),('native_commands',(n,16)),('poses',(n,2,4,4)),
        ('target_poses',(n,2,4,4)),('object_states',(n,2,13))]:
        if a[key].shape!=shape or not np.isfinite(a[key]).all():raise ValueError(('shape/finite',key))
    if contacts['support_force_N'].shape!=(n,2)or contacts['robot_bowl_force_N'].shape!=(n,):raise ValueError('contact timeline mismatch')
    if not all(np.isfinite(v).all()for v in contacts.values()):raise ValueError('nonfinite contact')
    actual=panda_fk(np.stack([a['states'][:,:7],a['states'][:,9:16]],1))
    target=panda_fk(np.stack([a['native_commands'][:,:7],a['native_commands'][:,8:15]],1))
    np.testing.assert_allclose(actual,a['poses'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(target,a['target_poses'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(a['robot_roots_world'][None]@actual,a['tcp_world'],atol=3e-6,rtol=0)
    np.testing.assert_allclose(a['openness'],(a['native_commands'][:,[7,15]]+1)/2,atol=1e-6,rtol=0)
    np.testing.assert_allclose(a['timestamps_seconds'],np.arange(n)/r['fps'],atol=1e-8,rtol=0)
    if a['cross_robot_contact_force_N'].max()>=.5:raise ValueError('cross robot collision')
    if hashlib.sha256(np.ascontiguousarray(a['initial_physics_state']).tobytes()).hexdigest()!=r['initial_physics_sha256']:raise ValueError('initial physics hash')
    for t in (0,n//3,2*n//3,n-2):
        action,state,mask=eef_actions(r,a,t,16)
        restored=apply_offsets(a['poses'][t],action.reshape(-1,2,7)[...,:6])
        np.testing.assert_allclose(restored,a['target_poses'][t:t+len(action)],atol=3e-6,rtol=0)
        if not mask.all():raise ValueError('action mask invalid')
    passed,checks=outcome(r,a,contacts)
    if not passed:raise ValueError(('bowl outcome rejected',checks))
    index,target_index=PROGRAMS[r['program']][0];negative={}
    if len(a['container_centers_world'])>1:
        negative['wrong_bowl_rejected']=not outcome(r,a,contacts,destination=1-target_index)[0]
    for name in ('held','floating','moving'):
        bad={k:v.copy()for k,v in a.items()}
        if name=='held':bad['grasp'][-1,:,index]=True
        if name=='floating':bad['final_object_states'][index,2]+=float(r['receptacle']['floor_profile']['rim_z_m'])+.05
        if name=='moving':bad['final_object_states'][index,7]=.2
        negative[name+'_rejected']=not outcome(r,bad,contacts)[0]
    lost_contact={k:v.copy()for k,v in contacts.items()};lost_contact['support_force_N'][:]=0
    negative['unsupported_rejected']=not outcome(r,a,lost_contact)[0]
    if not all(negative.values()):raise ValueError(('bad negative control',negative))
    pts=[]
    with av.open(r['video'])as c:
        for f in c.decode(video=0):
            if(f.width,f.height)!=(224,224):raise ValueError('policy video size mismatch')
            pts.append(float(f.pts*f.time_base))
    if len(pts)!=n:raise ValueError('policy frame count mismatch')
    np.testing.assert_allclose(pts,a['timestamps_seconds'],atol=1e-5,rtol=0);verify_native(r)
    return dict(passed=True,frames=n,physical_checks=checks,negative_controls=negative,receipt_sha256=sha(path),
        scope='All native actions, full policy/native video timelines, raw fruit geometry and actual bowl support forces.')
