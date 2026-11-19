"""Independent native-action, raw-geometry, video and counterexample checks."""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from .bootstrap import ROOT,sha,write
from ..hypericl_task_expansion.registry import PROGRAMS
from ..hypericlzq.eef import panda_fk,eef_actions,apply_offsets


def recording_integrity(receipt,arrays):
    if not receipt.get('fresh_recorder_per_episode'):raise ValueError('unverified episode buffer isolation')
    np.testing.assert_allclose(arrays['states'][0],receipt['initial_measured_states'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(arrays['object_states'][0],receipt['initial_measured_object_states'],atol=2e-6,rtol=0)
    # A relocation clip has no throws. A 10 cm one-frame displacement at 20 Hz
    # is a reset/splice warning, not a plausible native expert trajectory.
    jumps=np.linalg.norm(np.diff(arrays['object_states'][...,:3],axis=0),axis=-1)
    if jumps.max()>.10:raise ValueError('discontinuous object recording; possible reset/splice')


def outcome(receipt,arrays):
    """Derive outcome from raw meshes and states, never collector success bits."""
    goal=PROGRAMS[receipt['program']];selected={i for i,j in goal}
    half=np.asarray(receipt['container_geometry']['inner_half_xy_m'])
    floor=receipt['container_geometry']['floor_top_m'];checks=[]
    for index,destination in goal:
        spec=receipt['asset_specs'][index]
        points=np.asarray(trimesh.load(spec['collision'],force='mesh').vertices)
        state=arrays['final_object_states'][index]
        matrix=Rotation.from_quat(np.r_[state[4:7],state[3]]).as_matrix()
        world=points@matrix.T+state[:3];bounds=np.stack([world.min(0),world.max(0)])
        center=arrays['container_centers_world'][destination]
        inside=bool(np.all(bounds[0,:2]>=center[:2]-half-.002)and np.all(bounds[1,:2]<=center[:2]+half+.002))
        supported=abs(float(bounds[0,2])-floor)<.008
        stable=np.linalg.norm(state[7:10])<.025 and np.linalg.norm(state[10:13])<.4
        released=not arrays['grasp'][-1,:,index].any()
        persistent=np.linalg.norm(np.ptp(arrays['object_states'][-10:,index,:3],axis=0))<.008
        lifted=np.max(arrays['object_states'][:,index,2])-arrays['object_states'][0,index,2]>.08
        actually_grasped=arrays['grasp'][:,:,index].any()
        checks.append(dict(index=index,destination=destination,inside=inside,supported=bool(supported),
            stable=bool(stable),released=bool(released),persistent=bool(persistent),
            lifted=bool(lifted),grasped=bool(actually_grasped)))
    untouched=all(np.linalg.norm(arrays['object_states'][:,i,:3]-arrays['object_states'][0,i,:3],axis=-1).max()<.015
                  for i in range(2)if i not in selected)
    return bool(untouched and all(all(v for k,v in c.items()if k not in ('index','destination'))for c in checks)),checks


def verify(path):
    r=json.loads(path.read_text())
    for name,digest in r['sha256'].items():
        if sha(path.parent/name)!=digest:raise ValueError('payload hash mismatch')
    for spec in r['asset_specs']:
        for filename,digest in spec['files'].items():
            if sha(filename)!=digest:raise ValueError('source mesh changed')
    with np.load(r['arrays'],allow_pickle=False)as f:a={k:f[k]for k in f.files}
    recording_integrity(r,a)
    n=r['length']
    for key,shape in [('states',(n,18)),('native_commands',(n,16)),('poses',(n,2,4,4)),
                      ('target_poses',(n,2,4,4)),('object_states',(n,2,13))]:
        if a[key].shape!=shape or not np.isfinite(a[key]).all():raise ValueError(('shape/finite',key))
    actual=panda_fk(np.stack([a['states'][:,:7],a['states'][:,9:16]],1))
    target=panda_fk(np.stack([a['native_commands'][:,:7],a['native_commands'][:,8:15]],1))
    np.testing.assert_allclose(actual,a['poses'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(target,a['target_poses'],atol=2e-6,rtol=0)
    np.testing.assert_allclose(a['robot_roots_world'][None]@actual,a['tcp_world'],atol=3e-6,rtol=0)
    np.testing.assert_allclose(a['openness'],(a['native_commands'][:,[7,15]]+1)/2,atol=1e-6,rtol=0)
    np.testing.assert_allclose(a['timestamps_seconds'],np.arange(n)/r['fps'],atol=1e-8,rtol=0)
    if a['cross_robot_contact_force_N'].max()>=.5:raise ValueError('cross-robot collision')
    if hashlib.sha256(np.ascontiguousarray(a['initial_physics_state']).tobytes()).hexdigest()!=r['initial_physics_sha256']:
        raise ValueError('initial physics hash')
    for t in (0,n//3,2*n//3,n-2):
        actions,state,mask=eef_actions(r,a,t,16)
        restored=apply_offsets(a['poses'][t],actions.reshape(-1,2,7)[...,:6])
        np.testing.assert_allclose(restored,a['target_poses'][t:t+len(actions)],atol=3e-6,rtol=0)
        if not mask.all():raise ValueError('unexpected invalid action dimensions')
    passed,checks=outcome(r,a)
    if not passed:raise ValueError(('independent physical outcome rejected',checks))
    negatives={}
    if r['program']in ('pick_first_to_left','pick_first_to_right'):
        wrong=dict(r,program='pick_first_to_right'if r['program']=='pick_first_to_left'else'pick_first_to_left')
        negatives['wrong_destination_rejected']=not outcome(wrong,a)[0]
    i=PROGRAMS[r['program']][0][0]
    held={k:v.copy()for k,v in a.items()};held['grasp'][-1,:,i]=True
    negatives['unreleased_object_rejected']=not outcome(r,held)[0]
    floating={k:v.copy()for k,v in a.items()};floating['final_object_states'][i,2]+=.05
    negatives['floating_object_rejected']=not outcome(r,floating)[0]
    moving={k:v.copy()for k,v in a.items()};moving['final_object_states'][i,7]=.2
    negatives['moving_object_rejected']=not outcome(r,moving)[0]
    if not all(negatives.values()):raise ValueError(('negative control incorrectly accepted',negatives))
    stamps=[];images=[];ids=set(np.rint(np.linspace(0,n-1,8)).astype(int).tolist())
    with av.open(r['video'])as video:
        stream=video.streams.video[0]
        for index,frame in enumerate(video.decode(stream)):
            if(frame.width,frame.height)!=(224,224):raise ValueError('resolution changed')
            stamps.append(float(frame.pts*stream.time_base))
            if index in ids:images.append(frame.to_ndarray(format='rgb24'))
    if len(stamps)!=n:raise ValueError('video/label count mismatch')
    np.testing.assert_allclose(stamps,a['timestamps_seconds'],atol=1e-5,rtol=0)
    return r,dict(receipt=str(path),passed=True,physical_checks=checks,negative_controls=negatives,
                  frames=n,receipt_sha256=sha(path)),images


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'rich_asset_pilot_v1');args=p.parse_args()
    paths=sorted((args.root/'episodes').glob('*/receipt.json'))
    if not paths:raise ValueError('no admitted receipts')
    groups=defaultdict(list);checks=[];sample_rows=[]
    for path in paths:
        row,check,images=verify(path);checks.append(check);groups[row['group']].append(row)
        sample_rows.append((row,images))
    for group,rows in groups.items():
        if len({r['initial_physics_sha256']for r in rows})!=1:raise ValueError('goal-dependent initial physics')
        if len({r['initial_rgb_sha256']for r in rows})!=1:raise ValueError('goal-dependent initial raw RGB')
        if len({r['initial_decoded_rgb_sha256']for r in rows})!=1:raise ValueError('goal-dependent encoded initial RGB')
        if len({json.dumps(r['randomization'],sort_keys=True)for r in rows})!=1:raise ValueError('goal-dependent randomization')
    pages=[]
    for start in range(0,len(sample_rows),4):
        subset=sample_rows[start:start+4];sheet=Image.new('RGB',(8*224,len(subset)*250),'white');draw=ImageDraw.Draw(sheet)
        for j,(r,images)in enumerate(subset):
            y=j*250;draw.text((4,y+4),r['id']+' | '+','.join(s['name']for s in r['asset_specs']),fill='black')
            for k,rgb in enumerate(images):sheet.paste(Image.fromarray(rgb),(k*224,y+24))
        path=args.root/f'video_samples_{len(pages)+1:02d}.jpg';sheet.save(path,quality=95);pages.append(str(path))
    report=dict(passed=True,scope='Every admitted receipt, all native video frames, raw geometry and executed-command EEF round trips',
        episodes=len(paths),independent_layouts=len(groups),frames=sum(c['frames']for c in checks),checks=checks,
        sample_pages=pages,goal_independent_initial_physics_rgb_and_appearance=True,
        source_sha256=sha(__file__),visual_video_review_pending=True,
        not_a_general_contact_solver_or_real_world_calibration_certificate=True)
    write(args.root/'independent_verification.json',report)
    print(json.dumps({k:v for k,v in report.items()if k!='checks'}),flush=True)


if __name__=='__main__':main()
