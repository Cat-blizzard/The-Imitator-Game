"""Optional pretrained proposals; Panda adaptation and acceptance remain local.

No training. Run inference in a separate environment on a locked, idle GPU.
The model predicts its gripper BASE frame; our cache stores native Panda TCP.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from resimpair_grasp.common import ROOT,read,write,binding,resolve,digest

UPSTREAM=ROOT/'runtime/upstream/GraspGenX'
COMMIT='b9429097728cb1c430dd78b92edf17ba318aad03'


def tcp_transform(urdf):
    from resimpair_tabletop.geometry import urdf_origin
    tree=ET.parse(urdf).getroot();frames={'world':np.eye(4)}
    for _ in range(10):
        for joint in tree.findall('joint'):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if joint.get('type')=='fixed' and parent in frames:
                frames[child]=frames[parent]@urdf_origin(joint.find('origin'))
    return frames['panda_hand_tcp']


def infer(index,output,gpu):
    from corl2026_workshop.experiments.autonomous_pipeline_v4 import resources
    if output.exists():raise ValueError('Inference batch exists; recover instead of resubmit')
    entries=read(index)['entries']
    if not 1<=len(entries)<=6:raise ValueError('Inference sample limit is 6 assets')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip()!=COMMIT:
        raise ValueError('Unpinned upstream checkout')
    output.mkdir(parents=True)
    with contextlib.ExitStack() as stack:
        for p in resources.lock_paths(gpu):
            f=stack.enter_context(p.open('r'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=resources.gpu_snapshot(gpu);resources.check_idle(snapshot)
        write(output/'reservation.json',{'input':binding(index),'entries':entries,'started_unix':time.time(),
            'gpu':snapshot,'timeout_seconds':600,'upstream_commit':COMMIT,'api_limit':0,
            'model_loading_and_inference_only':True,'max_tries_per_asset':1,'seed':2026102688})
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=snapshot['uuid'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                 RESIMPAIR_GRASPGENX_RESERVATION=str((output/'reservation.json').resolve()))
        start=time.monotonic()
        with (output/'inference.log').open('x') as log:
            proc=subprocess.Popen([str(ROOT/'runtime/envs/graspgenx/bin/python'),'-u','-m','resimpair_pickplace.graspgenx','worker','--output',str(output.resolve())],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(output/'process.json',{'pid':proc.pid})
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code='timeout'
        result={'returncode':code,'elapsed_seconds':time.monotonic()-start,'api_calls':0,'physical_executions':0}
        write(output/'terminal.json',result);return result


def worker(output):
    if os.environ.get('RESIMPAIR_GRASPGENX_RESERVATION')!=str((output/'reservation.json').resolve()):
        raise ValueError('GPU supervisor required')
    import torch,trimesh
    torch.set_num_threads(2);torch.manual_seed(2026102688);np.random.seed(2026102688)
    from graspgenx import get_checkpoints_version_dir
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.grasp_server import GraspGenXSampler
    start=time.monotonic();checkpoint=get_checkpoints_version_dir()
    cfg=load_model_cfg(str(checkpoint/'gen'),str(checkpoint/'dis'))
    sampler=GraspGenXSampler(cfg,'franka_panda',assets_dir=str(UPSTREAM/'assets'))
    write(output/'model.json',{'generator':binding(cfg.eval.gen_checkpoint),'discriminator':binding(cfg.eval.dis_checkpoint),
        'load_seconds':time.monotonic()-start,'torch_version':torch.__version__,'device':torch.cuda.get_device_name(0)})
    for entry in read(output/'reservation.json')['entries']:
        d=read(resolve(entry['asset']));folder=output/d['id'];folder.mkdir()
        scene=trimesh.load(resolve(d['visual']),force='scene');mesh=scene.dump(concatenate=True)
        points,_=trimesh.sample.sample_surface(mesh,3500,seed=2026102688)
        start=time.monotonic()
        poses,scores=GraspGenXSampler.run_inference(np.asarray(points),sampler,grasp_threshold=-1.,num_grasps=256,
            topk_num_grasps=128,min_grasps=1,max_tries=1,remove_outliers=False)
        torch.cuda.synchronize()
        np.savez_compressed(folder/'raw.npz',poses=poses.cpu().numpy(),scores=scores.cpu().numpy())
        write(folder/'result.json',{'entry':entry,'raw':binding(folder/'raw.npz'),'elapsed_seconds':time.monotonic()-start,
            'count':len(poses),'coordinate_frame':'asset from model gripper base; NOT native TCP',
            'confidence_is_not_a_physical_success_label':True})
    return {'completed':len(read(output/'reservation.json')['entries'])}


def qualify(output):
    import trimesh
    from resimpair_tabletop.geometry import Clearance
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    from resimpair_grasp.grasps import GRIPPER
    urdf=UPSTREAM/'ext/gripper_descriptions/gripper_descriptions/assets/x_grippers/franka_panda/gripper.urdf'
    conversion=tcp_transform(urdf);entries=[];rows=[]
    for entry in read(output/'reservation.json')['entries']:
        d=read(resolve(entry['asset']));folder=output/d['id']
        if not (folder/'result.json').exists():
            rows.append({'source_id':d['source_id'],'status':'inference_unavailable'});continue
        raw=np.load(folder/'raw.npz',allow_pickle=False)
        mesh=trimesh.load(resolve(d['visual']),force='scene').to_geometry()
        ray=RayMeshIntersector(mesh);guard=Clearance(mesh);candidates=[];rejections={}
        def reject(reason):rejections[reason]=rejections.get(reason,0)+1
        for i,(base,score) in enumerate(zip(raw['poses'],raw['scores'])):
            T=base@conversion
            if T[2,2]>-np.cos(np.deg2rad(10)):
                reject('approach_not_tabletop');continue
            # Snap small tilt to the common vertical controller. Recompute
            # contacts and clearance after snapping; model score alone is unused.
            closing=T[:3,1].copy();closing[2]=0;closing/=np.linalg.norm(closing)
            approach=np.array([0.,0.,-1.]);T[:3,:3]=np.column_stack([np.cross(closing,approach),closing,approach])
            origins=np.array([T[:3,3]-.04*closing,T[:3,3]+.04*closing])
            loc,ids,faces=ray.intersects_location(origins,np.array([closing,-closing]),multiple_hits=False)
            if len(ids)!=2:reject('no_two_closing_contacts');continue
            order=np.argsort(ids);loc=loc[order];normals=mesh.face_normals[faces[order]]
            width=float(np.linalg.norm(loc[1]-loc[0]));center=loc.mean(0)
            cosines=[float(-normals[0]@closing),float(normals[1]@closing)]
            if not .012<=width<=.073 or min(cosines)<1/np.sqrt(1+.25**2):
                reject('width_or_antipodal_friction');continue
            if np.linalg.norm(center-T[:3,3])>.01:reject('off_center_contact_pair');continue
            T[:3,3]=center
            ok,gap,reason=guard.check(T,mesh.bounds[0,2],.002)
            if not ok:reject(reason);continue
            if T[2,3]+d['half_size'][2]<GRIPPER['minimum_tcp_above_table_m']:
                reject('legacy_tcp_height');continue
            candidates.append({'id':f'gx_{i:03d}','tcp_in_asset':T.tolist(),'width_m':width,
                'contacts_in_asset':loc.tolist(),'contact_normals':normals.tolist(),'friction_cosines':cosines,
                'table_clearance_m':gap,'raw_model_score':float(score),'raw_proposal_index':i,
                'proposal':'GraspGenX_Panda_base_to_native_TCP_then_vertical_antipodal_refinement',
                'physical_status':'unmeasured','source_offline_candidate_status':'unmeasured'})
        old=read(resolve(entry['cache']))
        cache={'schema':'resimpair_mesh_grasp_cache_v1','signature':{'descriptor':entry['asset'],'gripper':GRIPPER,
            'adapter':binding(__file__),'raw':binding(folder/'raw.npz'),'model_gripper_urdf':binding(urdf),
            'upstream_commit':COMMIT,'base_to_native_TCP':conversion.tolist()},'candidates':candidates,
            'physical_qualification':'unknown','api_calls':0}
        cache['key']=digest(cache['signature']);path=folder/'qualified_cache.json';write(path,cache)
        if candidates:entries.append({'asset':entry['asset'],'cache':binding(path)})
        rows.append({'source_id':d['source_id'],'raw_proposals':len(raw['poses']),
            'geometric_candidates':len(candidates),'retained_geometric_baseline_candidates':len(old['candidates']),
            'rejections':rejections,'physical_success':'unmeasured'})
    write(output/'cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':[]})
    result={'rows':rows,'ready_assets':len(entries),'api_calls':0,'physical_executions':0,
            'no_success_rate_claim':True};write(output/'qualification.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('infer');q.add_argument('--index',type=Path,required=True);q.add_argument('--gpu',type=int,default=2)
    for name in ['worker','qualify']:s.add_parser(name)
    for q in s.choices.values():q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='infer':r=infer(a.index,a.output,a.gpu)
    elif a.command=='worker':r=worker(a.output)
    else:r=qualify(a.output)
    print(json.dumps(r,indent=2))
