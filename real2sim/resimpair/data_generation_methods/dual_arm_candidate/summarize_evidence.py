"""Read-only CPU summary of real pilot records, including unsuccessful attempts.

This is an evidence extraction utility, not an independent qualification gate.
It never changes recorded acceptance results or claims successful throughput.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from ..human_to_robot_pipeline.planning_contract import bind,save_new


def summarize(folder):
    import av
    result=json.loads((folder/'result.json').read_text())
    inp=json.loads((folder/'input.json').read_text())
    completed=(folder/'episode/arrays.npz').exists()
    d=folder/('episode' if completed else 'failure')
    arrays_path=d/('arrays.npz' if completed else 'trace.npz')
    a=np.load(arrays_path,allow_pickle=False)
    objects=a['object_states'] if completed else a['objects']
    tcp=a['tcp_world'] if completed else a['tcp']
    stages=a['stage'];n=len(stages);fps=20
    cp=d/('physics_substep_contacts.npz' if completed else 'contacts_and_grasp.npz')
    c=np.load(cp,allow_pickle=False)
    contact=c['contact_each_physics_substep'] if completed else c['physics_substeps']
    grasp=a['grasp'] if completed else c['grasp']
    final_grasp=c['final_postcommand_grasp'] if 'final_postcommand_grasp' in c else None
    final_objects=(a['final_object_states'] if completed else
        c['object_states_each_physics_substep'][-1,-1] if 'object_states_each_physics_substep' in c else None)
    full_tcp=np.concatenate([tcp,a['final_tcp_world'][None]]) if completed else tcp
    moving=np.linalg.norm(np.diff(full_tcp[:,:,:3,3],axis=0),axis=-1)*fps>.01
    native_diag=np.stack([grasp[:,i,i] for i in range(2)],axis=1)
    video=d/'video_native512.mp4';pts=[];dimensions=set()
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            pts.append(float(frame.pts*frame.time_base));dimensions.add((frame.width,frame.height))
    segments=[]
    for i,stage in enumerate(stages.tolist()):
        if i==0 or stage!=stages[i-1]:segments.append({'stage':stage,'first_frame':i,'frames':1})
        else:segments[-1]['frames']+=1
    reports=[json.loads(p.read_text()) for p in sorted((folder/'plans').glob('*/guard.json'))]
    entry={'attempt':folder.name,'seed':inp['seed'],'mode':inp['mode'],
        'hold_strategy':inp.get('hold_strategy','full_close'),
        'original_passed':result['passed'],'original_failure':result.get('error'),
        'original_metrics':result.get('metrics'),'source_result':bind(folder/'result.json'),
        'source_input':bind(folder/'input.json'),'arrays':bind(arrays_path),'physics_contacts':bind(cp),
        'video':bind(video),'contact_sheet':bind(folder/'source_contact_sheet.png'),
        'frames':n,'simulated_duration_s':n/fps,'wall_seconds':result.get('timing',{}).get('total_wall_seconds',result.get('total_wall_seconds')),
        'TCP_motion_intervals_measured':len(moving),'TCP_motion_endpoint_scope':'postcommand final included' if completed else 'last precommand sample; final interval unknown',
        'both_TCPs_moving_s':float(moving.all(axis=1).sum()/fps),'each_TCP_moving_s':(moving.sum(axis=0)/fps).tolist(),
        'overlap_fraction_of_any_motion':float(moving.all(axis=1).sum()/max(1,moving.any(axis=1).sum())),
        'last20_recorded_precommand_native_grasp_all':native_diag[-20:].all(axis=0).tolist(),
        'final_native_grasp':None if final_grasp is None else [bool(final_grasp[i,i]) for i in range(2)],
        'final_native_grasp_missing_not_inferred':final_grasp is None,
        'last_precommand_height_gain_m':(objects[-1,:,2]-objects[0,:,2]).tolist(),
        'final_postcommand_height_gain_m':None if final_objects is None else (final_objects[:,2]-objects[0,:,2]).tolist(),
        'forbidden_contact_peak_N':dict(zip(c['contact_keys'].tolist(),contact.max((0,1)).tolist())),
        'physics_substeps_shape':list(contact.shape),'decode':{'frames':len(pts),'dimensions':sorted(dimensions),
            'timestamps_equal_20Hz':bool(len(pts)==n and np.allclose(pts,np.arange(n)/fps,atol=1e-8,rtol=0))},
        'stages':segments,'paired_guard':{'segments':len(reports),'all_passed':all(r['passed'] for r in reports),
            'dense_samples':sum(r['dense_samples'] for r in reports),'independent_recheck':False},
        'mesh_negative_control':json.loads((folder/'plans/inter_arm_overlap_negative_control.json').read_text()),
        'qualification_inherited_from_apple':False,'production_approved':False}
    if 'object_states_each_physics_substep' in c:
        s=c['object_states_each_physics_substep']
        entry['last20_commands_substep_velocity_max']={
            'linear_m_s':np.linalg.norm(s[-20:,:,:,7:10],axis=-1).max((0,1)).tolist(),
            'angular_rad_s':np.linalg.norm(s[-20:,:,:,10:13],axis=-1).max((0,1)).tolist(),
            'scope':'includes lift when execution stops before hold; not a stable-hold estimate'}
    return entry


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ledger=sorted((args.campaign/'attempts').glob('*.json'))
    rows=[summarize(Path(json.loads(x.read_text())['output'])) for x in ledger]
    result={'schema':'dual_arm_pilot_evidence_summary_v1','source':bind(__file__),
        'attempts':len(rows),'qualified_attempts':sum(x['original_passed'] for x in rows),
        'new_valid_training_pairs':0,'API_calls':0,'GPU':'GPU2 rendering only; CPU PhysX; shared serial lock',
        'scope':'two independent mplib plans synchronized in one env; no14DoF joint optimality claim',
        'decision':'retain synchronous execution/guard as candidate; reject measured fixed-aperture repair; do not promote any full task qualification',
        'throughput_speedup':None,'throughput_reason':'same-layout sequential attempt fails before final hold; no successful matched baseline. Extra substep logging also changes wall overhead.',
        'next':'force/geometry-aware grasping with separately budgeted qualification; do not retry seeds or weaken stable gates',
        'records':rows}
    save_new(args.output,result);print(json.dumps({'output':str(args.output),'attempts':len(rows),'qualified':result['qualified_attempts']}))


if __name__=='__main__':main()
