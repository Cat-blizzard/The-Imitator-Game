"""CPU evidence summary for the preserved failed dexterous-hand trajectory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .assets import ASSETS, WORKSPACE, binding, inspect_urdf, save


def summarize(root):
    root = Path(root).resolve()
    out = root/'evidence_summary'
    out.mkdir(exist_ok=False)
    replay = root/'contact_replay_02'
    baseline = root/'grasp_attempt_01'
    raw = np.load(replay/'trajectory.npz', allow_pickle=False)
    old = np.load(baseline/'trajectory.npz', allow_pickle=False)
    model = json.loads((replay/'runtime_model.json').read_text())
    for run in (baseline, replay):
        save(out/(run.name+'_trajectory_semantics.json'), {
            'trajectory': binding(run/'trajectory.npz'),
            'state_rows':301, 'executed_control_commands':300,
            'actions_row_0': 'Initialization placeholder in active_joint_names order; never passed to env.step.',
            'actions_rows_1_through_300': 'Actual commands in controller_joint_names order.',
            'qpos_and_qvel_order': 'active_joint_names',
            'mapping_source': binding(run/'runtime_model.json'),
            'valid_command_rows': {'inclusive_start':1, 'exclusive_end':301},
            'same_physical_run_cameras':['overview','close'],
            'counting':'One planned motion; two actual executions (original and measurement replay), not two new motions.'})
    np.save(out/'valid_action_mask.npy',np.r_[False,np.ones(300,dtype=bool)])

    urdf = root/'panda_allegro_model/panda_allegro_candidate.urdf'
    tree = ET.parse(urdf).getroot()
    links = {x.attrib['name']:x for x in tree.findall('link')}
    child_names = [x.find('child').attrib['link'] for x in tree.findall('joint')]
    inertia=[]; no_inertia=[]; source_equal=[]
    source_links={x.attrib['name']:x for x in ET.parse(ASSETS/'panda/panda_v3.urdf').getroot().findall('link')}
    source_links.update({'allegro_'+x.attrib['name']:x for x in ET.parse(ASSETS/'allegro/allegro_hand_right_glb.urdf').getroot().findall('link')})
    for name,link in links.items():
        elem = link.find('inertial')
        source_elem=source_links[name].find('inertial')
        def canonical(node):
            return None if node is None else (node.tag,sorted(node.attrib.items()),[canonical(x) for x in node])
        source_equal.append(canonical(elem)==canonical(source_elem))
        if elem is None:
            no_inertia.append(name);continue
        a={k:float(v) for k,v in elem.find('inertia').attrib.items()}
        matrix=np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]])
        inertia.append({'link':name,'mass_kg':float(elem.find('mass').attrib['value']),
                        'inertia_eigenvalues':np.linalg.eigvalsh(matrix).tolist()})
    save(out/'model_preflight.json',{
        'urdf':binding(urdf),'unique_links':len(links),'roots':sorted(set(links)-set(child_names)),
        'each_nonroot_has_one_parent':len(child_names)==len(set(child_names))==len(links)-1,
        'all_source_inertias_preserved':all(source_equal),'inertial_links':inertia,
        'positive_masses_and_inertias':all(x['mass_kg']>0 and min(x['inertia_eigenvalues'])>0 for x in inertia),
        'source_fixed_links_without_inertial':no_inertia,
        'fixed_mount_xyz_m':[0,0,.14],'fixed_mount_rpy_rad':[0,0,0],
        'global_self_collision_disabled':model['self_collisions_disabled'],
        'limits':'Runtime commands checked against loaded joint limits; no dense collision certificate.',
        'hardware_qualification':'unknown; adapter strength, mass, cable routing and physical calibration unvalidated'})

    extra = ['realman/inspire_urdf/realman.urdf','realman/inspire_urdf/realman_mobile_base.urdf',
             'ur10e_shadow/ur10e_shadow_package_keyword_replaced.urdf']
    expanded={str(base/rel):inspect_urdf(base/rel) for base in (ASSETS,WORKSPACE/'The-Imitator-Game/mani_skill/assets/robots') for rel in extra}
    save(out/'inventory_addendum.json',{'baseline':binding(root/'inventory.json'),
         'additional_static_inspections':expanded,
         'interpretation':['URDF non-fixed joints including mimic are not necessarily independent control dimensions.',
           'Realman class declares 28 primary controls; URDF has 40 non-fixed joints including 12 mimic. Not runtime-qualified here.',
           'UR10e package-replaced URDF meshes resolve; original package:// references are not evidence of missing assets.',
           'Only the new Panda+Allegro model was actually loaded and controlled in this experiment.']})

    obj=raw['object_pose'];time=raw['timestamps'];contact=raw['per_link_contact_substeps_N']
    groups=contact.sum(axis=-1); valid=raw['contact_substep_valid']
    close=np.flatnonzero(raw['stages']=='close_fingers')
    save(out/'measurement_correction.json',{
        'old_result':binding(baseline/'result.json'),'new_result':binding(replay/'result.json'),
        'original_contact_logger_missing_links':[f'allegro_link_{i}.0_tip' for i in (3,7,11,15)],
        'actual_command_rows_identical':bool(np.array_equal(raw['actions'][1:],old['actions'][1:])),
        'qpos_identical':bool(np.array_equal(raw['qpos'],old['qpos'])),
        'object_pose_identical':bool(np.array_equal(obj,old['object_pose'])),
        'middle_fixed_tip_force_peak_N':float(contact[valid,:,1,4].max()),
        'thumb_force_peak_N':float(groups[valid,:,3].max()),
        'simultaneous_thumb_and_other_substeps':int(((groups[valid,:,3]>.05)&(groups[valid,:,:3].max(axis=-1)>.05)).sum()),
        'cube_max_height_m':float(obj[:,2].max()),
        'cube_max_lift_m':float(obj[:,2].max()-obj[0,2]),
        'cube_xy_displacement_at_close_end_m':float(np.linalg.norm(obj[close[-1],:2]-obj[0,:2])),
        'physical_grasp_and_stable_lift':False,
        'finding':'Single middle fingertip pushed the cube during closure. No thumb opposition and no lift.',
        'baseline_preserved':True,'api_calls':0})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12})
    fig,axs=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
    axs[0].plot(time,(obj[:,0]-obj[0,0])*100,label='Cube x displacement',color='#1d4ed8')
    axs[0].set_ylabel('Shift (cm)');axs[0].legend(loc='upper left')
    axs[1].plot(time,(obj[:,2]-obj[0,2])*100,label='Actual cube lift',color='#1d4ed8')
    axs[1].axhline(5,color='#b91c1c',linestyle='--',label='Frozen lift requirement: 5 cm')
    axs[1].set_ylabel('Lift (cm)');axs[1].set_ylim(-.5,6);axs[1].legend(loc='upper left')
    ts=raw['contact_substep_times_s'][valid].reshape(-1)
    axs[2].plot(ts,groups[valid,:,1].reshape(-1),label='Middle finger (including tip)',color='#c2410c')
    axs[2].plot(ts,groups[valid,:,3].reshape(-1),label='Thumb',color='#1d4ed8')
    axs[2].set_ylabel('Contact (N)');axs[2].set_xlabel('Simulation time (s)');axs[2].legend(loc='upper right')
    for ax in axs:
        ax.axvspan(time[close[0]],time[close[-1]],color='#fef3c7',alpha=.5)
        ax.grid(alpha=.2)
    fig.suptitle('Panda + Allegro: controlled approach and closure; push, no grasp')
    fig.savefig(out/'contact_and_motion.png',dpi=170)
    fig.savefig(out/'contact_and_motion.svg')
    plt.close(fig)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(summarize(p.parse_args().root))
