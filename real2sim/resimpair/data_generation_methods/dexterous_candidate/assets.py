from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

WORKSPACE = Path(__file__).resolve().parents[2]
ASSETS = WORKSPACE/'PolicyGen/mani_skill/assets/robots'


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def inspect_urdf(path):
    path = Path(path)
    if not path.exists(): return {'path': str(path), 'exists': False}
    tree = ET.parse(path).getroot()
    joints = [j for j in tree.findall('joint') if j.attrib['type'] != 'fixed']
    meshes = []
    for mesh in tree.iter('mesh'):
        name = mesh.attrib['filename']
        p = (path.parent/name).resolve()
        meshes.append({'reference': name, 'exists': p.is_file(),
                       'binding': binding(p) if p.is_file() else None})
    return {'urdf': binding(path), 'exists': True, 'active_joint_count': len(joints),
            'active_joints': [{'name': j.attrib['name'], 'type': j.attrib['type'],
                              'limit': j.find('limit').attrib if j.find('limit') is not None else None,
                              'mimic': j.find('mimic').attrib if j.find('mimic') is not None else None} for j in joints],
            'links': len(tree.findall('link')), 'mesh_references': meshes,
            'all_direct_mesh_paths_resolve': all(x['exists'] for x in meshes)}


def inventory(output):
    rels = ['ur10e_shadow/ur10e_shadow.urdf', 'xarm7/xarm7_shadowhand.urdf',
            'xarm7/xarm7_ability_right_hand.urdf', 'panda/panda_v3.urdf',
            'allegro/allegro_hand_right_glb.urdf']
    result = {'schema': 'local_dexterous_asset_inventory_v1',
              'models': {str(root/rel): inspect_urdf(root/rel)
                         for root in (ASSETS, WORKSPACE/'The-Imitator-Game/mani_skill/assets/robots') for rel in rels},
              'ur5_combined_asset_found_in_these_roots': False,
              'ur10e_is_not_ur5': True,
              'known_legacy_evaluator_defect': {
                  'source': binding(WORKSPACE/'PolicyGen/mani_skill/agents/robots/ur10e_shadow/ur10e_shadow.py'),
                  'method': 'UR10eShadowHand.is_grasping', 'finding': 'unconditional return True; unusable as grasp evidence'},
              'api_calls': 0}
    save(output, result)
    return result


def compose_panda_allegro(output):
    """Compose existing meshes/inertia; declared fixed adapter, not a hardware CAD claim."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    panda_path = ASSETS/'panda/panda_v3.urdf'
    hand_path = ASSETS/'allegro/allegro_hand_right_glb.urdf'
    panda, hand = ET.parse(panda_path).getroot(), ET.parse(hand_path).getroot()
    removed = {'panda_hand'}
    while True:
        before = len(removed)
        for joint in panda.findall('joint'):
            if joint.find('parent').attrib['link'] in removed:
                removed.add(joint.find('child').attrib['link'])
        if len(removed) == before: break
    for child in list(panda):
        if (child.tag == 'link' and child.attrib['name'] in removed) or (
            child.tag == 'joint' and child.find('child').attrib['link'] in removed):
            panda.remove(child)
    dependencies = {str(panda_path): binding(panda_path), str(hand_path): binding(hand_path)}
    for tree, source, prefix in ((panda, panda_path, ''), (hand, hand_path, 'allegro_')):
        for mesh in tree.iter('mesh'):
            path = (source.parent/mesh.attrib['filename']).resolve()
            if not path.is_file(): raise FileNotFoundError(path)
            dependencies[str(path)] = binding(path)
            mesh.attrib['filename'] = str(path)
        if prefix:
            for link in tree.findall('link'): link.attrib['name'] = prefix+link.attrib['name']
            for joint in tree.findall('joint'):
                joint.attrib['name'] = prefix+joint.attrib['name']
                for side in ('parent', 'child'):
                    joint.find(side).attrib['link'] = prefix+joint.find(side).attrib['link']
                if joint.find('mimic') is not None:
                    joint.find('mimic').attrib['joint'] = prefix+joint.find('mimic').attrib['joint']
    for child in hand: panda.append(deepcopy(child))
    mount = ET.SubElement(panda, 'joint', name='candidate_panda_allegro_mount', type='fixed')
    ET.SubElement(mount, 'parent', link='panda_link8')
    ET.SubElement(mount, 'child', link='allegro_base_link')
    ET.SubElement(mount, 'origin', xyz='0 0 0.14', rpy='0 0 0')
    panda.attrib['name'] = 'candidate_panda_allegro_right'
    urdf = output/'panda_allegro_candidate.urdf'
    ET.indent(panda)
    ET.ElementTree(panda).write(urdf, encoding='utf-8', xml_declaration=True)
    result = inspect_urdf(urdf)
    if result['active_joint_count'] != 23 or not result['all_direct_mesh_paths_resolve']:
        raise ValueError('Composed model failed active-joint/mesh preflight')
    save(output/'manifest.json', {'schema': 'panda_allegro_composition_v1', 'result': result,
         'dependencies': list(dependencies.values()), 'removed_parallel_gripper_links': sorted(removed),
         'mount_transform_xyz_rpy': [0,0,.14,0,0,0], 'mount_is_declared_simulation_adapter': True,
         'inertia': 'Source arm and hand inertias preserved; fixed adapter has no additional mass model.',
         'self_collision_policy': 'No global self-collision disable; only simulator default adjacent-link exclusions.',
         'limitations': ['Mechanical adapter CAD/strength/mass not validated for hardware.',
                        'No collision-free trajectory or grasp qualification from composition alone.']})
    return urdf
