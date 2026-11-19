"""Independent CPU audit of frozen synthetic rig reports; never human acceptance."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

BASE = Path('/mnt/sdg/datasets/real_sim_human_pipeline')
V3 = BASE / 'harness_20261021_v3'
FROZEN = '37d7a8339980c8a560005f033f9620453854d9ce42bf99b40ad7aacfeb2540cd'


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError('Immutable output already exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def near(a, b):
    assert np.allclose(a, b, atol=1e-8, rtol=0), (a, b)


def project(point, camera):
    e = np.asarray(camera['camera_from_world'])
    q = e[:, :3] @ point + e[:, 3]
    if not camera['near'] < q[2] < camera['far']:
        return None, False
    uv = (np.asarray(camera['intrinsic_matrix']) @ q)[:2] / q[2]
    uv /= [camera['width'], camera['height']]
    return uv, bool(((uv >= 0) & (uv <= 1)).all())


def probe(output):
    """Execute only frozen geometry compiler, without its unsafe historical wrapper."""
    from data_generation_methods.r2h_pipeline import actor_geometry_candidate as candidate
    assert bound(candidate.__file__)['sha256'] == FROZEN
    output = Path(output)
    assert not output.exists()
    package = BASE / 'harness_20261021_v2/unified_anchor/validation/package.json'
    world = BASE / 'harness_20261021_v1/multiview/validation/plans/collision_world.json'
    protocol = V3 / 'independent_review/actor_geometry_protocol.json'
    data = candidate.unified.validate_package(package)
    _, table, _ = candidate._load_world(data, world)
    rows = []
    started = time.monotonic()
    for angle, height in itertools.product([0, 15, 30, 45], [-.05, 0., .05]):
        name = f'lean_{angle:02d}_waist_{round(height*100):+03d}cm'
        directory = output / name
        save(directory / 'rig.json', candidate.nominal_rig(data, table, lean=angle, waist_height=height))
        receipt = candidate.compile_geometry(package, directory/'rig.json', world, protocol,
                                             directory/'report.json')
        rows.append({'label': name, 'angle_degrees': angle, 'waist_height_m': height, **receipt})
    summary = {'schema': 'independent_frozen_geometry_layout_probe_v1',
               'source_package': bound(package), 'candidate_implementation': bound(candidate.__file__),
               'runner': bound(__file__), 'protocol': bound(protocol), 'results': rows,
               'wall_seconds': time.monotonic()-started,
               'api_calls': 0, 'gpu_runs': 0,
               'scope': 'g01 fixed-layout probe after freeze; previously reviewed source, not unseen-source holdout',
               'human_or_image_or_video_qualification': False}
    save(output/'summary.json', summary)
    print(json.dumps({'completed': str(output), 'rigs': len(rows)}), flush=True)


def audit_one(report_path):
    r = read(report_path)
    for v in r['bindings'].values():
        assert bound(v['path']) == v
    assert r['bindings']['implementation']['sha256'] == FROZEN
    package = read(r['bindings']['source_package']['path'])
    packet = read(package['inputs']['conditions']['path'])
    plan = read(package['actor_plan']['path'])
    inp = read(package['inputs']['execution_input']['path'])
    for v in [*package['inputs'].values(), package['actor_plan'], packet['bindings']['arrays']]:
        assert bound(v['path']) == v
    camera = packet['camera_calibration']
    rig = read(r['bindings']['rig']['path'])
    table = trimesh.load(r['bindings']['table_mesh']['path'], force='mesh', process=False)
    local = trimesh.load(r['bindings']['selected_object_mesh']['path'], force='mesh', process=False)
    a = np.load(packet['bindings']['arrays']['path'])
    selected = int(inp['proposal']['active_arm'])
    assert r['context'] == package['initial_context']
    assert r['context']['role_ids']['selected_object'].endswith('/object/'+str(selected))
    assert rig['actor_id'] == plan['actor']['id']
    assert rig['arms'] == {'upper_arm_m': .30, 'forearm_m': .26, 'shoulder_width_m': .36}
    torso = rig['torso']
    assert torso['dimensions_depth_width_height_m'] == [.24, .36, .60]
    assert torso['shoulder_height_from_waist_m'] == .54
    assert torso['lean_degrees'] in [0, 15, 30, 45]
    e = np.asarray(camera['camera_from_world'])
    eye = -np.linalg.solve(e[:, :3], e[:, 3])
    center = table.bounds.mean(0)
    fxy = eye[:2]-center[:2]
    fxy /= np.linalg.norm(fxy)
    waist_xy = center[:2] + (((table.vertices[:, :2]-center[:2])@fxy).min()-.15)*fxy
    waist = np.asarray(torso['waist_world_m'])
    near(waist[:2], waist_xy)
    height = waist[2]-table.bounds[1, 2]
    assert any(abs(height-h) < 1e-8 for h in [-.05, 0., .05])
    near(torso['facing_world_xy'], fxy)
    forward = np.r_[fxy, 0.]
    left = np.array([-fxy[1], fxy[0], 0.])
    theta = np.deg2rad(torso['lean_degrees'])
    axis = np.sin(theta)*forward + np.array([0., 0., np.cos(theta)])
    shoulders = {'left': waist+.54*axis+.18*left, 'right': waist+.54*axis-.18*left}
    near(r['waist_world_m'], waist)
    near(r['anatomical_left_world'], left)
    near(r['torso_axis_world'], axis)
    for hand, point in shoulders.items():
        near(r['shoulders'][hand], point)
        uv, ok = project(point, camera)
        assert r['projection']['shoulders'][hand]['in_frame'] == ok
        if uv is not None:
            near(r['projection']['shoulders'][hand]['normalized_uv'], uv)
    labels = {l: c for c in packet['conditions'] for l in c['labels']}
    event_labels = {'grasp_apple':'grasp_established', 'lift_from_source':'lift_clear',
                    'transport_to_target':'last_grasp_before_release', 'release_in_target':'released'}
    assignments = {x['event_id']: x for x in plan['event_assignments']}
    hands = assignments['grasp_apple']['hands']
    expected = sorted((event, labels[label]['frame']['frame_index_0based'], hand)
                      for event, label in event_labels.items() for hand in assignments[event]['hands'])
    assert sorted((x['event_id'], x['frame_index_0based'], x['acting_hand']) for x in r['event_targets']) == expected
    held = np.flatnonzero(a['grasp'][:, :, selected].any(axis=1))
    assert r['held_frame_count'] == len(held)
    assert sorted((x['frame_index_0based'], x['acting_hand']) for x in r['held_trajectory_targets']) == sorted(itertools.product(held.tolist(), hands))
    targets = r['event_targets'] + r['held_trajectory_targets']
    distances, nearest_distances = [], []
    for x in targets:
        i = x['frame_index_0based']; state = a['object_states'][i, selected]
        rotation = Rotation.from_quat(state[[4, 5, 6, 3]])
        vertices = rotation.apply(local.vertices) + state[:3]
        target = vertices[np.argmax(vertices[:, 2])]
        shoulder = shoulders[x['acting_hand']]
        near(x['target_world_m'], target); near(x['shoulder_world_m'], shoulder)
        near(x['object_pose_wxyz_world'], state[:7]); near(x['timestamp_seconds'], a['timestamps_seconds'][i])
        distance = float(np.linalg.norm(target-shoulder)); distances.append(distance)
        near(x['reach']['distance_m'], distance)
        expected_status = 'point_reachable' if .04-1e-8 <= distance <= .56+1e-8 else 'point_unreachable'
        assert x['reach']['status'] == expected_status
        if x['reach']['elbow_world_m'] is not None:
            elbow = np.asarray(x['reach']['elbow_world_m'])
            near(np.linalg.norm(elbow-shoulder), .30); near(np.linalg.norm(target-elbow), .26)
        posed = trimesh.Trimesh(vertices=vertices, faces=local.faces, process=False)
        _, ds, _ = trimesh.proximity.closest_point_naive(posed, shoulder[None, :])
        nearest_distances.append(float(ds[0])); near(x['closest_surface_distance_m'], ds[0])
        assert x['all_surface_points_beyond_maximum_reach'] == (ds[0] > .56+1e-8)
        uv, ok = project(target, camera)
        assert x['projection']['in_frame'] == ok
        if uv is not None:
            near(x['projection']['normalized_uv'], uv)
        assert x['human_contact_correspondence'] == 'unknown'
        assert x['robot_tcp_used_as_human_target'] is False
        assert x['reach']['joint_or_collision_feasibility_verified'] is False
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((np.cos(theta)*forward-np.array([0.,0.,np.sin(theta)]), left, axis))
    transform[:3, 3] = waist+.30*axis
    body = trimesh.creation.box(extents=[.24,.36,.60], transform=transform)
    near(r['torso_table']['torso_bounds_world_m'], body.bounds)
    # Independently run actual mesh surface test, not AABB-as-collision.
    from mplib.pymp.collision_detection import fcl
    with tempfile.TemporaryDirectory(prefix='independent-rig-') as tmp:
        path = Path(tmp)/'body.obj'; body.export(path)
        c1 = fcl.CollisionObject(fcl.load_mesh_as_BVH(str(path), np.ones(3)))
        c2 = fcl.CollisionObject(fcl.load_mesh_as_BVH(r['bindings']['table_mesh']['path'], np.ones(3)))
        collision = bool(fcl.collide(c1, c2, fcl.CollisionRequest()).is_collision())
    # For these actual rigs collision is surface crossing; do not generalize containment.
    assert collision == r['torso_table']['mesh_collision']
    reach_fail = any(x > .56+1e-8 or x < .04-1e-8 for x in distances)
    framing_fail = any(not project(s, camera)[1] for s in shoulders.values()) or any(not x['projection']['in_frame'] for x in targets)
    assert r['clears_geometry_preflight'] == (not reach_fail and not collision and not framing_fail)
    for key in ['human_anatomy_verified','generated_image_verified','physical_execution_verified',
                'quality_release_approved','training_admitted','can_generate_video']:
        assert r[key] is False
    assert r['scope_unknowns_do_not_require_unavailable_human_3d_ground_truth'] is True
    assert r['projection']['rendered_visibility_verified'] is False
    return {'report': bound(report_path), 'lean_degrees': torso['lean_degrees'], 'waist_height_m': round(height, 8),
            'held_frames': len(held), 'event_records': len(r['event_targets']), 'reach_failed': reach_fail,
            'mesh_collision': collision, 'framing_failed': framing_fail,
            'maximum_proxy_distance_m': max(distances), 'maximum_closest_surface_distance_m': max(nearest_distances),
            'clears_geometry_preflight': r['clears_geometry_preflight'], 'independent_recomputation_passed': True}


def audit(directory, output):
    summary = read(Path(directory)/'summary.json')
    rows = [audit_one(x['report']['path']) for x in summary['results']]
    assert len(rows) == 12
    assert sorted((x['lean_degrees'], round(x['waist_height_m'], 2)) for x in rows) == sorted(itertools.product([0,15,30,45],[-.05,0.,.05]))
    result = {'schema':'independent_actor_geometry_actual_audit_v1','summary':bound(Path(directory)/'summary.json'),
              'auditor':bound(__file__),'results':rows,'rigs':len(rows),
              'cleared':sum(x['clears_geometry_preflight'] for x in rows),
              'reach_failures':sum(x['reach_failed'] for x in rows),
              'mesh_failures':sum(x['mesh_collision'] for x in rows),
              'framing_failures':sum(x['framing_failed'] for x in rows),
              'api_calls':0,'gpu_runs':0,'human_or_visual_qualification':False,
              'interpretation':'Only the fixed camera-far waist grid and synthetic two-link rig are falsified; no conclusion that humans cannot perform the task.'}
    save(output,result)
    print(json.dumps({k:v for k,v in result.items() if k in ['rigs','cleared','reach_failures','mesh_failures','framing_failures']}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('probe');a.add_argument('--output',required=True)
    a=sub.add_parser('audit');a.add_argument('--directory',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='probe':probe(args.output)
    else:audit(args.directory,args.output)
