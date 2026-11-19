"""Execute native PD commands, then admit only released, stable, goal-correct objects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from .common import DEFAULT_ROOT, bind_gpu, source_pins, write, sha
from .registry import SCENES, PROGRAMS, goal_key, inventory


class QualificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise QualificationError(message)


def array(value):
    import numpy as np
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class Recorder:
    def __init__(self, env):
        self.env = env
        self.unwrapped = env.unwrapped
        self.rows = []
        self.frames = []
        self.commands = []
        self.stage = 'settle'
        self.stage_names = []
        self.recording = False
        self.robot_entities = {link._bodies[0].entity: i
                               for i, agent in enumerate(self.unwrapped.agent.agents)
                               for link in agent.robot.get_links()}

    def __getattr__(self, name):
        if hasattr(self.unwrapped, name):
            return getattr(self.unwrapped, name)
        return getattr(self.env, name)

    def measure(self):
        import numpy as np
        b = self.unwrapped
        agents = b.agent.agents
        force = 0.
        for contact in b.scene.get_contacts():
            indices = [self.robot_entities.get(body.entity) for body in contact.bodies]
            if None not in indices and indices[0] != indices[1]:
                force += sum(float(np.linalg.norm(point.impulse)) for point in contact.points) * b.sim_freq
        return dict(states=np.concatenate([array(a.robot.qpos).reshape(-1)[:9] for a in agents]).astype('float32'),
                    tcp_world=np.stack([array(a.tcp.pose.to_transformation_matrix()).reshape(4, 4) for a in agents]),
                    objects=np.stack([np.r_[array(o.pose.raw_pose).reshape(7),
                                           array(o.linear_velocity).reshape(3), array(o.angular_velocity).reshape(3)]
                                      for o in b.objects]),
                    grasp=np.array([[bool(array(a.is_grasping(o)).item()) for o in b.objects] for a in agents]),
                    cross_robot_contact_force_N=force)

    def step(self, action):
        import numpy as np
        command = np.concatenate([array(action[f'panda_wristcam-{i}']).reshape(8) for i in range(2)]).astype('float32')
        require(np.isfinite(command).all(), 'Nonfinite command')
        if self.recording:
            self.rows.append(self.measure())
            frame = array(self.obs['sensor_data']['zed2i']['rgb']).reshape(224, 224, 3)
            self.frames.append(frame.copy())
            self.commands.append(command)
            self.stage_names.append(self.stage)
        result = self.env.step(action)
        self.obs = result[0]
        return result

    def hold(self, steps, grippers=(1, 1)):
        import numpy as np
        q = self.measure()['states']
        action = {f'panda_wristcam-{i}': np.r_[q[i * 9:i * 9 + 7], grippers[i]] for i in range(2)}
        for _ in range(steps):
            self.step(action)

    def reset(self, seed, program):
        import numpy as np
        import copy
        self.recording = False
        # A new PhysX scene also clears contact warm-start caches. Merely restoring
        # poses/velocities produced goal-dependent grasp failures in reused scenes.
        self.obs, _ = self.env.reset(seed=seed, options=dict(layout_seed=seed, program=program, reconfigure=True))
        b = self.unwrapped
        self.robot_entities = {link._bodies[0].entity: i
                               for i, agent in enumerate(b.agent.agents)
                               for link in agent.robot.get_links()}
        if getattr(b, '_expansion_initial_seed', None) != seed:
            for obj in b.objects:
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))
            self.hold(20)
            b._expansion_initial_seed = seed
            b._expansion_initial_state = copy.deepcopy(b.get_state_dict())
            b._expansion_initial_controller = copy.deepcopy(b.agent.get_controller_state())
            # Apply the exact same serialized snapshot for the first program too.
            b.set_state_dict(copy.deepcopy(b._expansion_initial_state))
            b.agent.set_controller_state(copy.deepcopy(b._expansion_initial_controller))
            self.obs = b.get_obs()
            b._expansion_initial_observation = copy.deepcopy(self.obs)
        else:
            # Reset alone retains tiny contact/velocity differences across previous
            # executions. Restore one real, settled initial state for all goals.
            b.set_state_dict(copy.deepcopy(b._expansion_initial_state))
            b.agent.set_controller_state(copy.deepcopy(b._expansion_initial_controller))
            self.obs = copy.deepcopy(b._expansion_initial_observation)
        require(all(float(np.linalg.norm(array(o.linear_velocity))) < .03 for o in self.unwrapped.objects),
                'Objects not settled at start')
        self.initial = self.measure()
        self.initial_physics_state = array(b.get_state()).copy()
        raw = np.ascontiguousarray(self.initial_physics_state).tobytes()
        self.initial_physics_sha256 = hashlib.sha256(raw).hexdigest()
        if getattr(b, '_expansion_initial_hash', None) and b._expansion_initial_hash_seed == seed:
            require(self.initial_physics_sha256 == b._expansion_initial_hash, 'Initial snapshot restore differs across goals')
        b._expansion_initial_hash = self.initial_physics_sha256
        b._expansion_initial_hash_seed = seed
        frame = array(self.obs['sensor_data']['zed2i']['rgb']).reshape(224, 224, 3)
        self.initial_rgb_sha256 = hashlib.sha256(frame.tobytes()).hexdigest()
        self.recording = True


def solve(rec, program):
    import numpy as np
    import sapien
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
    b = rec.unwrapped
    planners = [PandaArmMotionPlanningSolver(rec, base_pose=a.robot.pose, multi_robot_id=i,
                    vis=False, debug=False, visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=.8, joint_acc_limits=.8) for i, a in enumerate(b.agent.agents)]
    events = []
    initial_joints = [array(a.robot.qpos).reshape(9).copy() for a in b.agent.agents]
    pack_offsets = np.zeros((2, 3))
    if program.startswith('pack'):
        # Separate slots in a real 23 cm wide tray. Maximum clearance, rather
        # than tightly fitting meshes, also leaves room for the opening fingers.
        pack_offsets[0, 0] = -.045
        pack_offsets[1, 0] = .045

    def move(planner, position, orientation, stage):
        rec.stage = stage
        goal = sapien.Pose(position, orientation)
        result = planner.move_to_pose_with_screw(goal, dry_run=True)
        bad = isinstance(result, int) and result == -1
        if bad or len(result['position']) > 240:
            result = planner.move_to_pose_with_RRTConnect(goal, dry_run=True)
            bad = isinstance(result, int) and result == -1
        require(not bad and len(result['position']) <= 300, 'Planner failed or excessive path: ' + stage)
        planner.follow_path(result, other_gripper_state=planners[1 - planner.multi_robot_id].gripper_state, refine_steps=3)

    for order, (obj_idx, bin_idx) in enumerate(PROGRAMS[program]):
        planner = planners[obj_idx]
        agent = b.agent.agents[obj_idx]
        obj = b.objects[obj_idx]
        obb = get_actor_obb(obj)
        closing_hint = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)[:3, 1]
        grasp = compute_grasp_info_by_obb(obb, approaching=[0, 0, -1],
                                        target_closing=closing_hint, depth=.025)
        require(float(grasp['extents'][1]) <= .077, 'Object too wide for Panda grasp')
        pose = agent.build_grasp_pose([0, 0, -1], grasp['closing'], grasp['center'])
        orientation = pose.q
        move(planner, pose.p + [0, 0, .16], orientation, f'{order}:approach')
        move(planner, pose.p, orientation, f'{order}:grasp_reach')
        rec.stage = f'{order}:close'
        planner.close_gripper(other_gripper_state=planners[1 - obj_idx].gripper_state, t=12)
        move(planner, pose.p + [0, 0, .19], orientation, f'{order}:lift')
        rec.stage = f'{order}:lift_settle'
        rec.hold(6, grippers=tuple(p.gripper_state for p in planners))
        require(bool(array(agent.is_grasping(obj)).item()), f'Object {obj_idx} not grasped after lift')
        current_mesh = obj.get_first_collision_mesh(to_world_frame=True)
        bounds = current_mesh.bounds
        center = bounds.mean(0)
        require(center[2] - rec.initial['objects'][obj_idx, 2] >= .08, 'Object did not lift')
        destination = b.goal_centers[bin_idx].copy()
        if program.startswith('pack'):
            destination += pack_offsets[obj_idx]
        destination[2] = b.floor_top + .5 * (bounds[1, 2] - bounds[0, 2]) + .005
        tcp = array(agent.tcp.pose.p).reshape(3)
        release = tcp + destination - center
        overhead = release.copy()
        overhead[2] = max(.25, release[2] + .12)
        move(planner, overhead, orientation, f'{order}:transport')
        # Re-estimate the held object's offset after transport, before release.
        rec.stage = f'{order}:transport_settle'
        rec.hold(6, grippers=tuple(p.gripper_state for p in planners))
        bounds = obj.get_first_collision_mesh(to_world_frame=True).bounds
        destination[2] = b.floor_top + .5 * (bounds[1, 2] - bounds[0, 2]) + .005
        release = array(agent.tcp.pose.p).reshape(3) + destination - bounds.mean(0)
        move(planner, release, orientation, f'{order}:place')
        rec.stage = f'{order}:release'
        planner.open_gripper(other_gripper_state=planners[1 - obj_idx].gripper_state)
        move(planner, overhead, orientation, f'{order}:retreat')
        # Return to the known initial joint configuration; redundant IK can choose
        # a very long branch when only the initial TCP pose is specified.
        rec.stage = f'{order}:return'
        result = planner.planner.plan_qpos_to_qpos([initial_joints[obj_idx]],
                    array(agent.robot.qpos).reshape(9).copy(), time_step=b.control_timestep)
        require(result['status'] == 'Success' and len(result['position']) <= 300, 'Joint return planning failed')
        planner.follow_path(result, other_gripper_state=planners[1 - obj_idx].gripper_state, refine_steps=3)
        rec.stage = f'{order}:settle'
        rec.hold(15)
        events.append(dict(order=order, object_index=obj_idx, container_index=bin_idx,
                           frame=len(rec.rows), actual_grasp_after_lift=True))
    rec.stage = 'final_settle'
    rec.hold(40)
    return events


def qualify(rec, program):
    import numpy as np
    b = rec.unwrapped
    final = rec.measure()
    placements = []
    for obj_idx, bin_idx in PROGRAMS[program]:
        obj = b.objects[obj_idx]
        bounds = obj.get_first_collision_mesh(to_world_frame=True).bounds
        center = b.goal_centers[bin_idx]
        inside_xy = bool(np.all(bounds[0, :2] >= center[:2] - b.inner_half - .002) and
                         np.all(bounds[1, :2] <= center[:2] + b.inner_half + .002))
        supported_z = bool(abs(float(bounds[0, 2]) - b.floor_top) < .008)
        released = not bool(final['grasp'][:, obj_idx].any())
        stable = bool(np.linalg.norm(final['objects'][obj_idx, 7:10]) < .025 and
                      np.linalg.norm(final['objects'][obj_idx, 10:13]) < .4)
        # Stable over the final half second, rather than one lucky endpoint.
        history = np.stack([r['objects'][obj_idx, :3] for r in rec.rows[-10:]])
        persistent = bool(np.linalg.norm(np.ptp(history, axis=0)) < .008)
        placements.append(dict(object_index=obj_idx, container_index=bin_idx,
                            full_mesh_inside_xy=inside_xy, supported_on_container_floor=supported_z,
                            released=released, stable=stable, persistent=persistent,
                            bounds_m=bounds.tolist(), linear_speed_m_s=float(np.linalg.norm(final['objects'][obj_idx, 7:10]))))
    selected = {o for o, _ in PROGRAMS[program]}
    untouched = all(float(np.linalg.norm(final['objects'][i, :3] - rec.initial['objects'][i, :3])) < .015
                    for i in range(2) if i not in selected)
    max_cross_force = max(r['cross_robot_contact_force_N'] for r in rec.rows)
    success = all(all(row[k] for k in ('full_mesh_inside_xy', 'supported_on_container_floor', 'released', 'stable', 'persistent'))
                  for row in placements) and untouched and max_cross_force < .5
    return dict(success=success, placements=placements, unselected_objects_untouched=untouched,
                max_cross_robot_contact_force_N=max_cross_force), final


def save_diagnostic(rec, folder):
    import numpy as np
    import imageio.v2 as imageio
    if not rec.rows:
        return
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'trace.npz', commands=np.stack(rec.commands),
                       states=np.stack([r['states'] for r in rec.rows]),
                       objects=np.stack([r['objects'] for r in rec.rows]),
                       tcp=np.stack([r['tcp_world'] for r in rec.rows]),
                       stage=np.array(rec.stage_names, dtype='U32'))
    with imageio.get_writer(folder / 'video.mp4', fps=rec.unwrapped.control_freq,
            codec='libx264', quality=7, ffmpeg_log_level='error') as writer:
        for frame in rec.frames:
            writer.append_data(frame)


def save(rec, folder, scene, program, split, seed, events, metrics, final, pins):
    import numpy as np
    import imageio.v2 as imageio
    import av
    from ..hypericlzq.eef import panda_fk
    folder.mkdir(parents=True, exist_ok=True)
    states = np.stack([r['states'] for r in rec.rows])
    commands = np.stack(rec.commands)
    n = len(states)
    require(16 <= n <= 3000, f'Invalid trajectory length {n}')
    poses = panda_fk(np.stack([states[:, :7], states[:, 9:16]], 1))
    targets = panda_fk(np.stack([commands[:, :7], commands[:, 8:15]], 1))
    roots = np.stack([array(a.robot.pose.to_transformation_matrix()).reshape(4, 4) for a in rec.agent.agents])
    tcp = np.stack([r['tcp_world'] for r in rec.rows])
    fk_error = float(np.max(np.abs(roots[None] @ poses - tcp)))
    require(fk_error < 3e-6, f'FK mismatch {fk_error}')
    arrays = dict(states=states, poses=poses.astype('float32'), target_poses=targets.astype('float32'),
        openness=(commands[:, [7, 15]] + 1) / 2, eef_mask=np.ones((n, 14), bool),
        pose_valid=np.ones((n, 2), bool), native_commands=commands, tcp_world=tcp.astype('float32'),
        object_states=np.stack([r['objects'] for r in rec.rows]).astype('float32'),
        grasp=np.stack([r['grasp'] for r in rec.rows]), final_object_states=final['objects'].astype('float32'),
        cross_robot_contact_force_N=np.asarray([r['cross_robot_contact_force_N'] for r in rec.rows], dtype='float32'),
        final_tcp_world=final['tcp_world'].astype('float32'), robot_roots_world=roots.astype('float32'),
        timestamps_seconds=np.arange(n, dtype='float64') / rec.control_freq,
        initial_physics_state=rec.initial_physics_state,
        container_centers_world=rec.unwrapped.goal_centers.astype('float32'),
        stage=np.array(rec.stage_names, dtype='U32'))
    np.savez_compressed(folder / 'arrays.npz', **arrays)
    video = folder / 'video.mp4'
    with imageio.get_writer(video, fps=rec.control_freq, codec='libx264', macro_block_size=16,
             ffmpeg_log_level='error', output_params=['-crf', '18', '-preset', 'fast', '-tune', 'zerolatency',
                 '-g', str(rec.control_freq), '-bf', '0', '-sc_threshold', '0']) as writer:
        for frame in rec.frames:
            writer.append_data(frame)
    # Decode every produced frame, checking count and PTS, not just file existence.
    stamps = []
    initial_decoded_rgb_sha256 = None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        require(abs(float(stream.average_rate) - rec.control_freq) < 1e-5, 'Video FPS mismatch')
        for frame in container.decode(stream):
            require(frame.width == 224 and frame.height == 224, 'Video shape mismatch')
            stamps.append(float(frame.pts * stream.time_base))
            if initial_decoded_rgb_sha256 is None:
                initial_decoded_rgb_sha256 = hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest()
    require(len(stamps) == n, 'Video and labels have different frame counts')
    require(np.max(np.abs(np.asarray(stamps) - arrays['timestamps_seconds'])) < 1e-5, 'Video PTS mismatch')
    pixels = np.stack(rec.frames[::max(1, n // 8)])
    require(float(pixels.std()) > 5 and float(pixels.mean()) > 4, 'Blank video')
    receipt = dict(id=folder.name, scene=scene, program=program, goal_key=goal_key(scene, program),
        group=f'{scene}:{seed}', layout_seed=seed, split=split, length=n, fps=int(rec.control_freq),
        simulation_freq=int(rec.sim_freq), arrays=str(folder / 'arrays.npz'), video=str(video),
        source='simulation_expansion', domain='robot', embodiment='dual_panda_wristcam',
        action_space='panda_eef14_v3', target_start_offset=0,
        label_kind='FK_of_executed_native_PD_joint_targets',
        video_alignment='frame[t] and measured state[t] immediately precede native command[t]',
        robot_state_dim=18, action_dim=14, fk_max_error=fk_error,
        initial_physics_sha256=rec.initial_physics_sha256, initial_rgb_sha256=rec.initial_rgb_sha256,
        initial_decoded_rgb_sha256=initial_decoded_rgb_sha256,
        encoding='H264 CRF18 zerolatency, no B frames or future lookahead, fixed keyframe interval',
        goal_id_is_not_model_input=True, initial_state_restored_from_common_real_snapshot=True,
        physx_scene_rebuilt_per_episode=True,
        container_geometry=dict(inner_half_xy_m=rec.unwrapped.inner_half.tolist(),
                                floor_top_m=rec.unwrapped.floor_top, wall_height_m=rec.unwrapped.wall_height),
        metrics=metrics, events=events,
        asset_specs=rec.asset_specs, source_sha256=pins,
        sha256={name: sha(folder / name) for name in ('arrays.npz', 'video.mp4')},
        storage_bytes=sum((folder / name).stat().st_size for name in ('arrays.npz', 'video.mp4')))
    write(folder / 'receipt.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--scenes', nargs='+', choices=list(SCENES), default=list(SCENES))
    p.add_argument('--programs', nargs='+', choices=list(PROGRAMS), default=list(PROGRAMS))
    p.add_argument('--layouts', type=int, default=1)
    p.add_argument('--seed-base', type=int, default=4150000)
    p.add_argument('--split', choices=['qualification', 'train', 'validation', 'test'], default='qualification')
    p.add_argument('--disk-floor-gib', type=float, default=60.)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--keep-going', action='store_true', help='Inspect all qualification candidates even when a goal fails')
    args = p.parse_args()
    gpu = bind_gpu(args.gpu)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from .envs import make_env
    require(0 <= args.shard_index < args.shard_count and args.layouts > 0, 'Invalid collection schedule')
    args.root.mkdir(parents=True, exist_ok=True)
    write(args.root / 'task_inventory.json', inventory())
    pins = source_pins()
    write(args.root / f'collector_sources_{args.shard_index}.json', pins)
    start = time.time()
    accepted = []
    failures = []
    stop = False
    for scene in args.scenes:
        env = make_env(gpu, scene, args.programs[0])
        try:
            for index in range(args.shard_index, args.layouts, args.shard_count):
                seed = args.seed_base + index
                for program in args.programs:
                    folder = args.root / args.split / scene / f'{seed}_{program}'
                    if (folder / 'receipt.json').exists():
                        r = json.loads((folder / 'receipt.json').read_text())
                        require(r['source_sha256'] == pins, 'Existing collector source changed; choose a new run root')
                        for name, digest in r['sha256'].items():
                            require(sha(folder / name) == digest, 'Existing trajectory modified')
                        accepted.append(r)
                        continue
                    if shutil.disk_usage(args.root).free < args.disk_floor_gib * (1 << 30):
                        stop = True
                        break
                    rec = Recorder(env)
                    try:
                        rec.reset(seed, program)
                        events = solve(rec, program)
                        metrics, final = qualify(rec, program)
                        require(metrics['success'], 'Physical placement qualification failed: ' + json.dumps(metrics))
                        receipt = save(rec, folder, scene, program, args.split, seed, events, metrics, final, pins)
                        accepted.append(receipt)
                        print(json.dumps(dict(status='accepted', scene=scene, program=program, seed=seed,
                            length=receipt['length'], seconds=time.time() - start), ensure_ascii=False), flush=True)
                    except Exception as exc:
                        failure = dict(scene=scene, program=program, seed=seed, error=repr(exc),
                                       frames=len(rec.rows), elapsed_seconds=time.time() - start)
                        if rec.rows:
                            failure['final_objects'] = rec.measure()['objects'].tolist()
                        failures.append(failure)
                        rejected_folder = args.root / 'rejected' / scene / f'{seed}_{program}'
                        write(rejected_folder / 'failure.json', failure)
                        # Retain representative failures without unbounded rejected-video storage.
                        existing = len(list((args.root / 'rejected' / scene).glob('*/video.mp4')))
                        if args.split == 'qualification' or existing < 8:
                            save_diagnostic(rec, rejected_folder)
                        print(json.dumps(dict(status='rejected', **failure)), flush=True)
                        if args.split == 'qualification' and not args.keep_going:
                            raise
                    finally:
                        del rec
                    write(args.root / f'progress_{args.split}_{args.shard_index}.json',
                          dict(status='collecting', accepted=len(accepted), rejected=len(failures),
                               elapsed_seconds=time.time() - start,
                               accepted_by_scene={s: sum(r['scene'] == s for r in accepted) for s in args.scenes}))
                if stop:
                    break
            if stop:
                break
        finally:
            env.close()
    result = dict(status='disk_floor_reached' if stop else 'finished', episodes=accepted, rejected=failures,
                  accepted=len(accepted), attempted=len(accepted) + len(failures),
                  elapsed_seconds=time.time() - start, source_sha256=pins, gpu=gpu)
    write(args.root / f'manifest_{args.split}_{args.shard_index}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('episodes', 'rejected', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
