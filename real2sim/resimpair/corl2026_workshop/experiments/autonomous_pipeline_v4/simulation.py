"""The permitted planning boundary: native execution and a complete typed export.

No human-video annotations or generation prompts are created by this planner.
All image/condition compilation happens later through the same public interface.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from .timing import span, Accumulator

import numpy as np

from .core import WORKSPACE, PYTHON, bind, checked, digest, identifier, implementation, read, require, write


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def case_inputs(root, case_id):
    from .intake import cases as registered_cases
    experiment = read(Path(root) / "experiment.json")
    cases = [c for c in registered_cases(root) if c["id"] == case_id]
    require(len(cases) == 1, "Unknown case")
    case = cases[0]
    return experiment, case, read(checked(case["scene"])), read(checked(case["task"]))


def physical_budget_ledger(root, experiment, case, reservation_paths):
    """Build a read-only bucket ledger; launch calls this while holding all locks.

    A reservation consumes a slot even when no terminal exists or execution failed.
    Original cohort membership takes precedence over mutable registration phases.
    """
    if "physical_caps" not in experiment:
        return None
    caps = experiment["physical_caps"]
    buckets = {"base", "qualified_base_trajectory_variants", "development_and_new_contacts"}
    require(isinstance(caps, dict) and set(caps) == buckets
            and all(type(limit) is int and limit >= 0 for limit in caps.values()),
            "Invalid physical_caps; all three nonnegative integer limits are required")
    cohort_path = Path(root) / "cohort.json"
    require(cohort_path.is_file(), "physical_caps requires the original cohort.json")
    cohort_binding = bind(cohort_path)
    cohort = read(checked(cohort_binding))
    require(isinstance(cohort, dict) and isinstance(cohort.get("groups"), list),
            "Malformed original physical cohort groups")
    base_ids = set()
    for group in cohort["groups"]:
        require(isinstance(group, dict) and isinstance(group.get("cases"), list),
                "Malformed original physical cohort cases")
        for member in group["cases"]:
            require(isinstance(member, dict), "Malformed original physical cohort member")
            member_id = identifier(member.get("id"))
            require(member_id not in base_ids, "Duplicate original physical cohort member")
            base_ids.add(member_id)

    def bucket(value):
        require(isinstance(value, dict), "Malformed reserved physical case")
        case_id = identifier(value.get("id"))
        if case_id in base_ids:
            return "base"
        if value.get("phase") == "variants":
            return "qualified_base_trajectory_variants"
        return "development_and_new_contacts"

    before = {name: 0 for name in sorted(buckets)}
    evidence = []
    for path in sorted(reservation_paths):
        path = Path(path)
        binding = bind(path)
        reservation = read(checked(binding))
        assigned = bucket(reservation.get("case"))
        require(reservation["case"]["id"] == path.parent.parent.name
                and reservation.get("attempt") == path.parent.name,
                "Physical reservation path does not match its case/attempt")
        if "physical_budget_ledger" in reservation:
            prior = read(checked(reservation["physical_budget_ledger"]))
            require(prior.get("schema") == "physical_bucket_reservation_v1"
                    and prior.get("cohort") == cohort_binding and prior.get("bucket") == assigned,
                    "Original physical cohort or reserved bucket changed")
        before[assigned] += 1
        evidence.append(binding)
    require(all(before[name] <= caps[name] for name in buckets),
            "Existing physical reservations exceed a configured bucket limit")
    selected = bucket(case)
    require(before[selected] < caps[selected], "Physical bucket budget exhausted: " + selected)
    after = dict(before)
    after[selected] += 1
    return {"schema": "physical_bucket_reservation_v1", "bucket": selected,
            "cohort": cohort_binding, "physical_caps": dict(caps),
            "counts_before": before, "counts_after": after,
            "total_before": len(evidence), "total_after": len(evidence) + 1,
            "simulation_limit": experiment["simulation_limit"],
            "prior_reservations": evidence,
            "classification": "original cohort.groups IDs first; nonbase phase=variants second; all others development/new contacts",
            "failed_and_unknown_reservations_consume_budget": True}


def launch(root, case_id, attempt):
    root = Path(root)
    experiment, case, _, _ = case_inputs(root, case_id)
    identifier(attempt)
    directory = root / "executions" / case_id / attempt
    if (directory / "terminal.json").exists():
        return {"status": "preserved_completed_execution", "terminal": bind(directory / "terminal.json")}
    from .resources import lock_paths, gpu_snapshot, check_idle
    locked_gpu = experiment['gpu']
    with contextlib.ExitStack() as stack:
        for path in lock_paths(locked_gpu):
            handle = path.open("r")
            stack.callback(handle.close)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Re-read inputs and account for every reservation only after all shared
        # GPU locks are held. Checking before locking permits two launchers to
        # spend the final slot. Keep the lock through reservation and execution.
        experiment, case, _, _ = case_inputs(root, case_id)
        require(experiment['gpu'] == locked_gpu, "GPU selection changed while acquiring locks")
        if (directory / "terminal.json").exists():
            return {"status": "preserved_completed_execution", "terminal": bind(directory / "terminal.json")}
        require(not (directory / "reservation.json").exists(), "Existing physical attempt; inspect its process, do not restart")
        reservations = sorted((root / "executions").glob("*/*/reservation.json"))
        require(len(reservations) < experiment["simulation_limit"], "Physical execution budget exhausted")
        ledger = physical_budget_ledger(root, experiment, case, reservations)
        snapshot = gpu_snapshot(experiment['gpu'])
        check_idle(snapshot)
        require(snapshot["uuid"] == experiment["gpu_uuid"], "GPU identity changed")
        require(shutil.disk_usage(root).free > 10 * 2**30, "Insufficient output space")
        ledger_binding = {}
        if ledger is not None:
            ledger_path = directory / "physical_budget_ledger.json"
            write(ledger_path, ledger)
            ledger_binding["physical_budget_ledger"] = bind(ledger_path)
        write(directory / "reservation.json", {"case": case, "attempt": attempt,
              "experiment": bind(root / "experiment.json"), "gpu_snapshot": snapshot,
              "implementation": implementation(), "started_unix": time.time(), "automatic_retry": False,
              **ledger_binding})
        # Run this exact release. PYTHONPATH also exposes the installed simulation package.
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", RESIMPAIR_RESERVATION=str(directory / "reservation.json"),
                   PYTHONPATH=os.pathsep.join([str(Path(__file__).parent.parent), str(WORKSPACE), str(WORKSPACE / "PolicyGen")]))
        env.pop("CUDA_VISIBLE_DEVICES", None)
        command = [str(PYTHON), "-B", "-u", "-m", "autonomous_pipeline_v4", "worker", "--root", str(root),
                   "--case", case_id, "--attempt", attempt]
        start = time.time()
        with (directory / "worker.log").open("x") as stream:
            proc = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            write(directory / "process.json", {"pid": proc.pid, "pgid": proc.pid, "command": command, "started_unix": start})
            timed_out = False
            try:
                code = proc.wait(timeout=experiment["simulation_timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    code = proc.wait(timeout=10)
        result = read(directory / "worker_result.json") if (directory / "worker_result.json").exists() else None
        terminal = {"returncode": code, "timed_out": timed_out, "elapsed_seconds": time.time() - start,
                    "result": result, "physical_attempt_count": 1, "new_training_runs": 0,
                    "status": "executed" if result and result.get("bundle") and result.get("planning_failure") is None else "execution_failed"}
        write(directory / "terminal.json", terminal)
        return {**terminal, "directory": str(directory)}


def register(scene_spec, performance=None):
    import sapien
    import torch
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.agents.multi_agent import MultiAgent
    from mani_skill.sensors.camera import CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.table import TableSceneBuilder
    from scipy.spatial.transform import Rotation

    @register_env("ReSimPairSharedTable-v4", max_episode_steps=4000)
    class SharedTableEnv(BaseEnv):
        SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
        agent: MultiAgent

        def __init__(self, *args, **kwargs):
            self.scene_spec = scene_spec
            self.operation_timing = performance if performance is not None else Accumulator()
            self.robot_init_qpos_noise = 0.
            self.objects = []
            super().__init__(*args, robot_uids=("panda_wristcam", "panda_wristcam"), reconfiguration_freq=0, **kwargs)

        def _step_action(self, action):
            with self.operation_timing.section("physics_control_and_contacts"):
                return super()._step_action(action)

        def _get_obs_sensor_data(self, apply_texture_transforms=True):
            with self.operation_timing.section("sensor_render_and_readback"):
                return super()._get_obs_sensor_data(apply_texture_transforms)

        @property
        def _default_sensor_configs(self):
            return [CameraConfig(name, sapien_utils.look_at(config["eye"], config["target"]),
                                 config["width"], config["height"], config["fov"], .01, 100)
                    for name, config in self.scene_spec["cameras"].items()]

        def _load_agent(self, options):
            super()._load_agent(options, [sapien.Pose(p) for p in self.scene_spec["robot_roots"]])

        def _load_lighting(self, options):
            appearance = self.scene_spec["appearance"]
            if appearance.get("schema") == "resimpair_visual_appearance_v1":
                from resimpair_tasks.appearance_v1 import load_lighting
                return load_lighting(self)
            self.scene.set_ambient_light(appearance["ambient_rgb"])
            self.scene.add_directional_light([1., -.5, -1.], appearance["key_light_rgb"],
                                             shadow=True, shadow_scale=5, shadow_map_size=2048)
            self.scene.add_directional_light([-1., .5, -1.], [.5, .5, .5])

        def _load_scene(self, options):
            self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0.)
            self.table_scene.build()
            # The stock wall is on the requested opposite camera side: move it
            # behind the robots, so both cameras have a clear physical sightline.
            self.table_scene.wall.actor.set_pose(sapien.Pose([-1.05, 0., .3], Rotation.from_euler("z", np.pi / 2).as_quat(scalar_first=True)))
            appearance = self.scene_spec["appearance"]
            if appearance.get("schema") == "resimpair_visual_appearance_v1":
                from resimpair_tasks.appearance_v1 import build_visuals
                build_visuals(self)
            else:
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(half_size=[.58, .94, .0002], material=appearance["table_rgb"] + [1.])
                builder.initial_pose = sapien.Pose([-.12, 0., .0003])
                builder.build_static("table_surface_visual")
                builder = self.scene.create_actor_builder()
                builder.add_box_visual(half_size=[.003, 1.2, .8], material=appearance["background_rgb"] + [1.])
                builder.initial_pose = sapien.Pose([-1.0, 0., .3])
                builder.build_static("background_visual")
            self.objects = []
            self.object_ids = []
            for item in self.scene_spec["objects"]:
                builder = self.scene.create_actor_builder()
                require(item["shape"] in {"box", "compound_boxes", "convex_mesh", "catalog_mesh"}, "Unsupported planning collision geometry")
                if item["shape"] == "catalog_mesh":
                    from resimpair_grasp.assets import add_geometry
                    add_geometry(builder, item)
                elif item["shape"] == "convex_mesh":
                    from .active_assets import validate_object
                    descriptor = validate_object(item)
                    local_pose = sapien.Pose(descriptor["mesh_local_translation"])
                    builder.add_convex_collision_from_file(str(checked(descriptor["collision"])), pose=local_pose,
                        scale=descriptor["runtime_scale"], density=descriptor["density_kg_m3"])
                    builder.add_visual_from_file(str(checked(descriptor["visual"])), pose=local_pose,
                                                 scale=descriptor["runtime_scale"])
                else:
                    parts = item.get("parts", [{"center": [0., 0., 0.], "half_size": item["half_size"]}])
                    for part in parts:
                        builder.add_box_collision(half_size=part["half_size"], pose=sapien.Pose(part["center"]))
                        builder.add_box_visual(half_size=part["half_size"], pose=sapien.Pose(part["center"]),
                                               material=part.get("rgba", item["rgba"]))
                builder.initial_pose = sapien.Pose(item["position"], Rotation.from_euler("z", item["yaw"]).as_quat(scalar_first=True))
                actor = builder.build(item["id"]) if item["kind"] == "movable" else builder.build_static(item["id"])
                if item["kind"] == "movable" and item["shape"] not in {"convex_mesh", "catalog_mesh"}:
                    actor.set_mass(item["mass_kg"])
                self.objects.append(actor)
                self.object_ids.append(item["id"])
            self.support_actors = []
            for item in self.scene_spec["supports"]:
                builder = self.scene.create_actor_builder()
                hx, hy, hz = item["half_size"]
                shapes = [([hx, hy, hz], [0., 0., hz])]
                if item["shape"] == "tray":
                    wall_half = item.get("wall_height", .03) / 2
                    shapes += [([.004, hy, wall_half], [sign * (hx + .004), 0., wall_half]) for sign in (-1, 1)]
                    shapes += [([hx, .004, wall_half], [0., sign * (hy + .004), wall_half]) for sign in (-1, 1)]
                for half, center in shapes:
                    builder.add_box_collision(half_size=half, pose=sapien.Pose(center))
                    builder.add_box_visual(half_size=half, pose=sapien.Pose(center), material=item["rgba"])
                builder.initial_pose = sapien.Pose(item["position"])
                self.support_actors.append(builder.build_static(item["id"]))
            self.support_ids = self.object_ids + [s["id"] for s in self.scene_spec["supports"]] + ["table"]
            self.support_bodies = {o._bodies[0].entity: i for i, o in enumerate(self.objects + self.support_actors + [self.table_scene.table])}
            self.object_bodies = {o._bodies[0].entity: i for i, o in enumerate(self.objects)}
            self.robot_bodies = {l._bodies[0].entity: i for i, a in enumerate(self.agent.agents) for l in a.robot.get_links()}
            self.support_impulses = np.zeros((len(self.objects), len(self.support_ids)))
            self.robot_object_force = np.zeros((2, len(self.objects)))
            self.cross_robot_force = 0.

        def _initialize_episode(self, env_idx, options):
            self.table_scene.initialize(env_idx)
            for i, agent in enumerate(self.agent.agents):
                agent.robot.set_pose(sapien.Pose(self.scene_spec["robot_roots"][i]))
                agent.reset(np.array([0., -.4, 0., -2., 0., 1.8, np.pi / 4, .04, .04]))
            for actor, item in zip(self.objects, self.scene_spec["objects"]):
                actor.set_pose(sapien.Pose(item["position"], Rotation.from_euler("z", item["yaw"]).as_quat(scalar_first=True)))
                if item["kind"] == "movable":
                    actor.set_linear_velocity(np.zeros(3))
                    actor.set_angular_velocity(np.zeros(3))

        def _before_control_step(self):
            self.support_impulses[:] = 0.
            self.robot_object_force[:] = 0.
            self.cross_robot_force = 0.

        def _after_simulation_step(self):
            for contact in self.scene.get_contacts():
                first, second = [body.entity for body in contact.bodies]
                impulse = sum(float(np.linalg.norm(p.impulse)) for p in contact.points)
                for a, b in [(first, second), (second, first)]:
                    if a in self.object_bodies and b in self.support_bodies:
                        self.support_impulses[self.object_bodies[a], self.support_bodies[b]] += impulse
                    if a in self.robot_bodies and b in self.object_bodies:
                        arm, obj = self.robot_bodies[a], self.object_bodies[b]
                        self.robot_object_force[arm, obj] = max(self.robot_object_force[arm, obj], impulse * self.sim_freq)
                if first in self.robot_bodies and second in self.robot_bodies and self.robot_bodies[first] != self.robot_bodies[second]:
                    self.cross_robot_force = max(self.cross_robot_force, impulse * self.sim_freq)

        def _get_obs_extra(self, info):
            return {}

        def evaluate(self):
            return {"success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)}

        def compute_dense_reward(self, obs, action, info):
            return torch.zeros(self.num_envs, device=self.device)
    return SharedTableEnv


class Recorder:
    def __init__(self, env, directory):
        self.env = env
        self.unwrapped = env.unwrapped
        self.directory = Path(directory)
        self.rows, self.commands, self.stages = [], [], []
        self.stage = "initial"
        self.recording = False
        self.writers = {}
        self.paths = []
        self.last_frames = {}
        self.performance = getattr(self.unwrapped, "operation_timing", Accumulator())

    def __getattr__(self, name):
        if hasattr(self.unwrapped, name):
            return getattr(self.unwrapped, name)
        return getattr(self.env, name)

    def measure(self):
        with self.performance.section("state_and_contact_measurement"):
            return self._measure()

    def _measure(self):
        b = self.unwrapped
        return {"states": np.concatenate([array(a.robot.qpos).reshape(9) for a in b.agent.agents]),
                "tcp_world": np.stack([array(a.tcp.pose.to_transformation_matrix()).reshape(4, 4) for a in b.agent.agents]),
                "object_states": np.stack([array(o.get_state()).reshape(13) for o in b.objects]),
                "grasp": np.array([[bool(array(a.is_grasping(o)).item()) for o in b.objects] for a in b.agent.agents]),
                "support_impulses": b.support_impulses.copy(), "robot_object_force_N": b.robot_object_force.copy(),
                "cross_robot_force_N": np.array(b.cross_robot_force)}

    def capture(self):
        from PIL import Image
        self.rows.append(self.measure())
        self.stages.append(self.stage)
        for name, writer in self.writers.items():
            cfg = self.unwrapped.scene_spec["cameras"][name]
            frame = array(self.obs["sensor_data"][name]["rgb"]).reshape(cfg["height"], cfg["width"], 3)
            with self.performance.section("video_encode_enqueue"):
                writer.append_data(frame)
            if len(self.rows) == 1:
                Image.fromarray(frame).save(self.directory / (name + "_initial.png"))
            self.last_frames[name] = frame.copy()

    def step(self, action):
        if self.recording:
            command = np.concatenate([array(action[f"panda_wristcam-{i}"]).reshape(8) for i in range(2)])
            require(np.isfinite(command).all(), "Nonfinite control command")
            self.capture()
            self.commands.append(command)
        result = self.env.step(action)
        self.obs = result[0]
        return result

    def hold(self, steps, grippers=(1, 1)):
        states = self.measure()["states"]
        action = {f"panda_wristcam-{i}": np.r_[states[i * 9:i * 9 + 7], grippers[i]] for i in range(2)}
        for _ in range(steps):
            self.step(action)

    def reset(self, seed):
        import imageio.v2 as imageio
        self.obs, _ = self.env.reset(seed=seed, options={"reconfigure": True})
        self.hold(25)
        self.initial = self.measure()
        self.initial_physics = array(self.unwrapped.get_state()).copy()
        self.initial_physics_sha = hashlib.sha256(np.ascontiguousarray(self.initial_physics).tobytes()).hexdigest()
        self.last_frames = {}
        for name in self.unwrapped.scene_spec["cameras"]:
            self.writers[name] = imageio.get_writer(self.directory / (name + ".mp4"), fps=self.control_freq,
                codec="libx264", macro_block_size=16, ffmpeg_log_level="error",
                output_params=["-crf", "20", "-preset", "fast", "-threads", "2"])
        self.recording = True

    def finish(self):
        from PIL import Image
        if self.recording:
            self.capture()
            self.recording = False
        for writer in self.writers.values():
            with self.performance.section("video_encode_finalize"):
                writer.close()
        for name, frame in self.last_frames.items():
            Image.fromarray(frame).save(self.directory / (name + "_final.png"))


def targets(scene, task):
    objects = {o["id"]: o for o in scene["objects"]}
    supports = {o["id"]: o for o in scene["supports"]}
    result = {}
    for operation in task["operations"]:
        oid = operation["object"]
        target = operation["target"]
        support = target["support"]
        if support == "table":
            center, top = np.zeros(3), scene["table_top_z"]
        elif support in supports:
            center, top = np.asarray(supports[support]["position"]), supports[support]["top_z"]
        elif support in result:
            center = np.array(result[support])
            top = center[2] + objects[support]["half_size"][2]
        else:
            raise ValueError("Support target must already be defined")
        p = center.copy()
        p[:2] += target["offset_xy"]
        p[2] = top + objects[oid]["half_size"][2]
        result[oid] = p.tolist()
    return result


def solve(rec, task):
    import sapien
    from scipy.spatial.transform import Rotation
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
    b = rec.unwrapped
    destinations = targets(b.scene_spec, task)
    with rec.performance.section("planner_initialization"):
        planners = [PandaArmMotionPlanningSolver(rec, base_pose=a.robot.pose, multi_robot_id=i, vis=False,
                    debug=False, visualize_target_grasp_pose=False, print_env_info=False,
                    joint_vel_limits=.65, joint_acc_limits=.65) for i, a in enumerate(b.agent.agents)]
    initial = [array(a.robot.qpos).reshape(9).copy() for a in b.agent.agents]
    def follow(arm, path, stage):
        require(not isinstance(path, int) and path.get("status") == "Success" and len(path["position"]) <= 350,
                "Planning failed: " + stage)
        rec.stage = stage
        rec.paths.append({"arm": arm, "stage": stage, "start_frame": len(rec.rows),
                          "positions": np.asarray(path["position"]).tolist(), "planner_status": path["status"]})
        planners[arm].follow_path(path, other_gripper_state=planners[1 - arm].gripper_state, refine_steps=5)
    def move(arm, pose, stage):
        with rec.performance.section("motion_plan_screw"):
            path = planners[arm].move_to_pose_with_screw(pose, dry_run=True)
        if isinstance(path, int) or len(path["position"]) > 350:
            with rec.performance.section("motion_plan_rrt"):
                path = planners[arm].move_to_pose_with_RRTConnect(pose, dry_run=True)
        follow(arm, path, stage)
    for op in task["operations"]:
        oid, arm, prefix = op["object"], op["arm"], op["id"]
        index = b.object_ids.index(oid)
        obj, agent, planner = b.objects[index], b.agent.agents[arm], planners[arm]
        with rec.performance.section("grasp_geometry"):
            item = b.scene_spec["objects"][index]
            depth = .012
            if item["shape"] == "convex_mesh":
                from .active_assets import validate_object
                depth = validate_object(item)["grasp_depth_m"]
            grasp = compute_grasp_info_by_obb(get_actor_obb(obj), approaching=[0, 0, -1],
                    target_closing=array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)[:3, 1], depth=depth)
        require(float(grasp["extents"][1]) <= .077, "Target exceeds gripper opening")
        pose = agent.build_grasp_pose([0, 0, -1], grasp["closing"], grasp["center"])
        move(arm, sapien.Pose(pose.p + [0, 0, .17], pose.q), prefix + ":approach")
        move(arm, pose, prefix + ":reach")
        rec.stage = prefix + ":close"
        planner.close_gripper(other_gripper_state=planners[1 - arm].gripper_state, t=12)
        move(arm, sapien.Pose(pose.p + [0, 0, .20], pose.q), prefix + ":lift")
        rec.hold(8, tuple(p.gripper_state for p in planners))
        require(bool(array(agent.is_grasping(obj)).item()), "Target not grasped after lift: " + oid)
        desired = np.eye(4)
        desired[:3, :3] = Rotation.from_euler("z", op["target"]["yaw"]).as_matrix()
        desired[:3, 3] = np.asarray(destinations[oid]) + [0, 0, .002]
        object_T = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
        tcp_T = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
        place = sapien.Pose(desired @ np.linalg.inv(object_T) @ tcp_T)
        move(arm, sapien.Pose(place.p + [0, 0, .18], place.q), prefix + ":transport")
        move(arm, place, prefix + ":place")
        if task.get("release_precondition_control"):
            from .release_control import readiness
            observations = []
            for correction in range(3):
                rec.hold(5, tuple(p.gripper_state for p in planners))
                actual = array(obj.pose.to_transformation_matrix()).reshape(4, 4)
                held = bool(array(agent.is_grasping(obj)).item())
                check = readiness(actual, destinations[oid], held)
                observations.append({"frame": len(rec.rows), "correction": correction, **check})
                if check["ready_to_release"] or not held or correction == 2:
                    break
                tcp = array(agent.tcp.pose.to_transformation_matrix()).reshape(4, 4)
                place = sapien.Pose(desired @ np.linalg.inv(actual) @ tcp)
                move(arm, place, prefix + ":place_feedback")
            write(rec.directory / (prefix + "_release_preconditions.json"),
                  {"observations": observations, "max_corrections": 2, "acceptance_thresholds_changed": False})
            require(check["ready_to_release"], "measured_release_precondition_failed:" + oid)
        rec.stage = prefix + ":open"
        planner.open_gripper(other_gripper_state=planners[1 - arm].gripper_state)
        move(arm, sapien.Pose(place.p + [0, 0, .18], place.q), prefix + ":retreat")
        with rec.performance.section("motion_plan_return"):
            path = planner.planner.plan_qpos_to_qpos([initial[arm]], array(agent.robot.qpos).reshape(9).copy(),
                                                    time_step=b.control_timestep)
        follow(arm, path, prefix + ":return")
        rec.hold(20)
    rec.stage = "final_settle"
    rec.hold(40)


def export(rec, root, case, task, scene, directory, failure=None):
    directory = Path(directory)
    if not rec.rows:
        return None
    data = {k: np.stack([row[k] for row in rec.rows]) for k in rec.rows[0]}
    data.update(native_commands=np.stack(rec.commands) if rec.commands else np.empty((0, 16)),
                timestamps_seconds=np.arange(len(rec.rows), dtype=float) / rec.control_freq,
                initial_physics_state=rec.initial_physics,
                robot_roots_world=np.stack([array(a.robot.pose.to_transformation_matrix()).reshape(4, 4) for a in rec.agent.agents]))
    np.savez_compressed(directory / "measurements.npz", **data)
    write(directory / "planned_paths.json", {"paths": rec.paths, "scope": "controller-planned paths, separate from measured task events"})
    cameras = {}
    for name, config in scene["cameras"].items():
        params = rec.unwrapped._sensors[name].get_params()
        cameras[name] = {"config": config, "parameters": {k: array(v).reshape((-1,) + tuple(array(v).shape[-2:]))[0].tolist() for k, v in params.items()},
                         "video": bind(directory / (name + ".mp4")),
                         "initial_image": bind(directory / (name + "_initial.png")),
                         "final_image": bind(directory / (name + "_final.png")),
                         "synchronous_with_measurements": True, "frame_count": len(rec.rows),
                         "fps": float(rec.control_freq)}
    bundle = {"schema": "resimpair_execution_bundle_v5", "case": case, "task_intent": task, "scene": scene,
              "event_protocol": bind(directory / "event_protocol.json"),
              "planning_declaration": bind(directory / "planning_declaration.json"),
              "measurements": bind(directory / "measurements.npz"), "planned_paths": bind(directory / "planned_paths.json"),
              "object_ids": rec.unwrapped.object_ids, "support_ids": rec.unwrapped.support_ids,
              "target_positions_world": targets(scene, task), "cameras": cameras,
              "initial_physics_sha256": rec.initial_physics_sha,
              "initial_object_state_sha256": digest(data["object_states"][0].tolist()),
              "motion_planning_agent_participation": getattr(rec, "motion_planning_agent_participation", True),
              "manual_generation_semantics_supplied": False, "manual_review_observations_supplied": False,
              "native_execution": True, "physical_execution_count": 1, "saved_state_rerender": False,
              "code": implementation(), "planning_failure": failure,
              "qualification_scope": "native dynamics, measured grasp/support/release, stability and monitored contacts; not strict continuous path qualification",
              "source_scene_reconstruction_claimed": False, "training_admitted": False}
    if any(o["shape"] == "convex_mesh" for o in scene["objects"]):
        from .active_assets import export_native
        bundle["native_asset_evidence"] = export_native(rec, directory / "native_asset_geometry.json")
    write(directory / "execution_bundle.json", bundle)
    return bind(directory / "execution_bundle.json")


def worker(root, case_id, attempt):
    root = Path(root)
    experiment, case, scene, task = case_inputs(root, case_id)
    if scene["schema"] == "resimpair_mechanism_workbench_v1":
        from .workbench import worker as workbench_worker
        return workbench_worker(root, case_id, attempt)
    if scene["schema"] == "resimpair_novel_scene_v1":
        from .novel_runtime import worker as novel_worker
        return novel_worker(root, case_id, attempt)
    directory = root / "executions" / case_id / attempt
    require(os.environ.get("RESIMPAIR_RESERVATION") == str(directory / "reservation.json"), "Supervisor reservation required")
    for value in read(directory / "reservation.json")["implementation"].values():
        checked(value)
    from examples.baselines.hypericl_task_expansion.common import bind_gpu
    gpu = bind_gpu(experiment["gpu"])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    import gymnasium as gym
    env, recorder = None, None
    failure = None
    bundle = None
    performance = Accumulator()
    try:
        from .planning_protocols import pick_place
        object_ids = [o["id"] for o in scene["objects"]]
        support_ids = object_ids + [s["id"] for s in scene["supports"]] + ["table"]
        write(directory / "event_protocol.json", pick_place(scene, task, object_ids, support_ids, targets(scene, task)))
        write(directory / "planning_declaration.json", {
            "event_protocol": bind(directory / "event_protocol.json"), "case": case,
            "scope": "before_physical_execution", "declared_unix": time.time(),
            "implementation": implementation(), "manual_generation_prose": False})
        with span(directory, "simulation_environment_setup"):
            register(scene, performance)
            env = gym.make("ReSimPairSharedTable-v4", sim_backend="physx_cpu", render_backend=gpu["render_backend"],
                           control_mode="pd_joint_pos", obs_mode="rgb", reward_mode="none", render_mode="rgb_array",
                           sensor_configs={"shader_pack": "rt-fast"})
        with span(directory, "simulation_reset_and_recorders"):
            performance.phase = "reset"
            recorder = Recorder(env, directory)
            recorder.reset(scene["seed"])
        with span(directory, "planning_execution_and_rendering"):
            performance.phase = "execution"
            solve(recorder, task)
    except Exception as exc:
        import traceback
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "trace": [{"file": f.filename, "line": f.lineno, "function": f.name}
                             for f in traceback.extract_tb(exc.__traceback__)]}
        write(directory / "planning_failure.json", failure)
    finally:
        performance.phase = "finalize"
        if recorder is not None:
            with span(directory, "simulation_video_finalize"):
                recorder.finish()
            with span(directory, "simulation_measurement_export"):
                bundle = export(recorder, root, case, task, scene, directory, failure)
        if env is not None:
            env.close()
        performance.persist(directory / "timing_breakdown.json")
    result = {"bundle": bundle, "planning_failure": failure, "frames": len(recorder.rows) if recorder else 0,
              "status": "planned_and_executed" if failure is None else "planning_or_execution_failed"}
    write(directory / "worker_result.json", result)
    if failure is None and bundle:
        write(root / "planner_handoffs" / (case_id + ".json"), {
            "schema": "resimpair_planner_handoff_v4", "case": case_id, "attempt": attempt,
            "execution_bundle": bundle, "manual_downstream_annotations": False})
    return result
