"""One explicit experimental layout, unchanged bowl expert and raw verifier.

Not a production collector. New actors are static physical distractors; they
enter both the rendering scene and the expert's collision-point-cloud input.
This adapter must pass a small native qualification before any scale-up.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import numpy as np

from .planning_contract import WORKSPACE, bind, check_binding, save_new, validate_contract
from .structured_randomization import screen_layout


def register(proposal):
    import sapien
    from scipy.spatial.transform import Rotation
    from mani_skill.utils.registration import register_env, REGISTERED_ENVS
    from examples.baselines.hypericl_data_factory import real_bowl
    from examples.baselines.hypericl_task_expansion.collect import array
    real_bowl.register()
    parent = REGISTERED_ENVS["HyperICLRealBowlTransfer-v1"].cls

    @register_env("HumanGroundedAppleClutterExperimental-v1", max_episode_steps=3000)
    class ExperimentalApple(parent):
        def _load_scene(self, options):
            super()._load_scene(options)
            self.added_static_distractors = []
            for spec in proposal["added_distractors"]:
                for path, digest in spec["asset_files"].items():
                    check_binding({"path": path, "sha256": digest})
                # The proposal specifies a bounding-volume centre, not mesh origin.
                import trimesh
                mesh = trimesh.load(spec["collision"], force="mesh")
                yaw = spec["yaw_rad"]
                rotation = Rotation.from_euler("z", yaw).as_matrix()
                local_center = np.asarray(mesh.bounds).mean(0)
                translation = np.asarray(spec["center"])-rotation@local_center
                quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
                pose = sapien.Pose(translation, np.r_[quaternion_xyzw[3], quaternion_xyzw[:3]])
                builder = self.scene.create_actor_builder()
                builder.add_visual_from_file(spec["visual"])
                builder.add_convex_collision_from_file(spec["collision"])
                builder.initial_pose = pose
                self.added_static_distractors.append(builder.build_static(spec["instance_id"]))
            self._extra_entities = {body.entity for actor in self.added_static_distractors
                                    for body in actor._bodies}
            self._extra_impulse = 0.
            self.extra_contact_force_N = 0.
            self.randomization["structured_proposal"] = proposal
            self.randomization["camera"] = proposal["camera"]
            self.randomization["extra_distractors_are_static_physical"] = True
            # Candidate-only opt-in appearance extension; no physics changes.
            if proposal.get("background"):
                from .candidate_scene import apply_background
                apply_background(self, proposal["background"])

        def _initialize_episode(self, env_idx, options):
            super()._initialize_episode(env_idx, options)
            self.goal_centers = np.array([proposal["roles"][r]["center"]
                                          for r in ("source_container", "destination_container")])
            for i, bowl in enumerate(self.bins):
                bowl.set_pose(sapien.Pose(self.goal_centers[i]))
            index = next(i for i, spec in enumerate(self.asset_specs) if spec["name"] == "apple")
            # Gym's constructor first resets with its own seed. The exact
            # proposal seed/role is checked after the recorder's explicit reset.
            pose = self.objects[index].pose
            self.objects[index].set_pose(sapien.Pose(
                self.goal_centers[0]+[0,0,self.floor_top+.006], array(pose.q).reshape(4)))
            self.randomization["receptacle_center_world_m"] = self.goal_centers[0].tolist()
            self.randomization["receptacle_centers_world_m"] = self.goal_centers.tolist()
            self.randomization["camera"] = proposal["camera"]

        def _before_control_step(self):
            super()._before_control_step()
            self._extra_impulse = 0.

        def _after_simulation_step(self):
            super()._after_simulation_step()
            for contact in self.scene.get_contacts():
                entities = {body.entity for body in contact.bodies}
                if entities & self._extra_entities and (
                        entities & self._robot_entities or entities & set(self._fruit_entities)):
                    self._extra_impulse += sum(float(np.linalg.norm(point.impulse)) for point in contact.points)

        def _after_control_step(self):
            super()._after_control_step()
            self.extra_contact_force_N = self._extra_impulse*self.control_freq


def run(proposal, contract, output, gpu):
    """Explicit experimental execution; no retries, no approval or training."""
    # Four physical pilots are archived. A later source audit found the custom
    # Screw planner ignores its collision result and can report stalled motion
    # as Success. Do not execute another batch until a separate strict planner
    # adapter has been implemented and physically qualified. There is no CLI
    # flag to bypass this missing implementation.
    raise RuntimeError(
        "Next simulation batch blocked: strict per-path FK/end-goal/collision "
        "validation is not installed and qualified. Archived pilot results "
        "remain available; plan_geometry_validator is retrospective only.")
    # Historical pilot implementation retained below for review and replacement.
    validate_contract(contract)
    if not screen_layout(proposal)["coarse_screen_passed"]:
        raise ValueError("Proposal no longer passes its coarse screen")
    if output.exists():
        raise FileExistsError("Immutable single-attempt output exists")
    sys.path.insert(0, str(WORKSPACE/"PolicyGen"))
    from examples.baselines.hypericl_task_expansion.common import bind_gpu, source_pins
    gpu_info = bind_gpu(gpu)
    import gymnasium as gym
    from examples.baselines.hypericl_data_factory import real_bowl, verify_bowl
    from examples.baselines.hypericl_data_factory.rich_runtime_v2 import VisibilityRecorder, save_native
    from examples.baselines.hypericl_data_factory.collect_dataset import visibility_quality
    from examples.baselines.hypericl_task_expansion.collect import save, save_diagnostic, require
    base = Path("/mnt/sdf/datasets/hypericl_data_factory_20261016_v1")
    fruit_path = base/"fruit_adapters_v2/catalog.json"
    fruit = json.loads(fruit_path.read_text())["assets"]
    bowls = json.loads((base/"asset_adapters_v2/catalog.json").read_text())["assets"]
    bowl = next(x for x in bowls if x["uid"].startswith("11a459d5"))
    profile = base/"receptacle_geometry_v1"/(bowl["uid"]+"_cavity_profile.json")
    pins = source_pins()
    for path in (Path(__file__), Path(real_bowl.__file__), Path(verify_bowl.__file__), fruit_path, profile):
        pins[str(path.resolve())] = bind(path)["sha256"]
    output.mkdir(parents=True)
    save_new(output/"input.json", {"proposal": proposal, "contract": contract, "source_sha256": pins,
                                  "scope": "Experimental candidate only; no production qualification"})
    class Recorder(VisibilityRecorder):
        def measure(self):
            result = super().measure()
            result["bowl_support_force_N"] = self.unwrapped.support_force.copy()
            result["robot_bowl_force_N"] = float(self.unwrapped.robot_bowl_force)
            result["extra_contact_force_N"] = float(self.unwrapped.extra_contact_force_N)
            return result

    program = "pick_first_to_right" if proposal["active_arm"] == 0 else "pick_second_to_right"
    env, rec = None, None
    episode = output/"episode"
    try:
        register(proposal)
        env = gym.make("HumanGroundedAppleClutterExperimental-v1",
                       asset_pool=fruit, bowl_asset=bowl, transfer_mode=True,
                       view_profile="balanced", data_split=proposal["split"], program=program,
                       sim_backend="physx_cpu", render_backend=gpu_info["render_backend"],
                       control_mode="pd_joint_pos", obs_mode="rgb+segmentation",
                       reward_mode="none", render_mode="rgb_array",
                       sensor_configs={"shader_pack": "rt-fast"})
        rec = Recorder(env)
        rec.reset(proposal["seed"], program)
        actual_selected = next(i for i, spec in enumerate(rec.unwrapped.asset_specs) if spec["name"] == "apple")
        require(actual_selected == proposal["active_arm"], "Native proposal-seed asset order differs")
        # Screen actual initial meshes against the original unselected dynamic object.
        other = 1-proposal["active_arm"]
        dynamic_bounds = rec.unwrapped.objects[other].get_first_collision_mesh(to_world_frame=True).bounds
        for actor in rec.unwrapped.added_static_distractors:
            bounds = actor.get_first_collision_mesh(to_world_frame=True).bounds
            overlap = np.minimum(bounds[1], dynamic_bounds[1])-np.maximum(bounds[0], dynamic_bounds[0])
            require(not (overlap > -.02).all(), "Added clutter conflicts with unselected dynamic object")
        b = rec.unwrapped
        original_bins = b.bins
        # The unchanged expert reads bins only to build its collision cloud.
        # Goal index and actual support bookkeeping remain the original two bowls.
        b.bins = original_bins+b.added_static_distractors
        try:
            events = real_bowl.solve(rec, program)
        finally:
            b.bins = original_bins
        metrics, final = real_bowl.qualify(rec, program)
        require(metrics["success"], "Unchanged native physical outcome rejected")
        extra_force = np.array([r["extra_contact_force_N"] for r in rec.rows])
        require(float(extra_force.max(initial=0)) < .5, "Robot/object contacted added static clutter")
        quality = visibility_quality(rec, program)
        require(quality["passed"], "Unchanged image quality gate rejected")
        receipt = save(rec, episode, "experimental_apple_clutter", program, proposal["split"],
                       proposal["seed"], events, metrics, final, pins)
        # Keep the base export before extending its receipt with native/extra QA.
        (episode/"receipt.json").rename(episode/"receipt.base.json")
        receipt.update(save_native(rec, episode))
        contact_path = episode/"bowl_contact.npz"
        np.savez_compressed(contact_path,
            support_force_N=np.array([r["bowl_support_force_N"] for r in rec.rows]),
            robot_bowl_force_N=np.array([r["robot_bowl_force_N"] for r in rec.rows]))
        extra_path = episode/"extra_contact.npz"
        np.savez_compressed(extra_path, force_N=extra_force, timestamps_seconds=np.arange(len(extra_force))/20.)
        receipt.update(
            goal_key="factory:transfer_apple_occupied_bowl_to_empty_bowl",
            image_quality=quality, randomization=b.randomization, receptacle=b.receptacle_spec,
            contact_data=str(contact_path), contact_sha256=bind(contact_path)["sha256"],
            extra_contact=bind(extra_path), max_extra_contact_force_N=float(extra_force.max(initial=0)),
            initial_measured_states=rec.initial["states"].tolist(),
            initial_measured_object_states=rec.initial["objects"].tolist(),
            fresh_recorder_per_episode=True, qualification_only=True, production_approved=False,
            human_pairing_requires_separate_review=True, training_admitted=False,
            semantic_scope="Complete apple transfer; added static distractors never change target")
        save_new(episode/"receipt.json", receipt)
        verification = verify_bowl.verify(episode/"receipt.json")
        save_new(output/"verification.json", verification)
        result = {"passed": True, "receipt": bind(episode/"receipt.json"),
                  "frames": receipt["length"], "max_extra_contact_force_N": receipt["max_extra_contact_force_N"],
                  "actual_extra_static_distractors": len(b.added_static_distractors),
                  "extra_objects_in_collision_cloud": True, "native_verifier": verification,
                  "still_pending": ["actual RGB task/occlusion review", "held-object swept-collision improvement"],
                  "production_approved": False, "training_admitted": False}
    except Exception as exc:
        if rec is not None and rec.rows:
            try:
                save_diagnostic(rec, output/"failure")
            except Exception as diagnostic:
                save_new(output/"diagnostic_error.json", {"error": repr(diagnostic)})
        result = {"passed": False, "error": repr(exc), "traceback": traceback.format_exc(),
                  "controller_phase": rec.stage if rec is not None else "environment_initialization",
                  "frames": len(rec.rows) if rec is not None else 0,
                  "attempt_preserved": True, "training_admitted": False}
    finally:
        if env is not None:
            env.close()
    save_new(output/"result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    cases = json.loads(args.proposals.read_text())["cases"]
    case = next(c for c in cases if c["case_id"] == args.case_id)
    result = run(case, json.loads(args.contract.read_text()), args.output, args.gpu)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
