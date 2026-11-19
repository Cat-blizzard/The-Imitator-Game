"""Strict, independently checked apple adapter; old production files unchanged.

Execution/export scaffold adapted from the exact archived four-case adapter.
Motion is implemented independently in strict_motion.py using official mplib.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
import uuid
import xml.etree.ElementTree as ET
import numpy as np
from .planning_contract import WORKSPACE, bind, save_new, validate_contract
from .structured_randomization import screen_layout
from . import experimental_apple_adapter as scene_module
from . import strict_motion as strict_module
from .experimental_apple_adapter import register
from .strict_motion import solve as strict_solve


def robot_dependency_paths(urdf):
    """Actual model and referenced mesh paths; no simulated episode is needed."""
    urdf=Path(urdf).resolve()
    paths={urdf,urdf.with_suffix(".srdf")}
    for mesh in ET.parse(urdf).getroot().iter("mesh"):
        path=(urdf.parent/mesh.attrib["filename"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return sorted(paths)

def run(proposal, contract, output, gpu, *, multiview=False):
    """Explicit experimental execution; no retries, no approval or training."""
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
    from examples.baselines.hypericl_data_factory import rich_runtime_v2
    from examples.baselines.hypericl_task_expansion import collect as base_collect
    from examples.baselines.hypericl_data_factory.collect_dataset import visibility_quality
    from examples.baselines.hypericl_task_expansion.collect import save, save_diagnostic, require
    base = Path("/mnt/sdf/datasets/hypericl_data_factory_20261016_v1")
    fruit_path = base/"fruit_adapters_v2/catalog.json"
    fruit = json.loads(fruit_path.read_text())["assets"]
    bowls = json.loads((base/"asset_adapters_v2/catalog.json").read_text())["assets"]
    bowl = next(x for x in bowls if x["uid"].startswith("11a459d5"))
    profile = base/"receptacle_geometry_v1"/(bowl["uid"]+"_cavity_profile.json")
    import mplib.planner as upstream_planner
    import mplib.pymp
    from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
    expected_urdf=Path(PandaWristCam.urdf_path).resolve()
    robot_paths=robot_dependency_paths(expected_urdf)+[Path(mplib.pymp.__file__).resolve()]
    recorder_base = VisibilityRecorder
    multiview_module = None
    execution_id = str(uuid.uuid4())
    recorder_sources = [Path(rich_runtime_v2.__file__), Path(base_collect.__file__)]
    if proposal.get("background"):
        from . import candidate_scene
        recorder_sources.append(Path(candidate_scene.__file__))
    if multiview:
        from . import multiview_recording as multiview_module
        class MultiviewRecorder(multiview_module.SynchronizedMultiviewMixin, VisibilityRecorder):
            pass
        recorder_base = MultiviewRecorder
        recorder_sources.append(Path(multiview_module.__file__))
    pins = source_pins()
    for path in (Path(__file__), Path(strict_module.__file__), Path(scene_module.__file__), Path(upstream_planner.__file__), Path(real_bowl.__file__), Path(verify_bowl.__file__), fruit_path, profile,*robot_paths,*recorder_sources):
        pins[str(path.resolve())] = bind(path)["sha256"]
    output.mkdir(parents=True)
    archive=output/"source_archive"
    archive.mkdir()
    archived=[]
    for path in (Path(__file__),Path(strict_module.__file__),Path(scene_module.__file__),
                 expected_urdf,expected_urdf.with_suffix(".srdf"),*recorder_sources):
        source=path.resolve()
        frozen=archive/(pins[str(source)]+"_"+source.name)
        frozen.write_bytes(source.read_bytes())
        if bind(frozen)["sha256"] != pins[str(source)]:
            raise RuntimeError("Source changed while pinning: "+str(source))
        archived.append({"original":str(source),"snapshot":bind(frozen)})
    save_new(archive/"manifest.json",{"files":archived,"frozen_before_environment_creation":True})
    save_new(output/"robot_model_dependencies_before_run.json",{
        "capture_timing":"before environment construction and episode execution",
        "expected_agent_class":"PandaWristCam","expected_runtime_urdf":str(expected_urdf),
        "runtime_identity_must_be_verified_before_commands":True,
        "legacy_pin_note":"Old source_pins panda_v2.urdf entry is an additional dependency, not the actual robot-model identity.",
        "dependencies":[bind(path) for path in robot_paths],
        "urdf_srdf_snapshots":str(archive),
        "referenced_meshes_and_runtime_binary":"hashed in place; not copied for each episode"})
    save_new(output/"input.json", {"proposal": proposal, "contract": contract, "source_sha256": pins,
                                  "execution_id": execution_id,
                                  "recorder_version": "synchronized_multiview_v1" if multiview else "legacy_visibility_v2",
                                  "opposite_camera": multiview_module.OPPOSITE_CAMERA if multiview else None,
                                  "scope": "Strict experimental candidate; independent official planner and pre-execution dense checks; no production qualification"})
    class Recorder(recorder_base):
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
        if multiview:
            env_id = multiview_module.register_multiview(proposal)
        else:
            register(proposal)
            env_id = "HumanGroundedAppleClutterExperimental-v1"
        env = gym.make(env_id,
                       asset_pool=fruit, bowl_asset=bowl, transfer_mode=True,
                       view_profile="balanced", data_split=proposal["split"], program=program,
                       sim_backend="physx_cpu", render_backend=gpu_info["render_backend"],
                       control_mode="pd_joint_pos", obs_mode="rgb+segmentation",
                       reward_mode="none", render_mode="rgb_array",
                       sensor_configs={"shader_pack": "rt-fast"})
        rec = Recorder(env)
        rec.reset(proposal["seed"], program)
        runtime_models=[]
        for agent in rec.unwrapped.agent.agents:
            actual=Path(agent.urdf_path).resolve()
            require(actual==expected_urdf,"Actual robot URDF differs from captured model")
            for dependency in robot_dependency_paths(actual):
                require(bind(dependency)["sha256"]==pins[str(dependency)],"Robot dependency changed before execution")
            runtime_models.append({"agent_class":type(agent).__name__,"actual_urdf":bind(actual)})
        save_new(output/"runtime_robot_models.json",{
            "models":runtime_models,"captured_dependencies_verified_before_motion":True})
        actual_added = [actor.name for actor in rec.unwrapped.added_static_distractors]
        expected_added = [item["instance_id"] for item in proposal["added_distractors"]]
        require(actual_added == expected_added, "Actual static body identities/count differ from proposal")
        save_new(output/"actual_scene_inventory.json", {"static_body_ids":actual_added,
            "static_distractor_count":len(actual_added),
            "static_categories":[item["category"] for item in proposal["added_distractors"]],
            "checked_after_reset_before_motion":True, "semantic_category_origin":"hash-bound asset catalog, not RGB detection"})
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
        events = strict_solve(rec, program, output/"plans")
        metrics, final = real_bowl.qualify(rec, program)
        require(metrics["success"], "Unchanged native physical outcome rejected")
        extra_force = np.array([r["extra_contact_force_N"] for r in rec.rows])
        require(float(extra_force.max(initial=0)) < .5, "Robot/object contacted added static clutter")
        quality = visibility_quality(rec, program)
        require(quality["passed"], "Unchanged image quality gate rejected")
        receipt = save(rec, episode, "strict_apple_clutter", program, proposal["split"],
                       proposal["seed"], events, metrics, final, pins)
        # Keep the base export before extending its receipt with native/extra QA.
        (episode/"receipt.json").rename(episode/"receipt.base.json")
        receipt.update(save_native(rec, episode))
        if multiview:
            receipt.update(multiview_module.save_multiview(rec, episode, execution_id))
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
                  "strict_plan_manifest_root": str(output/"plans"), "old_custom_planner_used": False,
                  "still_pending": ["actual RGB task/occlusion review",
                                    "gripper open/close relies on native physical contact verification; no separately planned finger trajectory",
                                    "snapshot geometry does not predict passive object falling after release",
                                    "source support has installed narrow contact guard; broader task/layout generalization remains unqualified",
                                    "formal continuous collision is not claimed"],
                  "production_approved": False, "training_admitted": False}
    except Exception as exc:
        if rec is not None and rec.rows:
            try:
                save_diagnostic(rec, output/"failure")
                np.savez_compressed(output/"failure"/"complete_contact_trace.npz",
                    grasp=np.array([r["grasp"] for r in rec.rows]),
                    bowl_support_force_N=np.array([r["bowl_support_force_N"] for r in rec.rows]),
                    robot_bowl_force_N=np.array([r["robot_bowl_force_N"] for r in rec.rows]),
                    extra_contact_force_N=np.array([r["extra_contact_force_N"] for r in rec.rows]),
                    timestamps_seconds=np.arange(len(rec.rows))/rec.control_freq,
                    camera_from_world=rec.extrinsic, intrinsic_224=rec.intrinsic)
                save_new(output/"failure"/"scene_evidence.json",{
                    "asset_specs": rec.unwrapped.asset_specs,
                    "randomization": rec.unwrapped.randomization,
                    "controller_stage_is_contact_gt": False})
                if multiview:
                    save_native(rec, output/"failure")
                    multiview_module.save_multiview(
                        rec, output/"failure", execution_id, diagnostic=True)
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
    parser.add_argument("--multiview", action="store_true",
                        help="Opt-in candidate: preserve zed2i/native and add synchronized opposite camera")
    args = parser.parse_args()
    cases = json.loads(args.proposals.read_text())["cases"]
    case = next(c for c in cases if c["case_id"] == args.case_id)
    result = run(case, json.loads(args.contract.read_text()), args.output, args.gpu,
                 multiview=args.multiview)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__":
    main()
