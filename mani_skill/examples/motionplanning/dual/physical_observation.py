"""Shared physical observations and native collision-model consistency checks.

Actor/TCP poses use world XYZ/WXYZ [7]; payload and finger spheres use each
robot base frame [x,y,z,r] in metres. Commands are two ordered finite [8] arm
vectors (seven absolute joint radians and one gripper scalar). Contact
separation is metres and impulse is N s. These observers retain original
physical tests and declared policy thresholds; proxy checks are not CCD.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

# Direct-script setup is test-only; normal package imports need no path changes.
if __name__ == "__main__" and not __package__:
    import mani_skill.examples.motionplanning.dual as _dual_package
    _dual_package.__path__ = [str(Path(__file__).parent), *_dual_package.__path__]
    __package__ = _dual_package.__name__

from .runtime_io import json_value
from .curobo_backend import pose_error, urdf_tcp_pose
from .curobo_scene import _matrix

ACTOR_NAMES = ("apple_out", "banana_out")

def interarm_contact_rows(contacts: Any, identities: dict[Any, tuple[int, str]]) -> list[dict[str, Any]]:
    """Read all physical interarm contact manifolds, including zero impulses."""
    result = []
    for contact in contacts:
        sides = [identities.get(body.entity) for body in contact.bodies]
        if len(sides) != 2 or any(side is None for side in sides) or sides[0][0] == sides[1][0]:
            continue
        points = [{"position_m": np.asarray(p.position).tolist(),
                   "normal": np.asarray(p.normal).tolist(),
                   "impulse_Ns": np.asarray(p.impulse).tolist(),
                   "separation_m": float(p.separation)} for p in contact.points]
        result.append({"bodies": [{"robot_index": s[0], "link": s[1]} for s in sides], "points": points})
    return result


def physical_contact_points(points: Any) -> bool:
    return any(p['separation_m']<=0 or any(x!=0 for x in p['impulse_Ns']) for p in points)


def classify_interarm_contacts(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    for row in rows:
        actual=physical_contact_points(row['points'])
        row.update(physical_contact=actual,
            positive_separation_zero_impulse_proximity=bool(row['points']) and not actual)
    return rows


def applied_action(action: dict[str, Any], *, mode: str, phase: str,
                   keys: list[str]) -> tuple[dict[str, np.ndarray], bool]:
    """Negative control changes only the right release gripper command."""
    if mode not in ("positive", "retain-right") or len(keys) != 2 or set(action) != set(keys):
        raise ValueError("Expected two canonical actions and an explicit mode")
    actual = {key: np.asarray(action[key], dtype=np.float32).copy() for key in keys}
    if any(a.shape != (8,) or not np.isfinite(a).all() for a in actual.values()):
        raise ValueError("Each action must have eight finite values")
    inject = bool(mode == "retain-right" and phase == "release" and actual[keys[1]][-1] == 1)
    if inject:
        actual[keys[1]][-1] = -1
    return actual, inject


def continue_success_only(terminated: Any, truncated: Any, info: dict[str, Any]) -> tuple[bool, bool]:
    """Return executor done signal and whether native success was continued."""
    def flag(value: Any) -> bool:
        array = np.asarray(json_value(value))
        if array.size != 1 or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
            raise ValueError("Expected one boolean signal")
        return bool(array.reshape(-1)[0])
    done, timeout = flag(terminated), flag(truncated)
    continued = bool(done and not timeout and flag(info.get("success", False)) and not flag(info.get("fail", False)))
    return bool(done and not continued), continued


def pose_matrix(pose: Any) -> np.ndarray:
    from transforms3d.quaternions import quat2mat
    values = np.asarray(pose, dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values[3:]), 1, atol=1e-5):
        raise ValueError("Expected one finite xyz/wxyz pose")
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = quat2mat(values[3:]), values[:3]
    return matrix


def held_substep_audit(rows: list[dict[str, Any]], references: Any, *, policy: Any = None,
                       target_names: Any = None) -> dict[str, Any]:
    from .lift_execution import _rotation_distance
    from .runtime_policy import LEGACY_STRICT, RuntimePolicy
    policy = LEGACY_STRICT if policy is None else policy
    if not isinstance(policy, RuntimePolicy):
        raise ValueError('A validated runtime policy is required')
    names = ACTOR_NAMES if target_names is None else tuple(target_names)
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError('Two canonical target names required')
    held = [r for r in rows if r["phase"] == "lift_and_hold" or r["phase"].startswith("carry_")]
    refs = np.asarray(references, dtype=float)
    if not held or refs.shape != (2, 4, 4) or not np.isfinite(refs).all():
        raise ValueError("Need paired actual attachment transforms and held substeps")
    objects = []
    for i, name in enumerate(names):
        dp, dr = [], []
        for row in held:
            pose = np.linalg.inv(pose_matrix(row["tcp_poses"][i])) @ pose_matrix(row["object_poses"][i])
            dp.append(float(np.linalg.norm(pose[:3, 3] - refs[i, :3, 3])))
            dr.append(_rotation_distance(refs[i, :3, :3], pose[:3, :3]))
        objects.append({"actor": name, "max_position_slip_m": max(dp), "max_rotation_slip_rad": max(dr)})
    return {"status": "passed" if all(o["max_position_slip_m"] <= policy.position_slip_m and o["max_rotation_slip_rad"] <= policy.rotation_slip_rad for o in objects) else "failed",
            "physical_substeps": len(held), "objects": objects, "position_limit_m": policy.position_slip_m, "rotation_limit_rad": policy.rotation_slip_rad,
            "scope": "both actual grasp-relative physical poses at100Hz; not collision checking orCCD"}


def contact_allowed(sides: list[dict[str, Any]], phase: str,
                    srdf: dict[int, set[frozenset[str]]]) -> bool:
    """Exact physical robot-contact permissions; no broad target/palm exception."""
    if len(sides) != 2:
        return False
    if any(s.get('link_name') == 'panda_link0' for s in sides) and any(s.get('actor') == 'table' for s in sides):
        return True
    if all(s.get('kind') == 'robot' for s in sides):
        index = sides[0]['robot_index']
        return sides[1]['robot_index'] == index and frozenset(s['link_name'] for s in sides) in srdf[index]
    if phase in ('descend', 'close_and_confirm', 'lift_and_hold', 'release') or phase.startswith('carry_'):
        for i in (0, 1):
            robot, target = sides[i], sides[1-i]
            if (robot.get('kind') == 'robot' and target.get('kind') == 'payload'
                    and robot['robot_index'] == target['target_index']
                    and robot['link_name'] in ('panda_leftfinger', 'panda_rightfinger')):
                return True
    return False


def install_full_robot_contact_capture(raw: Any, capture: Any, adapter: Any,
                                       output: list[dict[str, Any]]) -> None:
    """Supplement existing target/interarm observer with all robot/world contacts."""
    from .multitask_recording import contact_identities
    from .scene_collision import scene_entities
    agents = tuple(raw.agent.agents)
    identities, _ = contact_identities(raw, agents, adapter.target_names, adapter.targets)
    for name, entity in scene_entities(raw).items():
        identities.setdefault(entity, {'kind': 'actor', 'actor': name})
    srdf = {}
    for i, agent in enumerate(agents):
        tree = ET.parse(Path(agent.urdf_path).with_suffix('.srdf')).getroot()
        srdf[i] = {frozenset((node.attrib['link1'], node.attrib['link2']))
                   for node in tree.findall('disable_collisions')}
    original_record, original_step = capture.record_substep, capture.step

    def record() -> None:
        original_record()
        rows = []
        for contact in raw.scene.px.get_contacts():
            sides = [identities.get(body.entity, {'kind': 'unknown', 'actor': body.entity.name})
                     for body in contact.bodies]
            if not any(side.get('kind') == 'robot' for side in sides):
                continue
            points = [{'position_m': np.asarray(point.position).tolist(),
                       'normal': np.asarray(point.normal).tolist(),
                       'impulse_Ns': np.asarray(point.impulse).tolist(),
                       'separation_m': float(point.separation)} for point in contact.points]
            rows.append({'sides': sides, 'points': points,
                         'physical_contact': physical_contact_points(points),
                         'allowed': contact_allowed(sides, capture.phase, srdf)})
        output.append({'substep': len(output)+1, 'phase': capture.phase,
                       'control_step': capture.pending_control_step, 'contacts': rows})

    def step(action: dict[str, Any]) -> tuple[Any, ...]:
        first = len(output)
        outcome = original_step(action)
        if any(c['physical_contact'] and not c['allowed']
               for row in output[first:] for c in row['contacts']):
            raise RuntimeError('Forbidden actual robot/world contact at100Hz')
        return outcome

    capture.record_substep = record
    capture.step = step


def audit_payload_spheres(fk: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Read actual native base-frame payload spheres and compare real actor geometry."""
    spheres = np.asarray(fk['link_spheres']['tig_payload'], dtype=float)
    model, state = snapshot['payload_model'], snapshot['payload_state']
    if spheres.shape != (1, model['capacity'], 4) or not np.isfinite(spheres).all():
        raise ValueError('Invalid actual native payload sphere tensor')
    active = spheres[0, :, 3] > 0
    if state['attachment'] is None:
        if active.any():
            raise RuntimeError('Unattached model retains active payload collision geometry')
        return {'status': 'passed', 'active_spheres': 0}
    attachment = state['attachment']
    local = np.asarray(attachment['spheres'], dtype=float)
    transform = _matrix(attachment['object_pose_base_wxyz'])
    expected = local[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    actual = spheres[0, active]
    if actual.shape != local.shape:
        raise RuntimeError('Native attachment lost actual target collision spheres')
    error = float(np.max(np.abs(actual[:, :3] - expected)))
    if error > 1e-5 or not np.allclose(actual[:, 3], local[:, 3], rtol=0, atol=1e-7):
        raise RuntimeError(f'Native held geometry differs from actual measured actor: {error}')
    return {'status': 'passed', 'active_spheres': len(actual), 'max_center_coordinate_error_m': error}


def verify_finger_spheres(result: dict[str, Any], data: dict[str, Any],
                          queries: np.ndarray) -> list[dict[str, Any]]:
    """Compare native base-frame spheres[N,K,4] with independent URDF FK."""
    from scipy.spatial.transform import Rotation
    kin = data['robot_cfg']['kinematics']
    expected_links = ('panda_leftfinger', 'panda_rightfinger')
    native = result['link_spheres']
    rows = []
    for link in expected_links:
        local = kin['collision_spheres'][link]
        centers = np.asarray([s['center'] for s in local], dtype=float)
        radii = np.asarray([s['radius'] for s in local], dtype=float)
        spheres = np.asarray(native[link], dtype=float)
        if (spheres.shape != (len(queries), len(local), 4)
                or not np.isfinite(spheres).all()):
            raise ValueError(f'Invalid {link} sphere shape/values: {spheres.shape}')
        errors = []
        for q, row in zip(queries, spheres):
            joints = dict(zip(data['joint_names'], q.tolist())) | kin['lock_joints']
            pose = urdf_tcp_pose(Path(kin['urdf_path']), joints, tcp_link=link)
            matrix = Rotation.from_quat(np.r_[pose[4:], pose[3]]).as_matrix()
            expected = centers @ matrix.T + pose[:3]
            errors.append(float(np.max(np.abs(expected - row[:, :3]))))
        radius_error = float(np.max(np.abs(spheres[:, :, 3] - radii)))
        if max(errors) > 1e-5 or radius_error > 1e-6:
            raise RuntimeError(f'{link} collision geometry uses stale or wrong finger state')
        rows.append({'link': link, 'sphere_count': len(local),
                     'max_center_coordinate_error_m': max(errors),
                     'max_radius_error_m': radius_error})
    return rows


def check_fk(result: dict[str, Any], snapshot_data: dict[str, Any],
             query: np.ndarray) -> list[dict[str, Any]]:
    positions = np.asarray(result['positions'], dtype=float)
    quaternions = np.asarray(result['quaternions'], dtype=float)
    if positions.shape != (len(query), 3) or quaternions.shape != (len(query), 4):
        raise ValueError('cuRobo FK must return position[N,3] and quaternion[N,4]')
    names = snapshot_data['joint_names']
    urdf = Path(snapshot_data['robot_cfg']['kinematics']['urdf_path'])
    rows = []
    for i, (q, p, quat) in enumerate(zip(query, positions, quaternions)):
        expected = urdf_tcp_pose(urdf, dict(zip(names, q.tolist())))
        row = {'query': i, 'independent_urdf': pose_error(np.r_[p, quat], expected)}
        if i == 0:
            row['native_sapien'] = pose_error(np.r_[p, quat], snapshot_data['tcp_pose_base_wxyz'])
        for error in [v for v in row.values() if isinstance(v, dict)]:
            if error['position_m'] > 1e-5 or error['rotation_rad'] > 1e-4:
                raise ValueError(f'cuRobo/TIG TCP or joint mapping mismatch: {row}')
        rows.append(row)
    return rows


def measured_source_departure(checker: Any, *, refresh: bool = True) -> tuple[bool,bool]:
    """Read actual source departure without changing original support permissions.

    A never-registered tabletop exception deliberately leaves its original
    ``support_released`` flag false. Such a source is currently absent only
    when a fresh measured receipt proves every actual convex piece exceeds
    the original numeric band and no FCL piece touches the source. Candidate
    receipts are never read. This adapter has no latch; the existing physical
    recorder separately latches its >=3mm/no-native-contact transition.
    """
    from .table_support_collision import NUMERICAL_TOP_TOLERANCE_M
    if type(refresh) is not bool:
        raise ValueError('Measured source refresh policy must be an actual boolean')
    if checker is None or not getattr(checker,'attached',False):
        return False,False
    supports = getattr(checker,'_supports',())
    if len(supports) != 2:
        raise ValueError('Two canonical measured source-support checkers required')
    if all(getattr(support,'support_released',None) is True for support in supports):
        return True,True
    if refresh:
        checker.refresh()
        supports = checker._supports
    departure = []
    for support in supports:
        if getattr(support,'support_released',None) is True:
            departure.append(True)
            continue
        receipt = getattr(support,'_support_state_receipt',None)
        accepted = False
        if (getattr(support,'_initial_support_exists',None) is False
                and isinstance(receipt,dict) and receipt.get('mode') == 'measured_refresh'
                and receipt.get('all_pieces_separated') is True
                and receipt.get('fcl_touching_pieces') == []
                and receipt.get('separation_numeric_band_m') == NUMERICAL_TOP_TOLERANCE_M):
            pieces = receipt.get('pieces')
            targets = getattr(support,'_target_names',None)
            if isinstance(pieces,list) and pieces and targets:
                actual_names = [piece.get('target') for piece in pieces if isinstance(piece,dict)]
                if (len(actual_names) == len(pieces) == len(targets)
                        and all(isinstance(name,str) for name in actual_names)
                        and set(actual_names) == set(targets)):
                    try:
                        gaps = np.asarray([piece['bottom_minus_table_top_m'] for piece in pieces],dtype=float)
                        accepted = bool(np.isfinite(gaps).all() and np.all(gaps > NUMERICAL_TOP_TOLERANCE_M))
                    except (KeyError,TypeError,ValueError):
                        accepted = False
        departure.append(accepted)
    return tuple(departure)

def make_checker(raw: Any, solvers: list[Any], adapter: Any) -> Any:
    """Reuse each existing task's measured support adapter and common checker interface."""
    from .multitask_adapters import TASK_IDS as TASKS
    if adapter.task_id == TASKS[1]:
        from .pill_support_collision import PillDualPayloadCollisionChecker
        return PillDualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=adapter.targets)
    if adapter.task_id == TASKS[2]:
        from .support_surface_collision import SourceSupportDualPayloadCollisionChecker
        return SourceSupportDualPayloadCollisionChecker.from_solvers(raw, *solvers,
            targets=adapter.targets, source_supports=adapter.initial_supports)
    from .dual_payload_collision import DualPayloadCollisionChecker
    return DualPayloadCollisionChecker.from_solvers(raw, *solvers, targets=adapter.targets)


def source_departure_self_test() -> dict[str, Any]:
    """Retain the replay's twelve malformed measured-receipt negative cases."""
    import copy
    from types import SimpleNamespace
    def support() -> Any:
        return SimpleNamespace(support_released=False, _initial_support_exists=False,
            _target_names={'piece0'}, _support_state_receipt={
                'mode': 'measured_refresh', 'separation_numeric_band_m': 1e-6,
                'all_pieces_separated': True, 'fcl_touching_pieces': [],
                'pieces': [{'target': 'piece0', 'bottom_minus_table_top_m': 2e-6}]})
    refresh_calls = []
    checker = SimpleNamespace(attached=True, _supports=(support(), support()),
                              refresh=lambda: refresh_calls.append('measured_geometry_refresh'))
    assert measured_source_departure(checker) == (True, True)
    assert refresh_calls == ['measured_geometry_refresh']
    assert not any(item.support_released for item in checker._supports)
    departure_negatives = 0
    invalid_receipts = [None, {},
        {'mode': 'supplied_object_pose'}, {'mode': 'candidate'}]
    valid_receipt = copy.deepcopy(support()._support_state_receipt)
    invalid_receipts += [
        {**valid_receipt, 'mode': 'candidate'},
        {**valid_receipt, 'all_pieces_separated': False},
        {**valid_receipt, 'fcl_touching_pieces': ['piece0']},
        {**valid_receipt, 'pieces': []},
        {**valid_receipt, 'pieces': [{'target': 'other', 'bottom_minus_table_top_m': 2e-6}]},
        {**valid_receipt, 'pieces': [{'target': 'piece0', 'bottom_minus_table_top_m': 1e-6}]},
        {**valid_receipt, 'pieces': [{'target': 'piece0', 'bottom_minus_table_top_m': float('nan')}]},
        {**valid_receipt, 'separation_numeric_band_m': 0.}]
    for receipt in invalid_receipts:
        first, second = support(), support()
        first._support_state_receipt = receipt
        # A favorable candidate query cannot substitute for missing/invalid
        # actual measured evidence, even when its numbers look separated.
        first._last_support_validation = copy.deepcopy(valid_receipt)
        checker._supports = (first, second)
        assert measured_source_departure(checker) == (False, True)
        assert first.support_released is False
        departure_negatives += 1
    checker._supports = (support(), support())
    checker._supports[0]._initial_support_exists = True
    assert measured_source_departure(checker) == (False, True)
    checker._supports[0].support_released = True
    assert measured_source_departure(checker) == (True, True)
    checker.attached = False
    assert measured_source_departure(checker) == (False, False)
    assert departure_negatives == 12
    return {"status": "passed", "malformed_measured_receipt_negative_cases": departure_negatives,
            "original_source_support_permissions_unchanged": True,
            "physical_environment_steps": 0}



def self_test(archive: Path | None = None) -> dict[str, Any]:
    """Check physical observations using CPU arrays and optional real archives."""
    from dataclasses import replace
    from types import SimpleNamespace as NS
    from .runtime_policy import LEGACY_STRICT

    def rejects(function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            function(*args, **kwargs)
        except (ValueError, RuntimeError):
            return
        raise AssertionError(f"Expected rejection from {function.__name__}")

    keys = ["left", "right"]
    original = {key: np.r_[np.zeros(7), 1].astype(np.float32) for key in keys}
    actual, injected = applied_action(original, mode="retain-right", phase="release", keys=keys)
    assert injected and actual["right"][-1] == -1 and original["right"][-1] == 1
    assert applied_action(original, mode="positive", phase="release", keys=keys)[1] is False
    assert applied_action(original, mode="retain-right", phase="carry_mid", keys=keys)[1] is False
    rejects(applied_action, original, mode="unknown", phase="release", keys=keys)
    rejects(applied_action, {"left": np.zeros(7), "right": np.zeros(8)}, mode="positive", phase="release", keys=keys)
    assert continue_success_only([1], [0], {"success": [1], "fail": [0]}) == (False, True)
    assert continue_success_only([1], [0], {"success": [1], "fail": [1]}) == (True, False)
    assert continue_success_only([1], [1], {"success": [1]}) == (True, False)
    assert continue_success_only([0], [0], {}) == (False, False)
    rejects(continue_success_only, [0, 1], [0], {})
    rejects(continue_success_only, [np.nan], [0], {})
    points = lambda gap, impulse: [{"position_m": [0., 0., 0.], "normal": [0., 0., 1.],
                                    "impulse_Ns": impulse, "separation_m": gap}]
    assert not physical_contact_points(points(.001, [0, 0, 0]))
    assert physical_contact_points(points(0, [0, 0, 0]))
    assert physical_contact_points(points(.001, [0, 0, .1]))
    rows = classify_interarm_contacts([{"points": points(.001, [0, 0, 0])},
                                       {"points": points(-.001, [0, 0, 0])}, {"points": []}])
    assert rows[0]["positive_separation_zero_impulse_proximity"] and not rows[0]["physical_contact"]
    assert rows[1]["physical_contact"] and not rows[2]["physical_contact"]
    left, right, unknown = object(), object(), object()
    point = NS(position=np.zeros(3), normal=np.array([0, 0, 1]), impulse=np.zeros(3), separation=.001)
    contacts = [NS(bodies=[NS(entity=left), NS(entity=right)], points=[point]),
                NS(bodies=[NS(entity=left), NS(entity=unknown)], points=[point]),
                NS(bodies=[NS(entity=left), NS(entity=left)], points=[point])]
    manifold = interarm_contact_rows(contacts, {left: (0, "panda_hand"), right: (1, "panda_hand")})
    assert len(manifold) == 1 and manifold[0]["points"][0]["separation_m"] == .001
    robot = lambda i, link: {"kind": "robot", "robot_index": i, "link_name": link}
    payload = lambda i: {"kind": "payload", "target_index": i}
    srdf = {0: {frozenset(("a", "b"))}, 1: set()}
    assert contact_allowed([robot(0, "a"), robot(0, "b")], "pregrasp", srdf)
    assert not contact_allowed([robot(0, "a"), robot(1, "b")], "pregrasp", srdf)
    for phase in ("descend", "close_and_confirm", "lift_and_hold", "carry_mid", "release"):
        assert contact_allowed([robot(0, "panda_leftfinger"), payload(0)], phase, srdf)
    assert not contact_allowed([robot(0, "panda_hand"), payload(0)], "descend", srdf)
    assert not contact_allowed([robot(0, "panda_leftfinger"), payload(1)], "descend", srdf)
    assert not contact_allowed([robot(0, "panda_leftfinger"), payload(0)], "pregrasp", srdf)
    assert contact_allowed([robot(0, "panda_link0"), {"actor": "table"}], "pregrasp", srdf)
    tcp = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    objects = [np.array([.01, 0, 0, 1, 0, 0, 0]), np.array([0, .02, 0, 1, 0, 0, 0])]
    references = [pose_matrix(pose) for pose in objects]
    held = [{"phase": "carry_mid", "tcp_poses": [tcp.tolist(), tcp.tolist()],
             "object_poses": [pose.tolist() for pose in objects]}]
    audit = held_substep_audit(held, references, target_names=("pill_a", "pill_b"))
    assert audit["status"] == "passed" and [row["actor"] for row in audit["objects"]] == ["pill_a", "pill_b"]
    held[0]["object_poses"][0][0] += .006
    assert held_substep_audit(held, references)["status"] == "failed"
    assert held_substep_audit(held, references, policy=replace(LEGACY_STRICT, position_slip_m=.01))["status"] == "passed"
    rejects(held_substep_audit, held, references, target_names=("same", "same"))
    rejects(held_substep_audit, held, references, policy={})
    rejects(pose_matrix, [0, 0, 0, 2, 0, 0, 0])
    fk = {"link_spheres": {"tig_payload": [[[0., 0., 0., -.1], [0., 0., 0., -.1]]]}}
    snapshot = {"payload_model": {"capacity": 2}, "payload_state": {"attachment": None}}
    assert audit_payload_spheres(fk, snapshot)["active_spheres"] == 0
    fk["link_spheres"]["tig_payload"][0][0][3] = .01
    rejects(audit_payload_spheres, fk, snapshot)
    attachment = {"spheres": [[0., 0., 0., .01]], "object_pose_base_wxyz": [.1, .2, .3, 1, 0, 0, 0]}
    snapshot["payload_state"]["attachment"] = attachment
    fk["link_spheres"]["tig_payload"][0][0] = [.1, .2, .3, .01]
    assert audit_payload_spheres(fk, snapshot)["active_spheres"] == 1
    fk["link_spheres"]["tig_payload"][0][0][0] += .001
    rejects(audit_payload_spheres, fk, snapshot)
    probes = []
    if archive is not None:
        for phase in ("pregrasp", "lift_and_hold"):
            for arm in (0, 1):
                data = json.loads((archive / f"{phase}-arm-{arm}-snapshot.json").read_text())
                result = json.loads((archive / f"{phase}-arm-{arm}-fk.json").read_text())
                query = np.asarray(data["q_start"], dtype=float).reshape(1, -1)
                probes.append({"phase": phase, "arm": arm,
                               "fk": check_fk(result, data, query),
                               "fingers": verify_finger_spheres(result, data, query),
                               "payload": audit_payload_spheres(result, data)})
    return {"status": "passed", "physical_environment_steps": 0,
            "archive_probes": probes, "policy_parameters_preserved": True,
            "contact_and_failure_negative_controls": True,
            "live_observer_installation": "not attempted without a real environment",
            "source_departure": source_departure_self_test()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(self_test(args.archive), indent=2, allow_nan=False))
