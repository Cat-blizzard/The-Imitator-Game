"""Measured contact and held-object observations for synchronous execution.

Poses are world XYZ/WXYZ [7]; contact distances are metres and impulses are
N s. Geometric source release and observed physical support departure are
separate facts. These observers do not change collision masks or physics.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PHYSICAL_SUPPORT_CLEARANCE_M = 0.003


def physical_support_transition(previous: bool, geometry_released: bool, reference_z: float,
                                measured_z: float, source_contact: bool) -> bool:
    """Close initial physical support once, after a measured 3mm upward exit.

    Geometric collision checking retains its original independent release rule.
    Native contact manifolds describe the preceding narrow-phase state; their
    micro-scale startup contacts remain initial support until this explicit
    physical phase transition. No collision depth or force threshold changes.
    """
    if not np.isfinite([reference_z, measured_z]).all():
        raise ValueError('Finite support reference and measured origin heights required')
    return bool(previous or (geometry_released and not source_contact
                            and measured_z-reference_z >= PHYSICAL_SUPPORT_CLEARANCE_M))

def contact_identities(raw: Any, agents: Any, names: Any, targets: Any) -> tuple[dict, dict]:
    identities = {obj.entity: {'kind':'robot','robot_index':i,'link_name':link.name,'actor':link.name}
        for i,agent in enumerate(agents) for link in agent.robot.get_links() for obj in link._objs}
    for name,actor in raw.scene.actors.items():
        for obj in actor._objs:
            if obj not in identities: identities[obj] = {'kind':'actor','actor':name}
    for i,actor in enumerate(targets):
        for obj in actor._objs: identities[obj] = {'kind':'payload','actor':names[i],'target_index':i}
    robots = {entity:(value['robot_index'],value['link_name']) for entity,value in identities.items() if value['kind']=='robot'}
    return identities, robots

def support_name(actor: Any) -> str:
    return str(actor.name)

def contact_class(other: dict, index: int, released: bool, phase: str,
                  initial: tuple, destinations: tuple) -> str:
    if other.get('robot_index') == index and other.get('link_name') in ('panda_leftfinger','panda_rightfinger'):
        return 'own_finger_allowed'
    placement = phase in ('release','retreat','verify_placement','observe_task_outcome')
    if placement and other.get('actor') == destinations[index]: return 'destination_support'
    if not placement and not released and other.get('actor') == initial[index]: return 'initial_support'
    return 'forbidden'

def task_object_contact_rows(contacts: Any, identities: dict, released: tuple,
                             phase: str | tuple[str, str], adapter: Any, *,
                             include_destination_contents_proof: bool = True) -> list[dict]:
    rows = []
    if not isinstance(phase, str) and (len(phase) != 2 or any(not isinstance(p, str) for p in phase)):
        raise ValueError('Two canonical target phases are required')
    initial = tuple(support_name(a) for a in adapter.initial_supports)
    destinations = tuple(support_name(a) for a in adapter.destination_supports)
    contents_names = tuple(getattr(adapter, 'destination_contents_state_names', ()))
    for contact in contacts:
        sides = [dict(identities.get(body.entity, {'kind':'unknown','actor':str(getattr(body.entity,'name','unknown'))})) for body in contact.bodies]
        indices = [s['target_index'] for s in sides if s.get('target_index') is not None]
        if len(sides)!=2 or not indices: continue
        which = next(i for i,s in enumerate(sides) if s.get('target_index') is not None)
        index = sides[which]['target_index'];other = sides[1-which]
        points = [{'position_m':np.asarray(p.position).tolist(),'normal':np.asarray(p.normal).tolist(),
            'impulse_Ns':np.asarray(p.impulse).tolist(),'separation_m':float(p.separation)} for p in contact.points]
        actual = physical_contact_points(points)
        chosen = phase if isinstance(phase, str) else phase[index]
        row = {'bodies':sides,'target_indices':indices,'class':contact_class(other,index,released[index],chosen,initial,destinations),
            'support_released_snapshot':list(released),'initial_support_names':list(initial),'destination_support_names':list(destinations),
            'points':points,'physical_contact':actual,'positive_separation_zero_impulse_proximity':bool(points) and not actual}
        if include_destination_contents_proof and other.get('kind') == 'actor' and other.get('actor') in contents_names:
            proof = adapter.destination_contents_contact_proof(index, other['actor'], chosen, points)
            row['destination_contents_proof'] = proof
            if proof['allowed']:
                row['class'] = 'destination_contents'
        if not isinstance(phase, str): row['target_phases'] = list(phase)
        rows.append(row)
    return rows


def physical_contact_points(points: Any) -> bool:
    return any(p['separation_m']<=0 or any(x!=0 for x in p['impulse_Ns']) for p in points)


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
    from .measurements import _rotation_distance
    from .runtime_policy import DEFAULT_POLICY, RuntimePolicy
    policy = DEFAULT_POLICY if policy is None else policy
    if not isinstance(policy, RuntimePolicy):
        raise ValueError('A validated runtime policy is required')
    names = ("left_target", "right_target") if target_names is None else tuple(target_names)
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

def own_finger_contact_pair(sides: list[dict[str, Any]]) -> tuple[int, str] | None:
    """Identify only a physical finger touching its own paired target."""
    if len(sides) == 2:
        for i in (0, 1):
            robot, target = sides[i], sides[1-i]
            if (robot.get('kind') == 'robot' and target.get('kind') == 'payload'
                    and robot['robot_index'] == target['target_index']
                    and robot['link_name'] in ('panda_leftfinger', 'panda_rightfinger')):
                return robot['robot_index'], robot['link_name']
    return None


def contact_allowed(sides: list[dict[str, Any]], phase: str,
                    srdf: dict[int, set[frozenset[str]]], *,
                    retreat_contacts: set[tuple[int, str]] | None = None) -> bool:
    """Exact physical robot-contact permissions; no broad target/palm exception."""
    if len(sides) != 2:
        return False
    if any(s.get('link_name') == 'panda_link0' for s in sides) and any(s.get('actor') == 'table' for s in sides):
        return True
    if all(s.get('kind') == 'robot' for s in sides):
        index = sides[0]['robot_index']
        return sides[1]['robot_index'] == index and frozenset(s['link_name'] for s in sides) in srdf[index]
    pair = own_finger_contact_pair(sides)
    if phase == 'retreat':
        return pair is not None and retreat_contacts is not None and pair in retreat_contacts
    if phase in ('descend', 'close_and_confirm', 'lift_and_hold', 'release') or phase.startswith('carry_'):
        return pair is not None
    return False


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
