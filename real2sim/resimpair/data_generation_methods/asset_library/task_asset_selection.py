"""Candidate task-role asset selector, with separately scoped qualifications.

This module never downloads, runs a simulator, calls a VLM, or treats a catalog
name match as manipulation qualification. Unknown capabilities fail closed.
Static-preflight candidates can be selected explicitly for a small experiment;
their physics_qualified/training_admitted flags remain false.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ..human_to_robot_pipeline.household_assets import bind
from ..r2h_pipeline.common import file_sha256, write_json


class SelectionError(ValueError):
    """An explicit requirement could not be met; do not silently reduce it."""


def normalized(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def _check_files(asset):
    files = asset.get('files', {})
    if not files or any(asset.get(k) not in files for k in ('visual', 'collision')):
        raise SelectionError('Missing bound visual/collision files: '+asset.get('uid', '?'))
    for name, expected in files.items():
        if not Path(name).is_file() or file_sha256(name) != expected:
            raise SelectionError('Asset file missing or changed: '+name)
    if asset.get('asset_spec'):
        binding = asset['asset_spec']
        if file_sha256(binding['path']) != binding['sha256']:
            raise SelectionError('Asset sidecar changed')
        recorded = json.loads(Path(binding['path']).read_text())
        for key in ('uid', 'category', 'semantic_categories', 'visual', 'collision',
                    'files', 'qualification', 'extents_m', 'bounds_rotated_m', 'collision_policy'):
            if key in recorded and asset.get(key) != recorded[key]:
                raise SelectionError('Runtime asset declaration differs from bound sidecar: '+key)


def _active_evidence(asset, mechanism, required):
    """Validate a historical observation binding, not a new physics outcome."""
    # Closed mechanism registry: a sidecar may not relabel an apple episode as
    # evidence for lid turning, cabinet opening, or a new embodiment.
    if mechanism != 'apple_bowl_transfer':
        return None
    for proof in asset.get('capability_evidence', []):
        if proof.get('mechanism') != mechanism or not set(required) <= set(proof.get('capabilities', [])):
            continue
        if proof.get('status') != 'observed_passed_in_bound_episode':
            continue
        for key in ('receipt', 'independent_verification'):
            b = proof.get(key, {})
            if not b.get('path') or file_sha256(b['path']) != b.get('sha256'):
                raise SelectionError('Historical qualification evidence changed')
        receipt = json.loads(Path(proof['receipt']['path']).read_text())
        verifier = json.loads(Path(proof['independent_verification']['path']).read_text())
        if verifier.get('passed') is not True or verifier.get('receipt_sha256') != proof['receipt']['sha256']:
            raise SelectionError('Verifier does not pass the bound receipt')
        if (receipt.get('goal_key') != 'factory:transfer_apple_occupied_bowl_to_empty_bowl'
                or receipt.get('program') not in {'pick_first_to_right', 'pick_second_to_right'}
                or receipt.get('embodiment') != 'dual_panda_wristcam'):
            raise SelectionError('Bound episode is outside the registered mechanism/embodiment')
        subjects = list(receipt.get('asset_specs', []))+[receipt.get('receptacle', {})]
        subject = next((s for s in subjects if s.get('uid') == asset['uid']), None)
        if subject is None or subject.get('files') != asset.get('files'):
            raise SelectionError('Asset is not the geometry used in the qualification episode')
        if subject is receipt.get('receptacle'):
            expected_category, permitted = 'bowl', {'support_inside_cavity'}
        elif subject.get('name') == 'apple':
            expected_category, permitted = 'apple', {'grasp_lift_release'}
        else:
            raise SelectionError('Unselected episode object has no active manipulation qualification')
        if ({normalized(c) for c in asset.get('semantic_categories', [])} != {expected_category}
                or normalized(asset.get('category')) != expected_category
                or not set(proof.get('capabilities', [])) <= permitted
                or not set(required) <= permitted):
            raise SelectionError('Sidecar semantic label or capability does not match the actual episode role')
        return deepcopy(proof)
    return None


def role_candidates(role, assets, mechanism):
    if not isinstance(role, dict) or not role.get('category'):
        raise SelectionError('Each role needs an explicit category')
    if role.get('count', 1) != 1:
        raise SelectionError('Use separately named roles for multiple task instances')
    category = normalized(role['category'])
    capabilities = role.get('required_capabilities', [])
    if not capabilities:
        raise SelectionError('Active role needs explicit mechanism capabilities')
    result = []
    for a in assets:
        if a.get('usage') != 'active_task_object':
            continue
        if category not in {normalized(c) for c in a.get('semantic_categories', [])}:
            continue
        if role.get('allowed_asset_uids') and a['uid'] not in role['allowed_asset_uids']:
            continue
        if 'max_extent_m' in role:
            bounds = a.get('bounds_rotated_m', a.get('bounds_m'))
            if bounds is None:
                continue  # Unknown metric geometry cannot satisfy a size limit.
            dimensions = np.ptp(np.asarray(bounds, float), axis=0)
            if (dimensions.shape != (3,) or not np.isfinite(dimensions).all()
                    or dimensions.max() > float(role['max_extent_m'])):
                continue
        _check_files(a)
        proof = _active_evidence(a, mechanism, capabilities)
        if proof:
            result.append((a, proof))
    return sorted(result, key=lambda x:x[0]['uid'])


def select_distractors(assets, *, excluded_categories, count, seed,
                      permit_preflight_experiment=False, excluded_asset_uids=(), allowed_categories=None):
    if type(count) is not int or not 0 <= count <= 12:
        raise SelectionError('Bounded distractor count must be an integer in [0,12]')
    forbidden = {normalized(c) for c in excluded_categories}
    forbidden_ids = set(excluded_asset_uids)
    allowed = None if allowed_categories is None else {normalized(c) for c in allowed_categories}
    # No category-level substitutions: apple/bowl does not silently become pear/cup.
    candidates = []
    for a in assets:
        if a.get('usage') != 'static_distractor' or a.get('uid') in forbidden_ids:
            continue
        categories = {normalized(c) for c in a.get('semantic_categories', [a.get('category')])}
        if categories & forbidden:
            continue
        if allowed is not None and normalized(a.get('category')) not in allowed:
            continue
        if not permit_preflight_experiment:
            # This candidate intentionally has no global static-scene approval:
            # scene collision/reachability must be checked for every layout.
            continue
        if a.get('qualification', {}).get('static_geometry_preflight') != 'passed':
            continue
        _check_files(a)
        candidates.append(a)
    rng = np.random.default_rng(seed)
    # Canonical ordering makes input catalog order irrelevant to reproducibility.
    candidates.sort(key=lambda a:a['uid'])
    order = rng.permutation(len(candidates))
    chosen, used_categories, used_geometry = [], set(), set()
    for i in order:
        a = candidates[int(i)]
        category = normalized(a['category'])
        geometry = a['files'][a['visual']]
        if category in used_categories or geometry in used_geometry:
            continue
        chosen.append(deepcopy(a));used_categories.add(category);used_geometry.add(geometry)
        if len(chosen) == count:
            break
    if count == 0:
        return []
    if len(chosen) != count:
        raise SelectionError(f'Requested {count} distinct distractor categories; only {len(chosen)} eligible. No silent reduction.')
    return chosen


def select_task_assets(request, assets):
    mechanism = request.get('mechanism')
    if not mechanism or not isinstance(request.get('roles'), dict):
        raise SelectionError('Task request requires mechanism and explicit role map')
    resolved, blocked = {}, []
    for name, requirement in request['roles'].items():
        choices = role_candidates(requirement, assets, mechanism)
        if not choices:
            blocked.append({'role':name, 'category':requirement.get('category'),
                            'reason':'no_bound_historical_capability_evidence'})
            continue
        asset, evidence = choices[0]
        resolved[name] = {'asset':deepcopy(asset), 'capability_evidence':evidence,
                          'new_layout_requires_independent_physical_validation':True,
                          'instance_id':name}
    if blocked:
        return {'schema':'task_asset_selection_v1', 'status':'blocked', 'blocked_roles':blocked,
                'roles':resolved, 'distractors':[], 'training_admitted':False}
    clutter = request.get('distractors', {})
    exclusions = [r['category'] for r in request['roles'].values()]
    exclusions += clutter.get('exclude_categories', [])
    selected = select_distractors(assets, excluded_categories=exclusions,
        count=clutter.get('count', 0), seed=request.get('seed', 0),
        permit_preflight_experiment=clutter.get('permit_preflight_experiment', False),
        excluded_asset_uids=[r['asset']['uid'] for r in resolved.values()],
        allowed_categories=clutter.get('allowed_categories'))
    return {'schema':'task_asset_selection_v1', 'status':'candidate_ready_for_small_validation',
            'request':deepcopy(request), 'roles':resolved, 'distractors':selected,
            'simulation_executed':False, 'physics_qualified':False, 'training_admitted':False,
            'constraints':['Task role identities remain separate even if two roles use the same model.',
                           'No guarantee of scene reachability, visibility or contact before execution.',
                           'A historical asset success is not a new-task success.']}


def proposal_distractor(asset, *, instance_id, center_xy, yaw_rad=0.):
    """Emit exactly the existing loader's baked-metric static obstacle interface."""
    _check_files(asset)
    if asset.get('usage') != 'static_distractor':
        raise SelectionError('Only static distractors may enter this conversion')
    extent = np.asarray(asset['extents_m'], float)
    center_xy = np.asarray(center_xy, float)
    if (extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0)
            or center_xy.shape != (2,) or not np.isfinite(center_xy).all() or not np.isfinite(yaw_rad)):
        raise SelectionError('Invalid metric geometry/placement')
    return {'instance_id':instance_id, 'asset_uid':asset['uid'], 'category':asset['category'],
            'center':[float(center_xy[0]),float(center_xy[1]),float(extent[2]/2)],
            'radius_m':float(np.linalg.norm(extent)/2), 'yaw_rad':float(yaw_rad),
            'visual':asset['visual'], 'collision':asset['collision'], 'asset_files':deepcopy(asset['files']),
            'bounds_rotated_m':deepcopy(asset['bounds_rotated_m']), 'asset_spec':deepcopy(asset['asset_spec']),
            'must_remain_untouched':True, 'body_type':'static', 'collision_policy':asset['collision_policy'],
            'source_license':asset['license'], 'no_runtime_scaling':True,
            'qualification_scope':'CPU static preflight only; scene execution required'}


def historical_apple_assets(folder):
    """Import only objects actually bound to the archived apple verifier result."""
    folder=Path(folder)
    receipt_path=folder/'episode/receipt.json';verify_path=folder/'verification.json'
    receipt=json.loads(receipt_path.read_text())
    result=[]
    selected=next(a for a in receipt['asset_specs'] if a['name']=='apple')
    for spec, category, capabilities in [
        (selected,'apple',['grasp_lift_release']),
        (receipt['receptacle'],'bowl',['support_inside_cavity'])]:
        item=deepcopy(spec)
        item.update({'semantic_categories':[category], 'category':category,
            'usage':'active_task_object', 'physics_qualified':False, 'training_admitted':False,
            'capability_evidence':[{'mechanism':'apple_bowl_transfer',
                'capabilities':capabilities, 'status':'observed_passed_in_bound_episode',
                'receipt':bind(receipt_path), 'independent_verification':bind(verify_path),
                'scope':'Only this historical physical configuration, not arbitrary new layouts or manipulation.'}]})
        _check_files(item)
        if not _active_evidence(item, 'apple_bowl_transfer', capabilities):
            raise SelectionError('Cannot import unverified historical asset')
        result.append(item)
    return result


def load_static_catalog(path):
    result=[]
    for a in json.loads(Path(path).read_text())['assets']:
        a=deepcopy(a);a['usage']='static_distractor'
        _check_files(a)
        result.append(a)
    return result


def discover_raw(index, *, terms, sources=(), limit=20):
    """Name/category discovery only; rows cannot be passed as executable assets."""
    words=[normalized(t) for t in terms]
    result=[]
    for line in Path(index).open():
        r=json.loads(line)
        if r['kind']!='model' or (sources and r['source'] not in sources):
            continue
        text=normalized(r.get('name','')+' '+str(r.get('categories',[])))
        score=sum(w in text for w in words)
        if score:
            result.append({'asset_id':r['asset_id'], 'name':r['name'], 'source':r['source'],
                           'primary_path':r['primary_path'], 'indexed_sha256':r.get('sha256'),
                           'license':r.get('license'), 'score':score,
                           'usage':'raw_discovery_only', 'physics_qualified':False,
                           'required_next_step':'Explicit metric/collision/material preparation and role-specific qualification'})
    return sorted(result,key=lambda r:(-r['score'],r['asset_id']))[:limit]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-apple',type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    assets=load_static_catalog(args.catalog)
    if args.historical_apple:assets+=historical_apple_assets(args.historical_apple)
    request=json.loads(args.request.read_text())
    result=select_task_assets(request,assets)
    result['catalog']=bind(args.catalog);result['request_binding']=bind(args.request)
    write_json(args.output,result)
    print(json.dumps({'status':result['status'], 'roles':list(result['roles']),
                      'distractors':[a['category'] for a in result['distractors']]}))


if __name__=='__main__':main()
