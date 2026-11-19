"""Source-first allocation with the retained above-rim release geometry.

One cached variant is still selected per source before target allocation. The
new preflight checks the same first 12 poses and 3cm/1cm/15cm release contract
as the real driver. It does not run IK, motion planning, cooking or dynamics.
"""
from collections import Counter
from copy import deepcopy
from itertools import product
import math
import re

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve
from resimpair_grasp.grasps import load as load_cache
from resimpair_pickplace import above_rim_release_v1 as release
from resimpair_pickplace.requested_tasks import Rejected
from . import source_first_task_selection_v1 as previous

VERSION = 'source_first_task_selection_v2_release_geometry'
choose_objects = previous.choose_objects
task_contract = previous.task_contract
_partitions = previous._partitions
sizes = previous.sizes
DEFAULT_TASK_CONTRACT = previous.DEFAULT_TASK_CONTRACT
DEFAULT_SIZE_BIN_EDGES_M = previous.DEFAULT_SIZE_BIN_EDGES_M
MAX_CANDIDATES = 12
RELEASE_POLICY = {'mode': 'drop_from_above', 'allow_drop': True, 'fragile': False,
    'material_status': 'rigid_nonbreakable_simulation_assumption_not_real_item_claim',
    'bottom_above_rim_m': .03, 'hand_above_rim_m': .01, 'maximum_fall_m': .15}
EXPECTED_RELEASE_REJECTIONS = {'required_drop_exceeds_declared_maximum', 'release_waypoint_exceeds_contract_height'}


def check_operation(scene, operation):
    """Actual scene/operation preflight, preserving the cache order and limits."""
    item = next(row for row in scene['objects'] if row['id'] == operation['object'])
    cache_ref = scene['grasp_caches'][item['id']]
    cache = load_cache(cache_ref, item['asset_descriptor'])
    if cache['signature'].get('geometry_scope') != 'native':
        raise ValueError('native_geometry_grasp_cache_required_for_release_preflight')
    if operation['target']['support'] != 'table' and operation.get('release') != RELEASE_POLICY:
        raise ValueError('retained_3cm_1cm_15cm_release_contract_required')
    decisions = []
    for candidate in cache['candidates'][:MAX_CANDIDATES]:
        try:
            desired, proof = release.release_geometry(scene, operation, np.asarray(candidate['tcp_in_asset']))
            decisions.append({'candidate_id': candidate['id'], 'passed': True,
                'release_object_transform': desired.tolist(), 'release_geometry': proof})
        except ValueError as error:
            if str(error) not in EXPECTED_RELEASE_REJECTIONS: raise
            decisions.append({'candidate_id': candidate['id'], 'passed': False, 'reason': str(error)})
    return {'object': item['id'], 'asset': item['asset_descriptor'], 'cache': cache_ref,
        'support': operation['target']['support'], 'tested_candidates': len(decisions),
        'candidate_ids': [row['candidate_id'] for row in decisions],
        'passed_candidates': sum(row['passed'] for row in decisions),
        'passed': any(row['passed'] for row in decisions),
        'failure_reasons': dict(Counter(row['reason'] for row in decisions if not row['passed'])),
        'decisions': decisions, 'geometry_implementation': binding(release.__file__),
        'policy': deepcopy(operation.get('release')), 'IK_calls': 0, 'physics_steps': 0,
        'GPU_jobs': 0, 'API_calls': 0, 'physical_success': 'unmeasured'}


class ReleaseCompatibility:
    """Cache object/receiver height checks across all bounded subset partitions."""
    def __init__(self): self.results = {}

    def check(self, item, receiver_ref, receiver):
        entry, descriptor = item
        key = digest([entry['asset'], entry['cache'], receiver_ref, RELEASE_POLICY])
        if key not in self.results:
            if read(resolve(receiver_ref)) != receiver:
                raise ValueError('receiver_descriptor_must_equal_bound_artifact')
            if read(resolve(entry['asset'])) != descriptor:
                raise ValueError('payload_descriptor_must_equal_bound_artifact')
            scene = {'table_top_z': 0., 'objects': [{'id': 'payload',
                'asset_descriptor': entry['asset'], 'half_size': descriptor['half_size'],
                'position': [0., 0., descriptor['half_size'][2]], 'yaw': 0.}],
                'supports': [{'id': 'receiver', 'container_descriptor': receiver_ref,
                    'position': [0., 0., 0.], 'yaw': 0., 'top_z': receiver['cavity']['floor_z_m']}],
                'grasp_caches': {'payload': entry['cache']}}
            operation = {'id': 'release_preflight', 'object': 'payload',
                'target': {'support': 'receiver', 'offset_xy': receiver['cavity']['center_xy_m'], 'yaw': 0.},
                'release': deepcopy(RELEASE_POLICY)}
            result = check_operation(scene, operation)
            result.update(source_id=descriptor['source_id'], receiver=receiver_ref,
                receiver_source_id=receiver['source_id'], receiver_depth_m=receiver['cavity']['depth_m'],
                scope='height-only preflight; invariant to XY translation and table-normal yaw; no workspace/clearance/IK/physical claim')
            self.results[key] = result
        return self.results[key]

    def receipt(self):
        return {'schema': VERSION, 'policy': deepcopy(RELEASE_POLICY), 'maximum_cache_candidates': MAX_CANDIDATES,
            'unique_asset_receiver_checks': len(self.results), 'checks': list(self.results.values()),
            'retained_geometry': binding(release.__file__), 'source_selection': binding(previous.__file__),
            'runtime_rescaling': False, 'new_grasps': 0, 'IK_calls': 0,
            'physics_steps': 0, 'GPU_jobs': 0, 'API_calls': 0, 'physical_success': 'unmeasured'}


def plan_targets(group, bins, partition, seed, *, contract=None, balanced_two_groups_only=False):
    """Assign the same objects to 2–4 permitted targets; no object resampling.

    Subsets use the unchanged center/line/ring packing test. Container choices
    use their real floor and outside dimensions. Table circles are explicit
    goals inside the declared receiving lane, not synthetic container assets.
    """
    contract = task_contract(contract)
    release_checker = ReleaseCompatibility()
    n = len(group)
    if not 3 <= n <= 8 or any(d["partition"] != partition for _, d in group):
        raise ValueError("one_partition_three_to_eight_selected_objects_required")
    excluded = {d["source_id"] for _, d in group}
    admitted = []
    for ref, descriptor in bins:
        if (descriptor["partition"] != partition or descriptor["source_id"] in excluded
                or not descriptor.get("roles", {}).get("stationary_container")
                or re.search(r"\b(soup|filled|water|cereal|fruit|food)\b", descriptor.get("name", ""), re.I)):
            continue
        admitted.append((ref, descriptor))
    radii = [sizes.footprint(x) for x in group]
    subset_options = {}
    for mask in range(1, 1 << n):
        indices = [i for i in range(n) if mask & (1 << i)]
        pack = sizes.slots([radii[i] for i in indices], 1.)
        required = max(float(np.linalg.norm(x)) + r for x, r in zip(pack["offsets"], pack["radii_m"])) + pack["required_boundary_margin_m"]
        options = []
        if "container" in contract["allowed_target_kinds"]:
            compatible = [(ref, d) for ref, d in admitted if sizes.slots(
                [radii[i] for i in indices], d["cavity"]["flat_floor_radius_m"])["passed"]]
            # Minimum fitting exterior reduces scene area; source choices are
            # already frozen and do not depend on this ordering.
            compatible.sort(key=lambda x: (float(np.linalg.norm(np.asarray(x[1]["dimensions_m"][:2]) / 2)),
                digest([seed, mask, x[1]["source_id"], x[0]["sha256"]])))
            feasible = next(((ref, d) for ref, d in compatible
                if all(release_checker.check(group[i], ref, d)["passed"] for i in indices)), None)
            if feasible is not None:
                ref, descriptor = feasible
                options.append({"kind": "container", "indices": indices, "descriptor": ref,
                    "source_id": descriptor["source_id"], "partition": partition,
                    "outside_radius_m": float(np.linalg.norm(np.asarray(descriptor["dimensions_m"][:2]) / 2)),
                    "floor_radius_m": descriptor["cavity"]["flat_floor_radius_m"],
                    "packing": sizes.slots([radii[i] for i in indices], descriptor["cavity"]["flat_floor_radius_m"]),
                    "release_geometry": [release_checker.check(group[i], ref, descriptor) for i in indices]})
        region_radius = math.ceil((required + 1e-6) * 1000) / 1000
        if "table_region" in contract["allowed_target_kinds"] and region_radius <= contract["maximum_table_region_radius_m"]:
            options.append({"kind": "table_region", "indices": indices,
                "outside_radius_m": region_radius, "floor_radius_m": region_radius,
                "packing": sizes.slots([radii[i] for i in indices], region_radius),
                "visual_marker_status": "contract_coordinates_only_not_rendered_by_retained_compiler"})
        subset_options[mask] = options
    examined, best, best_key = 0, None, None
    lane = contract["receiving_lane_y_m"]
    for k in contract["receiving_target_counts"]:
        if k > n or (balanced_two_groups_only and k != 2):
            continue
        for partitioned in _partitions(list(range(n)), k):
            if balanced_two_groups_only and sorted(map(len, partitioned)) != [n // 2, (n + 1) // 2]:
                continue
            examined += 1
            options = [subset_options[sum(1 << i for i in indices)] for indices in partitioned]
            for targets in product(*options):
                used = {}; consistent = True
                for t in targets:
                    if t["kind"] == "container":
                        old = used.setdefault(t["source_id"], t["descriptor"])
                        consistent &= old == t["descriptor"]
                if not consistent:
                    continue
                span = sum(2 * t["outside_radius_m"] for t in targets) + (k - 1) * contract["target_gap_m"]
                if span > lane[1] - lane[0] or max(t["outside_radius_m"] for t in targets) >= .44:
                    continue
                key = (sum(t["kind"] == "table_region" for t in targets), k, span,
                       digest([seed, "target_assignment", targets]))
                if best_key is None or key < best_key:
                    best, best_key = deepcopy(targets), key
    if best is None:
        raise Rejected("no_targets_for_fixed_object_selection", contract=contract,
            selected_source_ids=[d["source_id"] for _, d in group], radii_m=radii,
            examined_group_partitions=examined, declared_object_count_preserved=n,
            subset_choice_limit="smallest fitting and release-compatible exterior per subset; not exhaustive container-combination proof",
            release_preflight=release_checker.receipt())
    span = best_key[2]; cursor = (lane[0] + lane[1] - span) / 2
    for i, target in enumerate(best):
        radius = target["outside_radius_m"]
        target["id"] = "receiving_" + str(i)
        target["xy"] = [round(.46 - radius - .012, 7), round(cursor + radius, 7)]
        target["yaw"] = 0.
        cursor += 2 * radius + contract["target_gap_m"]
    return {"contract": contract, "targets": list(best), "examined_group_partitions": examined,
        "requested_scene_objects": n, "actual_scene_objects": n, "removed_source_ids": [],
        "target_group_count": len(best), "target_kind_counts": dict(Counter(t["kind"] for t in best)),
        "task_definition_changed_from_fixed_two_containers": len(best) != 2 or any(t["kind"] != "container" for t in best),
        "physical_success": "unmeasured", "goal_grouping_uses_physical_success": False,
        "container_search_limit": "smallest fitting and release-compatible exterior per subset; all supported object group partitions searched",
        "release_preflight": release_checker.receipt()}

