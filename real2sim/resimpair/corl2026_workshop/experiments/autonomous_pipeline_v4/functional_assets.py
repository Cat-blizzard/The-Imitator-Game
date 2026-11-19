"""Geometry-derived functional metadata and preflight for native mechanisms.

Primitive volume is additive, including overlaps. This is the same mass model
used by the native shape-density builder, not a claim about a real object.
"""
import math
from copy import deepcopy
import numpy as np
from .core import require


def visual_goal_region(item):
    """A drawn goal region has appearance and coordinates, but no rigid ledge."""
    require(item['kind'] != 'movable' and item['mass_kg'] == 0,
            'visual_goal_region_must_not_remove_dynamic_object_physics')
    require(not item.get('grasp_feature') and not item.get('contact_feature'),
            'functional_contact_asset_cannot_be_relabelled_visual_region')
    result = deepcopy(item)
    result.update(kind='visual_region', semantic_role='nonphysical_goal_region', collision_enabled=False)
    result['geometry'] = geometry(result)
    return result


def geometry(item):
    if item["shape"] == "sphere":
        half = np.full(3, item["radius"])
        lower, upper = -half, half
        volume = 4. * math.pi * item["radius"] ** 3 / 3.
    else:
        parts = item.get("parts", [{"center": [0., 0., 0.], "half_size": item["half_size"]}])
        centers = np.array([p["center"] for p in parts], dtype=float)
        halves = np.array([p["half_size"] for p in parts], dtype=float)
        require(np.all(halves > 0.), "nonpositive_collision_dimension")
        lower, upper = (centers-halves).min(0), (centers+halves).max(0)
        volume = float(np.prod(2.*halves, axis=1).sum())
    return {"schema": "functional_geometry_v2", "shape": item["shape"],
            "bounds_local_m": [lower.tolist(), upper.tolist()],
            "bounds_center_local_m": ((lower+upper)/2).tolist(),
            "half_size_m": ((upper-lower)/2).tolist(),
            "actor_origin_is_bounds_center": bool(np.allclose(lower, -upper)),
            "primitive_sum_volume_m3": volume,
            "density_kg_m3": item["mass_kg"]/volume if item["kind"] == "movable" else None,
            "mass_kg": item["mass_kg"], "construction": "native primitives",
            "mass_model": "uniform density per primitive; overlapping volumes are additive",
            "size_source": "engineering design, not measured real object"}


def preflight(spec):
    rows = [];by_id={o["id"]:o for o in spec["objects"]}
    for item in spec["objects"]:
        g = geometry(item)
        require(item["geometry"] == g, "stale_functional_geometry:"+item["id"])
        require(type(item.get('collision_enabled', True)) is bool, 'collision_policy_must_be_boolean')
        if not item.get('collision_enabled', True):
            require(item['kind'] == 'visual_region' and item['mass_kg'] == 0
                    and item.get('semantic_role') == 'nonphysical_goal_region'
                    and not item.get('grasp_feature') and not item.get('contact_feature'),
                    'nonphysical_collision_policy_requires_explicit_visual_region')
        if item["kind"] == "movable":
            require(item["mass_kg"] > 0. and g["density_kg_m3"] > 0., "invalid_dynamic_mass")
        feature = item.get("grasp_feature")
        if feature:
            point = np.asarray(feature["center_local"])
            lo, hi = np.asarray(g["bounds_local_m"])
            require(np.all(point >= lo) and np.all(point <= hi), "grasp_feature_outside_geometry")
            require(0. < feature["grip_width_m"] < .075, "grip_width_exceeds_panda_capacity")
            require(feature["role"] == "tool_handle", "missing_functional_grasp_role")
            require("resting_support" in item, "tool_resting_configuration_missing")
        contact=item.get("contact_feature")
        clearance=None
        if contact and "entry_between" in contact:
            a,b=[by_id[k] for k in contact["entry_between"]]
            require(a["shape"]==b["shape"]=="sphere","entry_gap_requires_explicit_obstacle_geometry")
            half_gap=float(np.linalg.norm(np.asarray(a["position"])[:2]-np.asarray(b["position"])[:2])/2)
            radius=float(np.linalg.norm(item["parts"][contact["part_index"]]["half_size"][:2]))
            need=radius+max(a["radius"],b["radius"])+contact["clearance_margin_m"]
            require(need<half_gap,"contact_feature_too_wide_for_declared_entry_gap")
            clearance={"available_half_spacing_m":half_gap,"required_radius_plus_obstacle_and_margin_m":need,
                       "scope":"conservative initial planar gap; dynamic contact checked during execution"}
        rows.append({"id": item["id"], "geometry": g, "grasp_feature": feature,
                     "tip_local": item.get("tip_local"), "resting_support": item.get("resting_support"),"entry_clearance":clearance,
                     "reachable_grasp": "requires native planning and tracking evidence" if feature else "not_requested"})
    for joint in spec["joints"]:
        require(np.isclose(np.linalg.norm(joint["axis_world"]), 1.), "joint_axis_not_unit")
        require(joint["limits"][0] <= joint["initial_q"] <= joint["limits"][1], "initial_joint_out_of_limits")
    return {"schema": "functional_asset_preflight_v2", "scene": spec["id"],
            "mechanism": spec["mechanism"], "assets": rows,
            "scope": "structural compatibility only; physical success is measured independently"}
