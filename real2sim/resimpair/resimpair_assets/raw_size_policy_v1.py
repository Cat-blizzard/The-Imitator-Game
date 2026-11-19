"""Resolve original mesh units before preprocessing; never fit to a gripper.

Unknown units get a reproducible category-size hypothesis. Known metric sources
retain their original scale. This does not establish semantic or physical truth.
"""
from copy import deepcopy
import hashlib
import math
import re

import numpy as np

from resimpair_grasp.common import ROOT, binding, read
from .metric_inventory_v7 import metric_match
from .metric_semantics_v1 import profile, normalize

VERSION = 'original_geometry_size_policy_v1'
CONFIG = ROOT / 'configs/metric_population_v7.json'
METRIC_SOURCES = {'google_scanned_objects_original_fuel', 'ycb_official', 'polyhaven'}
CONTAINER_PROFILES = [
    ('wide_tray', r'\b(tray|platter|serving plate|baking sheet|roasting pan)\b', .42, [.25, .65]),
    ('shallow_basin', r'\b(basin|wash bowl|washbasin)\b', .40, [.25, .60]),
    ('storage_box', r'\b(storage box|storage bin|storage container|organizer|tote|crate|open box)\b', .45, [.28, .65]),
    ('basket', r'\bbasket\b', .40, [.25, .60]),
]


def decide_raw(row, original_obb, *, metadata_records=(), seed=20261031, role_override=None):
    dims = np.sort(np.asarray(original_obb, dtype=float))
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 1e-10):
        raise ValueError('finite_positive_original_OBB_required')
    if type(seed) is not int or role_override not in (None, 'container'):
        raise ValueError('recorded_integer_seed_and_supported_role_override_required')
    sid = row.get('id', row.get('source_id'))
    if not isinstance(sid, str) or ':' not in sid:
        raise ValueError('raw_source_identity_required')
    source = row.get('source', sid.split(':', 1)[0])
    cfg = read(CONFIG)
    chosen = profile(row, cfg)
    if role_override == 'container':
        if chosen['role'] != 'container':
            raise ValueError('container_override_requires_existing_container_role_evidence')
        name = normalize(row.get('name', ''))
        for category, pattern, anchor, interval in CONTAINER_PROFILES:
            if re.search(pattern, name):
                chosen = {**chosen, 'name': category, 'base_longest_m': anchor, 'longest_m': interval}
                break
    matched = metric_match(dims, metadata_records, cfg['metadata_uniform_fit_tolerance'])
    mass = None
    if source == 'abo' and matched['status'] == 'matched_uniform_metric_metadata':
        factor = float(matched['selected']['scale'])
        confidence = 'source_metadata_geometry_agreement'
        evidence = {'metadata_match': matched, 'source_metric_truth': 'product_dimensions_agree_with_mesh_shape'}
        weights = matched['selected']['record'].get('weights_kg', [])
        if weights and min(weights) > 0 and max(weights) / min(weights) <= 1.1:
            mass = float(np.median(weights))
        jitter = 1.
    elif source in METRIC_SOURCES:
        factor = 1.
        confidence = 'source_metric_convention_engineering_inference'
        evidence = {'source_convention': source, 'per_object_metric_truth': 'not_independently_measured'}
        jitter = 1.
    else:
        if chosen['name'] == 'role_scale_pending':
            raise ValueError('unresolved_role_and_original_units')
        # Stratification belongs to the sampler, not post-hoc mesh shrinking.
        size_key = row.get('source_sha') or sid
        number = int(hashlib.sha256(f'{VERSION}:{seed}:{size_key}'.encode()).hexdigest()[:16], 16)
        jitter = .9 + .2 * (number / (2**64 - 1))
        target = float(chosen['base_longest_m']) * jitter
        low, high = chosen['longest_m']
        if not low <= target <= high:
            raise ValueError('category_anchor_variation_outside_declared_interval')
        factor = target / float(dims[-1])
        confidence = 'engineering_category_size_unknown_source_units'
        evidence = {'category': chosen['name'], 'anchor_longest_m': chosen['base_longest_m'],
            'plausible_longest_m': chosen['longest_m'], 'source_hash_variation_factor': jitter,
            'variation_identity': size_key,
            'rationale': 'Category anchor with recorded source-uniform +/-10% variation; independent of gripper, container, old normalized scale and physics outcomes.',
            'source_metric_truth': 'unknown'}
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError('positive_finite_source_unit_conversion_required')
    longest = float(dims[-1] * factor)
    role = chosen['role']
    if chosen['name'] == 'role_scale_pending' and source in METRIC_SOURCES and longest <= .4:
        chosen = {**chosen, 'name': 'metric_bounded_rigid_proxy', 'role': 'manipulation'}
        role = 'manipulation'
    if role == 'manipulation' and longest > .6:
        role = 'background'
    return {'schema': VERSION, 'source_id': sid, 'role': role, 'category': chosen['name'],
        'source_unit_to_m': factor, 'original_obb_units': dims.tolist(), 'target_longest_m': longest,
        'metric_obb_m': (dims * factor).tolist(), 'size_confidence': confidence,
        'size_evidence': evidence, 'classification_evidence': chosen['classification_evidence'],
        'effective_density_kg_m3': float(chosen['density_kg_m3']), 'measured_mass_kg': mass,
        'fixed_seed': seed, 'size_depends_on_old_normalization': False,
        'size_depends_on_gripper_or_container_fit': False, 'source_metric_convention_preserved': source in METRIC_SOURCES,
        'semantic_truth_verified': False, 'physical_success_inferred': False,
        'implementation': binding(__file__), 'category_configuration': binding(CONFIG)}


def metric_policy(decision, evidence_ref):
    """Retained metric_variants_v2 policy after raw geometry is already scaled."""
    role = decision['role']
    size = {'mode': 'source_metric', 'source_unit_to_m': decision['source_unit_to_m'],
        'evidence': [evidence_ref],
        'rationale': 'Bound explicit raw-to-metre decision; engineering versus source confidence is retained in raw_size_decision.'}
    if decision['size_confidence'].startswith('engineering_'):
        size = {'mode': 'inferred_category', 'base_longest_m': decision['target_longest_m'],
            'plausible_longest_m': decision['size_evidence']['plausible_longest_m'],
            'evidence': [evidence_ref], 'rationale': decision['size_evidence']['rationale']}
    mass = {'mode': 'effective_density', 'density_kg_m3': decision['effective_density_kg_m3'],
        'evidence': [evidence_ref], 'rationale': 'Explicit category effective density over actual compound collision volumes; not a measured material density.'}
    if decision['measured_mass_kg'] is not None:
        mass = {'mode': 'measured_base_mass', 'base_mass_kg': decision['measured_mass_kg'],
            'evidence': [evidence_ref], 'rationale': 'Product metadata mass for the same uniform geometry-dimension match.'}
    return {'schema': 'bounded_metric_variants_v2', 'source_id': decision['source_id'],
        'semantic_role': role, 'semantic_category': decision['category'],
        'source_object_role': 'furniture' if role == 'background' else 'household_movable',
        'role_rationale': 'Retained name/category hierarchy with explicit source scale; no grasp success filtering.',
        'size': size, 'mass': mass, 'scale_factors': [1.], 'variant_ratio_bounds': [.9, 1.1],
        'raw_size_decision': evidence_ref, 'size_confidence': decision['size_confidence']}
