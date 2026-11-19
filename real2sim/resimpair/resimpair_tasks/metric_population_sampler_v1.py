"""Finite 3..8-object requests from a semantically admitted metric snapshot.

Source IDs, cached metric variants and partitions stay fixed. Asset, container,
layout and nuisance selection uses a recorded seed and geometric capacity;
the agent does not choose individual grasps or intervene in a scene program.
Failures remain in the sampling denominator; no physical outcome is consulted.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import time

import numpy as np

from resimpair_grasp.common import binding, digest, read, resolve, write
from resimpair_pickplace.requested_tasks import Rejected, check_separation, table_fit, compile_scene
from . import metric_task_factory_v2 as factory

VERSION = 'metric_population_multitask_sampler_v1'


def slots(radii, capacity, gap=.025, margin=.012):
    """Same size-only center/ring rule for every container, bounded to 8 roles."""
    r = np.asarray(radii, float)
    if (r.ndim != 1 or not 1 <= len(r) <= 8 or not np.isfinite(r).all() or np.any(r <= 0)
            or not np.isfinite([capacity, gap, margin]).all() or gap < .015 or margin < .008):
        raise ValueError('bounded_payloads_and_original_clearance_margins_required')
    n = len(r)
    if n == 1:
        p = np.zeros((1, 2))
    elif n == 2:
        p = np.array([[-r[1] - gap / 2, 0], [r[0] + gap / 2, 0]])
    else:
        # 7 and 8 objects use a center plus 6/7 ring. This conservative
        # construction is independent of container category and source ID.
        ring = n if n < 7 else n - 1
        radius = (2 * r.max() + gap) / (2 * math.sin(math.pi / ring))
        if n >= 7:
            radius = max(radius, 2 * r.max() + gap)
        p = np.array([[math.cos(a), math.sin(a)] for a in np.arange(ring) * 2 * math.pi / ring]) * radius
        if n >= 7:
            p = np.vstack(([0., 0.], p))
    remaining = capacity - np.linalg.norm(p, axis=1) - r
    gaps = [np.linalg.norm(p[i] - p[j]) - r[i] - r[j] for i in range(n) for j in range(i)]
    return {'passed': bool(np.min(remaining) >= margin - 1e-9 and all(g >= gap - 1e-9 for g in gaps)),
            'offsets': np.round(p, 7).tolist(), 'radii_m': r.tolist(), 'cavity_radius_m': capacity,
            'minimum_boundary_margin_m': float(remaining.min()),
            'minimum_inter_payload_gap_m': float(min(gaps)) if gaps else None,
            'required_boundary_margin_m': margin, 'required_gap_m': gap,
            'model': 'center_balanced_line_or_ring; no per-container hand-written slots',
            'physics_success': 'unmeasured'}


def admitted_pool(bundle):
    b = read(bundle)
    gate = b.get('source_semantics_checked_against_current_policy', {})
    if gate.get('all_cache_entries_passed') is not True:
        raise ValueError('current_semantic_and_scale_gate_required_before_automatic_sampling')
    resolve(gate['source_state_index']); resolve(gate['gate'])
    entries = read(resolve(b['native_cache_index']))['entries']
    for entry in entries:
        if entry.get('semantic_scale_admission', {}).get('passed') is not True:
            raise ValueError('per_variant_semantic_scale_admission_required')
        resolve(entry['semantic_scale_admission']['current_policy'])
    by_source = defaultdict(list)
    for e in entries:
        d = read(resolve(e['asset']))
        by_source[d['source_id']].append((e, d))
    bins = []
    for ref in read(resolve(b['containers']))['assets']:
        d = read(resolve(ref))
        if d['roles'].get('stationary_container'):
            bins.append((ref, d))
    return b, entries, by_source, bins


def propose(identity, group, bins, seed, static):
    rng = np.random.default_rng(seed)
    partition = group[0][1]['partition']
    radii = [float(np.linalg.norm(d['half_size'][:2])) for _, d in group]
    choices = []
    for ref, d in bins:
        if d['partition'] != partition:
            continue
        packing = slots(radii, d['cavity']['flat_floor_radius_m'])
        if packing['passed']:
            choices.append((ref, d, packing))
    if not choices:
        raise Rejected('no_same_partition_container_capacity_for_group', radii_m=radii,
                       maximum_available_radius_m=max((d['cavity']['flat_floor_radius_m']
                           for _, d in bins if d['partition'] == partition), default=None))
    choices.sort(key=lambda x: digest([seed, x[1]['source_id'], x[0]['sha256']]))
    chosen = choices[0]
    second = choices[1] if len(choices) > 1 else chosen
    selected = [chosen, second]
    containers, occupied = [], []
    for i, (_, d, _) in enumerate(selected):
        radius = float(np.linalg.norm(np.asarray(d['dimensions_m'])[:2] / 2))
        xy = np.array([.10 + rng.uniform(-.015, .015), (-1 if i == 0 else 1) * (.40 + rng.uniform(-.02, .02))])
        table_fit(xy, radius); check_separation(xy, radius, occupied)
        occupied.append(('container_' + str(i), xy, radius))
        containers.append({'id': 'container_' + str(i), 'source_id': d['source_id'],
                           'xy': np.round(xy, 7).tolist(), 'yaw': float(rng.uniform(-.4, .4))})
    objects = []
    for i, ((entry, d), radius) in enumerate(zip(group, radii)):
        accepted = None
        for proposal in range(128):
            xy = np.array([rng.uniform(-.36, -.14), rng.uniform(-.61, .61)])
            try:
                table_fit(xy, radius); check_separation(xy, radius, occupied)
                accepted = xy; break
            except Rejected:
                continue
        if accepted is None:
            raise Rejected('bounded_initial_object_layout_exhausted', object_index=i, attempts=128)
        oid = 'object_' + str(i)
        occupied.append((oid, accepted, radius))
        objects.append({'id': oid, 'source_id': d['source_id'], 'xy': np.round(accepted, 7).tolist(),
                        'yaw': float(rng.uniform(-math.pi, math.pi))})
    def placements(indices, c, offsets):
        return [{'object': objects[i]['id'], 'target': {'kind': 'container', 'container': containers[c]['id'],
                  'offset_xy_m': offset}} for i, offset in zip(indices, offsets)]
    all_ids = list(range(len(objects))); subset = all_ids[:max(2, len(objects) // 2)]
    collect = placements(all_ids, 0, chosen[2]['offsets'])
    subpacking = slots([radii[i] for i in subset], chosen[1]['cavity']['flat_floor_radius_m'])
    distributed = []; group_packing = []
    for c in range(2):
        ids = all_ids[c::2]
        packing = slots([radii[i] for i in ids], selected[c][1]['cavity']['flat_floor_radius_m'])
        if not packing['passed']:
            raise Rejected('subgroup_capacity_inconsistent')
        distributed += placements(ids, c, packing['offsets']); group_packing.append(packing)
    spec = {'id': identity, 'objects': objects, 'containers': containers,
        'variant_choices': {d['source_id']: e['variant_id'] for e, d in group},
        'distractors': {'count': int(rng.integers(3, 6)), 'max_dimension_m': .18,
                        'catalog': static[0], 'source_role_index': static[1]},
        'tasks': [{'id': 'select_one', 'placements': placements([0], 0, [[0., 0.]])},
                  {'id': 'collect_all', 'placements': collect},
                  {'id': 'collect_subset', 'placements': placements(subset, 0, subpacking['offsets'])},
                  {'id': 'distribute_groups', 'placements': distributed}]}
    return spec, {'collection': chosen[2], 'subset': subpacking, 'distribution': group_packing,
                  'selected_container_variants': [ref for ref, _, _ in selected]}


def build(bundle, output, *, scenes, seed, minimum=3, maximum=8, proposals=16):
    if not (1 <= scenes <= 1000 and 3 <= minimum <= maximum <= 8 and 1 <= proposals <= 16):
        raise ValueError('bounded_scene_and_object_and_proposal_counts_required')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('preserve_existing_sampling_failures')
    b, entries, source_pool, bins = admitted_pool(bundle)
    output.mkdir(parents=True)
    static = factory.static_source_catalog(resolve(b['static_catalog']), output / 'preflight_static')
    start = time.monotonic(); rows = []; accepted = []
    write(output / 'protocol.json', {'schema': VERSION, 'implementation': binding(__file__),
        'bundle': binding(bundle), 'seed': seed, 'requested_scenes': scenes,
        'object_count_range': [minimum, maximum], 'maximum_geometry_proposals_per_scene': proposals,
        'physical_attempts': 0, 'API_calls': 0, 'runtime_rescaling': False,
        'requested_tasks_per_accepted_scene': 4, 'source_selection_uses_physical_success': False})
    for i in range(scenes):
        directory = output / 'sampling' / f'{i:04d}'
        n = int(np.random.default_rng(seed + i).integers(minimum, maximum + 1))
        partitions = sorted({d['partition'] for xs in source_pool.values() for _, d in xs})
        partition = partitions[i % len(partitions)] if partitions else 'unavailable'
        pool = {s: xs for s, xs in source_pool.items() if xs[0][1]['partition'] == partition}
        attempts = []; chosen = None
        for trial in range(proposals):
            trial_seed = seed + i * 1009 + trial * 31
            trial_dir = directory / f'proposal_{trial:02d}'
            if len(pool) < n:
                attempts.append({'reason': 'not_enough_distinct_admitted_sources_in_partition',
                                 'requested': n, 'available': len(pool)})
                break
            sources = sorted(pool, key=lambda s: digest([trial_seed, s]))[:n]
            group = [min(pool[s], key=lambda x: digest([trial_seed, s, x[0]['variant_id']])) for s in sources]
            row = {'proposal': trial, 'seed': trial_seed, 'source_ids': sources,
                   'variant_ids': [e['variant_id'] for e, _ in group], 'passed': False}
            try:
                spec, allocation = propose(f'metric_group_{i:03d}', group, bins, trial_seed, static)
                write(trial_dir / 'request.json', spec); write(trial_dir / 'capacity.json', allocation)
                compile_spec = {k: v for k, v in spec.items() if k != 'variant_choices'}
                # Reuse exactly the same geometry compiler and clutter seed
                # that the frozen factory will later use.
                compile_scene(compile_spec, [e for e, _ in group],
                              [ref for ref, _ in bins], seed + len(accepted))
                chosen = spec; row['passed'] = True
                row['request'] = binding(trial_dir / 'request.json')
            except (Rejected, ValueError) as error:
                row['reason'] = getattr(error, 'code', type(error).__name__)
                row['details'] = getattr(error, 'details', {'message': str(error)})
            attempts.append(row)
            if chosen is not None:
                accepted.append(chosen); break
        write(directory / 'attempts.json', {'attempts': attempts})
        rows.append({'scene_index': i, 'object_count': n, 'partition': partition,
                     'geometry_accepted': chosen is not None, 'attempts': binding(directory / 'attempts.json')})
        print({'scene': i, 'objects': n, 'geometry_accepted': chosen is not None}, flush=True)
    write(output / 'sampling.json', {'scenes': rows, 'all_requested_scenes_retained': True})
    write(output / 'requests.json', {'schema': factory.VERSION, 'scenes': accepted})
    result = factory.compile_requests(bundle, output / 'requests.json', output / 'factory', seed) if accepted else None
    summary = {'requested_scenes': scenes, 'accepted_geometry_scenes': len(accepted),
               'compiled_factory': result, 'seconds': time.monotonic() - start,
               'physical_attempts': 0, 'API_calls': 0}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scenes', type=int, required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--min-objects', type=int, default=3); p.add_argument('--max-objects', type=int, default=8)
    p.add_argument('--proposals', type=int, default=16)
    a = p.parse_args()
    print(build(a.bundle, a.output, scenes=a.scenes, seed=a.seed,
                minimum=a.min_objects, maximum=a.max_objects, proposals=a.proposals))
