"""Canonical pick-place asset catalog: curated categories, texture colors, native grasps.

Manipulable geometry comes from the raw-asset rollout CPU stage. An asset that
failed only the 3 mm surface-fidelity gate is admitted under the v2 limit
(p95 <= 10 mm) through a derived descriptor; the original record is untouched.
Grasps are top-down antipodal proposals filtered by the retained native
open-hand clearance and finger closing-sweep checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import colorsys
import json
from pathlib import Path
import time
import traceback

import numpy as np

from resimpair_grasp.common import ROOT, binding, digest, read, resolve, write

VERSION = 'pp2_canonical_catalog_v1'
ROLLOUT = ROOT / 'runs/raw_asset_rollout_20261031_v1'
CONTAINER_BUNDLE = ROOT / 'runs/raw_candidate_consumer_20261031_v3/snapshot_02/reload_verified_v2/filtered/BUNDLE.json'
CURATION = ROOT / 'configs/pp2/canonical_assets_v1.json'
SURFACE_LIMIT_M = .010
OPENING_RANGE_M = (.012, .077)
MAX_CANDIDATES = 32
GRASP_FILTER = 'pp2_native_grasp_filter_v1'


def rollout_folders():
    rows = read(ROLLOUT / 'population.json')['sources']
    return {r['source_id']: Path(r['job']['path']).parent for r in rows}


def admit(source_id, folder, output):
    """Return (descriptor binding, admission record) or raise ValueError(reason)."""
    terminal = folder / 'cpu_terminal.json'
    if not terminal.exists():
        raise ValueError('no_cpu_terminal')
    status = read(terminal)['status']
    if status not in ('native_ready', 'native_qualification_failed'):
        raise ValueError('cpu_status_' + status)
    entry = read(folder / 'native_entry.json')
    if entry.get('support_status') != 'pass' or entry.get('settle_status') != 'pass':
        raise ValueError('native_support_or_settle_failed')
    surface = entry.get('surface', {})
    deviation = max(surface.get('visual_to_collision_p95_m') or 0., surface.get('union_external_p95_m') or 0.)
    if deviation > SURFACE_LIMIT_M:
        raise ValueError('surface_p95_above_v2_limit')
    record = {'original_status': status, 'surface_p95_m': deviation, 'surface_limit_m': SURFACE_LIMIT_M,
              'original_entry': binding(folder / 'native_entry.json')}
    if entry['geometry_qualified']:
        return entry['asset'], {**record, 'derived_descriptor': False}
    descriptor = read(resolve(entry['asset']))
    descriptor['roles'] = {**descriptor['roles'], 'rigid_manipulation_candidate': True}
    descriptor['pp2_admission'] = {**record, 'parent_descriptor': entry['asset'],
        'reason': 'only the 3 mm surface-fidelity gate failed; v2 admits p95 <= 10 mm'}
    path = output / 'derived' / digest([source_id, entry['asset']])[:24] / 'descriptor.json'
    write(path, descriptor)
    return binding(path), {**record, 'derived_descriptor': True}


def surface_samples(visual_path, count=6000, seed=0):
    """RGB samples on the textured visual surface, area weighted."""
    import trimesh
    from trimesh.visual.color import uv_to_color
    scene = trimesh.load(visual_path, force='scene', process=False)
    geometries = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    areas = np.array([g.area for g in geometries])
    rng = np.random.default_rng(seed); colors = []
    for geometry, area in zip(geometries, areas):
        n = max(50, int(count * area / areas.sum()))
        points, faces = trimesh.sample.sample_surface(geometry, n, seed=int(rng.integers(1 << 31)))
        visual = geometry.visual
        material = getattr(visual, 'material', None)
        image = getattr(material, 'baseColorTexture', None) or getattr(material, 'image', None)
        if getattr(visual, 'uv', None) is not None and image is not None:
            bary = trimesh.triangles.points_to_barycentric(geometry.triangles[faces], points)
            uv = (visual.uv[geometry.faces[faces]] * bary[:, :, None]).sum(1)
            # Texture only: some converted YCB materials carry a 0.4 base factor
            # that darkens rendering but is not the object's real color.
            rgb = uv_to_color(uv % 1., image)[:, :3] / 255.
        elif getattr(visual, 'kind', None) in ('vertex', 'face'):
            vertex = np.asarray(visual.vertex_colors, float)[:, :3] / 255.
            rgb = vertex[geometry.faces[faces]].mean(1)
        else:
            main = np.atleast_1d(getattr(material, 'main_color', None) if material is not None else None)
            rgb = np.tile(main[:3].astype(float) / 255. if main.size >= 3 else np.full(3, .5), (n, 1))
        colors.append(rgb)
    return np.clip(np.vstack(colors), 0, 1)


def color_name(rgb):
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in rgb])
    chromatic = (hsv[:, 1] > .28) & (hsv[:, 2] > .22)
    if chromatic.mean() >= .35:
        h = hsv[chromatic, 0] * 2 * np.pi
        hue = (np.degrees(np.arctan2(np.sin(h).mean(), np.cos(h).mean())) + 360) % 360
        value = float(np.median(hsv[chromatic, 2])); saturation = float(np.median(hsv[chromatic, 1]))
        if hue < 15 or hue >= 340: name = 'red'
        elif hue < 42: name = 'orange'
        elif hue < 70: name = 'yellow'
        elif hue < 165: name = 'green'
        elif hue < 255: name = 'blue'
        elif hue < 290: name = 'purple'
        else: name = 'pink'
        if name == 'orange' and value < .3: name = 'brown'
        elif name in ('orange', 'yellow') and saturation < .45 and value >= .5: name = 'beige'
        return name, {'hue_deg': hue, 'chromatic_fraction': float(chromatic.mean()), 'value': value,
                      'saturation': saturation}
    value = float(np.median(hsv[:, 2]))
    name = 'white' if value > .72 else 'gray' if value > .32 else 'black'
    return name, {'chromatic_fraction': float(chromatic.mean()), 'value': value}


def native_grasps(descriptor_binding, output):
    import trimesh
    from resimpair_grasp import grasps
    from resimpair_grasp.cartesian_clearance import VerticalClearance
    from resimpair_pickplace.closure_quality import ClosingSweep
    proposals = read(resolve(grasps.generate(descriptor_binding, output / 'proposals')))
    descriptor = read(resolve(descriptor_binding))
    mesh = trimesh.util.concatenate([trimesh.load(resolve(b), force='mesh') for b in descriptor['collisions']])
    clearance, closure = VerticalClearance(mesh), ClosingSweep(mesh)
    kept, rejected = [], {}
    for candidate in proposals['candidates']:
        pose = np.asarray(candidate['tcp_in_asset'], float); width = float(candidate['width_m'])
        reason = None
        if not OPENING_RANGE_M[0] <= width <= OPENING_RANGE_M[1]:
            reason = 'opening_range'
        else:
            ok, gap, why = clearance.check(pose, float(mesh.bounds[0, 2]), .002)
            if not ok:
                reason = str(why)
            else:
                sweep = closure.measure(pose, width)
                if not sweep['contact_pair_reachable']:
                    reason = 'closing_sweep_contact_mismatch'
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1; continue
        kept.append({**candidate, 'id': 'a%03d' % len(kept), 'closing_sweep': sweep,
                     'open_hand_clearance_m': gap, 'native_checks': 'open_hand_clearance+closing_sweep'})
        if len(kept) == MAX_CANDIDATES:
            break
    signature = {'descriptor': descriptor_binding, 'proposals': binding(Path(output / 'proposals') / (proposals['key'] + '.json')),
                 'gripper': grasps.GRIPPER, 'filter': GRASP_FILTER, 'opening_range_m': OPENING_RANGE_M,
                 'max_candidates': MAX_CANDIDATES}
    path = output / 'grasp_cache.json'
    write(path, {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature, 'key': digest(signature),
                 'candidates': kept, 'rejected': rejected, 'physical_qualification': 'unknown',
                 'scope': 'top-down antipodal proposals with native clearance and closing checks; execution still required'})
    return binding(path), len(kept), rejected


def build_manipulable(args):
    source_id, category, folder, output = args
    start = time.monotonic()
    try:
        out = output / 'manipulables' / digest(source_id)[:24]
        descriptor_binding, admission = admit(source_id, folder, out)
        descriptor = read(resolve(descriptor_binding))
        rgb = surface_samples(resolve(descriptor['visual']))
        color, color_evidence = color_name(rgb)
        cache, count, rejected = native_grasps(descriptor_binding, out)
        return {'source_id': source_id, 'role': 'manipulable', 'category': category, 'color': color,
                'color_evidence': color_evidence, 'partition': descriptor['partition'],
                'descriptor': descriptor_binding, 'cache': cache, 'grasps': count, 'grasp_rejections': rejected,
                'dimensions_m': descriptor['dimensions_m'], 'mass_kg': descriptor['mass_kg'],
                'admission': admission, 'seconds': time.monotonic() - start}
    except Exception as error:
        return {'source_id': source_id, 'role': 'manipulable', 'category': category, 'rejected': str(error),
                'trace': traceback.format_exc(limit=3), 'seconds': time.monotonic() - start}


def receptacles(curated):
    bundle = read(CONTAINER_BUNDLE)
    by_source = {}
    for b in read(resolve(bundle['containers']))['assets']:
        d = read(resolve(b)); by_source.setdefault(d['source_id'], (b, d))
    rows = []
    for source_id, (category, relation) in curated.items():
        if source_id not in by_source:
            rows.append({'source_id': source_id, 'role': 'receptacle', 'rejected': 'not_in_container_bundle'}); continue
        b, d = by_source[source_id]
        color, evidence = color_name(surface_samples(resolve(d['visual'])))
        rows.append({'source_id': source_id, 'role': 'receptacle', 'category': category, 'relation': relation,
                     'color': color, 'color_evidence': evidence, 'partition': d['partition'], 'descriptor': b,
                     'dimensions_m': d['dimensions_m'], 'floor_radius_m': d['cavity']['flat_floor_radius_m']})
    return rows


def build(output, workers=16):
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    curation = read(CURATION); folders = rollout_folders()
    jobs, missing = [], []
    for source_id, category in curation['manipulables'].items():
        if source_id in folders:
            jobs.append((source_id, category, folders[source_id], output))
        else:
            missing.append({'source_id': source_id, 'role': 'manipulable', 'rejected': 'not_in_rollout'})
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(build_manipulable, jobs))
    rows += missing + receptacles(curation['receptacles'])
    usable = [r for r in rows if 'rejected' not in r and (r['role'] == 'receptacle' or r['grasps'] > 0)]
    catalog = {'schema': VERSION, 'created_unix': time.time(), 'curation': binding(CURATION),
               'implementation': binding(Path(__file__)), 'rollout': binding(ROLLOUT / 'population.json'),
               'container_bundle': binding(CONTAINER_BUNDLE),
               'rules': {'surface_limit_m': SURFACE_LIMIT_M, 'opening_range_m': OPENING_RANGE_M,
                         'grasp_approach': 'top-down', 'max_candidates': MAX_CANDIDATES},
               'assets': usable, 'excluded': [r for r in rows if r not in usable]}
    write(output / 'catalog.json', catalog)
    return {'usable_manipulables': sum(r['role'] == 'manipulable' for r in usable),
            'usable_receptacles': sum(r['role'] == 'receptacle' for r in usable),
            'excluded': len(catalog['excluded'])}



POOL_BUNDLE = ROOT / 'runs/raw_candidate_consumer_20261031_v3/snapshot_02/reload_verified_v2/filtered/BUNDLE.json'
IN_NOUNS = ('bowl', 'basket', 'box', 'pan', 'pot', 'bin', 'bucket', 'crate', 'dish', 'cup', 'mug', 'jar', 'tub', 'container')
LONGEST_RANGE_M = (.04, .22)
# Curated noun policy for labelled pool assets: merge near-synonyms, drop items unsuitable for household tasks.
NOUN_MAP = {'chalice': 'goblet', 'champagne flute': 'wine glass', 'pet bowl': 'bowl', 'dog bowl': 'bowl',
            'wooden bowl': 'bowl', 'singing bowl': 'bowl', 'cooking pot': 'pot', 'plant pot': 'flower pot',
            'frying pan': 'pan', 'wood crate': 'crate', 'wooden crate': 'crate', 'cardboard tray': 'tray',
            'teacup': 'cup', 'lego brick': 'toy block', 'word block': 'toy block', 'toy hammer': 'hammer'}
DENY_NOUNS = {'warhammer', 'grenade', 'antlers', 'leaves', 'mineral specimen', 'banana peel', 'telegraph', 'skewer',
              'screw', 'pipe', 'canister'}


def noun_of(label):
    noun = label['noun'].strip().lower()
    return NOUN_MAP.get(noun, noun)


def label_ok(label):
    return bool(label and 'unknown' not in label and label['recognizable'] and label['plausible_real_size']
                and label['everyday_tabletop_item'] and label['single_clean_object'] and label['noun'].strip()
                and noun_of(label) not in DENY_NOUNS)


def pool_manipulable(args):
    """Admit one labelled pool asset: merged cuDex + antipodal native-checked grasps."""
    entry, label, output = args
    try:
        descriptor = read(resolve(entry['asset']))
        if not LONGEST_RANGE_M[0] <= max(descriptor['dimensions_m']) <= LONGEST_RANGE_M[1]:
            raise ValueError('longest_side_outside_tabletop_range')
        out = output / 'pool' / digest(entry['variant_id'])[:24]
        cudex = read(resolve(entry['cache']))['candidates']
        anti_binding, count, rejected = native_grasps(entry['asset'], out)
        anti = read(resolve(anti_binding))['candidates']
        from resimpair_grasp import grasps
        merged = [{**c, 'id': f'c{i:03d}', 'source': 'cudex'} for i, c in enumerate(cudex)] + \
                 [{**c, 'id': f'a{i:03d}', 'source': 'antipodal'} for i, c in enumerate(anti)]
        signature = {'descriptor': entry['asset'], 'gripper': grasps.GRIPPER, 'parents': [entry['cache'], anti_binding],
                     'merge': 'cudex_first_then_antipodal'}
        write(out / 'merged_cache.json', {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature,
              'key': digest(signature), 'candidates': merged, 'physical_qualification': 'unknown'})
        noun = noun_of(label)
        return {'source_id': entry['source_id'], 'role': 'manipulable', 'category': noun, 'color': label['color'],
                'color_evidence': {'source': 'vlm_label'}, 'partition': entry['partition'], 'descriptor': entry['asset'],
                'cache': binding(out / 'merged_cache.json'), 'grasps': len(merged), 'grasp_sources': [len(cudex), len(anti)],
                'dimensions_m': descriptor['dimensions_m'], 'mass_kg': descriptor['mass_kg'],
                'admission': {'derived_descriptor': False, 'pool': 'reload_verified_682', 'label': label}}
    except Exception as error:
        return {'source_id': entry['source_id'], 'role': 'manipulable', 'rejected': str(error)[:200]}


def build_v3(base_catalog, labels_path, output, workers=32):
    """v2 curated catalog + labelled reload-verified pool manipulables and containers."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(base_catalog); labels = read(labels_path)['labels']; bundle = read(POOL_BUNDLE)
    known = {a['source_id'] for a in base['assets']}
    jobs, skipped = [], []
    for entry in read(resolve(bundle['native_cache_index']))['entries']:
        label = labels.get('m_' + entry['variant_id'][:16])
        if entry['source_id'] in known:
            continue
        # Graspable open objects (cups, glasses) stay manipulables; receptacles come from the container bundle.
        if not label_ok(label):
            skipped.append({'source_id': entry['source_id'], 'label': label}); continue
        jobs.append((entry, label, output))
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(pool_manipulable, jobs))
    containers = []
    for ref in read(resolve(bundle['containers']))['assets']:
        d = read(resolve(ref)); label = labels.get('c_' + d['variant_id'][:16])
        if d['source_id'] in known or not label_ok(label) or not label['open_container']:
            continue
        noun = noun_of(label)
        containers.append({'source_id': d['source_id'], 'role': 'receptacle', 'category': noun,
                           'relation': 'in' if any(w in noun for w in IN_NOUNS) else 'on', 'color': label['color'],
                           'color_evidence': {'source': 'vlm_label'}, 'partition': d['partition'], 'descriptor': ref,
                           'dimensions_m': d['dimensions_m'], 'floor_radius_m': d['cavity']['flat_floor_radius_m'],
                           'admission': {'pool': 'reload_verified_containers', 'label': label}})
    usable = [r for r in rows if 'rejected' not in r and r['grasps'] > 0]
    catalog = {'schema': VERSION + '_v3', 'created_unix': time.time(), 'base_catalog': binding(Path(base_catalog)),
               'labels': binding(Path(labels_path)), 'pool_bundle': binding(POOL_BUNDLE),
               'implementation': binding(Path(__file__)),
               'rules': {**base['rules'], 'pool_label_filter': 'recognizable & plausible_real_size & everyday & single_clean',
                         'pool_longest_range_m': LONGEST_RANGE_M},
               'assets': base['assets'] + usable + containers,
               'excluded': base['excluded'] + [r for r in rows if r not in usable] + skipped}
    write(output / 'catalog.json', catalog)
    return {'base': len(base['assets']), 'pool_manipulables': len(usable), 'pool_containers': len(containers),
            'label_rejected': len(skipped), 'pool_failed': len(rows) - len(usable)}



CONTAINER_NOUNS = {'paper bowls': 'bowl', 'dog bowl': 'bowl', 'pet bowl': 'bowl', 'plant saucer': 'saucer',
                   'square tray': 'tray', 'cake pan': 'pan', 'planter': 'flower pot', 'plant pot': 'flower pot',
                   'tiered tray': 'tray', 'wooden bowl': 'bowl', 'decorative plate': 'plate'}
DENY_CONTAINERS = {'vase', 'lid', 'cup', 'mug', 'chamber pot', 'palette', 'coaster'}


def luminance(args):
    key, descriptor_ref = args
    try:
        d = read(resolve(descriptor_ref))
        rgb = surface_samples(resolve(d['visual']), count=3000)
        return key, float(np.median(rgb @ np.array([.2126, .7152, .0722])))
    except Exception:
        return key, None


def build_v4(v3_catalog, container_lists, container_labels, output, workers=32):
    """v3 + metric GSO/YCB/ABO containers (labelled) + per-asset texture luminance for contrast rules."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(v3_catalog); assets = [dict(a) for a in base['assets']]
    known = {a['source_id'] for a in assets}
    labels = {}
    for path in container_labels:
        labels.update(read(path)['labels'])
    added, rejected = [], []
    for path in container_lists:
        for key, ref in read(path):
            d = read(resolve(ref)); label = labels.get(key)
            if d['source_id'] in known or not label or 'unknown' in label or not label['open_container']:
                rejected.append({'source_id': d['source_id'], 'reason': 'known_or_unlabelled_or_closed'}); continue
            noun = CONTAINER_NOUNS.get(label['noun'].strip().lower(), label['noun'].strip().lower())
            if noun in DENY_CONTAINERS:
                rejected.append({'source_id': d['source_id'], 'reason': 'noun_not_a_receptacle:' + noun}); continue
            added.append({'source_id': d['source_id'], 'role': 'receptacle', 'category': noun,
                          'relation': 'in' if any(w in noun for w in IN_NOUNS) else 'on', 'color': label['color'],
                          'color_evidence': {'source': 'vlm_label'}, 'partition': d['partition'], 'descriptor': ref,
                          'dimensions_m': d['dimensions_m'], 'floor_radius_m': d['cavity']['flat_floor_radius_m'],
                          'admission': {'pool': 'pp2_metric_containers_v2', 'label': label,
                                        'size': 'source metric convention, factor 1'}})
            known.add(d['source_id'])
    assets += added
    with ProcessPoolExecutor(workers) as pool:
        lum = dict(pool.map(luminance, [(i, a['descriptor']) for i, a in enumerate(assets)]))
    for i, a in enumerate(assets):
        a['luminance'] = lum[i]
    catalog = {**base, 'schema': VERSION + '_v4', 'created_unix': time.time(), 'v3_catalog': binding(Path(v3_catalog)),
               'container_lists': [binding(Path(p)) for p in container_lists],
               'container_labels': [binding(Path(p)) for p in container_labels], 'assets': assets,
               'excluded': base['excluded'] + rejected, 'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'assets': len(assets), 'added_containers': len(added), 'rejected': len(rejected),
            'luminance_missing': sum(v is None for v in lum.values())}



def build_v5(v4_catalog, bench_rows, output, max_lever_m=.05):
    """Measured grasp ordering shared by all assets (pp2 grasp bench).

    Order: bench successes, then untested candidates with lever <= max_lever_m, both by lever;
    bench failures and long-lever candidates go last. Assets with no successful bench grasp
    become distractor-only. Lever = distance from asset COM to the finger closing axis.
    """
    from .grasp_bench import features
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(v4_catalog); bench = read(bench_rows)
    outcome = {}
    for r in bench:
        outcome.setdefault(r['source_id'], {})[r['candidate']] = r['outcome']
    assets, demoted = [], []
    for a in base['assets']:
        if a['role'] != 'manipulable':
            assets.append(a); continue
        measured = outcome.get(a['source_id'], {})
        reached = {c: o for c, o in measured.items() if o != 'no_reachable_collision_checked_grasp'}
        if reached and not any(o == 'success' for o in reached.values()):
            demoted.append({**a, 'role': 'distractor_only', 'bench': reached}); continue
        rows = features(a)
        def key(row):
            o = measured.get(row['candidate']['id'])
            tier = 0 if o == 'success' else 2 if (o and o != 'no_reachable_collision_checked_grasp') else \
                1 if row['lever_m'] <= max_lever_m else 3
            return (tier, row['lever_m'])
        ordered = sorted(rows, key=key)
        candidates = [{**r['candidate'], 'lever_m': r['lever_m'], 'bench_outcome': measured.get(r['candidate']['id'])}
                      for r in ordered]
        out = output / 'caches' / digest(a['source_id'])[:24]
        parent = read(resolve(a['cache']))
        signature = {**parent['signature'], 'parent_cache': a['cache'], 'ordering': 'pp2_measured_lever_v1',
                     'bench': binding(Path(bench_rows)), 'max_lever_m': max_lever_m}
        write(out / 'grasp_cache.json', {'schema': 'resimpair_mesh_grasp_cache_v1', 'signature': signature,
              'key': digest(signature), 'candidates': candidates, 'physical_qualification': 'bench_ordered'})
        assets.append({**a, 'cache': binding(out / 'grasp_cache.json'),
                       'bench': {'tested': len(measured), 'successes': sum(o == 'success' for o in measured.values()),
                                 'first_candidate_lever_m': candidates[0]['lever_m'] if candidates else None}})
    catalog = {**base, 'schema': VERSION + '_v5', 'created_unix': time.time(), 'v4_catalog': binding(Path(v4_catalog)),
               'grasp_bench': binding(Path(bench_rows)), 'assets': assets, 'distractor_only': demoted,
               'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'manipulables': sum(a['role'] == 'manipulable' for a in assets), 'demoted': len(demoted),
            'receptacles': sum(a['role'] == 'receptacle' for a in assets)}



def build_v7(base_catalog, output, low_lever_m=.03, fallback_lever_m=.05, min_keep=2):
    """Truncate each bench-ordered cache to low-lever grasps, so planners choosing by joint travel
    cannot pick a long-lever grasp. The asset's 'grasps' count is kept, so scene sampling is unchanged."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(base_catalog); assets = []; stats = Counter()
    bad = lambda c: c.get('bench_outcome') not in (None, 'success', 'no_reachable_collision_checked_grasp',
                                                   'no_ready_operation_has_complete_path')
    for a in base['assets']:
        if a['role'] != 'manipulable':
            assets.append(a); continue
        parent = read(resolve(a['cache'])); cands = [c for c in parent['candidates'] if not bad(c)]
        keep = [c for c in cands if c['lever_m'] <= low_lever_m]; tier = 'low'
        if len(keep) < min_keep:
            keep = keep + [c for c in cands if c not in keep and c.get('bench_outcome') == 'success'
                           and c['lever_m'] <= fallback_lever_m]; tier = 'fallback_success'
        if len(keep) < min_keep:
            keep = keep + [c for c in sorted(cands, key=lambda c: c['lever_m']) if c not in keep][:4 - len(keep)]
            tier = 'lowest_available'
        stats[tier] += 1
        out = output / 'caches' / digest(a['source_id'])[:24]
        signature = {**parent['signature'], 'parent_cache': a['cache'], 'truncation': 'pp2_low_lever_v1',
                     'low_lever_m': low_lever_m, 'fallback_lever_m': fallback_lever_m}
        write(out / 'grasp_cache.json', {**parent, 'signature': signature, 'key': digest(signature), 'candidates': keep})
        assets.append({**a, 'cache': binding(out / 'grasp_cache.json'), 'grasps_kept': len(keep), 'grasp_tier': tier})
    catalog = {**base, 'schema': VERSION + '_v7', 'created_unix': time.time(), 'v6_catalog': binding(Path(base_catalog)),
               'assets': assets, 'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return dict(stats)



def build_v8(base_catalog, output, demote_lever_m=.04):
    """Demote assets that have no reliable short-lever grasp (measured slip concentrates there).

    Rule, shared by all assets: tier 'lowest_available' (no >=2 short-lever and no bench-success grasp), or tier
    'fallback_success' whose best kept grasp lever exceeds demote_lever_m, becomes distractor-only.
    """
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(base_catalog); assets, demoted = [], list(base.get('distractor_only', []))
    for a in base['assets']:
        if a['role'] == 'manipulable':
            best = min((c['lever_m'] for c in read(resolve(a['cache']))['candidates']), default=9.)
            if a['grasp_tier'] == 'lowest_available' or (a['grasp_tier'] == 'fallback_success' and best > demote_lever_m):
                demoted.append({**a, 'role': 'distractor_only', 'demoted_by': 'pp2_short_lever_rule_v1', 'best_lever_m': best})
                continue
        assets.append(a)
    catalog = {**base, 'schema': VERSION + '_v8', 'created_unix': time.time(), 'v7_catalog': binding(Path(base_catalog)),
               'assets': assets, 'distractor_only': demoted, 'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'manipulables': sum(a['role'] == 'manipulable' for a in assets), 'distractor_only': len(demoted)}



def black_material(visual_path, threshold=.05):
    """True when every material multiplies its color by a near-zero base factor (renders as solid black)."""
    import trimesh
    scene = trimesh.load(visual_path, force='scene', process=False)
    factors = []
    for geometry in scene.geometry.values():
        material = getattr(geometry.visual, 'material', None)
        factor = getattr(material, 'baseColorFactor', None)
        if factor is None:
            return False
        rgb = np.asarray(factor[:3], float) / (255. if np.max(factor) > 1 else 1.)
        factors.append(rgb.max())
    return bool(factors) and max(factors) < threshold


def build_v9(base_catalog, output):
    """Drop receptacles whose materials render solid black (unsupported transmission/volume glass materials)."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(base_catalog); assets, dropped = [], []
    for a in base['assets']:
        if a['role'] == 'receptacle' and black_material(resolve(read(resolve(a['descriptor']))['visual'])):
            dropped.append({'source_id': a['source_id'], 'reason': 'solid_black_material'}); continue
        assets.append(a)
    catalog = {**base, 'schema': VERSION + '_v9', 'created_unix': time.time(), 'v8_catalog': binding(Path(base_catalog)),
               'assets': assets, 'excluded': base['excluded'] + dropped, 'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'dropped_black_receptacles': len(dropped), 'receptacles': sum(a['role'] == 'receptacle' for a in assets),
            'dropped': [d['source_id'] for d in dropped]}


def build_v10(base_catalog, audit_path, output):
    """Drop assets the ray-tracing audit (resimpair_pp2.rt_audit) found dark under ray tracing with no repair."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base, audit = read(base_catalog), read(audit_path); dark = set(audit['dark'])
    if audit['catalog']['sha256'] != binding(Path(base_catalog))['sha256']:
        raise ValueError('rt_audit_bound_to_a_different_catalog')
    keep = lambda rows: [a for a in rows if a['source_id'] not in dark]
    dropped = [{'source_id': a['source_id'], 'reason': 'dark_under_ray_tracing_no_repair'}
               for a in base['assets'] + base.get('distractor_only', []) if a['source_id'] in dark]
    catalog = {**base, 'schema': VERSION + '_v10', 'created_unix': time.time(), 'v9_catalog': binding(Path(base_catalog)),
               'rt_audit': binding(Path(audit_path)), 'assets': keep(base['assets']),
               'distractor_only': keep(base.get('distractor_only', [])), 'excluded': base['excluded'] + dropped,
               'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'dropped': [d['source_id'] for d in dropped], 'manipulables': sum(a['role'] == 'manipulable' for a in catalog['assets']),
            'receptacles': sum(a['role'] == 'receptacle' for a in catalog['assets']), 'distractor_only': len(catalog['distractor_only'])}


PLANNING_FAILURES = ('release_aperture', 'guard_rejected', 'collision_checked_grasp', 'complete_path')


def receptacle_planning_failures(cohorts):
    """Per target receptacle: transfers attempted, and planning failures charged to the transfer that failed."""
    stats = {}
    for cohort in map(Path, cohorts):
        for p in (cohort / 'acceptance/cases').glob('*.json'):
            verdict = read(p); execution = cohort / 'executions' / verdict['case']
            try:
                scene, task = read(execution / 'effective_scene.json'), read(execution / 'effective_task.json')
                done = read(execution / 'rolling_execution_summary.json')['completed_operation_ids']
            except (FileNotFoundError, KeyError):
                continue
            supports = {s['id']: s['pp2']['source_id'] for s in scene['supports'] if 'pp2' in s}
            failure = str(verdict['checks'].get('process_completed', {}).get('failure') or '')
            planning = any(k in failure for k in PLANNING_FAILURES)
            for op in task['operations']:
                source = supports.get(op['target'].get('support'))
                if source is None:
                    if planning and op['id'] not in done:
                        break
                    continue
                row = stats.setdefault(source, {'transfers': 0, 'planning_failures': 0}); row['transfers'] += 1
                if planning and op['id'] not in done:
                    row['planning_failures'] += 1
                    break
    return stats


def build_v11(base_catalog, cohorts, output, min_transfers=10, max_failure_rate=.25):
    """Demote receptacles whose measured planning-failure rate is high (a deep narrow bucket the hand cannot enter
    to release: 14/42). Rule-based on execution evidence, like v8's grasp-lever demotion; baseline is ~4%."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base = read(base_catalog); stats = receptacle_planning_failures(cohorts)
    bad = {s: r for s, r in stats.items() if r['transfers'] >= min_transfers
           and r['planning_failures'] / r['transfers'] > max_failure_rate}
    dropped = [{'source_id': a['source_id'], 'reason': 'receptacle_planning_failure_rate', **bad[a['source_id']]}
               for a in base['assets'] if a['source_id'] in bad]
    catalog = {**base, 'schema': VERSION + '_v11', 'created_unix': time.time(), 'v10_catalog': binding(Path(base_catalog)),
               'planning_evidence': {'cohorts': [str(c) for c in cohorts], 'min_transfers': min_transfers,
                                     'max_failure_rate': max_failure_rate, 'receptacles': stats},
               'assets': [a for a in base['assets'] if a['source_id'] not in bad],
               'excluded': base['excluded'] + dropped, 'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'dropped': dropped, 'receptacles': sum(a['role'] == 'receptacle' for a in catalog['assets']),
            'measured_receptacles': len(stats), 'total_transfers': sum(r['transfers'] for r in stats.values()),
            'total_planning_failures': sum(r['planning_failures'] for r in stats.values())}


def build_v12(base_catalog, intake_admitted, output):
    """Add the assets the fast intake (resimpair_pp2.asset_intake) admitted by measured pick-carry-place success.

    Labels are the sources' own names (no VLM); every new asset passed >= 3 of 4 bench grasps."""
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base, intake = read(base_catalog), read(intake_admitted)
    have = {a['source_id'] for a in base['assets']}
    keys = ('source_id', 'role', 'category', 'color', 'color_evidence', 'luminance', 'label_source', 'source_name',
            'profile', 'partition', 'descriptor', 'cache', 'grasps', 'dimensions_m', 'mass_kg', 'admission', 'bench')
    new = [{k: a[k] for k in keys if k in a} for a in intake['assets'] if a['source_id'] not in have]
    catalog = {**base, 'schema': VERSION + '_v12', 'created_unix': time.time(), 'v11_catalog': binding(Path(base_catalog)),
               'fast_intake': binding(Path(intake_admitted)), 'assets': base['assets'] + new,
               'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    manipulables = [a for a in catalog['assets'] if a['role'] == 'manipulable']
    return {'added': len(new), 'manipulables': len(manipulables), 'receptacles': sum(a['role'] == 'receptacle' for a in catalog['assets']),
            'partitions': dict(Counter(a['partition'] for a in manipulables)),
            'sources': dict(Counter(a['source_id'].split(':')[0] for a in manipulables))}


def build_v13(base_catalog, audit_path, output):
    """v12 with source names re-cleaned (asset_intake.clean_name) and ray-tracing-dark assets dropped (rt_audit)."""
    from .asset_intake import clean_name
    output = Path(output).resolve()
    if (output / 'catalog.json').exists():
        raise ValueError('preserve_existing_catalog')
    base, audit = read(base_catalog), read(audit_path)
    if audit['catalog']['sha256'] != binding(Path(base_catalog))['sha256']:
        raise ValueError('rt_audit_bound_to_a_different_catalog')
    dark = set(audit['dark']); assets, dropped, renamed = [], [], 0
    for a in base['assets']:
        if a['source_id'] in dark:
            dropped.append({'source_id': a['source_id'], 'reason': 'dark_under_ray_tracing_no_repair'}); continue
        if a.get('label_source') == 'source_name':
            name = clean_name(a.get('source_name'), a.get('profile'))
            renamed += name != a['category']; a = {**a, 'category': name}
        assets.append(a)
    catalog = {**base, 'schema': VERSION + '_v13', 'created_unix': time.time(), 'v12_catalog': binding(Path(base_catalog)),
               'rt_audit': binding(Path(audit_path)), 'assets': assets,
               'distractor_only': [a for a in base.get('distractor_only', []) if a['source_id'] not in dark],
               'excluded': base['excluded'] + dropped, 'implementation': binding(Path(__file__))}
    write(output / 'catalog.json', catalog)
    return {'dropped': len(dropped), 'renamed': renamed, 'manipulables': sum(a['role'] == 'manipulable' for a in assets),
            'receptacles': sum(a['role'] == 'receptacle' for a in assets)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=16)
    p.add_argument('--v3-base', type=Path); p.add_argument('--v3-labels', type=Path)
    p.add_argument('--v4-base', type=Path); p.add_argument('--container-lists', type=Path, nargs='*', default=[])
    p.add_argument('--container-labels', type=Path, nargs='*', default=[])
    p.add_argument('--v5-base', type=Path); p.add_argument('--bench-rows', type=Path)
    p.add_argument('--v7-base', type=Path); p.add_argument('--v8-base', type=Path); p.add_argument('--v9-base', type=Path)
    p.add_argument('--v10-base', type=Path); p.add_argument('--rt-audit', type=Path)
    p.add_argument('--v11-base', type=Path); p.add_argument('--cohorts', type=Path, nargs='*', default=[])
    p.add_argument('--v12-base', type=Path); p.add_argument('--intake', type=Path)
    p.add_argument('--v13-base', type=Path)
    a = p.parse_args()
    if a.v13_base:
        result = build_v13(a.v13_base, a.rt_audit, a.output)
    elif a.v12_base:
        result = build_v12(a.v12_base, a.intake, a.output)
    elif a.v11_base:
        result = build_v11(a.v11_base, a.cohorts, a.output)
    elif a.v10_base:
        result = build_v10(a.v10_base, a.rt_audit, a.output)
    elif a.v9_base:
        result = build_v9(a.v9_base, a.output)
    elif a.v8_base:
        result = build_v8(a.v8_base, a.output)
    elif a.v7_base:
        result = build_v7(a.v7_base, a.output)
    elif a.v5_base:
        result = build_v5(a.v5_base, a.bench_rows, a.output)
    elif a.v4_base:
        result = build_v4(a.v4_base, a.container_lists, a.container_labels, a.output, a.workers)
    else:
        result = build_v3(a.v3_base, a.v3_labels, a.output, a.workers) if a.v3_base else build(a.output, a.workers)
    print(json.dumps(result, indent=1))
