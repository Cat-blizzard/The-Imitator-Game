"""Bounded native-Panda pose proposals with a task-load proxy, on CPU.

This is an adaptation of the contact-optimization idea, NOT a reproduction
of cuDex-graspvla. Original candidates are measured under the same new model.
All outputs are new evidence; physical budgets and defaults are untouched.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
from resimpair_grasp.common import ROOT, read, write, binding, resolve, digest
from resimpair_tabletop.geometry import Clearance
from .closure_quality import ClosingSweep
from .contact_quality import assess, dfc_residual, quality_key


class ContactSearch:
    def __init__(self, descriptor, profile):
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        self.parts = [trimesh.load(resolve(p), force='mesh', process=False)
                      for p in descriptor['collisions']]
        self.mesh = trimesh.util.concatenate(self.parts)
        self.ray = RayMeshIntersector(self.mesh)
        self.profile = profile
        masses = np.asarray([abs(p.volume) for p in self.parts])
        if masses.sum() < 1e-12 or not np.isfinite(masses).all():
            raise ValueError('invalid_collision_volume_for_COM_proxy')
        self.center = np.average([p.center_mass for p in self.parts], axis=0, weights=masses)
        if not np.isfinite(self.center).all(): raise ValueError('nonfinite_COM_proxy')
        self.length = float(np.linalg.norm(self.mesh.extents))
        self.clearance = Clearance(self.mesh, profile['approach_distance_m'])
        self.sweep = ClosingSweep(self.mesh)

    def ray_contacts(self, T):
        direction = T[:3, 1]
        radius = self.length + np.linalg.norm(T[:3, 3]-self.mesh.centroid)
        origins = np.array([T[:3, 3]-radius*direction, T[:3, 3]+radius*direction])
        dirs = np.array([direction, -direction])
        loc, index, faces = self.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(index) != 2: return None
        order = np.argsort(index); points = loc[order]; normals = self.mesh.face_normals[faces[order]]
        width = float(np.linalg.norm(points[1]-points[0]))
        cosines = np.array([-normals[0]@direction, normals[1]@direction])
        threshold = 1/np.sqrt(1+min(self.profile['friction_coefficients'])**2)
        if not self.profile['min_width_m'] <= width <= self.profile['max_width_m']:
            return None
        if np.min(cosines) < threshold: return None
        return points, normals, width, cosines

    def measure(self, candidate):
        T = np.asarray(candidate['tcp_in_asset'], float)
        found = self.ray_contacts(T)
        if found is None: return None, 'native_contact_pair_or_friction'
        points, normals, width, cosines = found
        ok, gap, reason = self.clearance.check(T, self.mesh.bounds[0, 2], self.profile['table_clearance_m'])
        if not ok: return None, reason
        closing = self.sweep.measure(T, candidate['width_m'])
        if not closing['contact_pair_reachable']: return None, 'native_closing_sweep'
        score = assess(points, normals, self.center, self.length, self.profile)
        if not score['model_valid']: return None, 'static_solver_invalid'
        if score['worst_load_capacity'] < self.profile['minimum_capacity_for_ranking_pool']:
            return None, 'no_static_load_capacity_under_declared_assumptions'
        return {**candidate, 'native_contact_points': points.tolist(),
                'native_contact_normals': normals.tolist(), 'native_ray_width_m': width,
                'friction_cosines': cosines.tolist(), 'table_clearance_m': gap,
                'closing_sweep': closing, 'task_load_quality': score,
                'physical_status': 'unmeasured'}, None

    def local_proposals(self, candidates):
        """Finite low-discrepancy tangent/height/yaw neighborhoods, no retries."""
        profile = self.profile; seen = set(); seeds = []
        for c in candidates:
            T = np.asarray(c['tcp_in_asset'])
            # Equivalent finger-swap orientations share a seed; final poses
            # retain distinct camera/arm geometry and are checked separately.
            axis = T[:3, 1].copy()
            if axis[np.argmax(np.abs(axis))] < 0: axis *= -1
            key = tuple(np.round(np.r_[T[:3, 3]/.002, axis/.05]).astype(int))
            if key in seen: continue
            seen.add(key); seeds.append(c)
            if len(seeds) >= profile['local_seed_limit']: break
        design = qmc.Halton(3, scramble=True, seed=profile['seed']).random(profile['perturbations_per_seed'])*2-1
        proposals = []
        for seed in seeds:
            base = np.asarray(seed['tcp_in_asset'])
            for j, (offset, height, angle) in enumerate(design):
                T = base.copy()
                T[:3, 3] += base[:3, 0]*offset*profile['translation_bound_m']
                T[2, 3] += height*profile['height_bound_m']
                T[:3, :3] = Rotation.from_euler('z', angle*profile['yaw_bound_deg'], degrees=True).as_matrix()@base[:3, :3]
                found = self.ray_contacts(T)
                if found is None: continue
                p, n, width, cosines = found; T[:3, 3] = p.mean(axis=0)
                cheap = dfc_residual(p, n, self.center, self.length)
                cheap += np.linalg.norm(np.cross(p[1]-p[0], self.center-p.mean(axis=0)))/(width*self.length)
                proposals.append({'id': f"local_{seed['id']}_{j:02d}", 'tcp_in_asset': T.tolist(),
                    'width_m': width, 'contacts_in_asset': p.tolist(), 'contact_normals': n.tolist(),
                    'proposal': 'bounded_native_contact_refinement', 'parent_candidate': seed['id'],
                    'geometric_rank_cost': float(cheap), 'physical_status': 'unmeasured',
                    'approach_sweep_m': profile['approach_distance_m']})
        proposals.sort(key=lambda c: (c['geometric_rank_cost'], c['id']))
        return proposals[:profile['max_new_geometry_checks']], len(proposals)


def worker(entry, profile, output):
    started = time.perf_counter(); output.mkdir(parents=True, exist_ok=False)
    d = read(resolve(entry['asset'])); parent = read(resolve(entry['cache']))
    search = ContactSearch(d, profile); load_seconds = time.perf_counter()-started
    original = parent['candidates'][:profile['baseline_candidate_limit']]
    decisions = []; baseline = []; improved = []
    for c in original:
        result, reason = search.measure(copy.deepcopy(c))
        decisions.append({'id': c['id'], 'kind': 'original', 'retained': result is not None, 'reason': reason})
        if result is not None: baseline.append(result)
    baseline_seconds = time.perf_counter()-started-load_seconds
    t = time.perf_counter(); proposals, raw_count = search.local_proposals(original)
    for c in proposals:
        result, reason = search.measure(c)
        decisions.append({'id': c['id'], 'kind': 'local', 'retained': result is not None, 'reason': reason})
        if result is not None: improved.append(result)
    refine_seconds = time.perf_counter()-t
    ranked = sorted(baseline, key=quality_key)
    combined = sorted(baseline+improved, key=quality_key)
    # Prevent numerically identical poses from occupying the entire shortlist.
    selected = []; seen = set()
    for c in combined:
        T = np.asarray(c['tcp_in_asset']); yaw = np.arctan2(T[1, 1], T[0, 1])
        key = (*np.round(T[:3, 3]/.002).astype(int), int(round(yaw/.05)))
        if key in seen: continue
        seen.add(key); selected.append(c)
        if len(selected) >= profile['output_candidate_limit']: break
    write(output/'all_qualified_candidates.json', {'baseline': baseline, 'local': improved})
    signature = {**parent['signature'], 'parent_cache': entry['cache'],
                 'contact_refinement': binding(__file__), 'profile': profile,
                 'quality_model': binding(Path(__file__).with_name('contact_quality.py'))}
    if selected:
        write(output/'grasp_cache.json', {**parent, 'key': digest(signature), 'signature': signature,
              'candidates': selected, 'selection_policy': 'task_load_proxy_then_joint_travel_v1',
              'physical_qualification': 'unmeasured'})
    def brief(c):
        return None if c is None else {'id': c['id'], 'tcp_in_asset': c['tcp_in_asset'],
                'quality': c['task_load_quality'], 'closing_sweep': c['closing_sweep']}
    first = next((c for c in baseline if c['id'] == original[0]['id']), None) if original else None
    report = {'source_id': d['source_id'], 'entry': entry, 'profile': profile,
        'com_proxy_m': search.center.tolist(), 'com_model': 'uniform density of native collision parts; overlap counted; not measured real mass distribution',
        'baseline_checked': len(original), 'baseline_retained': len(baseline),
        'local_ray_candidates': raw_count, 'local_checked': len(proposals), 'local_retained': len(improved),
        'output_count': len(selected), 'baseline_first': brief(first),
        'rerank_only_best': brief(ranked[0] if ranked else None),
        'refinement_best': brief(selected[0] if selected else None),
        'decisions': decisions, 'timing_seconds': {'load': load_seconds, 'baseline_assessment': baseline_seconds,
             'local_refinement': refine_seconds, 'total': time.perf_counter()-started},
        'physical_executions': 0, 'api_calls': 0, 'gpu_jobs': 0,
        'claim_scope': 'proposal/static proxy comparison only, not a dynamic success-rate comparison'}
    write(output/'report.json', report)
    return report


def recover_index(root):
    """Reconcile completed work without resubmission; zero-pose is an outcome."""
    entries = []; failures = []; pending = []
    requests = sorted(root.glob('request_*.json'))
    for request_path in requests:
        request = read(request_path); entry = request['entry']
        suffix = request_path.stem.removeprefix('request_')
        terminal_path = root/f'terminal_{suffix}.json'
        folder = root/f'asset_{suffix}'; source = read(resolve(entry['asset']))['source_id']
        if not terminal_path.exists():
            pending.append({'source_id':source,'reason':'missing_terminal_query_before_retry'})
            continue
        terminal = read(terminal_path)
        if terminal.get('report'): resolve(terminal['report'])
        if terminal['returncode'] == 0 and (folder/'grasp_cache.json').exists():
            entries.append({'asset':entry['asset'],'cache':binding(folder/'grasp_cache.json'),'split':entry['split']})
        else:
            failures.append({'source_id':source,'terminal':binding(terminal_path),
                'reason':'no_qualified_pose_under_declared_model' if terminal['returncode']==0 else 'worker_failed_or_timed_out',
                'returncode':terminal['returncode']})
    return {'schema':'resimpair_grasp_cache_index_v1','entries':entries,'failures':failures,
            'pending':pending,'requested_sources':len(requests),'physical_executions':0,'api_calls':0}


def run(plan, output):
    spec = read(plan); profile = read(resolve(spec['profile'])); items = spec['entries']
    if output.exists(): raise ValueError('Preserve previous batch; recover recorded outcomes')
    if not 1 <= len(items) <= profile['max_assets']: raise ValueError('asset_budget_exceeded')
    if len({read(resolve(e['asset']))['source_id'] for e in items}) != len(items):
        raise ValueError('source_split_leak_or_duplicate_source')
    output.mkdir(parents=True)
    write(output/'reservation.json', {'plan': binding(plan), 'profile': profile,
          'implementation': [binding(__file__), binding(Path(__file__).with_name('contact_quality.py'))],
          'physical_executions': 0, 'api_calls': 0, 'gpu_jobs': 0, 'max_concurrent': 1,
          'prior_shared_physical_attempts': 24, 'prior_physical_budget_unchanged': True})
    entries = []; terminal = []
    for i, entry in enumerate(items):
        start = time.perf_counter(); folder = output/f'asset_{i:02d}'
        request = output/f'request_{i:02d}.json'; write(request, {'entry': entry, 'profile': profile})
        env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (output/f'asset_{i:02d}.log').open('x') as log:
            try:
                result = subprocess.run([sys.executable, '-m', 'resimpair_pickplace.contact_refinement',
                    'worker', '--request', str(request.resolve()), '--output', str(folder.resolve())],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=profile['timeout_seconds_per_asset'])
                code = result.returncode
            except subprocess.TimeoutExpired: code = 'timeout'
        row = {'source_id': read(resolve(entry['asset']))['source_id'], 'split': entry['split'],
               'returncode': code, 'elapsed_seconds': time.perf_counter()-start,
               'report': binding(folder/'report.json') if (folder/'report.json').exists() else None}
        write(output/f'terminal_{i:02d}.json', row); terminal.append(row)
        if code == 0 and (folder/'grasp_cache.json').exists():
            entries.append({'asset': entry['asset'], 'cache': binding(folder/'grasp_cache.json'), 'split': entry['split']})
    write(output/'cache_index.json', recover_index(output))
    write(output/'report.json', {'rows': terminal, 'physical_executions': 0, 'api_calls': 0})
    return terminal


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--plan', type=Path, required=True); r.add_argument('--output', type=Path, required=True)
    w = sub.add_parser('worker'); w.add_argument('--request', type=Path, required=True); w.add_argument('--output', type=Path, required=True)
    i = sub.add_parser('recover-index'); i.add_argument('--root', type=Path, required=True); i.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'run': print(run(a.plan, a.output))
    elif a.cmd == 'worker':
        request = read(a.request); print(worker(request['entry'], request['profile'], a.output)['source_id'])
    else:
        result=recover_index(a.root);write(a.output,result)
        print({'entries':len(result['entries']),'no_pose_or_failed':len(result['failures']),'pending':len(result['pending'])})
