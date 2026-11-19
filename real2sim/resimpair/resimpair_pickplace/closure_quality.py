"""Check whether physical fingers can reach a proposed contact pair.

Opposing ray hits can be behind another surface of the same object. This
bounded quasi-static sweep predicts first contact; it does not certify force
closure, deformation, or dynamic holding. Old grasp caches remain unchanged.
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh
from resimpair_grasp.common import read, write, resolve, binding, digest
from resimpair_tabletop.geometry import panda_meshes, bvh
from data_generation_methods.human_to_robot_pipeline.strict_motion import matrix_to_pose7


class ClosingSweep:
    def __init__(self, mesh):
        from mplib.pymp.collision_detection import fcl
        self.fcl = fcl
        self.geometry = bvh(mesh)
        self.object = fcl.CollisionObject(self.geometry)
        closed_parts = panda_meshes(0)[0]
        open_parts = panda_meshes(.08)[0]
        self.fingers = []
        for name in ['panda_leftfinger', 'panda_rightfinger']:
            # Each native finger has several collision elements. The first
            # contact is with their UNION, not every element simultaneously.
            mesh = trimesh.util.concatenate([m for n, _, m in closed_parts if n == name])
            opened = trimesh.util.concatenate([m for n, _, m in open_parts if n == name])
            geometry = bvh(mesh)
            offset = (opened.centroid - mesh.centroid) / .08
            self.fingers.append((name, geometry, fcl.CollisionObject(geometry), offset))

    def measure(self, pose, expected_width):
        T = np.asarray(pose, float)
        rows = []
        for name, _, finger, direction in self.fingers:
            def collides(width):
                delta = np.eye(4); delta[:3, 3] = direction * width
                finger.set_transformation(matrix_to_pose7(T @ delta))
                return self.fcl.collide(finger, self.object, self.fcl.CollisionRequest()).is_collision()
            previous = .08
            contact = None
            for width in np.linspace(.08, .008, 37):
                if collides(width):
                    lo, hi = float(width), float(previous)
                    for _ in range(5):
                        mid = (lo + hi) / 2
                        if collides(mid): lo = mid
                        else: hi = mid
                    contact = (lo + hi) / 2
                    break
                previous = width
            rows.append({'finger': name, 'first_contact_width_m': contact,
                         'predicted_minus_requested_width_m': None if contact is None else contact - expected_width})
        reachable = all(r['first_contact_width_m'] is not None and
                        abs(r['first_contact_width_m'] - expected_width) <= .004 for r in rows)
        return {'contact_pair_reachable': bool(reachable), 'expected_width_m': float(expected_width),
                'width_tolerance_m': .004, 'fingers': rows, 'physical_hold_status': 'unknown'}


def qualify(index, output, resample=False):
    if output.exists(): raise ValueError('Preserve previous closure evidence')
    output.mkdir(parents=True)
    entries = []; records = []
    for entry in read(index)['entries']:
        d = read(resolve(entry['asset'])); parent = read(resolve(entry['cache']))
        mesh = trimesh.util.concatenate([trimesh.load(resolve(b), force='mesh') for b in d['collisions']])
        sweep = ClosingSweep(mesh)
        candidates = parent['candidates']
        extra = None
        if resample:
            from resimpair_tabletop.geometry import propose
            from resimpair_grasp.common import ROOT
            profile = read(ROOT/'configs/tabletop_bulk_v2.json')
            profile.update(surface_samples=12000, max_grasps_per_pose=96, max_clearance_tests=400)
            visual = trimesh.load(resolve(d['visual']), force='scene', process=True).to_geometry()
            candidates, extra = propose(visual, profile, 2026102699)
            candidates = [{**c, 'id': f'resample_{i:03d}'} for i, c in enumerate(candidates)]
        accepted = []; decisions = []
        for c in candidates:
            r = sweep.measure(c['tcp_in_asset'], c['width_m'])
            decisions.append({'candidate': c['id'], **r})
            if r['contact_pair_reachable']: accepted.append({**c, 'closing_sweep': r})
        folder = output/(d['id']+'_'+digest(entry)[:10]); folder.mkdir()
        write(folder/'decisions.json', {'source_id': d['source_id'], 'rows': decisions,
                                      'resampling': extra, 'method': binding(__file__)})
        records.append({'source_id': d['source_id'], 'proposed': len(candidates), 'retained': len(accepted),
                        'evidence': binding(folder/'decisions.json')})
        if accepted:
            cache = {**parent, 'signature': {**parent['signature'], 'parent_cache': entry['cache'],
                     'closure_qualification': binding(__file__), 'resample': resample}, 'candidates': accepted}
            cache['key'] = digest(cache['signature']); path = folder/'grasp_cache.json'; write(path, cache)
            entries.append({'asset': entry['asset'], 'cache': binding(path)})
    write(output/'cache_index.json', {'schema': 'resimpair_grasp_cache_index_v1', 'entries': entries,
                                   'failures': [r for r in records if not r['retained']]})
    write(output/'report.json', {'rows': records, 'input': binding(index), 'resampled': resample,
                               'physical_execution_count': 0, 'api_calls': 0})
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--resample', action='store_true')
    a = p.parse_args(); print(qualify(a.index, a.output, a.resample))
