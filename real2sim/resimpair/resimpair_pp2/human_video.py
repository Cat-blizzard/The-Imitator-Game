"""Human reference videos for reference groups: request packages out, verified videos in (backend-agnostic).

For every program of groups/manifest.json whose human video is pending, `prepare` writes a request package from
its source sample (layout 0): rendered first and last frames per camera, per-object masks, step boundaries,
program and instruction. Any producer can answer it: an API first/last-frame model, a local open model, a
masked-inpainting model over a simulated proxy, or a person recording the same program. `register` records the
video and its verification verdict in the manifest; `pairs` lists (human video, robot sample, render variant)
training pairs for verified videos only, so an unverified or wrong-task video never reaches training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from resimpair_grasp.common import ROOT, binding, digest, read, write
from .dataset import relative, rewrite

VERSION = 'pp2_human_video_v1'
STATUSES = ('pending', 'requested', 'generated', 'verified', 'rejected')  # generated: produced, awaiting review


def request_id(group, program_key):
    return group + '__' + digest([group, program_key])[:10]


def prepare(manifest_path, output, camera='opposite_high'):
    """One request package per pending program; returns how many were written."""
    manifest_path, output = Path(manifest_path), Path(output); manifest = read(manifest_path)
    datasets = Path(manifest_path).parent.parent / 'dataset' / 'samples'
    written = []
    for group, programs in manifest['groups'].items():
        for program in programs:
            source = program['human_video_source']
            if program['human_video']['status'] != 'pending' or source is None or not source['renders']:
                continue
            meta = read(datasets / source['sample_id'] / 'meta.json')
            render = source['renders'][0]
            rid = request_id(group, program['program_key'])
            package = {'schema': VERSION + '_request', 'request_id': rid, 'reference_group': group,
                       'program_key': program['program_key'], 'program': meta['program'], 'instruction': program['instruction'],
                       'family': program['family'], 'source_sample': source['sample_id'], 'camera': camera,
                       'room': render['room'], 'room_seed': render['room_seed'], 'execution': meta['execution'],
                       'first_frame': render['keyframes'].get(f'{camera}_initial'),
                       'last_frame': render['keyframes'].get(f'{camera}_final'),
                       'object_masks': render['object_masks'].get(f'{camera}_object_masks'),
                       'sim_video': render['videos'].get(camera), 'fps': render['fps'],
                       'steps': meta['steps'], 'robot_samples': len(program['robot_samples']),
                       'acceptance': 'see verify(): end state from masks, K-way identification, human spot check'}
            write(output / rid / 'request.json', package)
            program['human_video'] = {'status': 'requested', 'request': relative(output / rid / 'request.json'),
                                      'path': None, 'requested_unix': time.time()}
            written.append(rid)
    rewrite(manifest_path, manifest)
    return {'requests': len(written), 'output': relative(output)}


def register(manifest_path, request, video, verdict, producer):
    """Record a produced video and its verification verdict ({'passed': bool | None, 'checks': {...}}) for one request;
    passed None (automatic checks only, review pending) registers it as 'generated'."""
    manifest = read(manifest_path); request = read(request)
    program = next(p for p in manifest['groups'][request['reference_group']] if p['program_key'] == request['program_key'])
    program['human_video'] = {**program['human_video'], 'status': {True: 'verified', False: 'rejected', None: 'generated'}[verdict['passed']],
                              'path': relative(video), 'video': binding(Path(video)), 'producer': producer,
                              'verdict': verdict, 'registered_unix': time.time()}
    rewrite(manifest_path, manifest)
    return program['human_video']['status']


def pairs(manifest_path):
    """(human video, robot sample, render variant) for every verified program."""
    manifest = read(manifest_path); datasets = Path(manifest_path).parent / 'dataset' / 'samples'; rows = []
    for group, programs in manifest['groups'].items():
        for program in programs:
            if program['human_video']['status'] != 'verified':
                continue
            for sample in program['robot_samples']:
                for render in read(datasets / sample / 'meta.json')['renders']:
                    rows.append({'human_video': program['human_video']['path'], 'robot_sample': sample,
                                 'render_variant': render['variant'], 'instruction': program['instruction']})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare'); a.add_argument('--manifest', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('register'); b.add_argument('--manifest', type=Path, required=True); b.add_argument('--request', type=Path, required=True)
    b.add_argument('--video', type=Path, required=True); b.add_argument('--verdict', type=Path, required=True)
    b.add_argument('--producer', required=True)
    c = sub.add_parser('pairs'); c.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.manifest, args.output)))
    elif args.command == 'register':
        print(register(args.manifest, args.request, args.video, read(args.verdict), args.producer))
    else:
        rows = pairs(args.manifest); print(json.dumps({'pairs': len(rows), 'first': rows[:3]}, ensure_ascii=False, indent=1))
