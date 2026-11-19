"""Generated human reference videos for pp2 tasks and blind task identification.

Per scene: one human first frame edited from the 2x simulator initial keyframe.
Per task: a human last frame edited from that task's simulator final keyframe
with the first frame as identity reference, then a first+last-frame video.
Identification is blind: a reviewer sees sampled frames and every task of the
scene as shuffled options. The same review on the robot's own rendered video
calibrates the reviewer. All calls go through the cumulative, serial,
single-submission budget; nothing is retried automatically.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from resimpair_grasp.common import ROOT, digest as plain_digest, resolve
from corl2026_workshop.experiments.autonomous_pipeline_v4.core import bind, checked, read, require, write
from . import grammar
from .render import room_seed

VERSION = 'pp2_reference_pilot_v1'
ANCESTOR = ROOT / 'runs/endpoint_conditioning_20261028'
LETTERS = 'ABCDEFGH'
SIDE_OF_ARM = {0: 'right', 1: 'left'}  # robot arm 0 sits at -y, the operator's right

FIRST_FRAME_PROMPT = (
    'Photorealistic edit of this tabletop scene. Remove both white robot arms completely, including their bases. '
    'In their place show exactly one adult person standing behind the far edge of the table, facing the camera, '
    'torso and both arms visible, both hands empty and resting near the table edge without touching any object. '
    'Keep everything else identical: the camera viewpoint, room, table, lighting, and every object on the table '
    'with the same identity, color, size and position. Do not add, remove, duplicate or move any object. '
    'No robot and no second person.')
LAST_FRAME_PROMPT = (
    'Image 1 shows the final arrangement of objects on a table; image 2 shows the person who did the task. '
    'Edit image 1: remove both white robot arms completely and show the same person as in image 2 (same face, hair, '
    'clothing and build) standing in the same place behind the table, facing the camera, both hands empty and '
    'withdrawn from the objects. Keep every object exactly where it is in image 1 with the same identity, color and '
    'size. Do not add, remove or move any object. No robot and no second person.')


def register(root, limits, usd_limit, parent=None, containment=None):
    """parent: a previous pp2 reference campaign whose started calls (including unknown ones) are inherited."""
    from corl2026_workshop.experiments.autonomous_pipeline_v4.budget import counts
    root = Path(root).resolve()
    if (root / 'budget.json').exists():
        raise ValueError('preserve_existing_reference_campaign')
    if parent is None:
        ancestor = read(ANCESTOR / 'budget.json'); status = read(ANCESTOR / 'status.json')
        require(status['status'] == 'closed', 'ancestor_campaign_must_be_closed')
        started = {k: ancestor['ancestor_started_calls'][k] + status['usage'][k] for k in ('image', 'video', 'review')}
        evidence = [bind(ANCESTOR / 'budget.json'), bind(ANCESTOR / 'status.json')]
    else:
        parent = Path(parent).resolve(); spec = read(parent / 'budget.json'); used = counts(spec['ledger_path'])
        require(containment is not None, 'parent_with_unknown_calls_requires_containment_evidence')
        started = {k: spec['ancestor_started_calls'][k] + used[k] for k in ('image', 'video', 'review')}
        evidence = [bind(parent / 'budget.json'), bind(containment)]
    write(root / 'config.json', read(ROOT / 'configs/providers.json'))
    spec = {'schema': 'resimpair_cumulative_budget_v4', 'campaign_id': root.name,
            'ledger_path': str(root / 'campaign_ledger'), 'new_call_limits': limits,
            'ancestor_started_calls': started,
            'lineage_cumulative_limits': {k: started[k] + limits[k] for k in limits},
            'ancestor_evidence': evidence,
            'max_concurrent': 1, 'timeout_seconds': 600, 'automatic_retry': False, 'currency_budget': None,
            'nominal_usd_admission_limit': usd_limit, 'observed_quota_stop_limit': int(usd_limit * 500000),
            'reserve_quota_before_new_call': {'image': 500000, 'video': 4000000, 'review': 500000},
            'authorization': 'Standing bounded API authorization (2026-10-24); user asked to run the pp2 six-step plan.'}
    write(root / 'budget.json', spec)
    return spec


def reserve_budget(root):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.billing import MeteredBudget, snapshot

    class ReserveBudget(MeteredBudget):
        """Refuse a new call unless observed spend + a conservative reserve stays under the stop limit."""
        def call(self, kind, request, operation):
            stage = self.directory(kind, request)
            if not (stage / 'result.json').exists():
                spent = 0
                for interval in self.ledger.glob('stages/*/credit_*_interval.json'):
                    if interval.name == 'credit_response_interval.json' and (interval.parent / 'credit_terminal_interval.json').exists():
                        continue
                    spent += read(interval).get('observed_token_quota_delta') or 0
                check = snapshot(self.root, self.root / 'reserve_checks' / f'{time.time_ns()}.json')
                quota = check.get('token_quota', {}); available = quota.get('total_available')
                reserve = self.spec['reserve_quota_before_new_call'][kind]
                require(spent + reserve <= self.spec['observed_quota_stop_limit'], 'campaign_quota_stop_limit_reached')
                # An unlimited-quota token reports negative availability; only the campaign stop limit applies then.
                require(quota.get('unlimited_quota') is True or type(available) not in (int, float) or available >= reserve,
                        'provider_credit_below_reserve')
            return super().call(kind, request, operation)
    return ReserveBudget(root)


def describe_steps(program, arms):
    phrases = []
    for k, (s, arm) in enumerate(zip(program['steps'], arms)):
        hand = SIDE_OF_ARM.get(arm, 'right')
        obj, target = grammar.descriptor_text(s['object']), grammar.descriptor_text(s['target'])
        if s['relation'] == 'in':
            place = f'puts it into {target}'
        elif s['relation'] == 'on':
            place = f'puts it onto {target}'
        else:
            place = f"sets it down on the table at the person's own {s['side']} side of {target}"
        lead = 'The person' if k == 0 else 'Then the person'
        phrases.append(f'{lead} uses the {hand} hand to pick up {obj}, {place}, releases it and withdraws the hand.')
    return ' '.join(phrases)


def video_prompt(item):
    return ('Endpoint routing: the first-frame image is the INITIAL state and the last-frame image is the FINAL state '
            'of the same single person, fixed camera and scene. Generate the continuous natural motion between them. '
            f"Task: {item['instruction']} {describe_steps(item['program'], item['arms'])} "
            'Only the named objects move, each exactly once and only while held by the hand; every other object '
            'stays where it is. Preserve object identity, count, color and size. One person only, no robot, '
            'no extra limbs. Static camera.')


def select(cohort, scenes, twin_tasks):
    """Accepted executions of the chosen scenes plus explicit twin tasks."""
    cohort = Path(cohort).resolve(); items = []
    accepted = {p.stem for p in (cohort / 'acceptance/cases').glob('*.json') if read(p)['passed']}
    cases = read(cohort / 'cohort.json')['cases']
    for case in cases:
        if case['id'] not in accepted:
            continue
        if case['scene_id'] in scenes or case['id'] in twin_tasks:
            execution = cohort / 'executions' / case['id']
            task = read(execution / 'effective_task.json')
            items.append({'case': case['id'], 'scene_id': case['scene_id'], 'execution': str(execution),
                          'program': task['pp2_program'], 'instruction': task['instruction'],
                          'arms': [op['arm'] for op in task['operations']]})
    options = {}
    for case in cases:
        options.setdefault(case['scene_id'], []).append(read(resolve(case['task']))['instruction'])
    return items, options


def keyframes(root, items, gpu):
    root = Path(root); done = {}
    for item in items:
        target = root / 'keyframes' / item['case']
        if not (target / 'keyframes.json').exists():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), OMP_NUM_THREADS='2')
            subprocess.run([sys.executable, '-m', 'resimpair_pp2.render', 'keyframes', '--execution', item['execution'],
                            '--output', str(target), '--seed', str(room_seed(item['scene_id']))],
                           cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        done[item['case']] = read(target / 'keyframes.json')
    return done


def image(root, budget, name, images, prompt):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.models import generate_image
    request = Path(root) / 'requests' / f'{name}.json'
    write(request, {'schema': VERSION, 'model_config': bind(Path(root) / 'config.json'),
                    'images': [bind(p) for p in images], 'prompt': prompt})
    output = Path(root) / 'images' / f'{name}.png'
    if output.exists():
        return output
    generate_image(root, budget, request, output)
    return output


def submit_video(root, budget, name, first, last, prompt, duration):
    from data_generation_methods.r2h_pipeline.providers import SeedanceClient
    from corl2026_workshop.experiments.task_contract_v3.direct_transport import call
    from corl2026_workshop.experiments.autonomous_pipeline_v4.models import configured, image_uri
    root = Path(root); config = configured(root)
    request = root / 'requests' / f'{name}.json'
    settings = {'duration': duration, 'resolution': '720p', 'ratio': 'adaptive', 'generate_audio': False,
                'watermark': False}
    write(request, {'schema': VERSION, 'model_config': bind(root / 'config.json'), 'first_frame': bind(first),
                    'last_frame': bind(last), 'prompt': prompt, 'settings': settings})
    payload = {**settings, 'model': config['seedance']['model'],
               'content': [{'type': 'text', 'text': prompt},
                           {'type': 'image_url', 'role': 'first_frame', 'image_url': {'url': image_uri(first)}},
                           {'type': 'image_url', 'role': 'last_frame', 'image_url': {'url': image_uri(last)}}]}
    require(len(json.dumps(payload)) < 3_000_000, 'video_payload_too_large')
    def operation(stage):
        def audited(method, url, **kwargs):
            require(kwargs['json'] == payload, 'video_payload_differs_from_request')
            write(stage / 'actual_input_audit.json', {'request': bind(request), 'payload_sha256': plain_digest(payload),
                  'roles': ['first_frame', 'last_frame'], 'payload_verified': True})
            kwargs['headers'] = {**kwargs['headers'], 'User-Agent': 'Mozilla/5.0', 'Connection': 'close'}
            raw = call(method, url, **kwargs)
            write(stage / 'provider_response.json', raw)
            return raw
        return SeedanceClient(dict(config['seedance'], timeout=budget.spec['timeout_seconds']), transport=audited).submit(payload)
    result, stage = budget.call('video', request, operation)
    return {'task_id': result['task_id'], 'request': bind(request), 'stage': str(stage)}


def wait_video(root, budget, submission, output, limit_seconds=1500):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.models import poll_video
    deadline = time.time() + limit_seconds
    while True:
        state = poll_video(root, submission, output)
        if state['status'] == 'downloaded':
            budget.settle_video(submission['stage']); return state
        if state['status'] in {'failed', 'expired', 'cancelled'}:
            budget.settle_video(submission['stage']); return state
        if time.time() > deadline:
            return {'status': 'still_running_resume_by_id', 'task_id': submission['task_id']}
        time.sleep(20)


def sample_frames(video, output, count=16, width=640):
    import av
    from PIL import Image
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with av.open(str(video)) as container:
        frames = [f.to_image() for f in container.decode(video=0)]
    picks = np.linspace(0, len(frames) - 1, count).round().astype(int)
    paths = []
    for k, i in enumerate(picks):
        im = frames[i]; im = im.resize((width, round(im.height * width / im.width)))
        path = output / f'frame_{k:02d}.png'
        if not path.exists():
            im.save(path)
        paths.append((path, float(i) / max(len(frames) - 1, 1)))
    return paths


def identify(root, budget, name, frames, options, actor):
    from corl2026_workshop.experiments.autonomous_pipeline_v4.models import reviewed_json
    order = sorted(range(len(options)), key=lambda i: plain_digest([name, i]))
    labels = {LETTERS[k]: options[i] for k, i in enumerate(order)}
    listing = '\n'.join(f'{k}. {v}' for k, v in labels.items())
    prompt = (f'These {len(frames)} frames are sampled in time order from one video of {actor} working at a table. '
              'At most one option below describes what is done in the video.\n' + listing + '\n'
              'Choose the single option that matches: which objects are moved, where each ends up (in, on, or next to '
              "which object, and on which side from the operator's own point of view), and in which order. If none "
              "matches, answer 'none'. Also count the distinct people visible in any frame, and say whether any object "
              'appears, disappears, duplicates or changes color or identity during the video.')
    schema = {'type': 'object', 'properties': {
        'chosen_option': {'type': 'string', 'enum': list(labels) + ['none']},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
        'moved_objects': {'type': 'array', 'items': {'type': 'string'}},
        'people_count': {'type': 'integer'},
        'object_identity_problem': {'type': 'boolean'},
        'explanation': {'type': 'string'}},
        'required': ['chosen_option', 'confidence', 'moved_objects', 'people_count', 'object_identity_problem', 'explanation']}
    request = Path(root) / 'requests' / f'{name}.json'
    write(request, {'schema': VERSION, 'model_config': bind(Path(root) / 'config.json'), 'prompt': prompt,
                    'images': [{'label': f'Frame {k + 1} (t={t:.2f} of video)', 'image': bind(p)}
                               for k, (p, t) in enumerate(frames)],
                    'response_schema': schema, 'option_labels': labels})
    review, evidence = reviewed_json(root, budget, request)
    return {'review': review, 'labels': labels, 'evidence': evidence}


def run(root, cohort, scenes, twin_tasks, twin_options_from, gpu, phase):
    root = Path(root).resolve(); budget = reserve_budget(root)
    items, options = select(cohort, scenes, twin_tasks)
    # Twins share programs with their source scene; their candidate list is the source scene's tasks.
    for item in items:
        if item['case'] in twin_tasks:
            item['options_scene'] = twin_options_from
    if phase == 'qualify':
        items = [i for i in items if i['scene_id'] == scenes[0]][:1]
    frames = keyframes(root, items, gpu)
    results = []
    for item in items:
        out = root / 'cases' / item['case']; out.mkdir(parents=True, exist_ok=True)
        if (out / 'result.json').exists():
            results.append(read(out / 'result.json')); continue
        scene_first = root / 'images' / f"{item['scene_id']}__human_first.png"
        if not scene_first.exists():
            image(root, budget, f"{item['scene_id']}__human_first", [resolve(frames[item['case']]['images']['initial'])],
                  FIRST_FRAME_PROMPT)
        # v2 protocol: the edited terminal frame (v1) reverted to the initial layout in qualification, so the
        # terminal frame is composited from simulator pixels instead of generated.
        kf = frames[item['case']]['images']
        last = out / 'human_last_composite.png'
        if not last.exists():
            info = composite_last_frame(scene_first, resolve(kf['initial']), resolve(kf['final']),
                                        resolve(kf['object_masks']), item['execution'], last)
            require(info['alignment']['passed'], 'human_first_frame_not_pixel_aligned_with_simulator')
            write(out / 'last_frame_composite.json', info)
        duration = 8 if len(item['program']['steps']) == 1 else 10
        first_small, last_small = out / 'video_first_768.png', out / 'video_last_768.png'
        for source, target in ((scene_first, first_small), (last, last_small)):
            if not target.exists():
                from PIL import Image
                Image.open(source).convert('RGB').resize((768, 768), Image.LANCZOS).save(target)
        prompt = video_prompt(item)
        video = out / 'human.mp4'
        submission_path = out / 'submission.json'
        if submission_path.exists():
            submission = read(submission_path)
        else:
            submission = submit_video(root, budget, f"{item['case']}__video", first_small, last_small, prompt, duration)
            write(submission_path, submission)
        state = wait_video(root, budget, submission, video)
        result = {'case': item['case'], 'scene_id': item['scene_id'], 'instruction': item['instruction'],
                  'program': item['program'], 'video_state': state['status'], 'prompt': prompt}
        option_scene = item.get('options_scene', item['scene_id'])
        if state['status'] == 'downloaded':
            human = identify(root, budget, f"{item['case']}__identify_human", sample_frames(video, out / 'human_frames'),
                             options[option_scene], 'a person')
            robot_video = Path(cohort) / '..' / 'renders' / item['case'] / 'opposite_high.mp4'
            robot = identify(root, budget, f"{item['case']}__identify_robot",
                             sample_frames(robot_video.resolve(), out / 'robot_frames'), options[item['scene_id']],
                             'two robot arms')
            for key, value in (('human', human), ('robot', robot)):
                chosen = value['review']['chosen_option']
                result[key] = {**value, 'chosen_instruction': value['labels'].get(chosen),
                               'correct': value['labels'].get(chosen) == item['instruction']}
        write(out / 'result.json', result); results.append(result)
        print(json.dumps({'case': item['case'], 'video': result['video_state'],
                          'human_correct': result.get('human', {}).get('correct'),
                          'robot_correct': result.get('robot', {}).get('correct')}), flush=True)
    write(root / 'usage' / f'{time.time_ns()}.json', budget.usage())
    return results



def identify_endpoints(root, budget, name, first, last, options):
    """Blind identification from the human start image and the composited end image only."""
    frames = [(Path(first), 0.), (Path(last), 1.)]
    return identify(root, budget, name, frames, options, 'a person (start and end images only)')


def run_static(root, cohort, scenes, twin_tasks, twin_options_from, gpu):
    """Video-free pilot: human first frames, composited terminal frames, blind identification."""
    root = Path(root).resolve(); budget = reserve_budget(root)
    items, options = select(cohort, scenes, twin_tasks)
    frames = keyframes(root, items, gpu); results = []
    for item in items:
        out = root / 'cases' / item['case']; out.mkdir(parents=True, exist_ok=True)
        if (out / 'result.json').exists():
            results.append(read(out / 'result.json')); continue
        kf = frames[item['case']]['images']
        first = root / 'images' / f"{item['scene_id']}__human_first.png"
        if not first.exists():
            image(root, budget, f"{item['scene_id']}__human_first", [resolve(kf['initial'])], FIRST_FRAME_PROMPT)
        last = out / 'human_last_composite.png'
        if not last.exists():
            info = composite_last_frame(first, resolve(kf['initial']), resolve(kf['final']),
                                        resolve(kf['object_masks']), item['execution'], last)
            require(info['alignment']['passed'], 'human_first_frame_not_pixel_aligned_with_simulator')
            write(out / 'last_frame_composite.json', info)
        option_scene = twin_options_from if item['case'] in twin_tasks else item['scene_id']
        robot_video = (Path(cohort) / '..' / 'renders' / item['case'] / 'opposite_high.mp4').resolve()
        def attempt(call):
            # A malformed but complete review is an unknown verdict; it is never parsed from prose or resent.
            try:
                return call()
            except Exception as error:
                return {'review': None, 'unknown': f'{type(error).__name__}:{getattr(error, "code", error)}'[:200]}
        endpoints = attempt(lambda: identify_endpoints(root, budget, f"{item['case']}__identify_endpoints", first, last,
                                                       options[option_scene]))
        robot = attempt(lambda: identify(root, budget, f"{item['case']}__identify_robot",
                                         sample_frames(robot_video, out / 'robot_frames'), options[item['scene_id']],
                                         'two robot arms'))
        result = {'case': item['case'], 'scene_id': item['scene_id'], 'instruction': item['instruction'],
                  'program': item['program'], 'ordered': item['program']['ordered']}
        for key, value in (('endpoints', endpoints), ('robot', robot)):
            if value['review'] is None:
                result[key] = {**value, 'correct': None}; continue
            chosen = value['review']['chosen_option']
            result[key] = {**value, 'chosen_instruction': value['labels'].get(chosen),
                           'correct': value['labels'].get(chosen) == item['instruction']}
        write(out / 'result.json', result); results.append(result)
        print(json.dumps({'case': item['case'], 'endpoints_correct': result['endpoints']['correct'],
                          'robot_correct': result['robot']['correct']}), flush=True)
    write(root / 'usage' / f'{time.time_ns()}.json', budget.usage())
    return results


def composite_last_frame(human_first, sim_initial, sim_final, masks_path, execution, output, dilate=4, moved_m=.005):
    """Deterministic terminal frame: paste moved objects' old and new pixels from the simulator final frame.

    Valid only because the human first frame preserves the simulator table pixels (checked by `alignment`).
    """
    from PIL import Image
    from scipy.ndimage import binary_dilation
    human = np.asarray(Image.open(human_first).convert('RGB')).copy()
    final = np.asarray(Image.open(sim_final).convert('RGB'))
    masks = np.load(masks_path)
    states = np.load(Path(execution) / 'measurements.npz')['object_states']
    moved = np.linalg.norm(states[-1, :, :3] - states[0, :, :3], axis=1) > moved_m
    region = (masks['initial'][moved].any(0) | masks['final'][moved].any(0)) if moved.any() else np.zeros(human.shape[:2], bool)
    region = binary_dilation(region, iterations=dilate)
    human[region] = final[region]
    Image.fromarray(human).save(output)
    return {'moved_objects': [str(x) for x in masks['object_ids'][moved]], 'pixels_replaced': int(region.sum()),
            'alignment': alignment(human_first, sim_initial)}


def alignment(human_first, sim_initial, rows=(480, 1000)):
    from PIL import Image
    a = np.asarray(Image.open(sim_initial).convert('L'), float)[rows[0]:rows[1]]
    b = np.asarray(Image.open(human_first).convert('L'), float)[rows[0]:rows[1]]
    F = np.fft.fft2(a - a.mean()) * np.conj(np.fft.fft2(b - b.mean()))
    r = np.fft.ifft2(F / np.maximum(np.abs(F), 1e-9)).real
    i, j = np.unravel_index(r.argmax(), r.shape)
    shift = [int(i if i < r.shape[0] // 2 else i - r.shape[0]), int(j if j < r.shape[1] // 2 else j - r.shape[1])]
    return {'shift_px': shift, 'mean_abs_diff': float(np.abs(a - b).mean()), 'passed': shift == [0, 0]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('register'); g.add_argument('--root', type=Path, required=True)
    g.add_argument('--image', type=int, required=True); g.add_argument('--video', type=int, required=True)
    g.add_argument('--review', type=int, required=True); g.add_argument('--usd', type=float, required=True)
    g.add_argument('--parent', type=Path); g.add_argument('--containment', type=Path)
    r = sub.add_parser('run'); r.add_argument('--root', type=Path, required=True); r.add_argument('--cohort', type=Path, required=True)
    r.add_argument('--scenes', nargs='+', required=True); r.add_argument('--twin-tasks', nargs='*', default=[])
    r.add_argument('--twin-options-from'); r.add_argument('--gpu', type=int, default=3)
    r.add_argument('--phase', choices=['qualify', 'all', 'static'], required=True)
    a = p.parse_args()
    if a.command == 'register':
        print(json.dumps(register(a.root, {'image': a.image, 'video': a.video, 'review': a.review}, a.usd,
                                  a.parent, a.containment), indent=1))
    elif a.phase == 'static':
        rows = run_static(a.root, a.cohort, a.scenes, a.twin_tasks, a.twin_options_from, a.gpu)
        print(json.dumps([{k: r[k] for k in ('case',)} | {'endpoints': r['endpoints']['correct'], 'robot': r['robot']['correct']} for r in rows]))
    else:
        print(json.dumps(run(a.root, a.cohort, a.scenes, a.twin_tasks, a.twin_options_from, a.gpu, a.phase), indent=1)[:4000])
