"""Seedance (Volcengine Ark) producer for human reference videos (human_video request packages in, videos out).

For each request: render the source sample's first and last frame *without the robots* (render.keyframes
--hide-robot, same room seed as the robot video), ask Seedance for a first/last-frame video of a person's
hands doing the program, download it, measure how well its ends match the requested frames, and register it
as 'generated' (human review decides verified / rejected; only verified videos reach `human_video.pairs`).

Paid-API rules:
  * one task at a time;
  * every submission is written to ledger.jsonl before the call and again with its outcome;
  * a submission whose outcome is unknown (no task id came back) is never resubmitted blindly: `reconcile` adopts
    the task the list shows for it, or records that none was created, before anything else is sent;
  * budget_cny caps the estimated spend (successful tasks at their billed tokens, unknown ones at the estimate).
The API key is read from private/volcengine.env.local; it is never printed or written anywhere else.

    .venv/bin/python -B -m resimpair_pp2.seedance check
    .venv/bin/python -B -m resimpair_pp2.seedance run --manifest runs/<run>/groups/manifest.json \\
        --budget-cny 150 [--model 2.0] [--resolution 720p] [--limit N] [--gpu 1]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from resimpair_grasp.common import ROOT, binding, read, write
from . import human_video

VERSION = 'pp2_seedance_v1'
BASE = 'https://ark.cn-beijing.volces.com/api/v3'
KEY_FILE = ROOT / 'private/volcengine.env.local'
MODELS = {'2.5': 'doubao-seedance-2-5-260628', '2.0': 'doubao-seedance-2-0-260128',
          '2.0-fast': 'doubao-seedance-2-0-fast-260128', '2.0-mini': 'doubao-seedance-2-0-mini-260615'}
# CNY per million output tokens, no input video (Ark model pricing, 2026-11); 1080p rows where they differ
PRICE = {('2.5', '720p'): 70., ('2.5', '1080p'): 77., ('2.0', '720p'): 46., ('2.0', '1080p'): 51.,
         ('2.0-fast', '720p'): 37., ('2.0-mini', '720p'): 23.}
PIXELS = {'480p': 854 * 480, '720p': 1280 * 720, '1080p': 1920 * 1080}  # 16:9 upper bound for the estimate
FINAL = ('succeeded', 'failed', 'expired', 'cancelled')
PROMPT = ('Realistic tabletop video from a completely fixed camera. A person standing on the camera side reaches into '
          'the frame with their hands and {task} The hands move naturally, pick up one object at a time, carry it and '
          'set it down gently, then withdraw out of the frame. The camera does not move. Only the objects the person '
          'handles move; everything else stays exactly where it is. No robot, no text, no other people.')


class BudgetExhausted(RuntimeError):
    pass


def api_key():
    if os.environ.get('ARK_API_KEY'):                 # Imitator 2.0: environment variable first
        return os.environ['ARK_API_KEY']
    if not KEY_FILE.is_file():
        raise KeyError('set ARK_API_KEY (or write it to private/volcengine.env.local)')
    for line in KEY_FILE.read_text().splitlines():
        if line.startswith('ARK_API_KEY='):
            return line.split('=', 1)[1].strip()
    raise KeyError('ARK_API_KEY missing from private/volcengine.env.local')


def call(method, path, body=None, timeout=60, retries=5):
    """One Ark call. Reads (GET) are retried on connection errors and timeouts; a create (POST) never is, because
    its outcome is then unknown and goes through `reconcile`."""
    import requests
    for attempt in range(retries if method == 'GET' else 1):
        try:
            response = requests.request(method, BASE + path, json=body, timeout=timeout,
                                        headers={'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'})
            break
        except (requests.ConnectionError, requests.Timeout):
            if method != 'GET' or attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    data = response.json() if response.content else {}
    if response.status_code >= 400:
        error = data.get('error', data) if isinstance(data, dict) else data
        raise RuntimeError(f'ark_http_{response.status_code}: {json.dumps(error, ensure_ascii=False)[:300]}')
    return data


def estimate_cny(model, resolution, seconds, fps=24):
    price = PRICE.get((model, resolution), PRICE.get((model, '720p')))
    return PIXELS[resolution] * fps * seconds / 1024 * price / 1e6


def duration_s(request, model):
    """About three seconds per step plus three for reaching in and out, within the model's range."""
    return int(min(30 if model == '2.5' else 15, max(5, 3 + 3 * len(request['steps']))))


def prompt(request):
    task = request['instruction'].strip()
    return PROMPT.format(task=task[0].lower() + task[1:] + ('' if task.endswith('.') else '.'))


class Ledger:
    def __init__(self, path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()] if self.path.exists() else []

    def add(self, **row):
        with self.path.open('a') as f:
            f.write(json.dumps({**row, 'unix': time.time()}, ensure_ascii=False) + '\n')

    def state(self):
        """request_id -> its latest row, and the spend that counts against the budget."""
        latest, spent = {}, 0.
        for r in self.rows():
            latest[r['request_id']] = r
            spent += r.get('cost_cny', 0.)
        spent += sum(r['estimate_cny'] for r in latest.values() if r['event'] in ('submitting', 'unknown', 'created'))
        return latest, spent


def frames(request, output, gpu):
    """Robot-free initial and final keyframes (2x resolution) in the robot video's room."""
    out = Path(output) / 'frames'
    if (out / 'keyframes.json').exists():
        return read(out / 'keyframes.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT))
    with (Path(output) / 'frames.log').open('a') as log:
        subprocess.run([sys.executable, '-B', '-m', 'resimpair_pp2.render', 'keyframes', '--execution', str(ROOT / request['execution']),
                        '--output', str(out), '--seed', str(request['room_seed']), '--scale', '2', '--camera', request['camera'],
                        '--hide-robot'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    return read(out / 'keyframes.json')


def data_url(path, quality=92):
    """JPEG data URL: two 1024px PNGs (~4 MB as base64) made the create call time out."""
    import io
    from PIL import Image
    buffer = io.BytesIO(); Image.open(path).convert('RGB').save(buffer, 'JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()


def reconcile(ledger, settle_s=120):
    """Resolve 'unknown' submissions from the task list: adopt the one task created for it, or record that none was.

    A task belongs to an unknown row when it has the row's model, was created after that submission started (a hung
    create call may still create its task late) and no ledger row already names it. Ambiguity (several candidates)
    stays unknown for a person to resolve."""
    rows = ledger.rows(); latest, _ = ledger.state(); known = {r.get('task_id') for r in rows}
    unknown = [r for r in latest.values() if r['event'] == 'unknown']
    if not unknown:
        return []
    tasks = call('GET', '/contents/generations/tasks?page_num=1&page_size=100').get('items', [])
    resolved = []
    for row in unknown:
        started = max(r['unix'] for r in rows if r['request_id'] == row['request_id'] and r['event'] == 'submitting')
        meta = {k: row[k] for k in ('request_id', 'model', 'resolution', 'duration_s', 'estimate_cny')}
        found = [t for t in tasks if t.get('model') == row['model'] and t['id'] not in known
                 and float(t.get('created_at', 0)) >= started - 5]
        if len(found) == 1:
            ledger.add(event='created', task_id=found[0]['id'], reconciled=True, **meta); resolved.append('adopted')
        elif not found and time.time() - row['unix'] > settle_s:
            ledger.add(event='not_created', reconciled=True, **meta); resolved.append('not_created')
    return resolved


def decode(video):
    import av
    with av.open(str(video)) as c:
        decoded = [f.to_ndarray(format='rgb24') for f in c.decode(video=0)]
        fps = float(c.streams.video[0].average_rate or 0)
    return decoded, fps


def step_objects(request):
    """Scene object id of each program step (descriptor = category, plus color when the program names one)."""
    objects = read(ROOT / request['execution'] / 'effective_scene.json')['objects']
    ids = []
    for step in request['program']['steps']:
        want = step['object']
        match = [o['id'] for o in objects if o['pp2']['category'] == want['category']
                 and ('color' not in want or o['pp2'].get('color') == want['color'])]
        ids.append(match[0] if len(match) == 1 else None)
    return ids


def departures(frames, start, masks, ids, threshold=.06):
    """Frame index after which each object's initial region no longer looks like the start frame (it was taken).

    The last frame still matching is used, so a hand passing over an object before or after it is picked up
    does not move its departure."""
    from PIL import Image
    h, w = frames[0].shape[:2]; index = {str(o): i for i, o in enumerate(masks['object_ids'])}
    out = []
    for oid in ids:
        if oid is None or oid not in index:
            out.append(None); continue
        region = np.asarray(Image.fromarray(masks['initial'][index[oid]]).resize((w, h)))
        diff = [float(np.abs(f[region] / 255. - start[region]).mean()) for f in frames]
        still = [t for t, d in enumerate(diff) if d < threshold]
        out.append(still[-1] + 1 if still else 0)
    return out


def compare(video, keyframes, request):
    """Mean absolute difference (0..1) of the generated ends against the requested ends, over the full frame and over
    the moved objects' final masks (where a wrong end state shows)."""
    from PIL import Image
    decoded, fps = decode(video); first, last, n = decoded[0], decoded[-1], len(decoded)
    h, w = last.shape[:2]
    target = lambda key: np.asarray(Image.open(ROOT / keyframes['images'][key]['path']).convert('RGB').resize((w, h))) / 255.
    masks = np.load(ROOT / keyframes['images']['object_masks']['path'])
    moved = [i for i, o in enumerate(masks['object_ids']) if (masks['initial'][i] != masks['final'][i]).any()]
    region = np.zeros((h, w), bool)
    for i in moved:
        region |= np.asarray(Image.fromarray(masks['final'][i]).resize((w, h)))
    end, start = last / 255., first / 255.
    order = None
    if len(request['steps']) > 1:
        left = departures(decoded, target('initial'), masks, step_objects(request))
        order = {'departure_frames': left, 'ordered_program': request['program']['ordered'],
                 'in_program_order': None if None in left else all(a < b for a, b in zip(left, left[1:]))}
    return {'frames': n, 'fps': fps, 'seconds': n / fps if fps else None, 'size': [w, h], 'order': order,
            'first_frame_mad': float(np.abs(start - target('initial')).mean()),
            'last_frame_mad': float(np.abs(end - target('final')).mean()),
            'moved_objects_final_mad': float(np.abs(end - target('final'))[region].mean()) if region.any() else None,
            'moved_objects': len(moved)}


def generate(request_path, ledger, model='2.0', resolution='720p', budget_cny=0., gpu=0, poll_s=15, wait_s=3600,
             attempts=3, settle_s=120):
    """One request -> output/video.mp4, or the reason it is not there.

    A create call that hangs (pilot: 5 of 21, while good ones answer in ~3 s) is reconciled after settle_s, and
    only resubmitted once the task list shows no task for it, up to `attempts` submissions per call."""
    for _ in range(attempts):
        row = submit_and_wait(request_path, ledger, model, resolution, budget_cny, gpu, poll_s, wait_s)
        if row['status'] != 'unknown':
            return row
        time.sleep(settle_s); reconcile(ledger, settle_s)
    return row


def submit_and_wait(request_path, ledger, model, resolution, budget_cny, gpu, poll_s, wait_s):
    request_path = Path(request_path); request = read(request_path); rid = request['request_id']
    output = request_path.parent
    if (output / 'video.mp4').exists():
        return {'request_id': rid, 'status': 'exists'}
    latest, spent = ledger.state(); last = latest.get(rid, {})
    if last.get('event') in ('submitting', 'unknown'):
        return {'request_id': rid, 'status': 'unknown_outcome_not_resubmitted'}
    keys = frames(request, output, gpu); seconds = duration_s(request, model)
    estimate = estimate_cny(model, resolution, seconds)
    if last.get('event') != 'created':
        if spent + estimate > budget_cny:
            raise BudgetExhausted(f'spent_or_reserved_{spent:.1f}_plus_{estimate:.1f}_exceeds_{budget_cny:.1f}_cny')
        body = {'model': MODELS[model], 'content': [
                    {'type': 'text', 'text': prompt(request)},
                    {'type': 'image_url', 'image_url': {'url': data_url(ROOT / keys['images']['initial']['path'])}, 'role': 'first_frame'},
                    {'type': 'image_url', 'image_url': {'url': data_url(ROOT / keys['images']['final']['path'])}, 'role': 'last_frame'}],
                'ratio': 'adaptive', 'resolution': resolution, 'duration': seconds, 'watermark': False, 'generate_audio': False}
        meta = {'request_id': rid, 'model': MODELS[model], 'resolution': resolution, 'duration_s': seconds, 'estimate_cny': estimate,
                'prompt': body['content'][0]['text']}
        ledger.add(event='submitting', **meta)
        try:
            task_id = call('POST', '/contents/generations/tasks', body, timeout=30)['id']
        except RuntimeError as error:  # an HTTP error answer: nothing was created, nothing is billed
            ledger.add(event='rejected', error=str(error), **meta); return {'request_id': rid, 'status': 'rejected', 'error': str(error)}
        except Exception as error:  # no answer: the task may or may not exist
            ledger.add(event='unknown', error=repr(error)[:300], **meta); return {'request_id': rid, 'status': 'unknown'}
        ledger.add(event='created', task_id=task_id, **meta); last = {**meta, 'task_id': task_id}
    meta = {k: last[k] for k in ('request_id', 'model', 'resolution', 'duration_s', 'estimate_cny', 'task_id')}
    started = time.time()
    while True:
        task = call('GET', f"/contents/generations/tasks/{meta['task_id']}")
        if task.get('status') in FINAL:
            break
        if time.time() - started > wait_s:
            return {'request_id': rid, 'status': 'still_running', 'task_id': meta['task_id']}
        time.sleep(poll_s)
    tokens = (task.get('usage') or {}).get('completion_tokens') or (task.get('usage') or {}).get('total_tokens') or 0
    if task['status'] != 'succeeded':
        ledger.add(event=task['status'], error=json.dumps(task.get('error'), ensure_ascii=False)[:300], cost_cny=0., **meta)
        return {'request_id': rid, 'status': task['status'], 'error': task.get('error')}
    import requests
    for attempt in range(5):  # the signed video URL stays valid for a day; a download is safe to repeat
        try:
            video = requests.get(task['content']['video_url'], timeout=300); video.raise_for_status(); break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))
    (output / 'video.mp4').write_bytes(video.content)
    model_key = next(k for k, v in MODELS.items() if v == meta['model'])
    cost = tokens * PRICE.get((model_key, meta['resolution']), PRICE.get((model_key, '720p'))) / 1e6
    ledger.add(event='succeeded', tokens=tokens, cost_cny=cost, **meta)
    write(output / 'producer.json', {'schema': VERSION, 'request': binding(request_path), 'task_id': meta['task_id'],
                                     'model': meta['model'], 'resolution': meta['resolution'], 'duration_s': meta['duration_s'],
                                     'prompt': prompt(request), 'keyframes': keys['images'], 'tokens': tokens, 'cost_cny': cost,
                                     'video': binding(output / 'video.mp4')})
    return {'request_id': rid, 'status': 'succeeded', 'cost_cny': cost}


MAX_LAST_FRAME_MAD = .05  # pilot: 0.012-0.017 for videos that end on the requested state


def verify(manifest, request_path):
    """Automatic checks -> verdict: a wrong end state or an ordered program done out of order is rejected; anything
    else is 'generated' and waits for a person's spot check."""
    from .dataset import rewrite
    request_path = Path(request_path); request = read(request_path)
    checks = compare(request_path.parent / 'video.mp4', read(request_path.parent / 'frames/keyframes.json'), request)
    problems = [p for p, bad in (('end_state', checks['last_frame_mad'] > MAX_LAST_FRAME_MAD),
                                 ('step_order', request['program']['ordered'] and (checks['order'] or {}).get('in_program_order') is False))
                if bad]
    verdict = {'passed': False if problems else None, 'problems': problems, 'checks': checks,
               'review': None if problems else 'pending human spot check'}
    producer = 'seedance:' + read(request_path.parent / 'producer.json')['model']
    status = human_video.register(manifest, request_path, request_path.parent / 'video.mp4', verdict, producer)
    rewrite(request_path.parent / 'registered.json', {'status': status, 'verdict': verdict})
    return {'registered': status, 'problems': problems}


def run(manifest, budget_cny, model='2.0', resolution='720p', limit=None, gpu=0, requests_dir=None):
    """Prepare request packages for pending programs, then produce and register videos one at a time."""
    manifest = Path(manifest).resolve(); requests_dir = Path(requests_dir or manifest.parent / 'human_videos')
    human_video.prepare(manifest, requests_dir)
    ledger = Ledger(requests_dir / 'ledger.jsonl'); rows = []; reconcile(ledger)
    todo = sorted(requests_dir.glob('*/request.json'), key=lambda p: (len(read(p)['steps']), p.parent.name))  # cheap first
    for path in todo[:limit]:
        try:
            row = generate(path, ledger, model, resolution, budget_cny, gpu)
        except BudgetExhausted as error:
            rows.append({'request_id': path.parent.name, 'status': 'budget_exhausted', 'detail': str(error)}); break
        if row['status'] in ('succeeded', 'exists') and not (path.parent / 'registered.json').exists():
            row = {**row, **verify(manifest, path)}
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    _, spent = ledger.state()
    return {'requests': len(todo), 'attempted': len(rows), 'spent_cny': round(spent, 2),
            'statuses': {s: sum(r['status'] == s for r in rows) for s in {r['status'] for r in rows}}, 'ledger': str(ledger.path)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    v = sub.add_parser('recheck', help='re-run the automatic checks on every produced video')
    v.add_argument('--manifest', type=Path, required=True); v.add_argument('--requests-dir', type=Path)
    r = sub.add_parser('run'); r.add_argument('--manifest', type=Path, required=True); r.add_argument('--budget-cny', type=float, required=True)
    r.add_argument('--model', default='2.0', choices=sorted(MODELS)); r.add_argument('--resolution', default='720p')
    r.add_argument('--limit', type=int); r.add_argument('--gpu', type=int, default=0); r.add_argument('--requests-dir', type=Path)
    a = p.parse_args()
    if a.command == 'check':  # authenticated, free: list one task
        data = call('GET', '/contents/generations/tasks?page_num=1&page_size=1')
        print(json.dumps({'auth': 'ok', 'tasks_total': data.get('total')}))
    elif a.command == 'recheck':
        directory = a.requests_dir or a.manifest.parent / 'human_videos'
        rows = [{'request_id': v.parent.name, **verify(a.manifest.resolve(), v.parent / 'request.json')}
                for v in sorted(directory.glob('*/video.mp4'))]
        print(json.dumps({'videos': len(rows), 'registered': {s: sum(r['registered'] == s for r in rows) for s in {r['registered'] for r in rows}},
                          'problems': [r for r in rows if r['problems']]}, ensure_ascii=False, indent=1))
    else:
        print(json.dumps(run(a.manifest, a.budget_cny, a.model, a.resolution, a.limit, a.gpu, a.requests_dir), ensure_ascii=False, indent=1))
