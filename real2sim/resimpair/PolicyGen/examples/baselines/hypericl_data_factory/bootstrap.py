"""Read-only audit of existing releases and running collectors; new artifacts only."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/mnt/sdf/datasets/hypericl_data_factory_20261016_v1')
WORKSPACE = Path('/mnt/sdf/dataset/zhouxunzhe/zhiqi')
OBJECTIVE = Path('/home/xzzhou/.codex/attachments/fd68de68-1c46-4cee-981a-4059774a4a3c/pasted-text-1.txt')
PRIOR = Path('/mnt/sdf/datasets/hypericl_ego_alignment_20261016_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        stat = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        cmd = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        return dict(pid=pid, state=stat[0], start_ticks=int(stat[19]), command=cmd,
                    live=stat[0] not in ('Z', 'X'))
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, live=False)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / 'OBJECTIVE.txt').exists():
        (ROOT / 'OBJECTIVE.txt').write_bytes(OBJECTIVE.read_bytes())
    assert sha(ROOT / 'OBJECTIVE.txt') == sha(OBJECTIVE)
    migration = Path('/mnt/sdc/datasets/_migration_20261016_ego_hot3d_v1/COMPLETION.json')
    m = json.loads(migration.read_text())
    links = {}
    for name in ('egodex', 'hot3d_clips', 'agibot_world'):
        p = Path('/mnt/sdf/datasets') / name
        links[name] = dict(path=str(p), is_symlink=p.is_symlink(), target=str(p.resolve()), exists=p.exists())
    scale = Path('/mnt/sdf/datasets/hypericl_ego_scaleup_20261016_v1/state.json')
    s = json.loads(scale.read_text())
    jobs = []
    for j in s.get('active_jobs', []):
        st = j.get('state', {})
        jobs.append(dict(name=j['name'], state=st, observed_process=process(st.get('pid', -1))))
    pins = {}
    for p in [PRIOR/'goal_release_v1/COMPLETION.json', PRIOR/'goal_release_v1/policy_manifest.json',
              PRIOR/'human_motion_decoder_ready_v1/motion/manifest.json', PRIOR/'full_screen_v1/metrics.jsonl',
              Path('/mnt/sdf/datasets/hypericl_sim_inventory_20261016_v1/tasks.csv')]:
        pins[str(p)] = dict(sha256=sha(p), bytes=p.stat().st_size)
    simulator = WORKSPACE/'simulator-main'
    sources = ['new_pipeline/tasks/template_task.py', 'new_pipeline/tasks/base_config_task.py',
               'new_pipeline/tasks/lid_pan.py', 'new_pipeline/scene_randomizer.py', 'add_objects.md']
    review = dict(source_root=str(simulator), files={n:sha(simulator/n) for n in sources},
        reusable_design=['Task-level layout templates and per-scene expert programs',
                         'Targets expressed relative to randomized container poses',
                         'Material/light/camera sampling independent of task goals',
                         'Stable pose, scale, grasp qualification before production'],
        missing_external_assets=['External object mesh/grasp dataset paths', 'External vMaterials library'],
        action='Reuse organization and adapt to the current SAPIEN/Panda collectors; no engine migration.',
        template_success_requires_real_override=True)
    write(ROOT/'simulator_review.json', review)
    report = dict(observed_unix=time.time(), objective_sha256=sha(OBJECTIVE), migration=m,
        datasets=links, storage={str(p):dict(zip(('total','used','free'),shutil.disk_usage(p)))
                                for p in ('/mnt/sdf','/mnt/sdc','/mnt/sdb')},
        prior_release_pins=pins, prior_scaleup_status=s['status'], live_jobs=jobs,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu',
                                     '--format=csv,noheader'],text=True).strip().splitlines(),
        existing_collectors_modified=False, benchmark_modified=False, model_training_started=False)
    write(ROOT/'initial_audit.json', report)
    requirements = {
        'existing_work_audit': dict(status='verified_initial_snapshot', evidence='initial_audit.json'),
        'simulator_task_design_review': dict(status='verified_source_review', evidence='simulator_review.json'),
        'batch_human_quality_and_task_search': dict(status='in_progress'),
        'separate_training_purpose_eligibility': dict(status='pending'),
        'stratified_video_and_semantic_review': dict(status='pending'),
        'task_driven_asset_and_background_acquisition': dict(status='pending'),
        'geometry_collision_and_interaction_qualification': dict(status='pending'),
        'reusable_task_programs_and_replanning': dict(status='pending'),
        'layered_randomization_and_negative_controls': dict(status='pending'),
        'qualified_batch_production_and_failure_accounting': dict(status='pending'),
        'eef_native_human_coordinates_and_time_checks': dict(status='pending'),
        'four_training_routes_and_split_leakage_checks': dict(status='pending'),
        'final_counts_samples_report_and_sampling_recommendation': dict(status='pending')}
    if not (ROOT/'requirements.json').exists():
        write(ROOT/'requirements.json',dict(original_scope_preserved=True, requirements=requirements,
              no_training_or_architecture_changes=True, completion_proven=False))
    print(json.dumps(dict(root=str(ROOT), migration_passed=m['passed'],
        sdf_free_gib=report['storage']['/mnt/sdf']['free']/2**30,
        live_jobs=[j['name'] for j in jobs if j['observed_process']['live']]),ensure_ascii=False))


if __name__ == '__main__':
    main()
