"""Snapshot the household library without conflating discovery and usable assets."""
import argparse
import collections
import datetime
import json
from pathlib import Path
import shutil
import time

DEFAULT = Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1')


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def json_rows(path, errors, key=None):
    if not path.exists():
        return []
    try:
        if path.suffix == '.jsonl':
            lines = path.read_bytes().splitlines(keepends=True)
            rows = []
            for i, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if i == len(lines)-1 and not line.endswith(b'\n'):
                        errors.append({'path': str(path), 'warning': 'incomplete_last_append_skipped'})
                    else:
                        raise
            return rows
        data = json.loads(path.read_text())
        return data.get(key, []) if key else data
    except (OSError, ValueError) as exc:
        errors.append({'path': str(path), 'error': str(exc)})
        return []


def snapshot(root):
    errors, records = [], []

    def add(source, row, source_file, source_id, kind, path, *, reused=False, size=None, qa=None, sha=None):
        records.append(dict(asset_id=f'{source}:{source_id}', source=source, source_id=source_id,
            kind=kind, name=row.get('name', source_id), categories=row.get('categories', row.get('category', [])),
            license=row.get('license'), primary_path=path, primary_path_exists=bool(path and Path(path).is_file()),
            sha256=sha or row.get('sha256'), recorded_payload_bytes=size,
            reused_prior_library=reused, validation=qa or {}, source_record_path=str(source_file),
            portable_primary_path=row.get('portable_model_xml'),
            portable_validation={'geometry':row.get('portable_geometry_readability'),
                'missing_dependencies':row.get('portable_dependency_missing')} if row.get('portable_model_xml') else None,
            physics_qualified=False, simulation_qualified=False, training_admitted=False))

    p=root/'objaverse/catalog.jsonl'
    checks={r['uid']: r for r in json_rows(root/'objaverse/sample_geometry_checks.jsonl', errors)}
    for r in json_rows(p, errors):
        add('objaverse',r,p,r['uid'],'model',r.get('geometry_path',r.get('local_path')),
            size=r.get('bytes'),qa={'structure':r.get('geometry_check'), 'CPU_sample':checks.get(r['uid'])})
    p=root/'abo/catalog.jsonl'
    checks={r['source_id']:r for r in json_rows(root/'abo/sample_geometry_checks.jsonl', errors)}
    for r in json_rows(p, errors):
        add('abo',r,p,r['source_id'],'model',r.get('geometry_path'),size=r.get('bytes'),
            qa={'structure':r.get('geometry_check'), 'CPU_sample':checks.get(r['source_id'])})
    p=root/'scanned_objects/objects.jsonl'
    for r in json_rows(p, errors):
        add(r['source'],r,p,r['source_id'],'model',r.get('primary_mesh_path',r.get('local_primarymesh_path')),
            reused=r.get('status')=='existing_local_checked',size=r.get('payload_bytes'),qa=r.get('geometry_check'))
    p=root/'environments/polyhaven/asset_catalog.json'
    for r in json_rows(p, errors, 'records'):
        add('polyhaven',r,p,r['source_asset_id'],r['kind'].lower(),r.get('main_path'),
            size=sum(f.get('bytes',0) for f in r.get('files',[])),
            qa={'geometry':r.get('geometry_readability'),'images':r.get('image_checks')})
    p=root/'environments/robocasa/asset_catalog.json'
    for r in json_rows(p, errors, 'records'):
        add('robocasa',r,p,r['asset_id'].removeprefix('robocasa:'),'MJCF_definition',r.get('model_xml'),
            qa={'geometry':r.get('geometry_readability'),'missing_dependencies':r.get('dependency_missing'),
                'declared_joint_types':r.get('declared_joint_types')},sha=r.get('model_xml_sha256'))

    # Repeat visits to a source ID are updated records, not extra objects.
    records=list({r['asset_id']:r for r in records}.values())
    by_source={}
    for source in sorted({r['source'] for r in records}):
        group=[r for r in records if r['source']==source]
        by_source[source]=dict(records=len(group),by_kind=dict(collections.Counter(r['kind'] for r in group)),
            new_models=sum(r['kind']=='model' and not r['reused_prior_library'] for r in group),
            reused_models=sum(r['kind']=='model' and r['reused_prior_library'] for r in group),
            missing_primary_model_files=sum(r['kind']=='model' and not r['primary_path_exists'] for r in group))
    digests=collections.defaultdict(list)
    for r in records:
        if r['sha256']:
            digests[(r['kind'],r['sha256'])].append(r['asset_id'])
    duplicates=[ids for ids in digests.values() if len(ids)>1]
    physical=partial=files=0
    for p in root.rglob('*'):
        if not p.is_file() or 'index' in p.relative_to(root).parts:
            continue
        try: n=p.stat().st_size
        except FileNotFoundError: continue
        physical+=n;files+=1
        if p.name.endswith('.part'):partial+=n
    disk=shutil.disk_usage(root)
    status=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),
        source_counts=by_source,source_record_count=len(records),actual_file_bytes=physical,
        partial_file_bytes=partial,actual_file_count=files,free_disk_bytes=disk.free,
        exact_primary_file_duplicate_groups=duplicates,errors=errors,
        interpretation='Counts are per source, NOT globally unique geometries. MJCF definitions, reused assets and backgrounds are separate. Archive bytes include preserved originals plus extraction. Structural/CPU readability is not simulation qualification.',
        physics_qualified_assets=0,training_admitted_assets=0)
    output=root/'index'
    atomic(output/'catalog.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    atomic(output/'STATUS.json',json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    lines=['# Household asset library snapshot','',f'Updated (UTC): {status["updated_utc"]}', '',
        f'Root: `{root}`', '', '| Source | New model records | Reused models | Types |', '|---|---:|---:|---|']
    for source,s in by_source.items():
        lines.append(f'| {source} | {s["new_models"]} | {s["reused_models"]} | {json.dumps(s["by_kind"])} |')
    lines+=['',status['interpretation'],'',f'Actual file bytes: {physical:,}; partial bytes: {partial:,}.',
        f'Free disk: {disk.free/2**40:.2f} TiB. Exact primary-file duplicate groups: {len(duplicates)}.',
        '','No asset has yet been admitted into physical simulation or model training.']
    atomic(output/'STATUS.md','\n'.join(lines)+'\n')
    print(json.dumps({k:status[k] for k in ['updated_utc','source_counts','actual_file_bytes','partial_file_bytes','errors']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--watch-seconds',type=float,default=0)
    parser.add_argument('--hours',type=float,default=24)
    args=parser.parse_args()
    deadline=time.monotonic()+args.hours*3600
    while True:
        snapshot(args.root)
        if args.watch_seconds<=0 or time.monotonic()+args.watch_seconds>deadline:break
        time.sleep(args.watch_seconds)


if __name__=='__main__':main()
