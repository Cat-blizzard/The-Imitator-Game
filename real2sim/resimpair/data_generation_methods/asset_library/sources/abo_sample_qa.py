"""Actual CPU geometry and embedded-image checks of completed ABO models."""
import argparse
import collections
import json
from pathlib import Path
import time
import numpy as np
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/sdg/datasets/open_assets/household_scaleup_20261020_v1/abo'))
    parser.add_argument('--maximum',type=int,default=16)
    args=parser.parse_args()
    rows=[json.loads(s) for s in (args.root/'catalog.jsonl').read_text().splitlines()]
    grouped=collections.defaultdict(list)
    for r in rows:grouped[tuple(r['categories'])].append(r)
    # Sample across product types before filling with further members.
    chosen=[]
    while grouped and len(chosen)<args.maximum:
        for c in list(grouped):
            chosen.append(grouped[c].pop(0))
            if not grouped[c]:del grouped[c]
            if len(chosen)>=args.maximum:break
    results=[]
    for r in chosen:
        result=dict(source_id=r['source_id'],name=r['name'],categories=r['categories'],
            local_path=r['geometry_path'],physics_qualified=False)
        try:
            scene=trimesh.load(r['geometry_path'],force='scene',process=False)
            finite=bool(len(scene.geometry) and np.isfinite(scene.bounds).all())
            vertices=faces=images=0
            for m in scene.geometry.values():
                vertices+=len(m.vertices);faces+=len(m.faces)
                finite &= bool(np.isfinite(m.vertices).all())
                if len(m.faces) and (m.faces.min()<0 or m.faces.max()>=len(m.vertices)):
                    raise ValueError('Mesh contains invalid indices')
                mat=getattr(m.visual,'material',None)
                for field in ('baseColorTexture','normalTexture','metallicRoughnessTexture','emissiveTexture','occlusionTexture','image'):
                    im=getattr(mat,field,None)
                    if im is not None:im.load();images+=1
            extent=scene.bounds[1]-scene.bounds[0]
            if not finite or not faces or not (extent>0).all():raise ValueError('Nonfinite or empty geometry')
            result.update(status='trimesh_and_embedded_images_readable',geometry_count=len(scene.geometry),
                vertices=vertices,faces=faces,decoded_material_images=images,bounds_raw_units=scene.bounds.tolist())
        except Exception as exc:result.update(status='failed',error=repr(exc))
        results.append(result)
    destination=args.root/'sample_geometry_checks.jsonl'
    temporary=destination.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results));temporary.replace(destination)
    report=dict(checked=len(results),passed=sum(r['status']!='failed' for r in results),
        time_unix=time.time(),scope='Category-spread sample of downloaded assets, not whole library physics qualification',
        categories=sorted({c for r in results for c in r['categories']}))
    (args.root/'sample_QA_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if report['passed']!=report['checked']:raise SystemExit(1)


if __name__=='__main__':main()
