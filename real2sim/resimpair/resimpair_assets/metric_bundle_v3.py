"""Latest source plan must admit each cache, independent of geometric success."""
import argparse
from pathlib import Path
from collections import Counter
import trimesh
from resimpair_grasp.common import binding,read,resolve,write
from resimpair_grasp.grasps import load
from .metric_admission_v1 import inspect


def build(state_index,old_bundle,cudex,small_containers,large_containers,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);states=read(state_index);by_source={e['source_id']:e for e in states['entries']}
    prior=read(Path(old_bundle)/'native_cache_index.json')['entries']
    candidates=prior+read(cudex)['entries'];unique={};audit=[];entries=[]
    for e in candidates:unique[(e['source_id'],e['variant_id'])]=e
    for e in unique.values():
        state=by_source[e['source_id']];gate=inspect(e,state)
        audit.append({'source_id':e['source_id'],'variant_id':e['variant_id'],'asset':e['asset'],
            'name':read(resolve(e['asset']))['name'],'admission':gate})
        if not gate['passed']:continue
        c=load(e['cache'],e['asset'])
        if len(c['candidates'])!=e['retained']:raise ValueError('cache_count_changed')
        entries.append({**e,'semantic_scale_admission':gate})
        state.setdefault('executable_variants',[]).append({'variant_id':e['variant_id'],
            'asset':e['asset'],'cache':e['cache'],'retained':e['retained']})
    static=[];background=[];static_audit=[]
    for ref in read(Path(old_bundle)/'static_catalog.json')['assets']:
        d=read(resolve(ref));state=by_source[d['source_id']]
        # Only descriptors already compiled under a still-current bound policy
        # enter this snapshot. Changed role/scale candidates await recompilation.
        visual=trimesh.load(resolve(d['visual']),force='scene',process=False).to_geometry()
        current_base=state.get('base_longest_m')
        ratio=float(max(visual.bounding_box_oriented.primitive.extents))/current_base if current_base else None
        if state.get('semantic_admission')=='pending' or state.get('role')!=d.get('semantic_role') or ratio is None or not .9-1e-4<=ratio<=1.1+1e-4:
            static_audit.append({'source_id':d['source_id'],'reason':'current_role_or_scale_policy_changed'});continue
        static.append(ref)
        if state['role']=='background':background.append(ref)
    small,large=read(small_containers),read(large_containers)
    container_catalog={'schema':'stationary_container_catalog_v1','assets':small['assets']+large['assets'],
        'rows':small['rows']+large['rows'],'parents':[binding(small_containers),binding(large_containers)],
        'physics_executions':0,'api_calls':0,'source_id_and_partition_preserved':True}
    for name,value in [('native_cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries}),
        ('containers.json',container_catalog),('static_catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':static}),
        ('background_catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':background}),
        ('source_state_index.json',states),('admission_audit.json',{'cache_entries':audit,'static_rejections':static_audit})]:write(output/name,value)
    summary={'raw_sources':len(states['entries']),'native_role_counts':states['summary']['roles'],
        'candidate_caches_checked':len(audit),'admitted_grasp_variants':len(entries),
        'admitted_grasp_sources':len({x['source_id'] for x in entries}),'poses':sum(x['retained'] for x in entries),
        'rejected_cache_reasons':dict(Counter(reason for a in audit for reason in a['admission']['reasons'])),
        'container_variants':len(container_catalog['assets']),'container_sources':len({read(resolve(r))['source_id'] for r in container_catalog['assets']}),
        'static_descriptors':len(static),'background_descriptors':len(background),
        'physical_success_inferred':False}
    write(output/'BUNDLE.json',{'schema':'source_metric_executable_bundle_v3','summary':summary,
        **{key:binding(output/name) for key,name in [('native_cache_index','native_cache_index.json'),('containers','containers.json'),
            ('static_catalog','static_catalog.json'),('background_catalog','background_catalog.json'),('source_state_index','source_state_index.json'),('admission_audit','admission_audit.json')]},
        'latest_source_plan':binding(state_index),'implementation':binding(__file__),
        'source_semantics_checked_against_current_policy':{'schema':'source_semantic_scale_gate_v1','source_state_index':binding(state_index),'gate':binding(Path(__file__).with_name('metric_admission_v1.py')),'all_cache_entries_passed':True},
        'admission_implementation':binding(Path(__file__).with_name('metric_admission_v1.py'))})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['state-index','old-bundle','cudex','small-containers','large-containers','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(build(a.state_index,a.old_bundle,a.cudex,a.small_containers,a.large_containers,a.output))
