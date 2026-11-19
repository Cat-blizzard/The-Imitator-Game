"""Build a consumer-ready snapshot from completed metric asset/cache stages."""
import argparse
from collections import Counter
from pathlib import Path
from resimpair_grasp.common import binding,read,resolve,write
from resimpair_grasp.grasps import load as load_cache


def build(batch,cudex,containers,output):
    batch,cudex,containers,output=map(Path,[batch,cudex,containers,output])
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    states=read(batch/'source_state_index.json');entries=[];static=[];background=[]
    by_source={e['source_id']:e for e in states['entries']}
    for state in states['entries']:
        for e in state.get('variants',[]):
            d=read(resolve(e['asset']))
            if e.get('role')!='manipulation':
                static.append(e['asset'])
                if e.get('role')=='background':background.append(e['asset'])
            elif e.get('geometry_qualified'):static.append(e['asset'])
            if e.get('retained',0)>0:entries.append(e)
    entries+=read(cudex)['entries'];unique={}
    for e in entries:
        key=(e['source_id'],e['variant_id'])
        if key in unique and unique[key]['asset']!=e['asset']:
            raise ValueError('same_source_variant_has_different_descriptor_binding')
        d=read(resolve(e['asset']));cache=load_cache(e['cache'],e['asset'])
        if d['source_id']!=e['source_id'] or len(cache['candidates'])!=e['retained']:
            raise ValueError('cache_source_or_count_mismatch')
        if read(resolve(e['native_support']))['status']!='pass':
            raise ValueError('executable_index_requires_passed_native_support')
        unique[key]=e
        if e['source_id'] in by_source:
            by_source[e['source_id']].setdefault('executable_variants',[]).append({
                'variant_id':e['variant_id'],'asset':e['asset'],'cache':e['cache'],
                'retained':e['retained'],'physical_status':'unmeasured'})
    catalog=read(containers)
    for row in catalog['rows']:
        if row['source_id'] in by_source:
            by_source[row['source_id']].setdefault('metric_container_results',[]).append(row)
    entries=list(unique.values())
    write(output/'native_cache_index.json',{'schema':'resimpair_grasp_cache_index_v1','entries':entries})
    write(output/'containers.json',catalog)
    write(output/'static_catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':static})
    write(output/'background_catalog.json',{'schema':'resimpair_mesh_catalog_v1','assets':background})
    write(output/'source_state_index.json',states)
    summary={'raw_source_rows':len(states['entries']),'executable_grasp_variants':len(entries),
        'executable_grasp_source_ids':len({e['source_id'] for e in entries}),
        'poses':sum(e['retained'] for e in entries),'metric_container_geometry':len(catalog['assets']),
        'static_descriptors':len(static),'background_descriptors':len(background),
        'remaining_states':dict(Counter(e.get('batch_status',e['status']) for e in states['entries'])),
        'all_raw_coverage_is_not_all_raw_executable':True,'grasp_execution_success_inferred':False,
        'independent_physics_qualification_still_required':True}
    bundle={'schema':'source_metric_executable_bundle_v1','summary':summary,
        'native_cache_index':binding(output/'native_cache_index.json'),
        'containers':binding(output/'containers.json'),'static_catalog':binding(output/'static_catalog.json'),
        'background_catalog':binding(output/'background_catalog.json'),
        'source_state_index':binding(output/'source_state_index.json'),
        'inputs':{'batch_state':binding(batch/'source_state_index.json'),'new_cudex':binding(cudex),
            'containers':binding(containers)},'implementation':binding(__file__),'API_calls':0}
    write(output/'BUNDLE.json',bundle);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['batch','cudex','containers','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(build(a.batch,a.cudex,a.containers,a.output))
