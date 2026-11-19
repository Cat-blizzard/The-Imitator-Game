"""Closed rolling-index schema plus end-to-end checks of the published fields.

Keep v1/v2 and their real fault regressions immutable. Consumers of this entry
point use RESULT.json only after both producer and final-consumer checks pass.
"""
import argparse
from pathlib import Path
from resimpair_grasp.common import binding,digest,read,resolve,write
from . import raw_candidate_bundle_v2 as retained

ROLLING_FIELDS=frozenset({'source_id','variant_id','asset','cache','retained','method',
    'relative_scale','semantic_scale_admission','source_aliases','source_state',
    'ingestion_group','optimizer_result'})


def rolling_contract(index):
    for entry in index['entries']:
        if set(entry)!=ROLLING_FIELDS:
            raise ValueError('rolling_index_must_not_override_native_producer_fields')


def check_published(root,bundle):
    value=read(bundle);entries=read(resolve(value['native_cache_index']))['entries'];rows=[]
    for entry in entries:
        descriptor=read(resolve(entry['asset']));retained.check_native_properties(entry,descriptor)
        job_path=Path(root)/'gpu'/digest(entry['source_id'])[:24]/'job.json'
        native=read(job_path)['entry']
        for key,expected in native.items():
            if entry.get(key)!=expected:
                raise ValueError('final_consumer_changed_registered_native_field:'+key)
        rows.append({'source_id':entry['source_id'],'asset':entry['asset'],
            'native_support':entry['native_support'],'native_properties':entry['native_properties'],
            'producer_job':binding(job_path),'all_native_fields_equal_producer':True})
    return rows


def publish(raw_root,baseline,container_catalog,output):
    root=Path(raw_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('preserve_existing_consumer_publication')
    rolling_contract(read(root/'native_cache_index.json'))
    value=retained.publish(root,baseline,container_catalog,output/'verified_producers')
    # Recheck the exact captured input and resulting entries, not a later live
    # index. New completed producers may arrive while the snapshot is prepared.
    captured=output/'verified_producers/captured_raw_root/native_cache_index.json'
    try:
        rolling_contract(read(captured))
        rows=check_published(root,resolve(value['bundle']))
    except Exception as error:
        write(output/'REJECTED.json',{'error':repr(error),'published_for_task_use':False,
            'preserved_attempt':str(output/'verified_producers')})
        raise
    write(output/'CONSUMER_FIELD_CHECKS.json',{'rows':rows,'exact_captured_input':binding(captured),
        'implementation':binding(__file__),'producer_checker':binding(retained.__file__),
        'final_entry_properties_bound_to_same_source':True,'original_geometry_rules_unchanged':True})
    result={**value,'consumer_field_checks':binding(output/'CONSUMER_FIELD_CHECKS.json'),
        'publication_integrity_passed':True,'task_execution_quality':'unmeasured'}
    write(output/'RESULT.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw-root','baseline','container-catalog','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(publish(a.raw_root,a.baseline,a.container_catalog,a.output))
