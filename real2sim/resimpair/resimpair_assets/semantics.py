"""Metadata role conformance, separate from already bound geometric evidence."""
import argparse,json,sqlite3
from pathlib import Path
from resimpair_grasp.common import binding,read,resolve,write


def reconcile(catalog,index,output):
    if output.exists():raise ValueError('Metadata revision exists')
    output.mkdir(parents=True);admitted=[];rejected=[];changes=[];index_b=binding(index)
    with sqlite3.connect(f'file:{index.resolve()}?mode=ro',uri=True) as db:
        for b in read(catalog)['assets']:
            d=read(resolve(b));row=db.execute('SELECT eligible,classification,reason FROM assets WHERE id=?',(d['source_id'],)).fetchone()
            if not row or not row[0]:
                rejected.append({'asset':b,'source_id':d['source_id'],'reason':row[2] if row else 'not_in_index',
                                 'scope':'reject role/scale interpretation, not a retroactive geometry failure'});continue
            c=json.loads(row[1]);v={**d,'previous_descriptor':b,'category':c['category'],'semantic_evidence':c,
                'semantic_role_eligible':not c.get('semantic_ambiguities'),
                'semantic_status':'unambiguous_metadata_candidate' if not c.get('semantic_ambiguities') else 'ambiguous_metadata',
                'semantic_revision':{'index_sha256':index_b['sha256'],'geometry_and_scale_preserved':True,
                                     'visual_confirmation_inferred':False}}
            for k in ['visual','collisions','dimensions_m','half_size','source_to_asset','mass_kg','density_kg_m3','roles']:
                assert v[k]==d[k]
            p=output/(d['id']+'.json');write(p,v);admitted.append(binding(p))
            if d['category']!=v['category'] or not v['semantic_role_eligible']:
                changes.append({'source_id':d['source_id'],'old_category':d['category'],'category':v['category'],'semantic_status':v['semantic_status']})
    result={'schema':'resimpair_mesh_catalog_v1','assets':admitted,'failures':read(catalog).get('failures',[]),
            'semantic_rejections':rejected,'semantic_changes':changes,'previous_catalog':binding(catalog),
            'classification_index':index_b,'scope':'same geometry; role/metadata admission revised, not new physical executions','api_calls':0}
    write(output/'catalog.json',result)
    return {'catalog':str(output/'catalog.json'),'admitted':len(admitted),'semantic_rejected':len(rejected),'changed_or_ambiguous':len(changes)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(reconcile(a.catalog,a.index,a.output),indent=2))
