"""Freeze the v4 native test and make its material/success scope explicit."""
from pathlib import Path
from resimpair_grasp.common import binding, read, resolve, write
from . import metric_native_v4 as retained


def compile_variant(descriptor_path, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    result=retained.compile_variant(descriptor_path,output/'retained_test')
    d=read(resolve(result['asset']));support=read(resolve(d['native_support']))
    support.update(retained_support=d['native_support'],
        physical_stability=('verified_only_for_bound_table_static_support_test'
                            if support['status']=='pass' else 'failed_bound_support_test'),
        material={'static_friction':.6,'dynamic_friction':.6,'restitution':0.},
        gravity_m_s2=[0,0,-9.81],settle_duration_seconds=1.5,
        grasp_transport_or_placement_success_implied=False,
        actual_scene_material_and_live_mass_properties_must_be_recorded_separately=True)
    write(output/'native_support.json',support)
    d.update(native_support=binding(output/'native_support.json'),
        support_scope_adapter=binding(__file__),retained_native_descriptor=result['asset'])
    write(output/'descriptor.json',d)
    result={**result,'schema':'metric_native_support_scope_v5','asset':binding(output/'descriptor.json'),
        'native_support':binding(output/'native_support.json')}
    write(output/'result.json',result)
    return result
