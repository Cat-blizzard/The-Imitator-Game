"""Current source-role/scale gate separate from native geometric grasp tests."""
import numpy as np
import trimesh
from resimpair_grasp.common import read,resolve


def inspect(entry,state):
    d=read(resolve(entry['asset']));reasons=[]
    if state.get('semantic_admission')!='engineered_role_and_scale_candidate':reasons.append('role_scale_evidence_pending')
    if state.get('role')!='manipulation':reasons.append('current_source_role_not_dynamic')
    if d.get('semantic_role')!='manipulation':reasons.append('descriptor_not_dynamic')
    if entry['source_id']!=d['source_id'] or state['source_id']!=d['source_id']:reasons.append('source_identity_mismatch')
    if not isinstance(d.get('size_provenance'),dict) or not isinstance(d.get('mass_provenance'),dict):
        reasons.append('unstructured_legacy_size_or_mass_provenance')
    mesh=trimesh.load(resolve(d['visual']),force='scene',process=False).to_geometry()
    longest=float(max(mesh.bounding_box_oriented.primitive.extents))
    expected=state.get('base_longest_m');ratio=longest/expected if expected else None
    if ratio is None or not .9-1e-4<=ratio<=1.1+1e-4:reasons.append('geometry_not_within_current_source_base_plusminus10percent')
    native=read(resolve(d['native_properties']))
    if (not np.isfinite(d['mass_kg']) or not .001<=d['mass_kg']<=3
            or not np.isfinite(native['inertia_kg_m2']).all()
            or abs(native['mass_kg']-d['mass_kg'])>1e-6):reasons.append('native_numeric_mass_or_inertia_invalid')
    return {'passed':not reasons,'reasons':reasons,'current_profile':state.get('profile'),
        'current_role':state.get('role'),'current_policy':state.get('policy'),
        'actual_longest_obb_m':longest,'current_base_longest_m':expected,
        'ratio_to_current_base':ratio,'size_confidence':state.get('size_confidence'),
        'semantic_truth_verified':False,'physical_execution_success_inferred':False}
