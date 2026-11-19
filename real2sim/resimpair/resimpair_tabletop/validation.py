"""Offline evidence checks, frozen independently of physics outcome labels."""
import numpy as np

VERSION='tabletop_pose_evidence_v1'


def check_pose(c,profile,bounds):
    T=np.asarray(c['tcp_in_asset'],float);points=np.asarray(c['contacts_in_asset']);normals=np.asarray(c['contact_normals'])
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('invalid_rigid_pose')
    if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<.999:raise ValueError('invalid_rotation')
    if not np.allclose(T[:3,2],[0,0,-1],atol=1e-6):raise ValueError('not_top_down')
    width=np.linalg.norm(points[1]-points[0]);axis=(points[1]-points[0])/width
    if abs(width-c['width_m'])>1e-6 or not profile['min_width_m']<=width<=profile['max_width_m']:raise ValueError('invalid_contact_width')
    if not np.allclose(T[:3,3],points.mean(0),atol=1e-6) or abs(abs(T[:3,1]@axis)-1)>1e-5:raise ValueError('pose_contact_mismatch')
    cone=1/np.sqrt(1+profile['friction_coefficient']**2)
    if min(-normals[0]@axis,normals[1]@axis)<cone-1e-6:raise ValueError('friction_cones_fail')
    if T[2,3]-bounds[0][2]<profile['min_tcp_height_m']:raise ValueError('insufficient_table_height')
    if c['table_clearance_m']<profile['table_clearance_m']:raise ValueError('gripper_table_clearance_fail')
    if c.get('physical_status')!='unmeasured':raise ValueError('offline_record_cannot_claim_physical_success')
    return True
