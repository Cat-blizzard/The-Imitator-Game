"""Task-load proposal ranking for the native Panda; never a success label.

cuDex-graspvla's parallel-gripper configuration optimizes a DFC residual.
This independent CPU implementation adds an explicit, bounded static load
model. It does not import cuDex, reproduce its optimizer, or certify dynamics.
Point-contact and soft-contact assumptions are reported separately.
"""
import numpy as np
from scipy.optimize import linprog

VERSION = 'native_panda_task_load_proposals_v1'


def contacts(points, outward_normals):
    p = np.asarray(points, dtype=float)
    n = np.asarray(outward_normals, dtype=float)
    if p.shape != (2, 3) or n.shape != (2, 3) or not np.isfinite([p, n]).all():
        raise ValueError('two_finite_contacts_required')
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-8) or np.linalg.norm(p[0]-p[1]) < 1e-6:
        raise ValueError('degenerate_contact_pair')
    return p, -n / lengths[:, None]


def dfc_residual(points, outward_normals, center, length):
    """Equal unit inward normals, with torques normalized by object length."""
    p, n = contacts(points, outward_normals)
    if length <= 0: raise ValueError('positive_length_required')
    wrench = np.r_[n.sum(axis=0), np.cross((p-center)/length, n).sum(axis=0)]
    return float(np.linalg.norm(wrench))


def load_capacity(points, outward_normals, center, force, mu, pad_radius=0.):
    """Maximum multiple of force with <= 1 N normal force PER finger.

    An inscribed eight-ray friction pyramid is used. Soft-contact generators
    share their normal-force budget with tangential generators, rather than
    independently assigning both their maximum friction. Torque is balanced
    about the supplied center. The contact patch radius is an assumption, not
    a measured pad/contact area. No arbitrary torque robustness is claimed.
    """
    p, n = contacts(points, outward_normals)
    center, force = np.asarray(center, float), np.asarray(force, float)
    if center.shape != (3,) or force.shape != (3,) or not np.isfinite([center, force]).all():
        raise ValueError('finite_center_and_load_required')
    if np.linalg.norm(force) < 1e-8 or not 0 < mu <= 1 or pad_radius < 0:
        raise ValueError('invalid_contact_model')
    length = max(float(np.linalg.norm(p[0]-p[1])), .01)
    columns = []; owners = []
    for i in range(2):
        ref = np.eye(3)[int(np.argmin(np.abs(n[i])))]
        u = np.cross(n[i], ref); u /= np.linalg.norm(u)
        v = np.cross(n[i], u)
        for theta in np.arange(8)*np.pi/4:
            f = n[i] + mu*(np.cos(theta)*u + np.sin(theta)*v)
            columns.append(np.r_[f, np.cross(p[i]-center, f)/length]); owners.append(i)
        if pad_radius:
            for sign in [-1, 1]:
                torque = np.cross(p[i]-center, n[i]) + sign*mu*pad_radius*n[i]
                columns.append(np.r_[n[i], torque/length]); owners.append(i)
    W = np.asarray(columns).T
    eq = np.column_stack([W, -np.r_[force, np.zeros(3)]])
    upper = np.zeros((2, W.shape[1]+1))
    for j, owner in enumerate(owners): upper[owner, j] = 1.
    objective = np.zeros(W.shape[1]+1); objective[-1] = -1.
    result = linprog(objective, A_ub=upper, b_ub=np.ones(2),
                     A_eq=eq, b_eq=np.zeros(6), bounds=(0, None), method='highs')
    if not result.success:
        return {'capacity': None, 'solver_status': int(result.status), 'valid': False}
    equality_error = float(np.max(np.abs(eq @ result.x)))
    upper_error = float(max(0., np.max(upper @ result.x - 1.)))
    valid = equality_error <= 1e-7 and upper_error <= 1e-7
    return {'capacity': max(0., float(result.x[-1])) if valid else None,
            'solver_status': int(result.status), 'valid': valid,
            'equilibrium_residual': equality_error, 'force_bound_residual': upper_error}


def assess(points, normals, center, length, profile):
    """Finite load/COM cases; score must not be used as task acceptance."""
    center = np.asarray(center, float)
    radius = profile['assumed_contact_patch_radius_m']
    worst_mu = min(profile['friction_coefficients'])
    force_cases = [np.asarray(x, float) for x in profile['load_directions']]
    rows = []
    for mu in profile['friction_coefficients']:
        for force in force_cases:
            rows.append({'mu': mu, 'force': force.tolist(), 'com_offset_m': [0., 0., 0.],
                         **load_capacity(points, normals, center, force, mu, radius)})
    for delta in profile['com_offsets_m']:
        if not np.any(delta): continue
        rows.append({'mu': worst_mu, 'force': [0., 0., 1.], 'com_offset_m': delta,
                     **load_capacity(points, normals, center+delta, [0, 0, 1], worst_mu, radius)})
    point = load_capacity(points, normals, center, [0, 0, 1], worst_mu, 0.)
    valid = all(x['valid'] for x in rows)
    return {'version': VERSION, 'model_valid': valid,
            'worst_load_capacity': min(x['capacity'] for x in rows) if valid else None,
            'point_contact_vertical_capacity': point['capacity'],
            'dfc_residual': dfc_residual(points, normals, center, length),
            'assumed_contact_patch_radius_m': radius, 'normal_force_budget_per_finger_N': 1.,
            'cases': rows, 'physical_status': 'unmeasured',
            'scope': 'static proxy only; friction, pad area and COM uncertainty are assumed'}


def quality_key(candidate):
    q = candidate['task_load_quality']
    if not q['model_valid'] or q['worst_load_capacity'] is None:
        return (float('inf'), float('inf'), candidate['id'])
    return (-q['worst_load_capacity'], q['dfc_residual'], candidate['id'])
