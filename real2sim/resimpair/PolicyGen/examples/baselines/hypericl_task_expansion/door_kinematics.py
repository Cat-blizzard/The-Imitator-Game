"""Pure kinematic candidates; actual native grasp/collision checks remain mandatory."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..hypericlzq.eef import panda_chain, panda_ik


def rotate_about_hinge(T, pivot, axis, delta, orientation_follow=1.):
    R = Rotation.from_rotvec(axis * delta).as_matrix()
    result = T.copy()
    result[:3, :3] = Rotation.from_rotvec(axis * delta * orientation_follow).as_matrix() @ T[:3, :3]
    result[:3, 3] = pivot + R @ (T[:3, 3] - pivot)
    return result


def grasp_arc_candidates(grasp, robot_root, pivot, axis, delta, current_q, orientation_follow=1.):
    bounds = np.asarray([b for _, active, b in panda_chain() if active])
    rng = np.random.default_rng(4959600)
    seeds = np.concatenate([current_q[None], rng.uniform(bounds[:,0]+.1, bounds[:,1]-.1, (47,7))])
    inverse = np.linalg.inv(robot_root)
    solutions, ok, _ = panda_ik(np.broadcast_to(inverse@grasp, (len(seeds),4,4)), seeds)
    solutions = solutions[ok]
    if not len(solutions):
        return []
    margins = np.min(np.minimum(solutions-bounds[:,0], bounds[:,1]-solutions),axis=1)
    results = []
    for q in solutions[np.argsort(-margins)[:12]]:
        initial=q.copy();path=[q.copy()];margin=float(np.min(np.minimum(q-bounds[:,0],bounds[:,1]-q)))
        for step in np.linspace(0.,delta,45)[1:]:
            new,good,_=panda_ik(inverse@rotate_about_hinge(grasp,pivot,axis,step,orientation_follow),q)
            if not good or np.max(np.abs(new-q))>=.2:
                break
            q=new;path.append(q.copy());margin=min(margin,float(np.min(np.minimum(q-bounds[:,0],bounds[:,1]-q))))
        if len(path)==45:
            results.append(dict(initial_q=initial, qpath=np.stack(path), min_joint_margin_rad=margin))
    return sorted(results,key=lambda x:-x['min_joint_margin_rad'])
