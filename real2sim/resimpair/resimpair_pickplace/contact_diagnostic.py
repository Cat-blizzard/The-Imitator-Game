"""Opt-in, read-only native contact attribution for bounded release diagnostics."""
import numpy as np

from resimpair_grasp.common import write


def install(rec):
    """Observe existing physics callbacks without changing actions or contacts."""
    env = rec.unwrapped
    original = env._after_simulation_step
    rows = []
    peak = {"force_N": 0.}
    counter = 0

    def observe():
        nonlocal counter
        original()
        counter += 1
        if rec.stage.split(":")[-1] not in {"open", "retreat", "return"}:
            return
        for contact in env.scene.get_contacts():
            entities = [body.entity for body in contact.bodies]
            robot = next((i for i, e in enumerate(entities) if e in env.robot_bodies), None)
            obj = next((i for i, e in enumerate(entities) if e in env.object_bodies), None)
            if robot is None or obj is None:
                continue
            force = sum(float(np.linalg.norm(p.impulse)) for p in contact.points) * env.sim_freq
            if force < .01:
                continue
            row = {"control_frame": len(rec.rows), "substep": counter, "stage": rec.stage,
                   "force_N": force, "bodies": [e.name for e in entities],
                   "arm": env.robot_bodies[entities[robot]], "object": env.object_bodies[entities[obj]],
                   "body_world": [e.pose.to_transformation_matrix().tolist() for e in entities],
                   "shapes": [{"type": type(s).__name__, "contact_offset": s.contact_offset,
                               "rest_offset": s.rest_offset,
                               "local_pose": s.local_pose.to_transformation_matrix().tolist()}
                              for s in contact.shapes],
                   "points": [{"position": p.position.tolist(), "normal": p.normal.tolist(),
                               "impulse": p.impulse.tolist(), "separation": p.separation}
                              for p in contact.points]}
            if len(rows) < 5000:
                rows.append(row)
            if force > peak['force_N']:
                peak.clear(); peak.update(row)
                peak['shape_meshes'] = [
                    {"vertices": s.vertices.tolist(), "triangles": s.triangles.tolist(),
                     "scale": s.scale.tolist()} if hasattr(s, 'vertices') else None
                    for s in contact.shapes]

    env._after_simulation_step = observe

    def persist():
        write(rec.directory / 'native_release_contacts.json',
              {"observer_only": True, "maximum_rows": 5000, "rows": rows, "peak": peak})

    return persist
