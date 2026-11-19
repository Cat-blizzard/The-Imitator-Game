"""Actual table bounds as a conservative static collision object in old MPlib."""
import numpy as np


def install(planner,env):
    import mplib
    table=env.table_scene.table
    mesh=table.get_first_collision_mesh(to_world_frame=True)
    bounds=np.asarray(mesh.bounds,dtype=np.float64).copy()
    actual=bounds.copy()
    # The table's top is the mounting plane of the fixed Panda base. Old MPlib
    # has no external-object ACM; erode just 1 mm below that plane so fixed
    # base contact is allowed. Moving links still cannot pass through the table.
    bounds[1,2]-=.001
    if np.any(bounds[1]<=bounds[0]):raise ValueError('Invalid physical table bounds')
    shape=mplib.collision_detection.fcl.Box(bounds[1]-bounds[0])
    obj=mplib.collision_detection.fcl.CollisionObject(shape,bounds.mean(0),np.array([1.,0.,0.,0.]))
    planner.planner.planning_world.set_normal_object('physical_table_volume',obj)
    return dict(actual_world_bounds_m=actual.tolist(),planning_world_bounds_m=bounds.tolist(),
        base_mount_plane_erosion_m=.001,
        method='Solid AABB of the actual table collision mesh; conservative inclusion of space between its legs.',
        physical_mesh_contact_offsets_and_success_rules_unchanged=True)


def first_command_collision(planner,positions,measured_qpos):
    """Check actual 7-joint command targets against the installed table box.

    Timing interpolation and the contact-step IK can introduce targets outside
    the discrete RRT path. Check those targets too; do not trust planning success
    alone. Fingers retain their measured configuration, matching a held aperture.
    This changes only the planner's collision model and restores it afterwards.
    """
    robot=planner.planner.robot
    world=planner.planner.planning_world
    measured=np.asarray(measured_qpos,dtype=np.float64).reshape(9)
    commands=np.asarray(positions,dtype=np.float64).reshape(-1,7)
    try:
        for frame,command in enumerate(commands):
            q=np.r_[command,measured[7:]]
            robot.set_qpos(q,True)
            world.set_qpos_all(q[:7])
            pairs=[dict(object1=item.object_name1,object2=item.object_name2,
                        link1=item.link_name1,link2=item.link_name2)
                   for item in world.collide_with_others()
                   if 'physical_table_volume' in (item.object_name1,item.object_name2)]
            if pairs:return dict(frame=frame,pairs=pairs)
    finally:
        robot.set_qpos(measured,True)
        world.set_qpos_all(measured[:7])
    return None
