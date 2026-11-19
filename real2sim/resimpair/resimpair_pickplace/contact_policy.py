"""Explicit small-tabletop collision-contact profile; legacy remains available.

PhysX's metric default creates contacts at a combined 4 cm separation. This
candidate limits each shape's contact-generation offset to 2 mm. Rest offsets,
collision geometry, materials, mass, drives, timestep and forces are unchanged.
This is a simulator configuration change, not a new acceptance criterion.
"""


def apply(env, policy):
    if policy not in {'native_default', 'small_tabletop_v1'}:
        raise ValueError('Unknown contact policy')
    rows = []
    actors = [*env.objects, *env.support_actors, env.table_scene.table]
    for agent in env.agent.agents:
        actors.extend(agent.robot.get_links())
    seen = set()
    for actor in actors:
        for body in actor._bodies:
            if body.entity in seen:
                continue
            seen.add(body.entity)
            for index, shape in enumerate(body.get_collision_shapes()):
                old = float(shape.contact_offset); rest = float(shape.rest_offset)
                new = min(old, .002) if policy == 'small_tabletop_v1' else old
                if new <= rest:
                    raise ValueError('contact_offset_must_exceed_retained_rest_offset')
                if new != old:
                    shape.set_contact_offset(new)
                rows.append({'body': body.entity.name, 'shape_index': index,
                    'type': type(shape).__name__, 'previous_contact_offset_m': old,
                    'contact_offset_m': float(shape.contact_offset), 'rest_offset_m': rest})
    return {'policy': policy, 'shapes': rows, 'rest_offsets_changed': False,
            'geometry_material_mass_drive_timestep_changed': False,
            'reference': 'https://nvidia-omniverse.github.io/PhysX/physx/5.1.2/docs/AdvancedCollisionDetection.html'}
