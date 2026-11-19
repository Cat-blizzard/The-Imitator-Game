# Current measured receiving-surface consumer

Use `container_affordance_consumer_v2.py` and the generated
`runs/container_affordance_consumer_20261030_v2/sampler_catalog.json` with sampler4.
Its 159 descriptors are byte-for-byte the same strict subset as v1: 148 sources,
using the independent unchanged 97-ray support decision. The 13 rejected
variants remain in the bound v1 rejection report. All 172 measured variants stay
in the index and CSV. Existing physical cohorts and task contracts are untouched.

The subset contains 91 deeper-opening geometry candidates, 12 shallow tray-like
cavities, 36 elevated shallow recess/support surfaces and 20 intermediate cases.
These bins describe measured geometry. They do not certify 148 deep containers.
Semantic `place_relation.value` stays unknown; geometry candidates and the
generic `on_support_surface` geometry fallback are separate fields. No entry is
automatically authorized as `into_open_container` from geometry or title alone.

V2 corrects two metadata limitations found by independent review while preserving
v1 evidence. Source/license metadata may live under `descriptor.provenance`;
the index now preserves it and records its location. A caller-supplied nominal
height is explicitly unverified, even when an optional task file is supplied.

`describe_bound_goal(index, scene_binding, task_binding, operation_id)` derives
the target from the real bound scene/task, measures the original collision
vertices, and reruns the retained payload-floor check on original visual geometry.
It returns a separate task/R2H sidecar with source bindings, nominal object-to-rim
margin, and XY-fit evidence. Nothing is written into the original task. Full
opening/swept clearance, semantic visual confirmation and actual physical success
remain unknown. A protruding object is not automatically declared a failure.
