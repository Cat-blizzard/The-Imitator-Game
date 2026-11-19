# Bound receiving-surface catalog

`container_affordance_consumer_v1.py` consumes the immutable 172-variant sampler
catalog and its independent native-support census. It exports a strict subset
using the **unchanged 97-ray support decision**, without editing descriptors,
meshes, source IDs, metric sizes, partitions, or existing tasks. All excluded
variants and complete rejection reasons remain in `rejections.json`.

The generated `sampler_catalog.json` can be passed directly to
`metric_population_sampler_v4.build(..., container_catalog=path)`, or its
`--container-catalog` CLI argument. A catalog change is part of v4's generation
contract and therefore changes the scene ID. Existing frozen cohorts stay intact.

`measured_affordance_index.json` retains **all 172 variants**, including failures.
`measured_affordances.csv` is the readable view. Measurements include body height,
floor height/body-height ratio, cavity depth, flat-floor radius, rim-to-body-top
margin, and flat-floor-to-wall radial margin. Payload-to-rim margin is initially
unknown because it requires a specific object and goal.

The independent descriptive bins are copied verbatim: shallow tray-like cavity,
elevated shallow recess/support, deeper opening candidate, and intermediate
unknown geometry. Their numeric boundaries are descriptive, not new admission
thresholds. Source names and the bins are **not semantic ground truth**.

`lookup(index_path, exact_descriptor_binding)` resolves a row without choosing a
different variant. `describe_goal(row, payload_height_m=..., payload=..., task=...)`
returns a task/R2H sidecar with nominal top-to-rim clearance and protrusion.
It never rewrites the task. `geometry_fallback=on_support_surface` allows a caller
to express only the measured receiving-surface geometry. `place_relation.value`
and semantic assertions remain unknown in this version; in particular a measured
deep opening, a positive height margin, or an encouraging source title does not
authorize `into_open_container`. The index records missing XY fit, opening/swept
clearance and bound visual/task semantic confirmation for a later evidence gate.

Native support qualification is not a certificate of full opening clearance,
stable dynamic release, whole-object containment, or generated-video success.
Objects may legitimately protrude above a shallow receptacle. This module makes
no API calls and starts no physics or GPU jobs.
