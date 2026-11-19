# Declared single-actor geometry candidate (2026-10-21)

Later on 2026-10-21, actual positive g00/g01 preparations exposed an outer/inner
source-splicing defect in preparation v2. Use
[preparation v3](ACTOR_GEOMETRY_PREPARATION_V3.md) for new work. The v1/v2 code,
results and descriptions below are retained as historical scope and regression
evidence; the numerical geometry model is unchanged. Fixed-left follow-up
results are described in the [third-round report](../HARNESS_RESULTS_20261021_V3.md).

`actor_geometry_candidate.py` adds a CPU preflight to the existing unified
anchor candidate. It binds the actual execution, camera, table and selected
object mesh to one synthetic waist/torso, two anatomical shoulders, fixed
two-link arms, and the task's acting-hand assignments. It neither reconstructs
a human from RGB nor certifies human anatomy or generated visual quality.

The old textual `far / upper center / towards camera` position was not a world
placement. In g00 the opposite camera is on +X, so its far edge is the actual
table's -X edge. The table spans about 1.2 m in X and 2 m in Y. Reusing an image
side label without this transform can place a single body beyond its declared
arm workspace.

## Frozen experiment and evidence

Independent numeric and stage protocols are under
`/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/independent_review/`.
They were agreed before running the 12-pose grid. Upper arm/forearm lengths
0.30/0.26 m, shoulder width 0.36 m, torso box 0.24×0.36×0.60 m and lean angles
are **declared engineering assumptions**, not population anthropometry.

The grid varies 0/15/30/45° waist-pivot lean and waist height relative to the
actual tabletop by -0.05/0/+0.05 m. Its waist XY remains outside the same actual
far table extent with a 0.15 m offset. Both shoulders rotate with one torso;
anatomical left is `cross(world_up, facing)`. No table scale, arm length, or
tolerance is tuned to the result. g00 is development; g01 belongs to the
independent auditor after source freeze.

Each result records four mapped task events and all 123 simulator-grasped
frames in g00. Each target is the highest world-Z vertex of the transformed
selected-object collision mesh, explicitly a surface proxy, **not** the robot
TCP or an observed human wrist. The additional exact closest-surface distance
can prove that *all surface points of that object* lie beyond the fixed
shoulder's maximum reach in a frame. Neither failure rules out a different
body posture or human interaction strategy. `withdraw_hand` has no human
trajectory evidence and stays unknown. The nonacting hand's pose, joint
limits, finger/arm collisions, balance and continuous human execution remain
outside this model.

Torso collision uses the complete saved table mesh, an AABB broad phase, FCL
triangle intersection and closed-mesh containment. AABB overlap alone is not
reported as an actual collision. Both shoulder centers and required target
points must project within the selected camera. This is a declared framing
constraint; torso clipping and 2D overlap remain heuristic risks, not rendered
occlusion truth.

g00 results are saved in
`harness_20261021_v3/actor_geometry/development_01/summary.json`:

| Failure / qualification | Fixed grid result |
|---|---:|
| Declared arm reach failure | 12/12 |
| Modeled torso/table mesh collision | 3/12 |
| Required point framing failure | 7/12 |
| Not falsified by declared model | 0/12 |

Every framing failure involves the shoulders leaving the top of the image;
the task target points stay in frame. At waist height 0, maximum nearest
surface distances decrease from 0.9573 m upright to 0.8327/0.7049/0.5835 m for
15/30/45°. All exceed the unchanged 0.56 m arm sum. Even the closest grid case,
45° and waist -0.05 m, reaches 0.5659 m and intersects the table. The whole
grid plus preparation/probe took 36.67 seconds, API/GPU calls 0. Counts of
event records and held-frame records overlap; they are not independent
physical trajectories or data pairs.

This is a useful rejection of the declared far-side rig. It does not show a
generated image improvement, a universally impossible human task, or a
validated human pose solution. A next distinct hypothesis is joint camera and
human-task geometry design while preserving object roles and required events.

## Preparation revision and actual dispatch boundary

Use **`actor_geometry_preparation_v2.py`** for preparation. The first frozen
wrapper in `actor_geometry_candidate.py` left a legal unified package even for
a rejected rig. Calling the old unified executor directly could bypass that
wrapper's new preflight. Its code and development outputs remain historical
evidence and must not be promoted as a safe preparation entry.

The v2 wrapper produces only a non-dispatchable
`diagnostic_request_spec.json` for rejected rigs, with no `package.json` or
`initial_request.json`. It preserves the actual initial-only prompt for
review, including static rig binding; future object states/trajectory points
are absent. `execute` recomputes bound geometry before any provider or budget
access. The actual development v2 preparation is in
`harness_20261021_v3/actor_geometry/development_preparation_v2/`; its execution
probe was blocked and no executable package exists there.

A future rig not falsified by this model may prepare an initial-image
candidate using the existing provider/budget adapters. Scope unknowns such as
human balance or withdrawal 3D ground truth do not impose impossible
prerequisites on initial RGB. Positive synthetic geometry never authorizes a
video: the unchanged independent `single_actor_v2.1` and `static_state_v1.0`
image reviews and matching generation provenance remain required. The old
unified CLI remains the old baseline and does not acquire geometry-check
semantics merely because this wrapper exists.

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.actor_geometry_candidate compile \
  --package SOURCE_PACKAGE --rig DECLARED_RIG --collision-world SAME_EXECUTION_WORLD \
  --protocol FROZEN_PROTOCOL --output NEW_REPORT_JSON
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.actor_geometry_preparation_v2 prepare \
  --report NEW_REPORT_JSON --output NEW_DIRECTORY --config CONFIG
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.actor_geometry_preparation_v2 execute \
  --preparation NEW_DIRECTORY/preparation.json --config CONFIG --budget EXPLICIT_CAMPAIGN_BUDGET
```

`run_actor_geometry_ablation.py` reproduces the frozen v1 numerical grid and
historical wrapper behavior; its output must remain diagnostic, not a default
generation entry. The current geometry tests cover actual g00, full held
trajectory coverage, camera reversal, foreign meshes, wrong actor/TCP mapping,
parameter inflation, reach bounds, closed-solid collision, report tampering,
and actual provider blocking. The v2 preparation test separately verifies
that a rejected rig leaves no package the old executor can dispatch.
