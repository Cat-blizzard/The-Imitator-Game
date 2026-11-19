# Geometry preparation v3: bind the reviewed source to the submitted request

The opt-in entry is now `actor_geometry_preparation_v3.py`. The frozen v1/v2
sources and results remain evidence of earlier limitations; neither is
silently repaired. Geometry dimensions, physical inputs, numeric rules,
independent image acceptance and provider budget behavior are unchanged.

The independent audit reproduced a real v2 routing failure: replacing only
the g00 outer preparation's `qualified_wrapper` with a valid g01 wrapper
reached the image dispatch boundary with g01 package metadata. The outer
geometry, diagnostic and executable-package fields still displayed g00.
The audit intercepted before any provider operation; this was a source
routing error, not observed paid overspending. Evidence is preserved under
`harness_20261021_v3/independent_review/geometry_positive_outer_inner_mismatch_01/`.

v3 binds and checks all of the following before accessing the budget:

- Outer geometry equals inner wrapper geometry, including actual file hashes.
- Outer executable package equals inner package.
- Outer `actual_initial_request` equals inner request and the package's actual
  `initial_request.json`.
- The request itself binds the same package.
- Diagnostic geometry equals outer geometry.
- Diagnostic source image, model, initial conditions and prompt exactly match
  those fields in the actual request.
- The complete diagnostic is rebuilt from the bound source, actor and rig,
  preventing coordinated same-source prompt edits with updated hashes.
- Wrapper implementations and all referenced artifact hashes remain valid.

Execution then uses the existing geometry/unified revalidation and stable
campaign budget. Rejected rigs continue to produce only diagnostic specs,
without an executable package or request. No proxy geometry result grants
human, image or video qualification.

The real g00 native/45°/+5 cm/left candidate is prepared separately in
`harness_20261021_v3/actor_geometry/left_hand_candidate/preparation_v3/`.
Only preparation is repeated; no physics, rig parameters or task source are
changed. The independent auditor owns the corresponding frozen g01 replay.

Ten targeted pytest cases and five subtests cover the actual cross-source
splice, outer package/request swaps, another rig, diagnostic reference and
four displayed-field changes, stale hashes, another implementation,
coordinated diagnostic/actual prompt edits, the real zero-budget positive
boundary, and rejected preparation with no package. These tests use full
package imports to match the repository-wide pytest invocation.

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.actor_geometry_preparation_v3 prepare \
  --report GEOMETRY_REPORT --output NEW_DIRECTORY --config CONFIG
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.actor_geometry_preparation_v3 validate \
  --preparation NEW_DIRECTORY/preparation.json --config CONFIG
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.actor_geometry_preparation_v3 execute \
  --preparation NEW_DIRECTORY/preparation.json --config CONFIG --budget EXPLICIT_CAMPAIGN_BUDGET
```

This is an opt-in wrapper contract. Calling the old unified baseline directly
does not invoke the additional geometry gate; v3 does not claim global
enforcement over the unchanged old CLI. Generated initial images still need
the unchanged independent actor and static-state evidence, and native
grasp/lift visibility remains unresolved.
