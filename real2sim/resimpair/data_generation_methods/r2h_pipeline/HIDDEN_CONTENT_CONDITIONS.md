# Hidden cabinet content conditions candidate v2

`hidden_content_conditions.py` prepares real source conditions and three image-request specifications from the existing right-door-opening episode. It makes no API call, imports no renderer, and cannot submit video. The existing scene proposal and cabinet contact-v7 physical review are reused; apple strict qualification is not inherited.

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.hidden_content_conditions \
  --proposal /mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/scene_proposals/cabinet_new_v2/proposal.json \
  --verification /mnt/sdf/datasets/hypericl_human_robot_task_library_20261019_v1/generation/pilot_review_CPU_v1/robot/cabinet_contact_v7_891914000/raw_check.json \
  --rgb-review /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/hidden_conditions/source_rgb_review_v2.json \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v3/hidden_conditions/NEW_UNOCCUPIED_OUTPUT
```

Omitting `--rgb-review` produces `prepared_pending_source_RGB_review`, no confirmed revealed reference, and no reveal/terminal request stages. A reviewed source still has `generation_allowed=false`, because an approved single-operator plan, generated-human-anchor reviews, a guarded cabinet execution adapter, and budget are absent. Invalid hashes, mismatched review object/slot/frame, sparse QA presented as a complete recording, changed dimensions, camera disagreement or unsupported task fail before preparation completes. An output directory is never reused.

The output contains:

- `hidden_conditions.json`: bound receipt/proposal/arrays/QA/video/mesh/adapter/physical-review evidence; object ID, original asset UID, local dimensions, physical middle-shelf slot, geometry relative to cabinet, initial/reveal/terminal source states and frame hashes.
- `visibility_timeline.npz`: every original frame, timestamp, instance pixel count, projected/visible QA bounds, measured object state and door angles.
- `image_request_spec.json`: closed initial, revealed keyframe and open terminal target/reference/prompt arguments compatible with `GeminiClient.generate_image_with_references`. The source target controls state; supporting images only supply the same visible object's shape, relative scale and slot.
- `payload_preflight.json`: the real provider's image encoding, separate reference Parts and payload-size checks run locally. `_call` is overridden to record hashes/sizes and stop before headers or transport; no fabricated provider response/image is returned. This proves interface compatibility only.

The closed initial stage has no open-scene supporting reference and must keep the object occluded. Closed RGB cannot transmit hidden content identity to a first-frame-only video. The intended later information path is **source revealed content → reviewed generated terminal human anchor → existing last-frame video condition**. The intermediate revealed human anchor may support later review/segmentation; this candidate does not establish an API or video strategy for it. The cabinet paid execution path remains unconnected and unauthorized.

## Evidence boundaries

The original asset is `block`, UID `robotwin_073_rubikscube_0`, with bound collision extents approximately 0.033878 × 0.040000 × 0.035277 m. These are simulator asset dimensions, not human-scene physical measurements. Object identity derives from the receipt index and bound visual/collision files; matching colors is never an identity pass.

The 499-frame native source at 20 fps is decoded completely and each PTS is checked against measured timestamps. Both camera matrices are checked against recorded look-at/FOV; 224-to-512 intrinsic scaling uses 1e-4 absolute pixel tolerance for float32 precision, matching the existing camera helper. The first attempt used 1e-5 and rejected a 1.5259e-5 float32 focal difference; that incomplete output was retained. Geometry bounds are projected geometry, including when occluded. They are explicitly not full visible extents. No size is invented from the closed image.

The QA timeline has first nonzero object pixels at frame 276 (9 native pixels). The existing scene-proposals adaptive threshold (8 pixels, three frames) also selects 276. The separately frozen v3 engineering reference rule (16 pixels, three consecutive recorded frames) selects 278. Neither threshold proves recognition. Direct inspection at 0/275/276/278/315/330/498 supports a visible small gridded cuboid at 330 and 498; 315 remains partly occluded. Frame 330 is the **first confirmed among these observations**; exact first RGB recognizability remains unknown.

The physical slot is the cavity above the cabinet's middle shelf, behind door 1 on cabinet-local negative Y. A vertical ray through object XY intersects the bound body mesh at shelf z=0.1859375 m and next upper surface z=0.328125 m. Their free vertical band is 0.1421875 m, and object local height / band is about 0.2481. This is a local geometric reference; it does not certify the entire cavity volume or exact support force. The earlier manual image ROI name `upper_compartment` referred to this upper image opening, while `middle` is the physical shelf support label. The original annotation is linked in the RGB observation record and is not renamed or overwritten.

The historical cabinet verifier passed opening, release, withdrawal, door stability, untouched content and contact checks. The task was right-door-open only, not insert/remove/close, and did not establish physically calibrated real-world dynamics. The receipt's `production_approved=false` is preserved. No execution UUID was exported, so the legacy execution ID and receipt hash are retained and UUID remains unknown. All derivatives of this one source execution stay in its original split.

## Local results and remaining work

`harness_20261021_v3/hidden_conditions/development_05/` is the current three-stage packet with explicit projection provenance. `development_04/` is the retained review-schema fix baseline. The earlier `development/` retained a superseded single-stage preparation; `development_02/` and `development_03/` retained intermediate review-schema versions. All remain historical evidence, not selected candidates. The 18 focused tests check sparse visibility, invalid timelines, surface intersections, actual source replay, review/media/slot/identity binding failures, omitted-review unknown propagation, immutable output, wrong-object rejection and real provider preflight with transport forbidden.

The original A video, A redraw, B comparison and VLM all-positive counterexample remain bound in `experiment_record.json`; v2's A-redraw coarse-size miss is retained. These are known failures and one source execution, not held-out generation gains. This work establishes a source-conditioned preparation path, not hidden-object repair success. Actual generated-condition identity/scale, single-person structure, later video preservation and other source/layout/asset generalization remain unverified. Independent v3 audit is separate from implementation.

Independent review found that the original v1 RGB observation record could accept a VLM string reviewer, omitted observation text, or conflicting duplicate frame entries. All three actual-source metadata counterexamples are retained under `harness_20261021_v3/independent_review/hidden_review_schema_negative_01/`. The new RGB observation schema v2 requires a typed reviewer (`human`, `assistant_visual`, or `vlm`), concrete per-frame observation text and unique frames. VLM-only yes answers retain unknown recognizability and cannot select a reference. Old string reviewer records are rejected, not silently upgraded. Reviewer provenance is a declaration rather than authenticated proof; independent image inspection remains necessary. An object with only transient low pixel counts and no 16-pixel/three-frame candidate now stops explicitly with unknown reference, rather than raising an incidental sorting error.

A frozen layout probe then selected seed `891914001` by the parent's preregistered first-nondevelopment-seed rule. The unchanged compiler SHA256 `b0430d3323c217b56d7c2c06f0f7d7f6d4fa388aed77ea0f2995930f8ab791f9` processed its 504-frame native video and original physical review in 2.03 seconds. This receipt reverses object ordering: pear index 0, cube index 1; selection used the existing cube UID rather than assuming index 0. QA-derived review frames were 0/273/274/280/340/503. Without any RGB observation record the output correctly remains `prepared_pending_source_RGB_review`, with no confirmed revealed reference and generation disabled. The probe used the original scene-proposal builder with a new source-only input, did not copy the development human binding, and did not retune code. This establishes metadata preparation on another layout of the same asset/task; RGB preservation and unseen-human-source generalization are unverified. Evidence is in `harness_20261021_v3/hidden_conditions/frozen_layout_891914001/`.

## Projection provenance correction

The parent's independent arithmetic audit found that the recorded QA hull projection is **not exactly reproduced by raw collision OBJ vertices**. The initial frame of seed000 differs by 0.0005332811105892565 in normalized coordinates (0.27303992862169935 native pixels). The original failed audit remains `root_arithmetic_decode_audit_01.json`. No tolerance was increased to make this comparison pass.

Conditions schema `r2h_hidden_cabinet_content_conditions_v2` separates three quantities:

- `recorded_QA_cooked_hull_box_normalized` is read from `qa_visibility.projected_bounds_224 / 224`. The original recorder obtains `get_first_collision_mesh(to_world_frame=False).convex_hull.vertices`; these runtime collision vertices were not exported. File binding is verified, but independent reconstruction and correspondence to raw OBJ vertices remain unknown. The recorder source establishes its origin; it does not establish why the projections differ.
- `projected_world_AABB` transforms the bound raw OBJ vertices with the measured pose, takes their world AABB and projects all eight corners. This is independently reconstructible and was the projection that matched the parent's recomputation. It is a different geometric quantity from the recorded hull.
- `raw_OBJ_vertices_projected_box_normalized` directly projects raw OBJ vertices. Its comparison against recorded QA retains `exact_equality_failed`, signed per-edge differences, maximum difference, `difference_cause=unknown` and `acceptance_tolerance_applied=null`.

The object/cabinet image ratio explicitly uses recorded QA hull bounds as numerator and cabinet-body world-AABB projection as denominator. It remains a source-camera geometric reference, not a generated-human measurement or independently reproduced runtime hull.

Compiler SHA256 `72b91a321d037beb7e023bbbb4e6d5422d62e1c2ec14790098a9b26b8dafc2d7` passed the same 18 focused tests, including retention of this failed comparison. The original seed001 invocation was replayed unchanged under `frozen_layout_891914001/conditions_projection_provenance_v2/`: metadata preparation remains pending RGB review, and the initial raw-versus-QA comparison also remains failed (0.0005228029092715358 normalized, 0.26767508954702635 native pixels). This is a replay of the same layout, not a second independent holdout. The original b0430d compiler snapshot and its seed001 result remain unchanged.
