# Native human-video ACT: ARX X5 validation

The human-video → native robot training → checkpoint → policy rollout loop ran
on **ARX X5, L0 `TwoRobotPickPillToRegions-v1`** on 2026-11-19. The fixed
6,000-update checkpoint achieved final official success on **3/3 development
seeds and 5/5 evaluation seeds**. These are small engineering checks of one
task and body, not a full TIG benchmark result.

A subsequent frozen batch of **twenty previously unused local-history seeds
also achieved 20/20 final success with the same checkpoint**. Read-only action
diagnostics and exact reconstruction of the earlier five rollouts are reported
below.

The implementation and portable commands are in [the workflow](native_video_act.md).
Exact hashes, per-seed results and limits are in
[the machine-readable record](native_video_act_validation.json).

## Data and training

The existing converted ARX dataset contains 23 actual robot trajectories:

| Role | Episode indices | Environment seeds | Control frames |
| --- | --- | --- | ---: |
| Training | 0–19 | 6000–6019 | 3,876 |
| Development | 20–22 | 7000–7002 | 581 |

All recorded seed fields, source H5 action lengths and converted episode bounds
were checked. A larger declared candidate seed pool was not counted as actual
training data. Normalizer statistics use only selected training rows, with
`bounds_q99` and the ARX native minimum range of `0.01`.

The official [task mapping](../examples/baselines/lerobot_dataset/task_mapping.json)
pairs the pill task with `human_H31`. Real zed2i human episodes 0–3 condition
training; episode 4 conditions development and episode 5 conditions evaluation.
These roles are this pilot's explicit allocation. The human episodes share a
LeRobot v3 MP4, so each clip uses its recorded global file-frame interval and ten
uniform frames within that interval. The DINOv2 encoder samples positions
`[0, 3, 6, 9]` from those ten frames. Each clip was actually decoded and has a
distinct content/sampling identity.

The network reuses the official visual video-conditioned ACT, with a pretrained
ResNet18 robot observation network and frozen DINOv2 ViT-L14 human video network.
The task adapter, task normalization and ACT policy are trainable. Frozen raw
video features are cached; adapted features retain their training gradient.
ARX inputs have 16 qpos coordinates and outputs have 14 native action
coordinates, ordered physical left then right, with no Panda padding.

The configuration uses 224×224 RGB, one observation frame, 16 action queries,
hidden dimension 256, four encoder layers, seven decoder layers and eight
attention heads. Training uses AdamW, batch 16, policy/task-adapter LR `1e-4`,
robot ResNet LR `1e-5`, weight decay `1e-4`, global gradient clip `1.0`, and
CUDA bfloat16 autocast. Learning rates remain constant. The objective is masked
L1 plus `10 × KL`, using the official sum over posterior coordinates for KL.

Training completed all **6,000 updates** in 25 entered epochs. The optimizer
loop took **834.23 seconds** on one RTX 5880 Ada GPU; this excludes initial
data/model loading. There are 337,877,518 total and 33,508,878 trainable
parameters after eager DINO loading. The first backward pass produced task
adapter gradient norm `0.112095`; the frozen backbone had no gradients. The
last minibatch loss was `0.0238379` (L1 `0.0191900`, KL `0.000464782`).
Loss was not used to assert convergence or choose the checkpoint.

## Real policy rollouts

| Checkpoint | Human episode | Environment seeds | Final official success |
| --- | ---: | --- | --- |
| 3,000, midpoint interface check | 4 | 7000 | 1/1 |
| 6,000, development | 4 | 7000–7002 | 3/3 |
| 6,000, fixed evaluation | 5 | 8000–8004 | 5/5 |

The final checkpoint was selected in advance. Every rollout ran 200 steps at
20 Hz control / 100 Hz physics, executing eight predicted actions before the
next observation. Each recorded episode contains 200 native 14-coordinate
actions, with its video and per-episode condition identity. All final evaluation episodes also
ended with both objects placed and ungrasped. Success was sampled through the
budget rather than stopping at the first successful frame.

The existing native action-space clamp remains part of execution. It affected
118–149 of 200 steps per final evaluation episode; per-seed counts are retained.
No extra physical rejection or quality gate was introduced. The policy receives
public RGB/qpos, without expert actions or intermediate simulator-state resets.
Physical measurements remain read-only and separate from official success.

A separate diagnostic holds one real development robot observation unchanged
and switches human episode 0 to episode 4. Both clips pass through the actual
pretrained encoder. Their task-vector hashes differ; normalized action mean
absolute difference is only `6.82218e-6` (maximum `5.35846e-5`). This verifies the
condition path but provides **no evidence that the policy needs the video to
solve this single task**. Discriminable tasks or goals are needed to test that.

## Source identities and artifacts

- Training runtime: `c1caeca8c116562556f613f8ca9abb3919c34b45`.
- Strict reload and rollouts: `537d086bd9e62665da7691df2efc810c293e8947`.
- Final runtime source: `9f2b06b`; the post-run change only rejects mixed or
  unidentified robot tasks and checks the recording/task pairing. It changes
  no network, optimizer or rollout behavior for this run.
- The initial pilot's applicable CPU regression suite passed **573 tests**. Real video
  decoding, pretrained-model training and simulator runs are additional evidence.
- All raw logs, configuration, manifests, provenance, checkpoints, action H5s,
  videos and summaries remain outside Git under
  `$WORKSPACE_ROOT/infrastructure/validation/native-video-act-20261119/`.
  The final weight file is `training/checkpoint_step_006000.pt`; its SHA256 is
  `625829ddebf75d2aaa36658e4fb5cb787f27b7b79ad79bcc4cafdb614108c038`.

This pilot uses ARX expert action supervision and a single L0 task. It does not
establish zero-shot embodiment transfer, cross-task video understanding, full
L0–L3 coverage, convergence at 6,000 updates or an official main-table result.
The five evaluation seeds were used previously in the non-video native ACT
engineering workflow; they are disjoint from this model's training data, not a
new untouched benchmark test set. Broader tests should use a newly frozen
protocol and retain failures as well as successes.

## Action-limit diagnosis and twenty new seeds

The follow-up froze seeds **18000–18019** before execution, after checking
retained local recording metadata, structured seed fields and command receipts.
No prior use or reservation was found within that audited history. Other hosts
and deleted records are outside the audit. All twenty seeds were executed once
with the same 6,000-update checkpoint and human episode 5, the original 200-step
budget, 20 Hz control, 100 Hz physics and eight executed actions per query.

Final official success was **20/20**, success at any point was **20/20**, and
there were **zero timeouts, completed task failures or interface errors**. All
twenty ended with both objects placed and ungrasped. Every video, action H5,
result and diagnostic array was retained; no seed was replaced or used to tune
the checkpoint. This is additional L0 engineering coverage for the same task.

The optional `--action-diagnostics` flag records existing normalized predictions,
denormalized requests and bounded executed actions. It defaults to off and
does not alter execution or scoring. Across the **4,000 directly recorded steps**,
the **twelve arm coordinates had zero exceedances**. Only the gripper commands
crossed their normalized `[-1,1]` bounds:

| Coordinate | Exceeding steps / 4,000 | Maximum command excess | P95 command excess | Maximum equivalent per-finger target correction |
| --- | ---: | ---: | ---: | ---: |
| Left gripper | 1,122 | 0.011229 | 0.006011 | 0.247 mm |
| Right gripper | 2,761 | 0.023214 | 0.010141 | 0.511 mm |

P95 is conditional on that coordinate exceeding a bound. At least one gripper
exceeded its bound in 2,764 steps; left/right counts overlap. Each episode had
117–155 affected steps. The controller maps a normalized gripper command `a` to
each finger's position target `q = 0.022 × (a + 1)` metres. The millimetre column
therefore describes the hypothetical target correction performed by the clamp,
not actual finger motion, joint-limit violation or physical penetration.
Mechanical-arm targets use radians and the prepared URDF's simulation limits;
these should not be described as manufacturer hardware limits.

All directly logged executed actions match their H5 records **bitwise**. They
also equal `np.clip(requested, low, high)` bitwise. An independent NumPy affine
inverse of every normalized action matches every recorded denormalized request
with **maximum error zero**. The saved training q01/q99 for both gripper action
dimensions are exactly `[-1,1]`, making their inverse scaling the identity.
The ACT action head is linear without a final tanh, so endpoint predictions can
slightly exceed these bounds. The observed corrections provide no reason to
retrain or change controller ranges; the existing execution clamp is retained.

The earlier seeds 8000–8004 were also reconstructed offline from their archived
public RGB and qpos, without constructing a simulator. With the original CUDA
RGB scaling, all **1,000 × 14 executed actions match their H5 values bitwise**,
and the per-seed clipping counts remain `[118,149,143,124,145]`. Arm exceedances
are zero. Left/right gripper maxima are `0.008996 / 0.018039`, equivalent to
`0.198 / 0.397 mm` per-finger target correction.

The initial offline reconstruction scaled saved RGB on CPU and was not bitwise
identical. A controlled seed8004 comparison traced this to CPU versus CUDA
division by 255: input values differ by at most one float32 ULP while contiguous
layouts agree. Matching the live renderer's CUDA preprocessing reproduces every
old action exactly. Both the initial reconstruction and corrected results are
retained, together with the device comparison; the original rollout, policy and
runtime preprocessing were not changed.

The follow-up runtime is `e00d4c92dba1039d13dc7b782ef7596a27cbfa17`. Its applicable
CPU suite passed **582 tests**, including diagnostics-on/off action, reward,
termination and success equivalence. Raw evidence is under
`clipping-seed-regression/` within the original local evidence directory. The
[machine-readable record](native_video_act_validation.json) retains all twenty
per-seed outcomes, fourteen-coordinate statistics, source identities and hashes.
