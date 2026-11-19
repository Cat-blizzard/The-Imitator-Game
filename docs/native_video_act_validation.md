# Native human-video ACT: ARX X5 validation

The human-video → native robot training → checkpoint → policy rollout loop ran
on **ARX X5, L0 `TwoRobotPickPillToRegions-v1`** on 2026-11-19. The fixed
6,000-update checkpoint achieved final official success on **3/3 development
seeds and 5/5 evaluation seeds**. These are small engineering checks of one
task and body, not a full TIG benchmark result.

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
- The final applicable CPU regression suite passed **573 tests**. Real video
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
