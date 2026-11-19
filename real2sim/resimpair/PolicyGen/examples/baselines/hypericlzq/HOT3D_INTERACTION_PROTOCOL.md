# HOT3D geometry supervision: development protocol, 2026-09-30

This is data preparation for the interaction research goal, not a demonstrated
robot-success improvement. HOT3D pilot labels have entered disposable integration
checks only. Full-cohort scientific training and robot transfer have not started.

## Why this next experiment

The matched EgoDex input diagnostic held the encoder fixed and trained each head
for five epochs. At 1 second, task/session-macro wrist errors were 5.099 cm with
all inputs, 5.324 cm without video, and 5.864 cm without wrist numerics. At
2 seconds they were 6.567, 6.824, and 7.516 cm. The complete-input rerun was
bit-identical to its original frozen-encoder control. These results indicate
that motion history supplies much of this particular head's benefit. Video
contributes, but wrist forecasting alone does not establish interaction learning.
All horizons, motion strata and clustered intervals are retained in
`interaction_research_20260930/human_input_analysis_v1`.

The local 4B teacher passes repeated-frame negatives but still makes object and
event-time errors on inspected development videos. Its labels remain excluded.
Stronger-model annotation is still in scope; zero external GPT API requests have
been sent, and model/service availability must be verified before claiming use.

HOT3D supplies tracked object poses in addition to hand poses. This allows
observed object-motion targets without asking a language model for 3D geometry.
It does not supply task success, force-verified contact, or reliable action verbs
through these geometry fields.

## Data and split

Existing root: `/mnt/sdf/datasets/hot3d_clips`. No repeated dataset download.
First extract all 1,516 official train-Aria clips (150 frames each, approximately
2.1 hours of RGB interaction). Quest3's 1,288 training clips have monochrome
views and are a later camera-domain extension, not silently mixed into this batch.
Official test archives remain unopened by this preparation.

Fixed participant split before outcome inspection:

- Training: P0001, P0003, P0009, P0010, P0011, P0012, P0014.
- Validation: P0002, P0015.

The geometry pilot has four clips from different sequences per participant:
36 clips, 9 participants, 36 sequences, all 33 modeled object categories.
It yielded 28 primary object-motion onset proposals in 20 clips; low/high
threshold sensitivity gave 32/24 proposals. These are candidate events, not
contact ground truth. Visual review of 27 frames across nine participants
checked geometry alignment; it is assistant review, not independent human GT.
Training on HOT3D cannot be described as reproducing EgoMAN's HOT3D-OOD evaluation.

## Coordinate and temporal contract

- Use `T_world_from_wrist` from UmeTrack. MANO's root translation is a different
  point and must not be substituted without a verified transformation.
- Keep each object UID stable; reject ambiguous duplicate instances.
- Store camera, wrist and object transforms in meters. Quaternions are wxyz at
  ingestion. Compare transformations/projections with pinned official code.
- An object's model origin is not necessarily its center or contact surface.
  Wrist-to-origin distance is not a surface contact distance.
- Aria images are fisheye. Any rectification must transform image projections
  consistently and preserve calibration provenance.
- Official RGB orientation is clockwise 90 degrees relative to the stored JPEG.
  Rectify to the official focal-scale-1 pinhole at clip-start intrinsics, rotate
  the image and camera-frame XYZ together, then resize to 224 with pixel-center
  coordinates. Source intrinsics can drift within a clip: compute each source
  warp using that frame's calibration, keeping the virtual camera fixed.
- Object-motion velocities use world poses. Head/camera motion cannot by itself
  create an object-motion event.
- Primary event proposal: object speed > 0.02 m/s OR rotation > 10 deg/s,
  following 0.2 seconds of quiet motion and persisting for 0.1 seconds; current
  modeled visibility >= 0.1 and camera distance <= 1 m. Quiet thresholds are
  half the moving thresholds. A bounded transition interval is saved.
- Proposals use smoothed finite differences and may lag actual first contact.
  Thresholds have not been calibrated as contact detectors.

## Prepared prediction task

`prepare_hot3d_forecast.py` samples queries at 5 Hz after 0.5 seconds of history.
Only ten past/current image indices and their actual relative times are
observation inputs. If a model is given an object-name goal, disclose this
condition; it does not identify an intended destination or complete task plan.

Geometry supervision is stored separately:

- Future object displacement at 0.1/0.2/0.5/1/2 seconds, in the frozen query-camera
  axes. Current object pose/region can serve as additional supervision.
- Future wrist displacement, with independent hand validity masks.
- Future wrist position in the object's coordinate frame frozen at query time.
  Predict object motion separately so that a moving frame cannot hide it.

Keep static objects and late queries; do not select the prediction cohort based
on future movement. Missing horizons are masked individually, preserving short
remaining interaction segments. Future object/hand poses, future frame indices
and future validity masks must never be model inputs. Labels are geometry, not
robot commands, verified contact, or task completion.

The 36-clip pilot contains 792 observation queries, versus 468 that could support
every two-second horizon. This comparison concerns query retention only; it is
not an accuracy comparison with the earlier EgoDex experiment, which had a
different history length, cohort and sampling schedule.

## Experiment gate and planned comparison

Before training: complete full extraction, verify cohort/split/hashes, audit
trajectory outliers and visibility, validate source-to-query coordinates, and
prepare visual feature caches with causal frame provenance. The pilot verifies
geometry, calibrated images, gradients and training/evaluation plumbing. Its
short integration runs are not performance evidence.

For a new matched pilot, preserve the proven robot controller and encoder
interface. Attach disposable human geometry heads to the shared video encoder;
do not treat hand trajectories as robot EEF commands. Compare, on the same
HOT3D queries and optimization budget, wrist-only supervision, wrist plus object
supervision, and a validity/distribution-matched wrong-target control. Include a
frozen-encoder head control. Define loss scaling and the target permutation before
seeing validation outcomes. Do not compare different data cohorts as if only
the auxiliary target changed.

Initially report direct geometric errors, movement-stratified performance,
event-window errors, and simple stationary/velocity baselines. A visual-only
or withheld-numeric control is necessary to measure how much the video learns.
Object names and past motion do not uniquely determine future intention; no
reasoning or task-understanding claim follows from good trajectory error alone.

After the pretraining gate, transfer only the encoder, then use matched original
robot exposures, action representation, human references and hold32 evaluation.
Compare against both the robot-only continuation and wrist-only pretraining;
confirm promising gains with another training seed and held-out rollouts.
Do not select checkpoint/thresholds on unseen robot task outcomes.

## Fixed first full-cohort training comparison

`train_interaction_motion.py` uses four arms: wrist, interaction,
shuffled_objects, and frozen. All use the same stable CNN-hyper20 encoder,
ten causal visual frames, pooled32 frozen DINOv2 features, seed42, batch128,
five query epochs, E learning rate 1e-5, head learning rate 3e-4, weight decay
0.01 and gradient clipping 1.0. A warmup of min(100, total_updates // 10) leads
into cosine decay to 10% of the peak. Save every five epochs and use the fixed
final endpoint; validation does not select a checkpoint. Each query appears
once per epoch. Mean-one participant/sequence/clip weights prevent long
recordings from dominating. Last partial batches are retained.

Each target is an XYZ displacement in meters in the upright query-camera axes.
Use masked SmoothL1 on meters / 0.1, beta=0.2. Wrist loss weight is 1.0;
object loss weight is 0.5, except wrist-only, where it is zero. Neither tracked
coordinates nor future-validity masks enter forward(): it receives only video
tokens. Category queries cover the training vocabulary; category names do not
supply current object positions, active-hand labels or future intent.

The wrong-object control permutes entire five-horizon object target vectors
within participant, category, exact horizon mask, total valid object-horizon
count and exact float32 sample weight. The total count matters because the
object loss normalizes each query by its valid-target count. Each group is a
bijection, preserving its loss-weighted joint target
marginal. Prefer donors from other sequences; retain and disclose singleton
groups. The wrist targets and all observations remain unchanged. Report donor
change, cross-clip/sequence change, and actual numerical target change
separately. The full-cohort fractions must be audited before interpreting this
control. The initial integration variant omitted total-count matching and is
superseded for scientific use; it never entered full-cohort training or robot
evaluation. A regression check now rejects this normalization confound.

Primary geometry diagnostics: one-second error on objects moving >2 cm and
one-second error on all wrists. Retain all five horizons, >2 cm/>5 cm strata,
and [-0.5,0) / [0,1] second windows around primary object-motion onset. These
windows are evaluation strata, never a success/contact label or an input.
Wrist event-window metrics include both hands because active hand is unknown.
Average clips within recording, recordings within participant, then participants.
Two validation people cannot support a broad participant-generalization claim.

The first heads predict translation. They do not yet predict object rotation,
articulated state, force/contact, target destination, or semantic subgoals.
Those remain separate extensions; low displacement error cannot establish
reasoning or manipulation competence. Dense256 features are cached alongside
pooled32 for a later matched spatial-resolution comparison, not mixed into
the initial four-arm experiment.

Preparation evidence: `hot3d_feature_source_v3/equivalence_to_v2.json` verifies
all 4,968 pilot frames and both feature grids byte-for-byte. CPU RGB caching
plus DINO took 264.0 s versus 932.2 s for sequential decoding, a shared-server
measurement, not a randomized performance benchmark. The first failed cache
retains its diagnosis: source intrinsics were incorrectly assumed constant.
Full preparation is source-pinned in `hot3d_full_preparation_source_v1`; it
does not start training or automatically restart failed jobs.

## Primary references

- [HOT3D clip format](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md)
- [Official hand tracking and camera toolkit](https://github.com/facebookresearch/hand_tracking_toolkit)
- [EgoMAN annotation and modeling](https://arxiv.org/html/2512.16907v1)

The pinned camera toolkit revision is
`f4123e1d663005e1a643359d251b513369dc0450`. The small source-only copy is isolated
under the research run; the shared Python environment was not modified.
