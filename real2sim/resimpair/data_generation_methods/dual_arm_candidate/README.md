# Two planners on one physical control clock — bounded candidate

This candidate extends the existing two-Panda rich fruit scene with one official
`mplib.Planner` per arm. Each arm proposes its own Cartesian-to-joint path; the
shorter path is stretched to the longer path's 20 Hz duration. One `env.step`
receives both native commands. This is not 14-DoF joint optimal planning.

The existing strict planner, recorder defaults, old data and archived media are
unchanged. Executed code imports the local **PolicyGen ManiSkill backend**. The
local The-Imitator-Game dual-fruit task, RoboTwin bimanual path loop and RoboDojo
multi-robot control queue informed the inspection; this campaign does not claim a
separate The-Imitator-Game backend execution.

## Frozen campaign and actual results

Artifacts:
`/mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dual_arm/`

Four physical attempts were allowed, maximum concurrency one, timeout 1200 s per
attempt, CPU PhysX and GPU2 rendering only. Every run took the shared
`report_control_20261021/gpu2.lock`, checked the exact GPU UUID and absence of other
compute users, and counted an immutable ledger entry before constructing an env.
The four-attempt budget is now exhausted; no API calls or training were performed.

| Attempt | Mode, seed, grasp policy | Recorded outcome |
|---|---|---|
| `synchronous_000_attempt01` | Synchronous, 901021000, full close | 160 frames / 8.00 s. Both TCPs moving for 4.45 s; apple/pear raised 0.186/0.169 m. Native grasp sustained in recorded tail, but velocity stability failed. |
| `synchronous_000_measured_attempt02` | Synchronous, same seed, measured finger aperture minus 1.5 mm | Stopped after 120 frames / 6.00 s when object 1 lost grasp during lift. Fixed aperture repair rejected. |
| `sequential_000_attempt03` | Arm 0 pick then arm 1 pick, same seed, full close | Stopped after 239 frames / 11.95 s when object 1 lost grasp after lift. No successful time-matched baseline. |
| `synchronous_001_attempt04` | Synchronous, preselected next layout 901021001, full close | Stopped after 116 frames / 5.80 s when object 0 lost grasp after lift. No new task/source/mechanism qualification. |

Object indices change across layouts: seed 000 has apple/pear in object 0/1,
while seed 001 has pear/apple in object 0/1. The posthoc
`object_identity_correspondence.json` binds every actual exported runtime mesh
by exact SHA256 to the UID-bearing attempt 01 receipt; it verifies this index
swap without treating object color as identity. This is source simulation mesh
correspondence, not generated RGB identity acceptance.

**0/4 complete qualifications.** Attempt 01 establishes actual concurrent arm
movement and simultaneous fruit lifting in this configuration; it does not
establish stable held objects or a promoted production method. Its final speeds
were 0.091/0.153 m/s and 4.90/1.83 rad/s, exceeding the frozen 0.03 m/s / 0.3 rad/s
limits. Small displacement between 20 Hz frames cannot erase these velocity
failures. All source clips, failed traces, requests and code snapshots remain.

The complete synchronous episode took 12.08 s measured wall time (including
initialization, planning, physics, rendering and export). The failed sequential
attempt took 13.93 s. These are not comparable successful throughput measurements:
the baseline did not reach the target, and later runs additionally sampled object
and own-finger evidence at 100 Hz. No speedup ratio is reported.

## Collision and evidence boundaries

The independent geometric guard moves both robots' exported physical link meshes
and held-object meshes together at each densified paired sample (maximum joint
step 0.015 rad). It checks inter-arm links, cross objects, full table and bins.
Only the exact base/table mount, own hand/finger interaction and narrow initial
lift support exception are allowed. This is finite sampling, not a continuous
collision proof. Both official planners also check their own self collision.

A counterfactual coincident-arm mesh input is rejected by the same paired guard.
It never changes the physical environment and is not an extra physical attempt.
Runtime contact impulses are inspected after **every 100 Hz physics substep**;
the stored arrays cover inter-arm, wrong-object, robot/environment and fruit/fruit
forces. These forbidden-contact peaks were zero in all four actual traces.

Frame/state/command `k` is pre-command. `physics_substeps[k]` contains the five
physics transitions after command `k`. The completed receipt has post-command
final object/TCP states. Attempt 01 did not persist its post-command native-grasp
matrix: its claim about the exact final 20 samples has an independently auditable
missing final field, which must stay unknown. The later runner implements final
grasp export for completed episodes, but all three later attempts stopped early,
so that success-path export remains unexercised. Those attempts do export 100 Hz
object state and own-finger forces in their failure traces.
Failed runs preserve their recorded native-grasp sequence; no success is inferred
from the missing post-command native-grasp field.

`protocol.json` froze thresholds and the four-attempt budget before running.
`controller_amendment_v2.json` records the fixed-aperture hypothesis, unchanged
qualification thresholds and extra monitoring. The first protocol runner hash
predates budget-ledger enforcement; every run's `input.json` and `source_archive/`
pin the code actually executed. This discrepancy is recorded rather than hidden.

## Reproduction and CPU inspection

From the workspace root, the historical physical command shape is:

```bash
timeout 1200s PolicyGen/.venv/bin/python -m data_generation_methods.dual_arm_candidate.run \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dual_arm/NEW_UNIQUE_NAME \
  --seed 901021000 --mode synchronous --hold-strategy full_close
```

It is intentionally rejected now because the cumulative campaign budget is
exhausted. Any future physical replay needs a separately authorized, versioned
campaign; do not erase ledger entries or increase this sealed protocol's budget.
Use the saved source archive and input bindings when reconstructing a historical
implementation. No supplied replay command touches old outputs.

CPU-only inspection remains usable without extra physical attempts:

```bash
PolicyGen/.venv/bin/python -m pytest -q data_generation_methods/tests/test_dual_arm_candidate.py
PolicyGen/.venv/bin/python -m data_generation_methods.dual_arm_candidate.summarize_evidence \
  --campaign /mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dual_arm \
  --output /tmp/dual_arm_evidence_NEW.json
PolicyGen/.venv/bin/python -m data_generation_methods.dual_arm_candidate.plot_raw_traces \
  --episode /mnt/sdg/datasets/real_sim_human_pipeline/report_control_20261021/dual_arm/synchronous_000_attempt01/episode \
  --output /tmp/dual_arm_traces_NEW.png
```

Eight CPU tests cover shared interpolation, fixed-arm baseline, paired dense
time, invalid input and measured-aperture action mapping. These tests do not stand
in for physics evidence. `ppt_results.json` is a read-only summary of all four
real records, including video decode/PTS, contacts, heights and failures. It is
not an independent acceptance implementation; independent audit is separate.

## Decision and next falsifiable question

Keep same-clock execution and paired collision evidence as an experimental
candidate. Do not promote the complete grasp-and-stable-hold task. Abandon the
fixed measured-aperture repair for this setup: it actually drops the pear.

Failure first appears in grasp/lift contact behavior, while all paired collision
guards pass and no forbidden runtime contacts are measured. The current OBB
grasp and fully closed Panda target are insufficient for robust fruit holding;
high-frequency contact behavior and changing pear grip geometry are plausible
explanations, not proven causes. A later budgeted experiment should distinguish
geometry-aware grasp placement from feedback on finger contact/force, holding
the same assets/layouts and stable thresholds. Repeating seeds or changing the
acceptance thresholds would not resolve this diagnosis.
