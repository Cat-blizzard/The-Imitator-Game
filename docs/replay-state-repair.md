# Action-replay initialization repair

The integrated original-task action replay now preserves an already exact seeded
reset instead of writing the same recorded initial state back through the native
simulator. The repair is `c17c461232c656d112c065ed1a4bb517a60cc594` on
`integration/current-features`. It changes replay initialization in the original
task and shared-example callers; generation, recorded actions, control/physics
frequencies, collision rules, quality thresholds and official task conditions are
unchanged.

## Cause and intervention

For the saved Panda/cuRobo pill episode 541, all reset state leaves already
equaled H5 S0. The unconditional `set_state_dict(S0)` then shifted both fixed
robot roots in z by `1.4901161193847656e-08` m. Native fixed-root pose read/write
is not numerically idempotent here. Robot velocities diverged at S4 and the first
payload differences appeared at S65 during closing contact. Replaying the 171
actions still passed official success and execution but counted only 18 of the
required 20 stable placement observations.

A single-variable diagnostic first checked exact reset/S0 equality, then omitted
only the redundant setter. All 171 actions and all 172 states became exactly
equal to generation, and stability recovered to 20/20. Separate production runs
of the committed fix reproduced that result without bypassing the initializer.
The original failing result and diagnostic control remain separate artifacts.

The production helper compares the entire recorded state tree with exact value
and shape equality, allowing only the single-environment leading axis removed by
H5 recording. Exact matches produce `initialization.method = seeded_reset` and
`state_write_count = 0`; any difference retains the original one-time S0 setter
and reports `recorded_state` / `1`. No later archived state is applied and no
extra action is executed. This does not make arbitrary nonmatching initial-state
restoration exact.

## Verification

The applicable CPU suite passes **528 tests** (17 existing warnings), including
17 new initialization cases covering exact reset preservation, one-ULP mismatch,
nested NumPy/Torch state, leading environment axes, key/shape differences,
unchanged input objects, exception propagation and no physics advancement.

Eight existing integration episodes pass production replay, with original
actions and all T+1 recorded state arrays exactly equal and maximum state-array
error zero:

| Route | Seed | Actions / states | Task / execution / quality | Full state equality |
| --- | --- | --- | --- | --- |
| Panda pill, cuRobo archive | 541 | 171 / 172 | pass / pass / pass | exact |
| Panda pill, cuRobo archive | 542 | 183 / 184 | pass / pass / pass | exact |
| Panda pill, MPLib | 6000 | 162 / 163 | pass / pass / pass | exact |
| Piper pill, MPLib | 6000 | 193 / 194 | pass / pass / pass | exact |
| ARX X5 pill, MPLib | 6000 | 194 / 195 | pass / pass / pass | exact |
| UR5+WSG pill, MPLib | 6000 | 184 / 185 | pass / pass / pass | exact |
| xArm6+Robotiq pill, MPLib | 9300 | 190 / 191 | pass / pass / pass | exact |
| Piper apple/banana baskets, MPLib | 450 | 191 / 192 | pass / pass / pass | exact |

The known Piper450 negative still rejects a forbidden robot/payload contact at
action 150, physical substep 748. Official task success remains true. Its old
unconditional-initialization run reported substeps 749/750 instead; the complete
reports are not equal, and that change is recorded rather than hidden. The
diagnostic's phase-classification override and input archive were unchanged.

The fixed new-seed plan was frozen before collection (SHA256
`f4d12ca600c753c96fc22db13cc19642fa1843cb69587d9222d945afc91088b5`).
Each of seven routes attempted seeds **6100 and 6101 once**, in one two-episode
process per route. No seed was replaced and `--only-count-success` was absent.
The audit found no prior use in the inspected local history; this is not a claim
about deleted or remote history. CPU physics remained 100 Hz, native joint
control 20 Hz, and the original-task horizon 200 steps. Two separately locked
idle GPUs ran the MPLib and cuRobo lanes; total wall time was 485.5 seconds.
This wall time is a collection receipt, not a comparable planner benchmark.

| Fresh route | Actions at 6100 / 6101 | Quality and replay |
| --- | --- | --- |
| Panda pill, MPLib | 162 / 165 | both pass; all states exact |
| Piper pill, MPLib | 192 / 196 | both pass; all states exact |
| ARX X5 pill, MPLib | 194 / 194 | both pass; all states exact |
| UR5+WSG pill, MPLib | 184 / 185 | both pass; all states exact |
| xArm6+Robotiq pill, MPLib | 191 / 199 | 6100 passes; 6101 quality fails (9/20), diagnostic states exact |
| Piper apple/banana baskets, MPLib | 194 / 189 | both pass; all states exact |
| Panda pill, cuRobo | 183 / 199 | both pass; all states exact |

All **14** attempts were recorded with T actions/T+1 states, and all **14** passed
official success and execution. **13/14** passed the expert quality requirements;
those thirteen also passed production replay with all actions and every recorded
state value exactly equal. No physical state divergence was observed.

The remaining xArm seed6101 was already quality-failed during generation:
`verify_placement` ended at action 199 after observing only 9/20 consecutive
stable steps. Production replay rejects failed stage metadata before reset or
any action, so this is not a state mismatch. A separate, explicitly diagnostic
replay used the same environment construction, production initializer and all
199 original actions, without editing metadata, replanning or applying later
states. Its 200 states and all actions were exactly equal. It remains quality
failed and is not counted as a production replay pass.

A separate CPU checker directly re-read all source/observed H5 datasets and
verified **14/14 exact recordings**, including that one diagnostic-only case.
The first checker report's anchored log regex missed success tokens appended to
`tqdm` progress lines. That original report is retained; a separate correction
verifies exactly two completed attempts in each of seven logs and changes only
log parsing and its aggregate integrity flag. Arrays and simulations were not
rerun for the correction. The corrected audit reports no artifact integrity
issues; archive, runtime and harness digests match their receipts.

## Historical shared-example boundary

The separate 400-step shared Piper example's old seed401 archive passes all 261
actions and task/execution/quality, but its later states are not exact. Its S0
matches and it uses zero state writes. This is a legacy functional compatibility
case, not a same-version determinism test: the archive predates the Piper native
finger tendon introduced in `ae98a07`, which changes its physical model. The old
metadata records a dirty base rather than a complete source snapshot. Static
inspection found no direct unrecorded physics step or intermediate root setter,
but there was no causal experiment isolating tendon as the only source of the
legacy difference. This case is excluded from the original-task exact-state
counts above and the fixed new-seed batch.

## Reproduction and evidence

Generation and replay use the documented original-task CLI in
[the integration guide](integration.md). Replay requires only the saved archive
and simulator; it does not replan or need cuRobo. The repair exposes its
initialization choice in each replay result. Exact-state verification observes
the real production replay, copying numeric state after reset/initialization and
each `env.step`; it never changes actions, rewards, termination, success or
world state. Persisted observed arrays were compared independently with H5.

Machine-readable summaries, source identities and archive digests are in
[replay-state-validation.json](replay-state-validation.json). Full local evidence
is retained under
`infrastructure/validation/replay-state-20261119/`: baseline and single-variable
control, production replay receipts, observed-state H5 arrays, known negative,
fixed plan/history scan, frozen runtime/dependency identities and new generation
H5/metadata/videos. Large artifacts are not committed. Earlier results in
[integration-validation.json](integration-validation.json) are historical and
have not been rewritten.

This is bounded infra development validation, not a learned-policy benchmark,
general success-rate estimate or planner speed comparison. The repair is on the
integration branch; the separate source feature branches retain their previous
published heads.
