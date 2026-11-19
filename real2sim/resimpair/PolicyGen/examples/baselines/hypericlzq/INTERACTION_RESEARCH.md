# Interaction and human-video transfer research, 2026-09-30

## Objective and scope

Improve seen and unseen closed-loop task success by investigating interaction-critical training,
strong-model semantic supervision, and information from human video. Every claimed gain must have
matched robot exposure, a reproducible evaluation protocol, and an explicit uncertainty estimate.
All three research directions remain in scope; an event-weighting pilot alone does not complete this goal.

## Starting evidence

- Stable candidate: CNN observation + relative EEF14 + human-reference HyperNet + two-step Refiner;
  no parameter/function expert supervision or test-time gradient adaptation.
- Checkpoint: `/mnt/sdf/datasets/hypericlzq_runs/seen_gap_causal_20260928/full_policy_visual_confirmation/cnn_hyper/epoch_0020.pt`.
- Robot manifest: `.../seen_gap_causal_20260928/action_representation_ablation/manifest_eef.json`.
- Historical results have repeatedly been inspected. Treat their rollout seeds as development evidence,
  not untouched test data. Never pick new hyperparameters by repeatedly querying unseen test success.
- Raw original robot Parquet has joint states, commanded joint/gripper targets, timestamps, episode/frame/task IDs.
  It does not provide contact forces or object poses in those columns. Command closure is a proxy, not contact GT.
- Prepared EgoDex: 6,173 train clips / 586 sessions, 708 validation clips / 56 sessions.
  Wrist-valid hand frames: 96.07% train, 95.08% validation; pinch-valid: 11.78% / 7.24%.
  The task `clip_unclip_papers` has no validation clips in the inherited split; report this gap explicitly.
  No verified object-centric contact annotations exist in this build.

## Experiments, separated by scientific question

### E1: Does emphasizing command-transition windows improve the existing policy?

Three matched arms continue the exact same stable hyper20 checkpoint:

1. `uniform`: ordinary action loss.
2. `event`: mean-one per-episode action importance around observed gripper command open/close transitions.
3. `shifted`: the same weight histogram, circularly shifted within each episode using a deterministic hash.

Proposals use hysteresis (open >= .8, closed <= .2), record transition endpoints, and reset at invalid gaps.
The profile is the maximum triangular support around events: .5 seconds before, 1 second after,
raw weight 1 + 2 * support. Normalize over all query frames of that episode. No dropping easy frames.
The shift can overlap real events: measure overlap per task, never describe it as guaranteed non-contact.

Pilot budget: 5 additional exhaustive original-data epochs, same effective batch 256, LR 1e-5,
warmup 100 updates, seed 42, same batch ordering/reference draws/FM noise; save epoch5.
Benchmark microbatch256 / accumulation1 first; use microbatch32 / accumulation8 only as a memory fallback.
The first microbatch32 probe passed but required about 7 seconds/update; it is unsuitable for the full campaign.
All formal arms use the same final batch configuration, selected using speed/memory only.
Keep CNN/base frozen, preserve all existing hyper-stage auxiliary losses, normalization and label masks.
Weights affect only action loss; auxiliary losses and validation metrics stay unweighted.
First three microbatches hash all non-intervention inputs and RNG states. A failed equality check blocks that arm.

This is a continuation experiment, not a from-scratch or full multi-seed result. Failure of E1 does not
falsify contact learning: the labels are only command-transition windows and the base policy is frozen.
Success must exceed both the equal-budget uniform continuation and the shifted control, not just hyper20.

### E2: Does human motion supply transferable information beyond more optimization?

Compare robot-only vs human motion pretraining vs temporally mismatched human targets with the same
valid samples and update budgets. Keep human wrist position/rotation/pinch heads distinct from robot commands.
Measure current/previous-motion baselines at fixed physical horizons, including 0.5/1/2 seconds.
Use session-separated validation and task-balanced aggregation; no full future-video feature as query input.
Motion accuracy is a diagnostic. Test downstream robot success with a matched robot base/hyper schedule.
First compare representation/conditioning transfer; sharing action-DiT weights is a separate intervention.

### E3: Does explicit interaction information make human video useful?

Target a 2x2 control: robot-only / human+robot crossed with interaction supervision off / on.
Candidate shared targets: active object/part, interaction region, approach/manipulate/release interval,
relative hand-object motion, observed object-state change, completion evidence and uncertainty.
3D values come from tracking/calibration with masks; a language model may not invent metric coordinates.
When only RGB/2D tracks exist, supervise those and retain the coordinate type. Never fill missing contact with zero.
Predict temporal/region/motion targets causally from current/past observation and an explicit goal.
Use human data to train a shared interaction representation; robot action loss trains its control connection.
Keep low-level control and gripper semantics embodiment-specific unless calibrated transfer is independently verified.

### E4: Strong-model teacher and later high-level agent

First use available strong vision-language models offline for batched structured annotation, candidate filtering,
atomic intent/object/state-change extraction, and uncertainty. Store model ID/version, exact prompt, input
frame timestamps, raw response, parser version and review evidence. Do not assume a model's geometry is accurate
because its reasoning is strong. Review a stratified batch, including open/push/wipe/fold/grasp and occlusions.
Compare teacher annotations against independent held-out review, and use label-confidence masks.
Distill structured targets into the small model; leave the proven fast controller responsible for execution.
Online skill selection/replanning is a later separate experiment with model latency, calls and failures recorded.
Actual API/model availability and cost must be checked before claiming any specific GPT version was used.

## Annotation contract

Every event: source episode + session + split, active hand/tool, object/part, atomic intent,
event type, [earliest, latest] time interval, visible evidence frames, label source,
confidence/calibration status, contact observability, valid 2D/3D coordinate frame and transform provenance.
Keep `command_transition`, `visual_manipulation_onset`, `geometric_proximity`, `verified_contact` distinct.
Offline teachers may inspect future frames for labels; deployed prediction inputs cannot contain that future.
Reference demonstrations from other episodes may be complete, and their role must be recorded.
Batch numeric proposal generation, then sparse semantic inspection, then local temporal refinement.
Object motion and task outcome matter: wrist movement alone does not establish successful interaction.

## Evaluation contract

- Preserve human reference selection, policy-parameter hold32, horizon16, FM10, EEF-to-IK conversion,
  temporal aggregation, 500-step limit, simulator source and random seeds across compared arms.
- Primary metric: success at episode end. Also report ever-success, per-task/per-level results,
  failures rescued/lost, tracking/action errors by proposed event support, and runtime.
- Screening uses known development seeds 2,3,4,5. Freeze candidate before using held-out initializations
  1100..1109 for confirmation; verify no existing result uses them. Unseen tasks remain excluded from
  original robot training. External related tasks are allowed but disclose overlap explicitly.
- Confirm promising changes with at least two independent continuation/training seeds; distinguish
  continuation seeds from independent initial pretraining seeds. Estimate paired differences with
  task/environment/episode grouping, not IID video windows. Intervals crossing zero remain inconclusive.
- Log all arms, including negative results; do not select a checkpoint on unseen benchmark outcomes.
- Stage annotations, human forecasting and robot rollout success are separate endpoints.

## Primary literature consulted

- [VRB, CVPR 2023](https://arxiv.org/abs/2304.08488): contact regions and post-contact trajectories from human
  video can serve as a robot-learning interface; motivates object interaction rather than raw wrist imitation.
- [HAMSTER, ICLR 2025](https://arxiv.org/abs/2502.05485): high-level visual-language path predictions condition
  a low-level policy. Coarse semantic/spatial plans do not replace closed-loop precise control.
- [EgoMAN](https://arxiv.org/abs/2512.16907): separately pretrain reasoning and a conditional motion generator,
  then align predicted conditions. Human forecasting evidence alone does not establish robot transfer.
- [AffordanceWAM, September 2026 preprint](https://arxiv.org/html/2609.22332v1): reports a robot/human by
  affordance-supervision factorial control; human video helps with its interaction representation and hurts
  in its RGB-only control. Uses object/interaction annotations, much more data and a large video backbone;
  architecture and effect sizes must not be assumed to transfer to our small model. Not independently reproduced here.
- [A4A, September 2026 preprint](https://arxiv.org/html/2609.05892v1): predicts goal-conditioned future motion
  of interaction-relevant 3D points and transfers internal representations while retaining robot-specific
  interfaces. Motivates evaluating object/part motion against generic scene motion and wrist-only targets.
- [BridgeACT, April 2026 preprint](https://arxiv.org/html/2604.23249v1): uses VLM task/entity extraction,
  object masks, point tracking and geometry, representing tool-target roles and relative motion.
  Estimated depth or masks are pseudo-labels; retain uncertainty and validate before treating them as metric GT.

## Artifacts and current scope

Research outputs: `/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930`.
Code: `interaction_events.py`, `train_interaction_weighting.py`; optional `action_importance` in `model.py`
defaults to one and does not alter inference. Unit tests cover mask/transition gaps, normalized loss mass,
time-shift distribution, unchanged auxiliary losses and unchanged sampled actions.
This document is a research protocol and backlog, not a claim of completed training or improved success.

## E2 implementation and preregistered first pilot

`prepare_motion_forecast.py` retains 38,886 training queries from 5,180 clips / 437 sessions,
and 4,549 validation queries from 599 clips / 42 sessions. The source remains 6,173/708 clips.
Queries have ten past/current frames over 0.9 seconds; targets are measured wrist displacements
at 0.1/0.2/0.5/1/2 seconds in the frozen query camera. Historical tracking masks are input;
future tracking masks are loss-only. A shared world transform cannot alter these inputs.
Thirty-four real query geometries were independently recomputed; all matched within 2 micrometers
numerically. This does NOT establish the physical accuracy of the underlying sensor labels.

Three arms, fixed five epochs / 1,520 updates / batch128 / seed42:

- `true`: update the existing video encoder E and a separate human-wrist forecasting head.
- `shifted`: identical inputs/masks and all target vectors, deranged in time within each clip.
  Exact valid-hand patterns are preserved. Median shift is 4 seconds; minimum is 1 second.
- `frozen`: correct targets with E frozen, testing how much the small added head alone explains.

E starts from the stable CNN hyper20 checkpoint. E LR is 1e-5; head LR is 3e-4, 100-update
warmup and cosine decay, AdamW weight decay .01. Loss is macro-weighted masked SmoothL1
on displacement/0.1m, beta .2, with gradient norm clipped at 1. The output starts at zero,
equal to last-position prediction. This is a deterministic forecasting diagnostic, not a
reproduction of EgoMAN's flow-matching method. Human position outputs are not robot commands.
A small trainable mean word embedding supplies the existing intent description; it is not
a pretrained language model or the strong-model teacher being investigated separately.

Epoch means one exhaustive pass over the selected queries. Loss balances task→session→clip→query;
the weighted effective query count is about 13,530, not 38,886 independent observations. Report
clip→session→task means and pooled errors separately, including >2cm and >5cm future-motion strata.
The inherited validation split still has no `clip_unclip_papers`. Stationary/CV baselines are
recomputed on exactly this subset; do not compare its scores directly to the earlier full audit.

Important selection limit: complete 2-second futures and derangeable within-clip groups exclude
993 training and 109 validation clips. This can underrepresent short contact interactions.
E3 must retain short clips using masked shorter horizons or alternative matched controls;
an E2 negative result cannot establish that all human video or contact information is useless.

The first launch (`human_pilot_seed42`) was interrupted for verified random-disk IO contention,
not a model/loss failure. No completed five-epoch checkpoint exists for it. Logs, source copy,
initial validation and interruption reason are retained. `human_pilot_seed42_v2` stages the exact
input tensors into roughly 28.5GB of tmpfs using sequential source reads, checks every staged
episode block, verifies both real loaders, and matches the original first-three-batch tensor/RNG
audit before restarting from the same initialization. No intervention/budget was changed.

`export_motion_encoder.py` exports ONLY E after the fixed endpoint, and verifies every other robot
tensor is unchanged. Downstream true/shifted transfer still requires the same five additional robot
epochs and paired rollouts as E1 uniform. The frozen arm's exported encoder must equal the original.
None of these forecasting diagnostics establishes improved robot success.

## E4 interface and availability

`interaction_teacher.py` defines bounded stage intervals, supplied evidence frames, an optional
normalized object/part box, visible state change, and explicitly uncalibrated confidence. Unknown
or truncated phases are legal. Invented metric coordinates and unseen evidence frame IDs are rejected.
Six schema/evidence tests pass. Grammar/bounds validation is not semantic or physical verification.
The teacher does not receive gripper-command proposals, preventing annotation from simply repeating
the weighting heuristic. Raw output, image bytes' hashes, actual model and prompts are retained.

48 GPT-6 Astra Responses requests were prepared locally, but no external API was called; ordinary
API credentials were not present. Model access information was requested asynchronously. A separate
six-clip local Qwen2-VL-2B pilot uses the existing pinned model cache to assess the annotation pipeline.
It is not GPT-6, and no local/generated pseudo-label is automatically supplied to training.


## Teacher diagnostic and selection coverage audit

The controlled six-clip development diagnostic is complete: full-schema metadata/no-metadata each0/6 structurally valid, compact no-metadata1/6 and metadata2/6. Prompt complexity changes the grammar task, so these are not comparable semantic accuracy estimates. Assistant visual review finds wrong active objects, repeated interactions and broad evidence citations even among structurally valid responses. No labels enter training. The other42 review clips remain untouched for a fixed-prompt check; independent semantic review is still required.

Strict E2 eligibility excludes993 training clips (431 too short;562 without a derangeable valid group), with median length3s and wrist-valid hand-frame fraction95.56%. Every excluded clip still has a measured16-step wrist target, but those counts do not establish sufficient causal history. E3 must keep short interaction clips and mask each future horizon separately. Do not change E2 enrollment after outcomes are seen, and do not equate its eligibility with semantic quality.

`analyze_interaction_campaign.py` refuses comparisons until all three complete80-episode cohorts match exactly. It verifies execution/source signatures, reference/state/RGB hashes, and missing/duplicate/error outcomes. End success is primary; ever success secondary. Task-cluster and environment/seed intervals are descriptive;5 tasks and1 continuation seed cannot support broad population or training-variance claims.

Evaluation-only dependency repairs are additive and documented in research-root addenda. Existing899 snapshot-pinned files are unchanged. Actual short L0/L3 preflight is separate from500-step benchmark evaluation and cannot establish a performance improvement.

## Fixed E2 human endpoints and downstream transfer (2026-09-30)

All three human arms completed the preregistered five epochs /1,520updates. Their
elapsed training times after identical-input RAM staging were171.5s(true),189.2s(shifted),
225.9s(frozen), on shared hardware; these are not isolated speed benchmarks.
`analyze_motion_forecast.py` independently recomputes all saved predictions against the
pinned validation arrays, validates per-clip and task-macro scores, verifies train/validation
session separation and matching input/RNG audits, and records checkpoint/source hashes.
Five analysis tests cover invalid predictions, mask/cohort drift, macro weighting, identical-arm
intervals and invalid independent resampling of sessions shared across tasks.

Position error in centimeters, clip→session→task averaging on the exact enrolled cohort:

| Horizon | True, update E | True, frozen E | Shifted, update E | Stationary | Constant velocity |
|---|---:|---:|---:|---:|---:|
|0.1s|0.815|0.826|1.025|0.952|0.689|
|0.2s|1.579|1.605|1.931|1.822|1.777|
|0.5s|3.347|3.416|4.131|3.933|5.714|
|1s|4.997|5.099|6.323|6.031|11.942|
|2s|6.454|6.567|7.809|7.621|22.506|

Updating E contributes only about0.10–0.11cm at1–2s relative to freezing it in this
particular pilot. Much of the gain over trivial baselines is achievable using the existing
encoder plus trained prediction heads; this does not isolate numeric history, text and visual
features from each other. The shortest-horizon model is worse than constant velocity.
Task-bootstrap and fixed-task/session-bootstrap intervals are descriptive, exclude training-seed
variance, and include only8tasks; clean_tableware has just1validation session.
All horizons and >2cm/>5cm motion strata are retained, without selecting a favorable endpoint.

`motion_transfer_seed42_v1` verified E-only exports and actual three-update robot smokes.
Frozen-E export leaves every robot model tensor bit-identical to the original, so E1uniform is
its exact scheduled control. True/shifted exports now receive five matched robot epochs each,
using uniform action weighting, the same frozen trainer, reference draws, RNG and evaluation.
The first three robot batches matched the existing E1 audit. There is no new robot success result yet.

`analyze_motion_transfer.py` requires complete80-episode paired cohorts, identical training
configuration except initializer/output, fixed5epoch/20190update checkpoints, verified E-only
provenance, and matching execution/reference/state/RGB signatures. Two new transfer-analysis
tests and the five existing rollout-analysis tests pass. The frozen `paired_analysis_source_v2`
watcher checks live producer handles and runs E1/E2 analysis after their controllers finish;
failure or a missing handle produces an attention record, never an automatic restart.

## Teacher model and input-resolution diagnostics

Qwen3-VL-4B-Instruct at revision`ebb281ec70b05090aa6165b016eac8ec08e71b17` was downloaded
with each file verified against that fixed Hugging Face revision. A separately pinned
Transformers4.57.3/tokenizers0.22.1/hub0.36.0 runtime leaves ongoing robot training unchanged.
On the same six development clips, the compact prompt is6/6 structurally valid; the full prompt
is5/6 valid, including2empty action lists and several visibly incorrect stage times. One full
response reaches the1800token limit. Assistant review finds invented bottle motion and wrong
release timing. Grammar success is not semantic accuracy;0labels are admitted to training.

`teacher_resolution_pilot_v1` then compares12actual inferences, native-source224/448 square
resizes, identical frame indices, model and compact prompt. All132source frame timestamps
match the previous selection; direct-source PNG versus compressed review cells differs by
at most1.62mean RGB levels per frame. Robot sources are224×224, so resizing adds no captured
detail; human sources are1920×1080. Both conditions are6/6 structurally valid.448helps identify
the lid and separates a blue object/yellow bottle in the cleaning clip, but still invents
red-bottle transport and conflates fruit transfers. No reliable phase-label pipeline is established.
Reviews are assistant development diagnostics, not independent human ground truth. The other42
stratified review clips remain untouched. No external GPT API requests have been sent.

Next teacher check should separate object/state-change discovery from local temporal localization,
use short windows with before/after evidence and identical-frame negative controls, then require
independent semantic review on a frozen holdout. Higher pixel count alone did not solve the
observed failures. E3interaction2×2, stronger-teacher distillation and online agent evaluation
remain open, alongside unseen and independent-training-seed robot confirmation.


## 第五次更新：2026-09-30 06:21 +0800

四组固定五轮的人类手腕输入来源对照已完成，每组 1,520 更新；完整输入重跑与原冻结 E 结果逐位相同。重新从预测和标签计算所有时长、运动幅度分层、session/任务平均及配对抽样区间。1 秒误差：完整 5.099 cm、去掉视频 5.324、去掉手腕数值 5.864、仅文字 6.064；2 秒分别为 6.567、6.824、7.516、7.643。说明当前冻结视觉设置中数值历史贡献更大，视频有一定贡献。单种子、八个验证任务；2 秒视频增益的按任务抽样区间略跨零。不能推断视频无用，也不能据此声称机器人成功率提高。结果见 `/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/human_input_analysis_v1/输入来源对照.md`。

局部前后状态标注/原生视频对照完成 56 次推理：两种输入接口各有 14 个真实窗口、14 个重复帧窗口。28 个重复帧窗口都没有变化输出；每种接口在 7/14 个真实窗口输出事件，这不是准确率。检查四个开发窗口仍见明显事件提前、物体转移方向错误，原生视频输入本身没有解决时间定位。继续保持 0 个教师标签进入训练、0 次外部 GPT 请求，其余 42 个留存审查片段未用于改提示词。

新盘点利用已经下载的 HOT3D，未重复下载数据。36 段数值抽查覆盖 9 人/36 场录制/33 类物体，手和物体位姿完整。主阈值提出 28 个物体运动开始候选（20 段视频），低/高阈值 32/24，均不称为真实接触。按人物固定留出 P0002/P0015，官方 test 不读取。官方工具投影核验 27 帧，坐标变换误差小于 3.4e-8 m；助手视觉检查通过基本对齐，不是独立人工真值。

已启动全部 1,516 段 train-Aria 的几何提取（当前 245/1516，实际进程 7191 存活）。数据写入 `/mnt/sdf/datasets/hot3d_clips/processed/`。固定源代码监听器在提取通过后自动构建因果查询；不会自动训练模型。查询保留不完整未来时长，36 段产生 792 个观测查询，其中 468 个能覆盖完整 2 秒。未来几何/有效性均为监督，观测索引仅过去和当前；独立从原始 tar 与官方相机工具核对了物体位移。尚未产生 HOT3D 特征缓存或训练结果。

E1 实际训练进度：uniform 2.11/5, event 2.11/5, shifted 2.11/5；人类预训练后的同预算机器人迁移：true 0.50/5, shifted 0.50/5。训练和配对评估监听器均核对实际进程存活，尚无这些新干预的闭环成功率结论。研究目标继续有效。


## 第六次更新：2026-09-30 07:22 +0800

全部 1,516 段 train-Aria 几何提取和因果查询已完成，覆盖 9 人、124 场录制、33 类物体。按人物划分训练 1,276 段/28,072 个查询，验证 240 段/5,280 个查询；查询是重叠观察窗口，不能当作同等数量的独立轨迹。零重复录制帧、零重复查询。主阈值提出 1,212 个物体运动开始候选，分布于 836 段视频；不是接触或成功真值。

检查中发现并修正了相机处理问题：原始 Aria 图像需要顺时针转90度，XYZ坐标同时旋转；源相机内参在片段内存在小幅变化，必须逐帧使用源标定，输出仍固定到片头虚拟相机。首个错误假设版本已保留失败记录，没有进入训练。修正后的两种缓存实现对36段/4,968帧的 dense256 和 pooled32 特征均逐文件字节相同。批量CPU预处理+GPU特征缓存用时264.0秒，串行932.2秒，约3.53倍提速；这是共享机器顺序实测，非随机性能基准。

全量质量检查只有1段触发>5m/s物体速度复查：验证片段clip-003226的记号笔在0→1、1→2帧位姿跳变。该段最早监督query为15帧、速度基线历史最早12帧；跳变不进入任何预测目标或速度基线，过去RGB也不含跟踪位姿输入。保留所有数据，不以删除困难验证样本改善指标。原始位姿物理准确性未据此判定。复查在数据目录 audit_v1/outlier_review.json。

首版错配控制漏考虑了损失按每query有效物体目标数归一化的分母，已在开发试跑中发现并修正。新控制同时匹配人物、物体、未来mask、样本权重与分母，整体打乱5个时长的目标向量，保持真正的加权监督分布。旧版本未进入全量训练或机器人评估。新增4项针对性检查通过；修正版四组真实数据完整训练/验证/保存试跑通过，初始权重、输入/RNG及最终采样器状态一致，非错配三组与修正前逐tensor相同，只有错配组改变。冻结E保持不变，构造迁移时所有非E机器人权重逐tensor一致。试跑各只有5次更新，不当作效果证据。

全量RGB预处理正在运行，随后自动生成DINO特征。正式四组实验已固定并排队：wrist、interaction、shuffled_objects、frozen；各5个query epoch、batch128、1,100次更新，仅图像输入，手腕/物体3D位移作监督，不作为机器人动作。每5轮保存，固定最终checkpoint，不按验证挑选。错误目标控制需达到预定最低改变比例，否则停止复查；不自动重启失败任务。训练结束保留所有预测、按人物/录制平均的各时长/运动分层/运动开始前后指标，后续仍需匹配机器人训练和闭环评估。

实际机器人训练：E1 uniform 2.48/5, event 2.48/5, shifted 2.48/5; E2 true 0.87/5, shifted 0.87/5. 所有训练和分析监听器均核对实际进程存活，目前没有这些新干预的完整闭环成功率。外部GPT调用仍为0，本地教师标签仍未准入。目标保留，强模型语义、机器人迁移、多种子与unseen验证等工作仍未完成。


## 第七次更新：2026-09-30 07:55 +0800

已完成独立检查点重放分析器：四组小规模试跑各176个验证查询重新从checkpoint推理，保存预测与重放预测逐位相同；独立重算坐标、当前物体可见mask、误差、按人物/录制平均及输入/RNG核验均通过。正式分析监听器实际存活，待固定五轮完成后自动重放全部验证预测。试跑仅验证流程，不作为提升证据。对象误差只比较训练过对象头的interaction/frozen/shuffled组，不能把wrist组未训练的零对象头当作公平的表征对照。

完成全部1,516段的输入可观测性审计。当前手腕原点投影在实际输入视野内且源相机建模手表面可见比例>=0.1作为代理判据；该判据不是人工遮挡真值，缺失可见性仍为未知。训练手腕query中30.52%不满足当前代理、23.87%整个历史都不满足；验证分别24.86%和17.68%。只看有效1秒预测目标，历史不满足比例为训练23.82%、验证17.84%。隐藏手腕仍可能从物体/手臂上下文推断，不能称为不可学习，也不因此删样本。

名义0.5秒的HOT3D历史最长0.66665秒已查明：4段训练视频有4处较大的原始时间戳间隔，影响9/28,072个查询(0.0321%)，验证集0处。全部时间戳来自源记录，历史无未来帧，跨间隔的未来监督正确屏蔽；有效候选未来时间与要求时间之差绝对值最大约2.58毫秒。没有证据把此小比例差异当作整体迁移问题的主因，保持预定队列不变。详情在hot3d_timing_audit_v1，包含全部受影响clip/frame。

1,212个主阈值物体运动候选的规则时间括号中位宽度约0.267秒；训练95分位0.533秒、验证0.585秒。这是规则支持区间，不是校准置信区间或真实接触误差。主候选在同物体±0.25秒内能匹配低/高阈值候选的比例：训练77.35%/81.54%，验证60.00%/84.76%。最近邻匹配可多对一，因此只说明阈值敏感性，不是事件准确率。后续阶段学习应保留类型、区间和不确定性，不能把这些候选升级为精确接触GT。当前四组预训练的监督是连续手腕/物体位移，这次审计不改变它们。

输入时间跨度也存在值得后续单独检验的域差异：750条原机器人训练任务对应的人类参考视频，以10帧覆盖完整示范，时长中位3.73秒；当前HOT3D预训练10帧覆盖约0.5秒，编码器不接收真实相对时间。这不等于已证明的失败原因；此前EgoDex手腕预训练的历史是0.9秒，不应与HOT3D混淆。固定四组结果出来后，才考虑匹配预算的多时间尺度输入对照。

新增可见性/时间间隔二级分析：当前可见、仅历史可见、全历史未满足代理三组互斥且覆盖所有有效手腕目标；按clip→录制→人物汇总，配对区间在既定人物内重采样录制。4项针对性测试通过，小规模实际四组预测重算与主要指标一致，冻结副本再次逐指标复现。CPU监听器已启动并核对实际进程，待主要重放核验完成后自动运行；验证没有时间间隔样本时明确缺少该组，不制造比较。它不会挑选checkpoint、改变主要终点或据此作因果结论。

当前实际进度：E1 uniform 2.58/5, event 2.58/5, shifted 2.58/5；E2 true 1.00/5, shifted 1.00/5。HOT3D RGB缓存761/1516，之后自动生成DINO特征并运行固定四组。全部训练、预处理与两级分析监听器已核对/proc身份和存活；没有重启现存任务。仍无本轮干预完整闭环成功率。外部GPT请求0、准入教师标签0；强模型语义增益、机器人迁移、独立种子及unseen确认仍待完成。


## 第八次更新：2026-09-30T08:47:35.767348+08:00

完成固定14段训练开发视频/7人、7个完全重复帧窗口、三种输入的63次Qwen3-VL-4B推理。原始输出重新解析，全部真实输入与processor预检一致，未截断输出。图像组合法12/14、非空11/14；图像+追踪合法14/14、非空9/14；仅追踪14/14合法但全部为空。三组各7个静止对照均无变化输出。物体运动声明与被指认物体的近静止轨迹冲突：图像7/11，图像+追踪6/10。这是输入事实一致性诊断，不能作为独立语义准确率；空输出也不能作为正确识别。四个预定抽查片段仍有错误物体UID、左右手、转移事件。全部教师标签继续未准入。详细结果在teacher_geometry_analysis_v2。

前一次长输入OOM已保留。未改变模型、帧数、448分辨率、BF16或10.5GiB上限，修复采用可扩展CUDA分配器并在推理之间清理缓存。最长三种输入各强制生成600token均通过，峰值分配约10.06GiB；完整63次随后通过。模型复制到共享内存并逐文件核对源hash，12文件约8.3GiB；不是重新下载或更换权重。

官方33个简化物体GLB和UmeTrack手网格代码已按固定版本下载并校验，共约3.1MB物体资产。4段16帧48组手/物体配对完成投影检查；官方FK与独立手腕变换的最大差1.72e-7m。四张叠加图已目测，基本对齐。距离定义为离散手网格顶点到物体三角面的最小无符号距离，不是连续双表面精确距离、穿透深度或物理接触真值。

批量扩展到同一14段开发片段，每三帧取样并保留原定审查帧，共784帧/8736组配对。两进程用时103.9秒（共享机器观察值）。1969组计算精确的上述采样距离，其余6767组通过保守包围盒下界证实远于10cm，保留缺失值/下界而不是填零。0.5/1/2cm阈值下持续至少0.2秒且至少3个采样的区间分别33/37/38段，伴物体位姿变化的区间30/30/29段。阈值变大可合并区间，数量不必单调。全部14段原注释聚合hash重新核对；先前48个距离点中24个精确值逐位一致、24个正确排除为远距。6项距离/间隔/共同运动测试通过。

1cm条件有14段左边界和16段右边界被片段边缘截断，不能强行赋予接触/释放时间。长柄工具例中手表面在约1.7–1.8秒靠近把手，物体随后明显运动，手腕到物体原点仍相距几十厘米。花瓶右手持续持握的可见片段在约3.7–3.9秒发生距离阈值短暂越界，不能据此标释放。长柄工具持握区间的手腕相对物体位移范围约17.9cm，包含转腕、调整握姿和追踪误差的可能影响；不适合直接用刚性手腕-物体相对位姿阈值硬判抓取。图表和审核在hot3d_proximity_analysis_v1。

发现并纠正助手目测锚点自身的错误：clip-002366原写左手拿起牛奶盒，补充投影和世界坐标轨迹后该拿起判断证据不足；物体全片追踪位移范围仅约1.24mm，投影支持盒子仍在原桌面，手靠近后收回。原锚点与原教师审核按hash保留，纠正记录在teacher_geometry_analysis_v2/anchor_correction_after_geometry_review.json。这个例子进一步说明当前助手目测不能被当作独立真值来报告教师准确率。

实际进度：E1三组约2.67/5轮，E2两组约1.21/5轮，训练进程身份均再次核对。全部1516段RGB缓存完成，用时约90.5分钟。DINO缓存控制器实际存活，GPU3当前剩余约6.2GiB，低于既定8GiB门槛，因此等待；正式四组训练和两级分析监听器保持存活，未重启。尚无这些新干预的闭环成功率。外部GPT请求0、准入教师标签0。下一步以更短、明确关联目标物体的证据窗口诊断语义标注，并继续完成同预算机器人对照；不把候选区间数量或离线预测误差替代成功率。


## 第九次更新：2026-09-30T09:20:14.926648+08:00

完成明确目标、短窗口的42次本地Qwen3-VL-4B诊断：7个训练开发片段，各含运动目标、同画面静止目标、完全重复帧三种窗口，raw/marked两种输入。两组都提供目标名称和像素框，marked只额外画框。与三维追踪代理一致的运动目标分别4/7和5/7，静止目标都为4/7，重复帧都为7/7；42/42格式合法且无输出截断。加框仅多对1例，不据此声称总体改进；没有独立语义准确率。全部真实输入与预检一致，原始输出重新解析，评分轨迹标签重新计算。

重新看了6个诊断窗口的原始/加框画面，包含正例与失败例：放下鼠标并撤手在两组中均有粗粒度视觉支持；移动壶的片段中手仍在把手，marked已声称撤手；静止花纹杯被误当成手中正在移动的白色容器；食品包装和手机的运动no虽然正确，文字却描述了不受画面支持的持握/释放。加框在白色容器双手配合顺序、桌面手机状态描述上也有局部改善。六例是读输出后选的诊断，助手目测不是独立人工GT；不统计为语义准确率。

全部21个窗口、42条输出、源图和追踪代理已打包为可离线浏览的 teacher_target_analysis_v1/review.html（约16.5MB）。21张嵌入图片逐张hash与原始审核图一致，42条结构化输出完整对应；六例点评和所有证据hash保存在semantic_diagnostic_review.json。重复帧不报运动，不能证明对象、接触或阶段语义可靠。这批教师标签准入仍为0；停止在同7段上继续调提示词，后续强教师需要独立人物/录制的字段级语义评审，不能用运动一致性替代。

GPU3空余显存低于原8GiB门槛时，已把尚未启动的HOT3D后续工作改排到GPU1。只终止了经身份/空子进程检查确认空闲的4个监听器，未终止运行中的缓存或训练子进程；复用已完成并核验的RGB和审计。训练计划的变化仅gpu、上游进程身份及资源变更记录；模块、初始化、数据、batch128、5轮/1100次更新、损失与随机种子全部不变。两次先前安全检查中止记录仍保留。此次独立复查所有79项HOT3D固定文件hash和前后计划通过。新控制器1354706/1354829/1355351/1355412及特征子进程1354767均已核验实际身份；子进程CUDA_VISIBLE_DEVICES为1。

当前E1：uniform 2.92/5，event 2.92/5，shifted 2.92/5；E2机器人迁移：true 1.58/5，shifted 1.58/5。HOT3D DINO缓存524/1516，RGB1516已完成；四组正式预训练等待缓存结束。新增机器人闭环成功率仍未产生。训练状态的ETA基于近期每步计时，未覆盖所有checkpoint/I/O开销，不作精确完成时间承诺。外部GPT请求0、准入教师标签0；完整目标继续保留，机器人迁移、多种子、unseen与强模型/agent收益均未证明。


## 第十次更新：2026-09-30T09:36:25.484264+08:00

已完成全1,516段HOT3D的旋转候选监督与审计，耗时394.1秒、CPU单进程，无新增GPU训练。现有InteractionMotion手腕/物体头只监督XYZ，不能直接表达姿态变化。本次分别保存0.1/0.2/0.5/1/2秒的手腕与物体旋转增量，约46.4MB，位于hot3d_rotation_audit_v1/targets；没有接入现有训练。

定义为固定当前竖直相机中的空间旋转增量 C·R_future·R_current^T·C^T，C为world→当前相机旋转。未来相机不参与定义，固定局部坐标轴重定义会抵消；不意味着人手已标定成机器人TCP。掩码沿用当前查询与物体视野规则，跨时间缺口的目标屏蔽，无效旋转保留NaN与mask。输入不增加未来几何。4项测试覆盖世界/局部坐标变化、相机自运动、恒角速度、时间缺口、缺失未来、互斥分层与人物/录制权重。

全量独立矩阵变换核对最大差5.55e-16；另按预定hash各选9人中的1段，从原始tar四元数独立复算525个有效目标，矩阵最大差2.98e-8、角度最大差1.16e-9度，固定容差通过；9段原注释聚合hash均与提取记录一致。全部1,516个保存文件再次核对hash、有限/无效mask、SO(3)和统计分母。这些检查证明数值转换一致，不证明传感器物理误差为零。

1秒预测目标中，按位移≤2cm且旋转≥10°的诊断规则，训练手腕2154/45916=4.69%，验证361/8640=4.18%；训练物体911/93011=0.98%，验证144/16368=0.88%。短0.1秒训练手腕为6.51%、物体2.27%。这些是重叠query/实体暴露数，不是独立任务或动作数量；位移原点和阈值会影响比例，完整3×3阈值敏感性已保留。旋转存在补充信息，但不足以认定它是机器人成功率下降的主因。

1秒时训练/验证物体中约78.01%/77.59%同时低于2cm和10°阈值。不能称这些物体整段没有操作，也不能把“变化小”当作负接触标签。它提示全场景平均指标易被变化较小的目标主导，必须继续报告预定运动/事件分层；当前固定四组研究已经包含这些分析，保持协议不变。较大旋转多数同时伴位移，不能凭旋转统计直接宣称新损失会改善任务成功率。

在完全相同的可用历史目标上，1秒验证手腕的保持朝向基线误差36.65°，恒角速度外推63.67°；物体分别11.68°和18.20°。0.1秒外推有一定优势，较长时间表现变差。该基线使用特权过去追踪姿态，不是图像模型的公平性能对照；结果仅说明长时轨迹不能简单按当前角速度延伸，不能直接归因于某一种操作语义。

当前E1：uniform 3.17/5, event 3.17/5, shifted 3.15/5；E2机器人迁移：true 1.88/5, shifted 1.86/5；HOT3D特征缓存972/1516。训练与分析控制器身份再次核查，旋转审计进程已正常结束，未重启。新增闭环成功率仍未产生，教师阶段标签准入0、外部GPT调用0。下一步先完成既定对照并检查运动片段/发生变化物体的结果，再决定是否加旋转监督及其错配控制；现有旋转候选文件可复用。研究目标未完成。


## 第十一次更新：2026-09-30T10:07:02.750777+08:00

完成已存档策略命令的接触诊断工具replay_policy_contacts.py。未运行专家示范、未调用策略网络，也没有新增成功率评估。冻结627个Python源文件，51个任务环境源码与原评估hash一致。对L0搅拌任务的成功seed2和失败seed5各执行500步，分别开/关只读物理观察器：每步关节状态与原存档逐值相同、原末步各项指标匹配；开关两次共1002组完整物理状态/RGB hash逐一相同，1000步info记录逐字节相同。每个控制步5个物理子步，共保留5000个接触样本，原生抓取标志与记录力的必要条件相符。特意避免额外调用evaluate/get_info，因为搅拌evaluate会累计进度。

两个案例都约2.9秒抓到勺子、6.6秒首次到达碗。成功seed2在碗附近且保持抓取的累计计数87步，最后计数样本10.90秒，抓取持续约10.10秒；失败seed5只有24步，最后计数样本7.75秒，抓取持续约6.60秒；旧benchmark门槛80步。失败案例离开碗后仍持有勺子，并在抓取结束前回到原位附近，因此这一个案例更支持“有效操作持续不足、较早返回”，不支持简单归为“初次抓取失败”。两者接触前勺子均从第一个控制后样本移动约3.185cm，说明仅以物体运动作为接触起点会误判；记录只包含目标与手指，无法单独确定重力/支撑/其他碰撞中的哪项原因。

这是按成功/失败选出的两例开发集诊断，不是总体原因占比或因果干预；没有试验延长操作是否挽救失败。搅拌成功标准实际上数“抓取且位于碗附近”的步数，并非独立的真实搅拌质量评估。原档案缺少逐步完整物体状态，因此接触轨迹属于经过严格一致性验证的补充重放，不能声称已核对原始每个接触。图与数值：policy_contact_analysis_v1/seed2_timeline.png、seed5_timeline.png、summary.json、task_state_review.json、independent_verification.json。图已目视检查，无标签加入训练。

已在tmux zq_contact_l0_0930启动完整L0开发子集20回合的同一诊断，覆盖5任务×4个原存档种子；复用上述2例、追加18例，单工作进程、GPU2仅用于渲染，物理使用CPU。所有初态、动作状态、结果与观察器开关检查不变，任一失败即停、不自动重启。该队列用于扩大描述性检查，不是新的策略成功率或训练数据。当前状态replaying，已完成1/20。

同时批量检查原始数据导出的元信息及Parquet schema。训练集3000条、60个场景目录，列仅包含关节状态、关节/夹爪命令和索引/时间，没有物体位姿、接触力、任务进度字段。完整manifest还有240条validation，对其仅读schema/元信息，未读图像、动作序列或评估结果。不能把当前夹爪命令事件当成物理接触标签；若进一步监督接触后持续操作，需要来自训练域的可靠标注或轨迹信号，不能混入本次评估接触记录。审计目录original_robot_annotation_schema_audit_v1。

当前E1：uniform 3.49/5, event 3.49/5, shifted 3.49/5；E2机器人迁移：true 2.18/5, shifted 2.18/5。HOT3D特征状态complete，1516/1516；预训练控制器状态full_cohort_preflight。已核查进程身份和最新步数。新增E1/E2闭环结果尚未产生，教师语义标签准入0、外部GPT调用0；目标保持未完成。下一步继续既定训练/评估与全L0接触诊断，再决定是否有证据增加进度/持续操作监督，不能由这两例直接宣布某个新分支有效。

补充运行观察：10:10:23，E1已到3.52/5，E2到2.18/5；进程仍存活，但多个数据加载worker处于folio_wait_bit_common（磁盘页读取等待），故当前速度估计不能只按GPU更新耗时推算。证据io_wait_1790734223.json；未因等待重启进程，后续需检查是否持续I/O瓶颈。


## 第十二次更新：2026-09-30T11:02:30.054026+08:00

HOT3D四组固定5轮/1100更新全部完成。训练28,072查询/1276片段/7人；验证5280查询/240片段/2人22录制，官方test未用。四组所有验证预测由checkpoint重放逐位相同，独立标签/指标复算通过；四组最终CPU/CUDA/sampler RNG一致。检查本次50项主分析和17项辅助分析证据hash。以下为预先约定的1秒主要指标，单位cm，越低越好；物体只列位移>2cm分层。

| 方法 | 手腕全部 | 发生运动的物体 |
|---|---:|---:|
| 只学手腕 | 14.68356 | 20.58366 |
| 手腕+物体 | 14.68510 | 20.57285 |
| 错配物体 | 14.68321 | 20.59377 |
| 冻结编码器 | 14.64437 | 20.57670 |
| 静止预测 | 14.53335 | 20.58366 |

联合训练相对冻结编码器的手腕差+0.407mm，95%录制聚类区间[-0.206,+1.058]mm；运动物体差-0.039mm，区间[-0.217,+0.122]mm。没有足够证据支持有实际意义的提升。相对错配物体标签组，真标签在运动物体上低0.209mm，区间[-0.410,-0.023]mm，虽条件区间排除0，但相对于约206mm误差很小，不应宣传成有用的交互知识或机器人成功率提高。只学手腕组的物体头未经训练，等于零预测，不能作为训练过的物体对照。手腕模型也没有优于静止预测。

区间固定于当前两个验证人物、按22个录制整体重采样，不覆盖训练种子、新人物/新任务或追踪标签误差。辅助可见性分层已完成，当前手腕可观测代理组中联合/冻结/静止误差15.698/15.674/15.604cm；不能把整体无收益简单归咎于手被遮挡。分层宏平均使用各自分母，不可直接按目标个数加权还原全体。全部时长和事件分层保留，无按结果选择checkpoint。资料hot3d_motion_analysis_v1、hot3d_secondary_analysis_v1。本HOT3D模型尚未迁移机器人，不能与正在进行的EgoDex E2实验混淆。

针对弱结果增加训练域拟合诊断：按固定hash从7个训练人物各选两个不同录制的一段视频，每段均匀4查询，共14段56查询，未按运动大小或模型误差挑选。标签、视频和loss与原预训练相同；仅此均衡小样本采用等查询权重、恒定lr，300次全batch更新，因此这里300轮不是完整数据集300轮。两组初始模型字节相同，可训练E和冻结E分别拟合；3项选择/分组测试通过。独立NumPy重建56条实际输入、全部目标和mask及320个指标单元，checkpoint重载预测逐位相同。首步零输出头使E梯度为0，第二步起可训练组E梯度非零，冻结组始终为0且权重不变。

正式5轮模型在这些训练样本的1秒手腕误差13.49cm，静止14.10cm；小样本反复拟合后可训练E为0.208cm、冻结E为0.113cm，运动物体分别0.612cm和0.831cm。可训练E的loss由0.38589降到0.000522；交换同人物的不同录制视频后手腕误差回到23.24cm，倒放帧序为9.37cm，重复末帧为9.69cm。说明当前计算链路能拟合并依赖视觉输入，不支持“完全没有梯度/恒定输出”的解释；这是训练记忆能力诊断，绝不能当成泛化或机器人收益，也未排除所有代码/建模问题。冻结E同样可拟合，说明仅凭小样本拟合不能证明E学到了新可迁移知识。资料hot3d_learnability_v1，图learnability_diagnostic.png。不得把这些小样本checkpoint用于机器人评估主线。

20回合L0存档策略接触诊断全部完成，17/20是原存档基线结果，非新模型评估。逐步关节状态与原存档一致，开关观察器完整状态/RGB及info一致；独立重建10000控制步成功判定，50000物理子步接触记录通过。折毛巾4/4、文件夹3/4、杯架4/4、盘架3/4、搅拌3/4。三例失败均seed5：文件未记录手指与目标文件接触，另一手打开文件夹；盘子有接触但没有达到原生抓取判定；勺子被抓住且运动，但有效操作仅24步，门槛80。它们分别支持接近、抓取建立、抓取后持续操作的描述性诊断，不能据此估总体失败原因占比，未进行因果恢复干预。抓取是仿真力/角度启发式，原成功条件各任务不同；搅拌判定不等于独立真实搅拌质量。资料policy_contact_cohort_analysis_v1/summary.json、episodes.csv、cohort_timeline.png、independent_verification.json。评估接触记录未用于训练。

I/O审计发现训练数据位于机械盘/dev/sdg1。原先整个人类参考文件常驻比例低，不等于实际输入页冷：10帧采样工作集约519MB在审计时已全部常驻；主要冷页是机器人RGB约13.5GB。新增只读单进程缓存预热，nice10/ionice7，限32GiB读取、64MiB/s上限、至少192GiB可用内存，无mlock、无清缓存、无改训练采样/RNG或重启。私有临时文件冷页功能检查通过。一版因启动路径带../与固定绝对路径键不符而在读取数据前安全中止，记录保留；改为规范路径后v3运行。当前状态warming，读7.50GiB、检查3956/11266文件。共享I/O仍有争用，不能宣称预热已提速或给精确ETA。

当前E1：uniform 3.96/5, event 3.96/5, shifted 3.96/5；E2机器人迁移：true 2.72/5, shifted 2.72/5。训练、控制器、分析监听器均按实际/proc身份核查，没有因等待而重启。新增闭环结果尚未完成；外部GPT调用0、准入教师阶段/接触标签0。

下一步先完成这两类既定机器人对照，同时调查预训练的训练量、运动/静止损失构成，以及任务定义是否适合参考编码器。实际hypernet看完整示范（中位跨度3.73秒），本轮只看过去约0.5秒并预测未知未来；完整示范的运动重建和手物体关系学习更贴近其输入角色，值得独立受控试验，但这只是待验证假设。若使用完整视频观察目标帧，应明确叫重建，不能冒充因果未来预测。当前负结果不支持直接把这批HOT3D模型迁移后宣传提升；强教师、更多种子、unseen成功率及跨域共享交互表征仍属未完成目标。


## 第十三次更新：2026-09-30T11:45:11.542933+08:00

完成新的“观察到的示范运动重建”四组研究。数据位于/mnt/sdf/datasets/hot3d_clips/processed/observed_reference_motion_20260930_v2：1516段（1276训练、240验证），仍为7训练人物、2验证人物/22录制，官方test未用。按真实时间从每段缓存范围均匀取10帧，覆盖中位4.700秒、约原片94.63%；缓存末尾约0.27秒未被观察，不将其声称为完整任务结束。每帧token实际为32×1024，打包输入约0.925GiB。新准备脚本初版误将通道写为384，在训练前报错；失败源和输出保留v1，改为从真实缓存读取形状并逐段验证，v2通过全量检查。原有缓存/训练不受影响。

标签为后9个已观察帧相对首帧的手腕/物体原点世界位移，再旋转到首帧竖直相机坐标，单位米。用每个观察时刻的相机投影检查物体可见性，同时运动坐标始终固定，避免直接把头部运动当物体运动；这不消除视觉推断本身的难度。保留追踪/间隔mask，物体还要求两端可见代理有效。无效值与mask配套，未引入接触/阶段伪真值。独立实现按齐次变换重新计算全部27236个手腕、55044个物体有效标签，以及15160个真实输入帧，输入逐位一致；6项几何、冻结/梯度和宏平均测试通过。

四组为只学手腕、手腕+物体、错配整段视频、冻结E。错配保留标签、mask、每片loss分母、人物与精确样本权重，99.92%换片、57.21%跨录制；它改变整段视觉对应关系，不是上一轮的仅物体标签错配，不能当成同一个对照。每组110轮×1276片段=140360次编码器样本暴露，batch128、1100更新，与上一轮22查询/片×5轮的编码器暴露和更新数相同。零初始化临时预测头，E lr1e-5、head3e-4，原稳定机器人checkpoint初始化、同seed42；只更换临时重建头的输出长度，E结构不变。初始权重与前三批监督/RNG检查匹配，错配组只改变输入视频。110轮端点在结果前固定，没有按验证选择ckpt。

全部四组和独立分析已结束。四组各240个验证输入从checkpoint重放逐位相同，最终RNG相同，逐片误差和录制聚类统计独立复算。以下是预定“已观察9个时间点的轨迹平均误差”，单位cm；它不是上一轮1秒未来误差，二者不能横向比数值判断方法优劣。

| 方法 | 手腕全部 | 位移>2cm的物体 |
|---|---:|---:|
| 只学手腕 | 24.2658 | 32.9597 |
| 手腕+物体 | 24.3697 | 33.2843 |
| 错配整段视频 | 24.3209 | 33.2539 |
| 冻结编码器 | 23.1365 | 32.9884 |
| 静止预测 | 21.6546 | 32.9597 |

联合组手腕比冻结E高1.233cm，固定两人物的录制聚类95%区间[0.736,1.740]cm，两个人方向都更差；运动物体高0.296cm，区间[-0.078,0.706]cm。相对错配视频也没有可信优势。所有主指标仍未优于静止基线。因此在本数据、特征、优化预算下，单纯把短历史预测改成覆盖目标帧的长视频重建也没有证明有效。不能据此宣称所有3D预训练无用，也不能认定唯一原因是视频长度、DINO冻结或样本数；但没有证据支持自动迁移这批E权重。重复使用原两位开发人物，不是全新的最终测试；区间不涵盖训练种子/新人物/标注误差。

固定hash各选一个验证人物片段clip-002031和clip-003243，在读取预测结果前固定可视化选择，未按最好/最差效果筛选；图显示跟踪轨迹、联合、冻结两组的双腕位移，坐标是相对各腕首帧的位移，不是绝对手位置。图已目视检查，位于hot3d_reference_seed42_v1/analysis/hash_selected_examples.png，完整数值summary.json/per_clip.json。

补充损失构成审计：训练物体有效时间点中位移>2cm占23.78%，按实际每片权重/分母计占29.48%；但预测零位移时，99.9568%的物体loss来自这些运动目标。不能把“静止目标数量多”直接等同于“loss数值被静止目标主导”。这不排除静止目标对梯度和最优预测分布的影响，也不证明需要哪种重加权。监督审计仅描述既定目标，没有改动训练。

另外核查现有机器人task多正例对比损失：每个anchor有m个正例时，平均负对数概率的下界是log(m)，再乘pair_confidence并平均。因此原始task loss不应以0为统一收敛目标，batch组成变化也会改变下界。在uniform第4轮的固定训练probe（128条、每批16条），实测1.055302，下界1.055011，超额仅0.000291；验证probe（256条、每批16条）实测6.106939，下界0.521107，超额5.585832。验证gap值得继续检查，但不是闭环成功率；训练batch256的下界又不同，不能拿probe数值直接与训练日志比较。此审计只使用已存probe的标签和confidence计算，没有训练或改动模型。源hash与实际运行版本一致，证据在task_contrastive_floor_audit_v1。

只读缓存预热已正常结束，进程已退出；检查完计划文件，实际读取9.10GiB（9,766,749,444 bytes），触及6077文件，用时2898.8秒。工作时其他训练也会加载/驱逐缓存页，因此总读取可小于初始冷页量，个别文件检查后仍有冷页，不声称全工作集现在永久驻留，也没有因果提速结论。未改采样、参数、训练源码或RNG。

当前E1：uniform 4.75/5, event 4.71/5, shifted 4.73/5；E2：true 3.52/5, shifted 3.49/5。这些机器人训练进程与控制器身份重新验证，新闭环评估尚未完成。外部GPT请求0、准入教师语义/接触标签0。

后续优先完成E1/E2成功率对照，再据结果选择方向。本次重建失败使“仅仅缺少未来信息”不足以解释全部问题；下一步需要受控检查空间定位、相机/手物体相对表征、保留原动作监督的co-training，而不是无条件增加3D loss或扩大训练。强教师/agent、可靠语义标注、多种子及unseen/跨域成功率仍未完成，完整研究目标保持有效。


## 第十四次更新：2026-09-30T12:12:30.868973+08:00

针对两次视觉3D预训练未获收益，完成新的当前图像手腕定位诊断，不是预测未来，也没有迁移机器人。这次直接检查冻结DINO网格是否保留可读的手腕空间信息；临时定位头绕开参考编码器E，因此不能据此断言E内4槽聚合足够。

数据/mnt/sdf/datasets/hot3d_clips/processed/spatial_probe_20260930_v1。原1516片段每片从之前10个采样帧中按固定SHA256选择1帧，不按误差/可见性筛片。1276训练帧/7人物，240验证帧/2人物；1777训练、361验证有效腕点，验证225帧至少有一个有效腕点。无有效标签的70训练/15验证帧保留；前者该项loss为0，后者不进入定位指标。标签是每帧世界腕位姿经当前相机逆变换、顺时针竖直旋转、固定目标针孔内参投影得到的(u+.5)/224。mask要求追踪有限、正深度、腕原点在图内、建模可见比例>=0.1。这是可见性代理，不是物理接触或遮挡真值。独立齐次矩阵实现复算全部3032个腕坐标/mask，最大归一化坐标差2.98e-8；全部3032个选定输入特征切片与源缓存逐位一致。记录源提取整文件hash及读取前后stat，本次只重验选定切片，未重新读遍未用的约百GB缓存。3项几何/屏蔽梯度/采样与宏平均测试通过。9人物各一张hash选择标签预览已目视检查，未作为追踪精度真值。

四组为密集16×16、原池化4×8及各自同人物精确权重组内错配图像控制；99.92%改变片段、57.21%跨录制，监督、mask、权重、样本顺序不变。相同67778参数逐token头（LayerNorm1024→Linear64→GELU→Linear2），输出左右腕softmax热力图、soft-argmax位置。Gaussian目标sigma=0.06归一化图宽，用KL监督；没有腕坐标、姿态或mask进入forward。每组40轮/800更新，batch64、lr3e-4，seed42/43/44，共12次训练；固定端点，没有按验证选ckpt。只训练临时定位头，DINO和机器人模型保持原样。输入特征及热图分辨率同时变化，这是整条定位路径对照，不能单独隔离二者。

12次训练全部完成；同seed初始参数、标签、mask、权重、采样顺序及最终RNG相同。独立重新加载12个checkpoint，各240验证预测逐位复现，逐样本像素误差、录制/人物宏平均和5000次录制聚类区间复算通过。训练显存峰值1.09GiB，训练循环总计76.4秒（不包括数据准备/验证等）。GPU1随后因机器人eval仅余917MiB，已验证诊断训练进程退出、控制器无活跃子进程且正等分析显存，停止其等待后仅将独立分析移至空闲GPU0；原plan和全部训练不改。终止后即时cmdline身份检查遇到退出竞态报错，随后/proc不存在再次核实退出；没有重复kill或重启训练。资源变更完整保留analysis_resource_amendment.json。图生成曾因分析未完成被前置检查拒绝，分析完成后正常生成，不影响数值结果。

主指标是224×224图上的平均腕点欧氏距离，先按有效手/片段，再录制、人物平均，最后平均三个种子，越低越好：

| 方法 | 误差(px) |
|---|---:|
| 16×16密集网格 | 15.300 |
| 当前4×8池化网格 | 13.424 |
| 错配图像，16×16 | 28.476 |
| 错配图像，4×8 | 29.634 |
| 训练坐标均值常数 | 29.735 |
| 训练坐标几何中位数常数 | 29.752 |

密集减池化为+1.876px，固定两人物/三个种子的录制聚类95%区间[1.063,2.643]px，两个验证人物及三个种子均池化更好。池化相对其错配控制改善16.210px，区间[14.531,18.042]px。这表明在该有限定位问题上，冻结DINO与4×8池化仍保留可利用的位置线索，不支持“只加patch便能补救3D预训练失败”。不据此证明冻结DINO在所有任务都足够、密集特征普遍较差、原E聚合无损或机器人成功率会提升。腕部定位、对象表面接触、世界3D运动、任务推理是不同目标；两个重复使用的开发人物也不代表新任务最终泛化。可视化沿用预先按hash选择的两张验证图、固定seed42，见spatial_probe_campaign_v1/analysis/hash_selected_predictions.png，绿色圈为追踪投影、橙色十字为预测；图已目视检查。

E1三个原始机器人组均完成5轮/20190更新，训练进程退出，epoch_0005与last checkpoint hash一致；uniform已自动进入80回合（5任务×4级×4seed）开发评估，保持生成参数32步、500步上限，评估checkpoint与源文件hash已再次核查，进程身份及heartbeat表明正在运行。event/shifted等待顺序评估。当前没有完整成功率比较，不能用loss或定位误差代替闭环成功率。E2仍正常训练：true 3.94/5, shifted 3.94/5，两个训练进程、控制器和配对分析等待器重新核实存活。

后续优先完成机器人成功率对照，同时将“冻结输入网格是否含有空间信息”与“E聚合之后是否仍易读、3D/相机运动目标是否适合”分开诊断，再决定保留原动作监督的辅助训练。强教师/agent的可验证使用、可靠语义/接触标注及unseen跨域提升仍未完成；外部GPT请求0，准入语义/接触教师标签0，完整研究目标保持active。


## 第十五次更新：2026-09-30T12:27:23.710454+08:00

完成新的参考编码器聚合诊断，根目录/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/aggregation_probe_campaign_v1。上一轮证明冻结4×8 DINO网格可通过逐token热力图头定位手腕；本轮进一步检查同一个稳定机器人checkpoint内的E聚合之后，位置是否仍易解码。机器人checkpoint保持原样，未训练E、policy或hypernet。

沿用同1516段/每段固定hash选1帧的手腕标签、1276训练/240验证和人物切分；采用已经独立验证的10帧完整参考特征，不再重新筛样本。三种表征均为1024维：1) 将E.in_proj的32×256 token按原4×8位置，均分成2×2四区域、每区域平均得到4×256；2) 原E.spatial_pool四个学习query得到4×256；3) 原完整E对10个观察帧编码后，取目标帧对应的4×256 H。前两组输入相同当前帧，第三组使用整段10帧，是有上下文的次要比较，不是同样的信息预算。三种每维均值/std仅按训练片段计算，std下限0.001，没有用标签/验证归一化。

冻结E的全量1516×3表征，使用另一条路径（显式linear、query/key归一化、attention、矩形区域平均、时序输入）重新计算，全量逐位一致；选定帧与上一轮单帧输入逐位一致，训练独立归一化复验通过。数据/mnt/sdf/datasets/hot3d_clips/processed/aggregation_probe_20260930_v1。对实际E的16帧projected token反序，四query输出最大变化1.55e-6，与注意力集合池化的排列不变性一致。不能因此说没有空间信息：DINO token值本身仍可能编码位置。3个针对区域顺序、训练专用归一化/无效mask、集合池化不变性的测试通过。

每种表征都加同人物、同精确权重的错配控制，共六组×seed42/43/44=18次训练。定位头统一Linear1024→64→4（GELU/sigmoid），65860参数；坐标SmoothL1 beta0.05、原人物/录制/片段权重，batch64、lr3e-4、40轮/800更新。只更新定位头。固定端点，不按验证挑模型。同seed参数初始值、监督、mask、权重、样本顺序及最终RNG全部一致。GPU0上训练循环94.75秒，峰值显存0.083GiB；提取E表征另有开销。启动器曾在子进程exec前记录空cmdline，随后用训练进程自身记录的PID/start_ticks/完整cmdline对/proc核实身份；训练和控制器正常退出，无重启。原记录保留，不将空命令hash冒充稳定进程身份。

18个checkpoint的全部1276训练与240验证预测均独立重放逐位一致，指标和按固定人物内录制聚类的5000次bootstrap复算通过。下表误差为224×224像素，先平均有效左右手、片段、录制、人物，再平均3种子；错配模型的训练误差也用正确图像计算，因此不能解释为其错配训练标签的拟合误差：

| 表征 | 正确输入训练误差(px) | 验证误差(px) |
|---|---:|---:|
| 固定2×2图像区域 | 5.339 | 27.326 |
| 现有4个注意力槽 | 15.191 | 34.055 |
| 时序编码后的对应4槽 | 21.779 | 32.365 |
| 错配图像，固定区域 | 32.599 | 36.949 |
| 错配图像，注意力槽 | 32.129 | 36.691 |
| 错配视频，时序槽 | 30.332 | 33.779 |

训练标签几何中位数常数预测在验证为29.752px。预定主比较（注意力槽减固定区域）+6.729px，95%录制聚类区间[3.414,10.302]px，两个验证人物和三个种子方向一致。时序槽减注意力槽-1.691px，区间[-3.555,0.241]，不稳定；相对各自错配控制，固定区域改善9.622px，注意力槽改善2.635px，时序槽改善1.414px且区间跨0。四槽两种表征都未优于常数预测。固定区域训练5.34px而验证27.33px，也有明显跨人物开发集差距，不能称为足够好的运动表征。

谨慎结论：在匹配维度、定位头和训练预算下，原注意力任务摘要对腕部绝对图像位置的可读性更弱；训练误差也更高，差异不只体现在换人物后的验证。它不证明位置在信息论意义上不可恢复，也不证明增加n_pool必然有效或这是机器人成功率下降的原因。原参考编码器旨在提取跨场景任务信息，丢弃部分绝对图像位置可能符合任务不变性的需求。上一轮13.42px来自逐token热力图路径，本轮换了通道投影、聚合和读出结构，因此两轮差值不能直接当成某一个模块的独立因果效应。更值得下一步受控测试的候选是保留任务摘要，同时给局部空间/手物体关系增加独立辅助路径，并保留原动作监督；本轮没有自动开始该co-training或把失败3D权重迁移到机器人。

定性图沿用上一轮在预测前按hash选定的两位验证人物片段，固定seed42，未按本轮最好/最差筛选。位置/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/aggregation_probe_campaign_v1/analysis/hash_selected_predictions.png；绿色圈是追踪投影，橙色十字是预测。图已目视检查。当前三种子仍只覆盖同两位重复使用的开发人物，不是新的最终测试，不代表unseen任务或接触准确率。

机器人E1三组训练均结束，uniform闭环eval进程正常，当前完成3/20环境，正在L0_TwoRobotPlacePlateRack-v1；其余两组按原控制器排队。每组目标80回合，生成参数保持32步，协议未改，完整成功率比较未出。E2当前true 4.19/5, shifted 4.16/5，两个训练进程、控制器及配对分析等待器重新核实存活。外部GPT请求仍0、准入教师语义/接触标签仍0。继续推进机器人闭环结果、可靠空间/接触监督及强agent/跨任务泛化，完整目标保持active。


## 第十六次更新：2026-09-30T12:49:21.424744+08:00

完成局部几何辅助分支的首轮受控研究，位于/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/local_interaction_campaign_v1。这是可用于后续动作co-training的候选模块，尚未改机器人模型或启动该迁移。与前两次世界3D位移预训练负结果不同，本轮先保留每个4×8局部token的位置，只监督当前帧可追踪的手腕/已知物体原点及评估相对位置，未要求现有任务摘要解码所有空间细节。

数据/mnt/sdf/datasets/hot3d_clips/processed/local_interaction_20260930_v1：沿用同一1516固定hash图像/1276训练/240验证、7训练人物/2验证人物，不重新按难度选样本。33种物体词表只从训练片段建立；每帧左右腕+33物体通道，每个通道输出二维位置。有效训练标签1777腕点、5959物体点、8404共可见腕物体对；验证361腕点、1167物体点、1784对。对应原点距离<=0.3m次要子集训练2307对/验证483对，该距离不等于表面距离或接触。所有物体在训练至少69个有效点，因此预定“至少5点类别”子集与全集数值相同。

标签由世界位姿通过当前相机逆变换、竖直坐标旋转和与图像一致的针孔投影得到。mask要求有限追踪、正深度、原点入图、建模可见比例>=0.1；它不是遮挡或接触真值。独立齐次变换实现复算全部1516×35坐标与mask、相对几何和近距子集，9264个有效点全部核查，最大归一化坐标差2.98e-8、相机XYZ差5.96e-8米；旧腕标签及样本权重逐位不变。原点可能不在物体表面，也不代表接触位置。模型只接收DINO token，不接收追踪坐标、相机姿态、物体可见性mask或盒子。只对有标签通道评分，不评估目标是否存在或误报，不能称为完整物体检测器。

架构为稳定机器人E.in_proj的1024→256线性投影，加独立LayerNorm256→Linear64→GELU→Linear35逐token头，输出35个4×8热力图与期望位置。DINO一直冻结；任务摘要、T/D、CNN和机器人policy未参与这次离线训练。可训练投影262400参数，局部头19235参数。用sigma0.06图宽Gaussian目标KL，腕loss+0.5×物体loss，每片有效目标归一化、原人物/录制/片段权重。投影lr1e-5、头lr3e-4，AdamW wd0.01，梯度裁剪1，40轮×800更新、batch64。共四组（只学腕、联合、物体标签错配、冻结投影）×seed42/43/44=12次，固定端点，未按验证挑checkpoint。

物体错配仅在同人物、精确float32样本权重、完全相同物体mask组内置换整个物体坐标向量；图像、腕标签、所有mask与loss分母不变。70.77%片段换了供体、77.24%有效物体点改变，其余未改变，因而这是不完全破坏的控制；不能把其差异解释为完全移除物体对应关系。初始全模型权重、采样顺序、腕监督、mask、权重及最终RNG在同seed四组相同；只在指定组改变物体标签或冻结投影。2项精准mask/权重错配和投影梯度/冻结测试通过。

全部12次结束，训练循环113.68秒、峰值显存0.178GiB（数据准备/验证另计）。12个checkpoint在全部1276训练/240验证输入上独立重放逐位一致；进一步用float64逐目标/逐腕物体对实现，独立复算17280个验证metric cell及所有种子宏平均，最大差0.00001000px。源文件、数据、checkpoint等hash重新核验。过程无需重启；启动器这次等待子进程完成exec再记录完整命令身份，所有研究进程已正常退出。

下表是224×224图像上的像素误差，先按每片有效点/对，再按录制、人物平均，最后平均3个种子。相对位置定义为物体预测位置减腕预测位置，和对应标注向量相比；不是只比较距离大小。

| 方法 | 手腕(px) | 物体原点(px) | 腕物体相对向量(px) |
|---|---:|---:|---:|
| 联合监督，训练投影 | 14.970 | 25.248 | 30.309 |
| 联合监督，冻结投影 | 17.152 | 27.140 | 33.265 |
| 错配物体标签，训练投影 | 14.934 | 32.674 | 36.873 |
| 训练坐标均值常数 | 29.735 | 41.506 | 50.712 |

联合训练相对冻结投影：腕误差改善2.182px，固定人物/种子的录制聚类95%区间[1.655,2.785]；物体改善1.891px，[1.554,2.257]；相对向量改善2.956px，[2.460,3.467]。两个验证人物和三个训练种子方向一致。相对物体标签错配，物体改善7.425px、相对向量改善6.563px，而腕误差基本相同（+0.036px，区间跨0），支持物体分支确实利用了正确图像—物体对应关系。

存在清楚的多任务权衡：只学手腕为14.420px，联合为14.970px，差+0.550px，区间[0.325,0.794]，联合物体监督没有额外改善腕定位。只学腕组的物体输出未受该loss训练，虽保留完整文件，不把它当合格物体专家基线。近距原点对子集联合22.893px、冻结24.747px；这个次要子集依然不能当作接触精度。多终点是探索性研究，没有家族错误率多重校正；区间只条件于同两个反复使用的开发人物、同三种子，不能代表新人物/任务总体或追踪误差。

结果使局部几何监督成为比直接向全局任务摘要压3D位移更有依据的候选，但尚不是机器人收益。联合组的共享投影参数相对初始参数L2范数变化约7.83–7.91%，并非证明推理特征或动作也只变这些比例；直接替换机器人接口可能改变原策略表现，因此本轮不迁移。下一步应保留原动作loss与原训练样本量，在独立辅助分支上做匹配的无辅助/腕/联合/错配co-training，检查梯度干扰、输入RNG一致性和真实闭环成功率。

图沿用预测前按hash选定的两张验证图、固定seed42，比较joint与frozen；只画有追踪标签的可见通道，不暗示模型能检测目标存在。绿色圈为追踪原点，橙色十字为预测；已目视检查。文件/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/local_interaction_campaign_v1/analysis/hash_selected_predictions.png。已观察到物体误差仍明显，不能称为足够精细的抓取定位。

当前机器人uniform评估完成7/20环境，正在L1_TwoRobotPlaceMugRack-v1，进程身份与heartbeat重新核实；其他E1组按原控制器排队。E2为true 4.51/5, shifted 4.51/5，两训练进程、两个机器人控制器和配对分析等待器均重新核实。完整机器人成功率比较仍待完成。外部GPT请求0、准入教师语义/接触标签0；可靠接触/3D关系/时间信息、强agent和unseen跨域闭环提升仍在完整目标范围，目标保持active。


## 第十七次更新：2026-10-01T00:13:14.008832+08:00

完成局部几何辅助分支的机器人联合训练预检，实验目录：/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/local_cotraining_preflight_v4。先后保留了三份失败的控制器记录：v1训练短跑完成，但独立验证脚本从绝对路径启动时未把源码快照放入`PYTHONPATH`；v2诊断对编译反向图多次保留计算图，触发PyTorch AOT donated-buffer限制；v3把可变的`__pycache__`也纳入源码固定哈希，导入时缓存更新导致完整性守卫停止。没有把这些工程问题当成模型效果。v4关闭诊断进程的编译缓冲区复用，并只固定实际源文件哈希，完成了全部预检。

对照保持原机器人CNN超网络、eef14动作、样本顺序和损失不变，批量256，seed42，原稳定超网络初始化，8步短跑。无辅助、腕点影子、腕点加物体点影子、错配物体点影子共4组。影子组会计算辅助几何loss和梯度，但不会把它加入优化目标。每组都核对前三批机器人输入与RNG、708个原机器人张量、原优化器状态和RNG；三组都与无辅助基线逐位相同，导出模型也可由原策略类严格加载。辅助梯度确实到达共享`E.in_proj`。

安全范围在看数据前定为：辅助梯度相对原机器人总投影梯度，中位数不超过0.25，单步最大不超过1；候选系数为0.01、0.003、0.001、0.0003、0.0001。三种监督的未缩放梯度与原动作梯度中位数比分别约为21048、21346、21218倍；加权后的梯度余弦接近0。即使最小候选0.0001，三组仍未满足安全阈值，因此控制器没有启动真正更新辅助参数的短跑，更没有启动正式训练。该预检只覆盖每组8个训练batch，不是任务效果测试；它证明当前尺度设置不适合直接混合训练，不证明所有归一化方式都无效。

下一步若继续，应新建独立训练抽样的标定计划，预先规定更低权重或梯度归一化方法，再做活动梯度短跑和匹配闭环评估。不能在这批样本或验证集上事后挑权重。E1的uniform机器人评估已完成20/20环境；event与shifted待评。E2的80回合配对评估已完成：true motion transfer结束成功率0.800，对照0.825，差−0.025，95%分层bootstrap区间[−0.100,0.050]；目前不支持运动预训练带来机器人提升。局部2D原点预测仍不是表面接触标签，既有离线定位收益不能等同机器人成功率。

目标继续保持active：还需完成其余E1评估，并研究可信接触/表面/3D时间监督、人类视频信息和强语义agent能否带来可复现的闭环收益。


## 第十八次更新：2026-10-01T00:19:24.653926+08:00

E1评估控制器此前因GPU3空闲显存约11.5GiB、低于原16GiB门槛而等待。uniform已完成20/20环境，event和shifted训练都已完成。核实原控制器仍是预期PID、启动ticks和命令哈希，处于资源等待且无子进程后，停止了这个空闲控制器，并用新恢复脚本把待测评任务迁至GPU0（启动时空闲约21.3GiB）。训练模型、评估任务、seed2/3/4/5、500步上限、策略参数每32步更新、图像编码器和检查点都保持不变。

event组评估进程身份已核验，当前刚启动、0/20环境；完成后同卡顺序评估shifted组。恢复脚本只允许接续已经完成的训练，不会自动重训或覆盖已存在评估目录。GPU迁移仅改变资源位置，不是新的模型干预。此前局部几何共训练预检仍表明当前辅助梯度尺度不安全，未开展正式更新。


## 第十九次更新：2026-10-01T00:38:16.805432+08:00

把HOT3D“手部表面接近+物体运动”候选从14段可行性样本扩为70段训练集试点，目录/mnt/sdf/datasets/hot3d_clips/processed/hot3d_proximity_batch_70_20261001_v1。样本在7位训练参与者中按固定SHA规则每人10段，覆盖48个不同序列；split全部为train。验证split元数据只用于确认participant分组，没有打开验证几何/原始标注。完整输入列表先冻结，再计算距离。处理3570帧、41616个手—物体候选对；对9358对做UmeTrack手网格采样顶点到BOP物体三角网格的距离，另外32258对由AABB下界证明离阈值足够远后跳过。运行约360.7秒、2 workers。

距离小于等于5/10/20毫米且至少3个采样点、持续至少0.2秒的区间分别198/204/214个；区间内物体运动证据182/183/177个。运动定义为区间内物体平移超过2cm或旋转超过10度，5mm组占91.9%，1cm组89.7%，2cm组82.7%。5mm下66个区间两端都可见、71个左删失、74个右删失；采样帧约10Hz，边界只落在约0.1秒窗口。区间中位持续时间1.65秒，故它表示持续近表面段，不是单帧接触时刻。各阈值的区间彼此有重叠，不能相加成独立事件数；70段全是按互动数据来源选出，也不能估计一般视频中的事件发生率。

独立审计逐段重算帧/位姿对齐、精确/缺失mask、最近点距离、间隙处理与区间统计；再次核对36个模型/工具资产hash，并重新读取全部70个原始tar重算标注成员hash。审计通过。这里的“精确距离”仅针对离散UmeTrack手顶点到CAD三角面，不是两个连续真实表面之间的精确距离；未计算穿透深度。模型尺寸、跟踪误差和手网格误差会让距离接近零，阈值尚未对真实接触校准。一个继承的计划说明还误留了“14段”的旧文字，已用带hash的scope correction声明实际70段范围。

在查看图像之前，按哈希从每位训练参与者选1段同时满足5mm、区间内物体运动、前后边界都可见的例子，生成/mnt/sdf/datasets/hypericlzq_runs/interaction_research_20260930/hot3d_proximity_batch_audit_20261001_v1/visual_review/seven_participant_surface_interval_review.jpg。七段都能看到对应物体互动，其中有一段手/物体关联较不清楚；这只是正例视觉合理性检查，样本预选规则只挑正例，所以不是precision或contact准确率估计，没有人工标签进入训练。原帧、事件时间窗和选择hash保存在review_manifest.json。

基于这次批处理可行性与时间估计，已经启动全量离线准备：1276个train片段用于之后拟合，240个独立参与者validation片段只用于评估；明确不含test。当前处理到34/1516段，预计约1.5–2.5小时，尚未开始任何模型训练。数据一旦完成，先用人级验证评估阶段候选检测，再决定是否训练时序局部交互头。

E1事件加权评估仍在原协议下GPU0运行，当前3/20环境，进程身份与心跳已核验；shifted组排在它之后。
