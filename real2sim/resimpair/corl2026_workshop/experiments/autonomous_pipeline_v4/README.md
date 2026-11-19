# ReSimPair：共享场景中的多任务自动流程

<!-- parameterized-batch-20261025 -->
2026-10-25 最新：[批量场景、接触机制与实际产数](BATCH_GENERATION_20261025.md)。82 次物理尝试中，80 次主机制执行经独立验收与 holdback 后导出 70 条；另 2 次控制探针单列。8 类机制、16 个场景、60 个场景任务组合，10 条同目标轨迹变体；新增载荷杠杆抬升与支撑抽离落料。两个人类视频原审核一过一拒，独立诊断发现运动误拒及终态幅度漏检，两条均暂缓质量准入。新机制来源绑定缺失，正式配对仍 0。

新增入口：`batch_sampling`（布局／任务／轨迹变体）、`batch_campaign`（登记、冻结分母与有界采集）、`contact_extensions`（间接接触）、`batch_dataset`（仅从独立验收导出）、`batch_human_queue`（共享首图／逐任务生成）、`review_schema_qualification`（接口资格和已下载视频的显式复审）。模块均位于当前目录，命令必须显式指定本轮 `--root`，不使用历史 DEFAULT_ROOT。

[合格仿真播放页](/mnt/sdg/datasets/real_sim_human_pipeline/batch_generation_20261025_v1/data/native_v3/index.html) · [人类视频及诊断](/mnt/sdg/datasets/real_sim_human_pipeline/batch_generation_20261025_v1/data/human_candidates_v1/index.html) · [完整证据索引](/mnt/sdg/datasets/real_sim_human_pipeline/batch_generation_20261025_v1/results.json)。本轮调用 image 1 / video 2 / review 6，名义额度成本 12.099994 USD；累计 29 / 22 / 50，人民币实付 unknown。原生实际版本至 `native_r4`，人类实际生成至 `human_r3`；后续预约保护与预算谱系修复单独保存在 `operational_r5`。本批已结束，未训练、未全局晋级，旧数据和旧版本完整保留。
<!-- /parameterized-batch-20261025 -->

<!-- motion-review-wire-20261024 -->
2026-10-24 最新：[运动参考实测与审核接口修复](MOTION_REFERENCE_AND_REVIEW_TRANSPORT_20261024.md)。四类新布局 4/4 原生合格，2,351 状态、8 路同步视频、12 个反例拒绝。开发资格样本实际生成一个运动参考视频并通过采样检查；四个新布局均在首图审核遇到 schema 失败，因此 A/B 视频 0/8 提交，跨机制收益尚未验证。完整响应隔离为 unknown，未重新生成/审核失败案例。

`motion_reference.py` 实现完整原生视频缩放、字节一致的媒体上传、显式 reference_image/reference_video、单次提交和累计额度准入；`motion_comparison.py` 保存冻结分母及共享输入，候选仍 opt-in。实际格式探针原参数 0/2、`responseSchema` 2/2；`review_transport.py` 和 `models.reviewed_json` 已接入需探针证据的可选参数，原验收语义与默认参数不变。`review_wire_r1` 55 个模块、148 项测试通过，真实历史图片人物子集 2/2 格式合格并正确拒绝双人反例，尚未验证完整 49 帧审核。

本轮合计 image 4 / video 1 / review 11，名义消费 6.287660 USD，人民币实付 unknown；累计 image 28 / video 20 / review 44。主试验与两个诊断批次均结束，正式配对仍 0，无训练或后台任务。下一步按报告登记新的完整审核资格与新布局对照；不能把 `comparison` 命令当作重抽旧失败入口。[资格播放页](/mnt/sdg/datasets/real_sim_human_pipeline/motion_reference_20261024_v1/qualification_gallery/index.html) · [新布局与首图](/mnt/sdg/datasets/real_sim_human_pipeline/motion_reference_20261024_v1/comparison_gallery/index.html)。以下各轮结论按时间和 release 理解。
<!-- /motion-review-wire-20261024 -->

<!-- general-evidence-20261024 -->
2026-10-24 最新本地候选：[跨任务过程证据与适用边界](/mnt/sdf/dataset/zhouxunzhe/zhiqi/corl2026_workshop/experiments/autonomous_pipeline_v4/GENERAL_EVIDENCE_20261024.md)。`general_r1` 根据已有合同中的事件、关系和持续约束选择证据，已接入合同与生成准备入口；十类机制、28 个原生合格分支回放完成，六份旧生成条件与提示保持一致。133 项代码测试通过，51 个冻结模块和 355 个证据绑定复核。两条旧搅动提点轨迹仍不足以自动确认完整圈数；固定颜色提点没有泛化成通用视觉识别器。

本轮 API／物理执行／新视频均为 0，未更改原审核或晋级质量管线，也未验证新运动参考生成。下一步按机制和新布局做条件对照，检查有效产率、人工介入和实际成本；只迁移任务必要约束，不要求人类逐帧复制机器人动作。上一付费轮完整账本保留，长期 API 授权有效。以下各轮“当前”结论按其历史日期与 release 理解。
<!-- /general-evidence-20261024 -->

<!-- paid-novel-20261024 -->
2026-10-24 最新付费轮已完成：[新机制人类视频与实际账单](/mnt/sdf/dataset/zhouxunzhe/zhiqi/corl2026_workshop/experiments/autonomous_pipeline_v4/PAID_NOVEL_HUMAN_PILOT_20261024.md)；[六任务对照播放页](/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/gallery_v2/index.html)。3 个场景各 2 个任务，共 3 张共享首图、6 条实际人类视频，18 次模型调用（image 3 / video 6 / review 9）。自动审核 3 过、2 拒、1 unknown；自动通过中的顺时针搅动又出现重复圈数疑点，已暂缓晋级，因此只有两条钩取有当前未发现冲突的采样支持。蓝按钮也有审核误拒疑点。原判定、失败与争议分开保留，不能报“三条可靠配对”。

实际消费 15,644,837 额度单位，按服务商公开单位名义折合 **31.289674 USD**；24 条消费记录、阶段额度变化和总额度计数一致，人民币现金实付 unknown。串行墙钟 83.34 分钟，API 阶段约 68.12 分钟，另含故障诊断／恢复。视频中位 4.65 分钟，49 帧视频审核中位 4.12 分钟。

当前候选 `human_r4`（49 个模块、117 项完整本地测试通过），实际恢复了已扣费审核输出丢失后的其余任务，以及同一次首图调用的 CDN 图片产物；没有重抽旧终态。完整记录在 `/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1`。本批已结束、调用额度已用完，无在途任务；这轮没有新增物理执行或训练。新机制仍缺真人来源绑定，正式 Real→Sim→Human 配对为 0，未全局晋级，论文正文未更新本轮结果。

**长期授权已改变：为生成任务数据可自主调用 API，不再要求用户逐轮许可。** 新实验由 agent 登记有限上限、累计谱系、并发、超时和实际消耗，失败也记账；不重开本批。已接通的付费入口是 `paid_pilot.py`，其他通用付费命令在新调用前也必须接入 `MeteredBudget` 或等价记账。下一步优先解决小幅动作／圈数审核和闭环动作过程条件，再扩大来源／布局。以 `CURRENT_STATE.json`、`HARNESS.md` 和实际进程／锁核对现场；以下零调用、旧 release 和“下一步”只属于各自历史轮次。
<!-- /paid-novel-20261024 -->

2026-10-24 最新：[十类新机制场景](TEN_NOVEL_SCENES_20261024.md)。已按“十类全部首试→统一复盘→30 分支验证→10 次登记修复”完成 50 次物理尝试，28/30 个任务分支合格：8 个场景各 3 个，杠杆和滚珠场景各 2 个。保留 17 次失败；按同一验收 v3 回放，99/99 个已构造反例拒绝。修正功能几何、质量／惯量、夹持方向和指令限位、工具进入间隙；新人物视频与正式配对仍 0，API 0，物理 50/50。已验证原生数据来自 r2/r4；r6 仅含后续输入类型修正和恢复保护，组合滚动未重新物理验证。不是十个写实房间，也没有验证任意真人来源／随机布局。 [合格任务播放页](/mnt/sdg/datasets/real_sim_human_pipeline/novel_scenes_20261024_v1/qualified_gallery/index.html)。当前无后台任务、无训练；下一步是双球组合、杠杆复位和新布局验证。以下记录保留各自时间范围。

2026-10-24 最新候选：[同场景多任务试产 `scene_pilot_r1`](SCENE_TASK_PILOT_20261024.md)。按场景资产自动枚举和筛选，冻结 3×3 任务后串行实测：8 条仿真合格、1 条牛奶盒落放倾倒拒绝；三组初态与两路首帧一致，14 个错合同反例拒绝。90 项冻结测试通过，9/9 物理额度结束，API 0。`prepare-scene-pilot`、`scene-batch`、`scene-package` 分别登记输入、逐项运行、输出精确绑定；当前登记器 campaign ID 固定，不能靠换目录启动新额度。新合同完成性检查要求双视角完整解码。共 3 份共享首图准备和 8 份任务条件，无新的人类视频／正式配对。详情与复现见报告；下方为保留的历次记录，旧“当前”均按其日期解释。

2026-10-24 最新候选：[`active_asset_r2` 主动资产与细分计时](ACTIVE_ASSET_AND_TIMING_20261023.md)。新增有绑定的 `convex_mesh` 动态资产、CPU 碰撞网格稳定预处理、实际 PhysX 几何／动力参数门禁；`prepare-assets` 准备本地有界资格实验。3 次物理尝试保留 1 次几何拒绝，牛奶盒修复实例通过，原保留布局未执行。80 项冻结测试通过，API 0。相机渲染、物理控制、状态接触采集和规划已分别实测；候选资格只绑定实际通过的开发 episode，不修改原静态 catalog 或全局质量结论。

最新实测：[`stratified_r1` 分层能力基准](STRATIFIED_BENCHMARK_20261023.md) 冻结 9 项后完成 3 次新布局物理执行，3 条通过原数值验收；另 3 个资产缺主动适配、3 个指令探针缺来源确认。71 项冻结代码测试通过，API 0，新物理预算 3/3。新入口 `benchmark-prepare --ancestor-root ...` 和 `benchmark-preflight` 保存候选池／分母／阻塞原因；禁止结果后换样本。实际仿真阶段计时已接通，规划／控制／渲染内部及新云端调用计时仍待细分和验证。旧报告按其原时间范围保留。

后续计时审计：[`observability_r1` 的能力边界与耗时报告](PIPELINE_RUNTIME_AUDIT_20261023.md) 核对 33 次调用、17 次仿真尝试、6 条人类视频，新增只读 `profile` 命令和关键阶段计时。65 项本地测试通过，零新增 API/仿真；既有手侧与验收规则不变。最新候选在 `pipeline_profile_20261023_v1/releases/observability_r1/`，真实新实验中的计时覆盖待验证。

2026-10-23 后续增补：候选 `hand_correspondence_r1` 修复旧插接合同 `arm0→left` 的几何映射错误；当前双 Panda 实际应 `arm0→right、arm1→left`。共享 `hand_correspondence.py` 贯穿初图、事件和终态，`hand_review.py` 提供版本化结构证据检查。可用 `replay-hand-bindings --root <历史目录> --output <独立审计目录>` 做零调用回放。

55 项本地测试通过，2 条工作台与 12 条其他原生记录回放完成，原 7 个物理负例仍拒绝。**未重新生成视频；视觉收益与远程新审核 schema 未验证，仍为候选。** [详细证据、边界与复现命令](HAND_CORRESPONDENCE_20261023.md)。以下 r4 记录按原时间范围保留；开发树已更新，旧源码 hash 不再描述当前候选。

agent 参与场景、物理机制、motion planning 和原生测量导出；来源理解、状态编译、首图、视频及审核由统一协议完成。生成阶段不接受逐任务手填来源映射、人体坐标、repair prompt 或审核 JSON。内容失败保留为终态；修改共享版本不会自动重抽原样本。

当前实验根为 [autonomous_mechanisms_20261023_v5](/mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5)，[播放页](/mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5/gallery/index.html)、[完整证据](/mnt/sdf/dataset/zhouxunzhe/zhiqi/corl2026_workshop/evidence/autonomous_mechanisms_v5_20261023.json)、[场景任务分组](/mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5/task_groups/mechanism_workbench_C.json)。代码目录名保持 v4，规划导出使用 v5。**DEFAULT_ROOT 仍指向上一轮，运行必须显式指定 --root。**

## 本轮实测

一个共享工作台包含 9 个刚体、铰链盖板、空心盒和桌面。双 Panda 同侧，两机位同步录制。同一初态执行三类目标：多米诺连锁、持续双臂插接、盖板开合。

- 5 次物理尝试中 2 次完成并通过数值协议：连锁 426 帧、插接 795 帧，共 1,221 帧、4 条同步视频。三次盖板失败为 1/1/158 帧，分别是类型错误、连续 IK 不可行、实际抓持不成立；失败不抹掉，不当独立新任务。
- 五次初始物体、物理和铰链状态一致，两机位首帧 SHA 一致。成功轨迹互换任务协议都被拒绝，表明任务目标和动作确实不同。
- 三条新真人来源通过固定 24 帧模型协议得到 articulate/push/insert 技能；插接还复用旧自动 pick_place 来源。柜门到盖板是明确技能迁移，不是原视频已演示新工作台。
- 两个人类视频实际从同一张首图生成，规划后无人工语义补丁。连锁方向反了；插接被自动审核以左右手分工拒绝，但直接查看与其判断冲突。共享人体侧别说明复核仍未解决。原审核 0/2 通过，未放行训练配对。

历史 [多布局实验](/mnt/sdg/datasets/real_sim_human_pipeline/autonomous_multitask_20261023_v4/gallery/index.html) 有 10 次完成执行、10,182 帧、7 种动态序列、20 条同步视频和 4 个人类视频（旧版采样 2 通过／2 拒绝）。历史五机制回放不计为本轮新执行；旧五任务不重跑。

## 从新来源到共享场景任务组

`intake-source` 登记原生文件、窗口、来源会话和划分；`planning-input --execute` 用固定模型 prompt/schema 提取实体、技能、事件、事实和不确定性。相同技能可迁移至新布局或组合为任务，但映射必须明确。

规划器执行前写入 event_protocol.json 和 planning_declaration.json，执行后交付 execution_bundle.json。它必须完整导出实体与功能几何、事件偏序、初终逐手状态、持续持握／禁接触区间，以及原生关节、物体、TCP、抓持、接触、时钟和相机信息。stage 字符串、手填事件帧号或 supplied success flag 不代替原生谓词。

事件解释器支持被动接触传播、关节运动、插入几何和持续支撑，生成／审核不按 task ID 分派。外部插件用 `register-planning --case <id> --bundle <path> --make-reference` 接入。新机制仍需要物理与测量适配器，属于允许的规划参与；不能声称任意任务无需规划开发。

`prepare-workbench` 提供本轮共享场景插件：多脚本共享 scene.json，目标和操作分开声明。两初始手均空闲时，首图角色归一为 either，以复用同初态首图；执行合同仍保留操作侧别。`run --case <id> --execute` 自动编译 → 首图或复用 → 审核 → 视频 → 审核 → 终态。缺测量明确拒绝；已有 provider task ID 只恢复，终态不跨版本重抽。

`tools/audit_shared_scene_group.py` 从实际导出建立 task_groups：核对初态、资产、两路 RGB，保留全部失败，执行错任务数值对照，关联人类候选和来源会话。整场景组、共享来源会话、一次物理执行的派生物应划在同一 split。反向连锁、另一插孔目前只有条件层反例，没有相应新物理执行或人类视频。

## 共享修订与适用范围

workbench_r2 已实际修复完整 49 帧结构化审核请求的 wire 兼容问题：checks、frame_census、target_instances 用数组表示，decoder 不展开每项，程序仍严格检查 ID、计数和精确帧覆盖。单纯改非流式未解决旧 HTTP 400，精确服务根因仍未知。旧三次新轮 HTTP 400 保留并计数。

相同审核检出了旧堆叠 38 帧额外手，仍漏掉 f48 红块复制。插接的解剖侧别判断与直接查看有冲突，r4 仅加共享肩—肘—腕说明的复核仍拒绝，**不是已修好的质量问题**。原视频和原拒绝均未变。

当前 workbench_r4 增加：

- grounding.py 自动把事件目标与局部功能几何投影到原生相机，输出行列端点、顺序、选定插孔及固定身份规则；不用手填坐标。反向事件图和另一插孔的本地对照通过，原首图条件、原任务和原数值标准一致。生成改善尚未验证。
- 抓持前检查实际 TCP 到目标的误差。盖板第三次在闭爪前偏差 15.47 mm，未形成原生抓持；新增 5 mm / 0.05 rad 门禁和逐指姿态／力向量导出仅本地准备，未再物理执行。
- 共享 prompt 明确人体左右与图像左右的区别。实际复核未解决插接争议，后续需结构化肢体归属与冻结正负例评价。

模型采样判断不能代替物理资格，也不能证明所有帧的人体连续性、人体三维或力。自动流程跑完、仿真成功、生成正确和审核正确分别统计。

## 冻结版本与恢复

source_api_r1 完成三来源；workbench_r1 完成连锁和插接及首盖板失败；r2 修复审核 wire 并生成两视频；r3 做有限抓取方向搜索；r4 是当前共享候选。25 个开发模块与 r4 SHA 一致，35 项本地测试通过，测试源码有不可变快照。

以下只恢复现有终态，不新增付费操作：

```bash
export PYTHONPATH=/mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5/releases/workbench_r4:/mnt/sdf/dataset/zhouxunzhe/zhiqi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PolicyGen/.venv/bin/python -B -m autonomous_pipeline_v4 run --root /mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5 --case mechanism_workbench_C__insertion
PolicyGen/.venv/bin/python -B -m autonomous_pipeline_v4 audit-reviews --root /mnt/sdg/datasets/real_sim_human_pipeline/autonomous_mechanisms_20261023_v5
```

本轮上限 image3/video3/review12；实际 **image1/video2/review12**，谱系 **image21/video13/review24**。余 image2/video1/review0，物理 5/5；无在途、无训练、无 subagent。不可只花剩余生成额度而跳过审核。后续用户授权范围内的新实验须登记累计预算和独立验证目标，不清零、不修改 frozen release、不删除失败。



## 2026-10-24 付费新机制候选

用户已长期授权为任务数据生成自主调用 API，无需每轮重问。`paid_pilot.py` 复用原生验收合格的按钮、钩取、搅动数据，固定 3 场景 × 2 任务，单次共享首图，首图失败不生成派生视频。首次付费执行入口是 `/mnt/sdg/datasets/real_sim_human_pipeline/novel_human_pilot_20261024_v1/releases/human_r2/`（后续以 human_r3/r4 恢复，见顶部最新报告），运行状态见同根 `progress.json`。

新条件由 `mechanism_conditions.py` 自动读取既有物体颜色、构件和路径谓词；两种搅动方向不能只保留相同终态。`mechanism_review.py` 追加关节中途运动／周期方向与圈数检查，保留原人数、身份、顺序和解剖学手对应规则。此为未配对 Sim→人类质量实验，原生合同的真人来源缺失和禁止正式配对字段不变。

`billing.MeteredBudget` 复用不可清零的累计调用账本，额外记录每个提交／审核前后额度，视频到终态再核对。后续新付费实验需接入该计费记录或等价机制；额度计数差、名义美元单位、人民币实付和逐请求账单必须区分。此次上限 image 3 / video 6 / review 9，并发 1，包含失败；语义失败不重抽。`human_r1` 仅在本地编译时发现共享首图的同像素异路径绑定问题，没有付费调用；已保留并在 `human_r2` 修正。
