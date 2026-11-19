# 2026-10-21 接手、同步多视角与单人约束

> 2026-10-21续研勘误：本文两布局“同一来源”的表述不准确。实际分别绑定2174/6057两个历史真人窗口与录制会话，均非全新来源盲测；旧v1快照及原产物未改变。另在续研中发现本文旧单人候选账本随预算文件路径变化的问题，新统一候选已修复；旧实现不用于新付费实验。详见[续研报告](HARNESS_RESULTS_20261021_V2.md)。以下保留当时的记录与结论范围。

本轮保留原 strict v3、旧生产记录器、历史失败和全部付费媒体。新 API 调用 0，没有训练。源码快照位于 `runs/20261021_harness_v1/baseline/`，接手基线 605 项测试通过；第一轮候选集成后 655 项，第二轮集成后 **667 项通过**。候选没有切换为默认生产实现。

## 已完成的实际改进

### 同次执行记录原视角和对面视角

`strict_apple_adapter.py --multiview` 使用隔离记录器，在一个 `env.step` 的命令前观察中保存 zed2i 224、原 native 512、opposite 512。前两者为同机位双分辨率，因此是 **2 个机位、3 路流**。新增帧映射、共享动作/状态/时间轴、执行 UUID、原始 RGB hash、每相机内外参、媒体 hash，失败路径也保留诊断记录。旧默认入口和严格物理门不变。

冻结方案后在 GPU 2 串行执行两个小样：

| 样本 | 实际帧数 | 原生物理验收 | 独立路径重放 | 同次三流同步 | 闭爪目标≥16像素，原512→对面512 |
|---|---:|---|---|---|---|
| 开发 g00 baseline | 305 | 通过 | 7 段 / 365 点通过 | 三流完整解码、PTS误差0、映射一致 | 0/16 → 16/16 |
| 保留布局 g01 clutter4 | 277 | 通过 | 7 段 / 283 点通过 | 三流完整解码、PTS误差0、映射一致 | 16/16 → 16/16 |

g01 未用于本轮记录器调参，但历史 strict 已使用，且与 g00 同真人来源；不是新来源泛化。固定 16 像素只沿用旧诊断阈值，不是接触或视觉质量真值。开发例还观察到提升阶段原 native 22/39 帧达到该阈值，对面为 39/39；本轮实际查看两个布局的闭爪原/对面帧，开发例中原视角目标被夹爪遮住，对面能看到红苹果。

两次运行分别约 30.31、33.52 秒，仅指本机这一仿真命令的墙钟时间，不能据此估计完整管线效率或多视角的增量耗时。没有做同期单相机耗时对照。

**未通过的复现项不能藏掉。** 与 20261020E 对应历史同 seed 执行比较，命令/关节状态最大差为 `6.617444900424222e-24`；物体状态、时间、TCP、阶段完全相同。历史字节一致门仍失败，综合审计 `passed=false`，同次同步与相机保持单列为通过。没有舍入后重新判通过，也没有把新的字节指纹当新行为。独立审计同意在保留此失败的前提下完成原定第二小样，未扩大至第三次。

首次审计早停导致后续指标未输出，已保留失败与源码快照，再完整输出独立分项；v1.1 又补查 receipt→多视角 manifest 的绑定。原报告保留。基于真实记录的 7 项篡改反例全部拒绝：错相机帧索引、错时间、漏控制步、改原相机、重复机位、错执行 ID、过期 manifest hash。这验证记录接口与证据完整性，不是图像检出率。

入口：

- [记录器说明](human_to_robot_pipeline/MULTIVIEW_RECORDING.md)
- [本轮冻结协议](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/protocol.json)
- [开发独立审计](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/development/independent_multiview_audit_v1_1.json)
- [保留布局独立审计](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/validation/independent_multiview_audit_v1_1.json)
- [开发对面视频](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/development/episode/video_opposite512.mp4)
- [保留布局对面视频](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/multiview/validation/episode/video_opposite512.mp4)

### 单人约束与历史重审

`single_actor_candidate.py` 在首图前要求一个操作者的位置、朝向、左右手同人归属、初始角色和逐任务事件分工；允许手的进入位置改变。视频准备和提交都重新核验独立 `single_actor_v2.1` 证据与首图/计划 hash。仅手数合格、VLM肯定或未知归属不能放行。预算按整个 campaign 累计，未知提交阻断，异步视频在终态查询前占并发槽。

规则由独立分支编写；v2.1 补严格 JSON Schema 验证和视频实际帧数上界，没有改变视觉标签或放宽标准，v2.0 证据保留。19 项候选测试、22 项独立规则测试及已有回归通过；还用真实源执行零预算阻断、用两张历史首图执行实际 gate，付费 call marker 为0。

人工结构计划不是视觉证据。苹果、书本、柜门3个真实开发来源只完成离线计划与请求准备；没有生成新首图，没有证明模型会服从单人结构，也未实现人体姿态/可达性求解。新版候选必须先获得合格首图才能尝试视频，尚未晋级。

直接查看两张旧共同首图和8条旧视频的56张采样帧后：

- 苹果/书本两首图和5条派生视频均呈现两人。苹果修终图的视频仍为两人，回跳修复不等于人物修复。
- 柜门3条视频可见一只手，无法确认完整单人/左右手归属，保留 unknown。
- 10份人物观察为7 reject、3 needs_review；另结合柜门A原/重抽的对象身份失败，8条视频为7 reject、1 needs_review。没有新规则下合格的视频可作为有效配对分子。

固定回归集绑定131个文件引用；10份标注决策全部重放符合预期。冻结边界检测在苹果原/修复两片上实际重跑，各193帧、192邻帧差，结果仍为 `[192]` 与 `[]`。这属于已知开发反例回归，不是自动人体检测率或盲测检出率。预留杯盖新来源仅为后续资格候选，尚无对应合成媒体或已资格的机器人布局。

入口：

- [单人候选说明](r2h_pipeline/SINGLE_ACTOR_CANDIDATE.md)
- [独立规则](r2h_pipeline/SINGLE_ACTOR_ACCEPTANCE_V2.md)
- [候选实际接线证据](/mnt/sdg/datasets/real_sim_human_pipeline/single_actor_20261021_candidate_v2/runtime_qualification_final.json)
- [回归清单](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/revision_v2_1/manifest.json)
- [回归结果](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/regression/revision_v2_1/run_01/result.json)

### 后续循环：从实际状态编译条件，而非沿用相机左右词

第一轮可见性检查直接显示：切对面相机时源碗、目标碗的画面左右关系反转。新增 `sim_state_conditions.py` 复用 `scene_proposals`，仅支持已资格的 strict 苹果换碗；自动从共享状态、真实 grasp、碰撞 mesh、支撑力及相机标定导出初态、抓持建立、提升过碗沿、最后持物、释放和录制末帧6个条件。

开发样本的 grasp 建立为71帧、过沿101、最后持物193、释放194、末帧304。释放194时支撑力为0，不能当作已落稳终态；末帧304单独调用未修改的原物理 outcome 核验，支撑力约0.674 N。旧事件 `frame=305` 是控制执行计数，编译器没有把不存在的第305张录制帧当作终图。

角色固定在执行 UUID + 对象/容器索引及世界位置；画面左右只由相机投影派生。碗的 renderer actor ID 未导出，明确 unknown；支撑力按物体聚合所有碗，终态依赖目标几何和聚合力共同支持，不伪称拥有逐目标碗接触标签。身份、尺寸、ROI只作为仿真参考约束，不能直接给生成图授予通过。

| 独立检查 | 实际结果 | 限制 |
|---|---|---|
| 同源码两个布局编译 | 开发和g01均完成，每例6张条件图 | 两例仍同真人来源、同苹果机制 |
| 条件图/时间轴 | 12张图与原视频重新解码逐像素一致 | 仅仿真媒体，未验证生成图 |
| 相机角色方向 | opposite视角开发左→右，g01右→左 | 不按相机名称硬编码任务方向 |
| 7类真实证据变异 | 错角色、末态回源、无抓取、未释放、无支持、复制错误标定、短视频错时间轴全部拒绝 | 使用副本重绑一致hash，测试实际入口语义；不是新物理失败分母 |

这完成了实际可运行的上游状态条件准备，不等于已接通自动人类锚图生成/审核。两个包均保留 `generation_allowed=false`；新单人首图、终图身份/比例、后续视频接触与连续性仍须独立检查，当前有效新合成人类配对数为0，因此每有效配对时间/成本尚不可计算。

- [编译器与运行说明](r2h_pipeline/SIM_STATE_CONDITIONS.md)
- [开发条件包](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/state_conditions/development/conditions.json)
- [保留布局条件包](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/state_conditions/validation/conditions.json)
- [独立条件与负例审计](/mnt/sdg/datasets/real_sim_human_pipeline/harness_20261021_v1/state_conditions_independent/run_01/audit.json)

## 发现、决策与后续循环

保留同步记录候选用于后续小样；物理门、原相机保持和同次同步已有实际证据，历史跨进程字节一致失败继续记录，不晋级默认。单人候选保留为预算前门控，收益目前是阻断已知错误传播，视觉生成改善未验证。旧报告的局部“支持”只能按原日期、原标准解释。

上述第二本地循环已完成，状态编译候选在苹果两布局的条件准备上局部有效，未晋级默认。下一步将结构计划与状态包接入同一个严格绑定的锚图请求，明确源与目标角色、相机版本和缺失条件；随后用新真人来源/其他机制检验。新增收费生成仍需明确累计预算，不能把准备完成计为图像或视频收益。隐藏对象身份自动检测、其他接触机制严格规划和新来源泛化均未解决。

长期规则与动态状态已分别落到根目录 [START_HERE.md](../START_HERE.md)、[HARNESS.md](../HARNESS.md)、[CURRENT_STATE.json](../CURRENT_STATE.json)。已有资产索引进程未干预；旧续采状态文件虽显示运行阶段，但没有匹配活进程，不自动恢复旧采集队列。没有新增付费预算，旧账单仍未核对。

本轮 GPU2 两次资格预算已耗尽，结束时显存恢复到21 MiB；GPU1/3原训练进程保持。重复启动已完成开发样本的实际探针被入口拒绝，开始记录仍只有2条，没有新增物理尝试。源码和记录器的晋级需要新的明确实验决定，不能直接重用本轮目录或无限重试。
