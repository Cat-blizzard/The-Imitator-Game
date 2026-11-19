# 2026-10-20C：关键状态条件与事件纠错的实施结果

本页记录本轮真实完成的模块、运行证据与尚未接通的部分。当前目标是验证 real→sim→合成人类参考的数据管线；没有启动训练，也没有新增真实人类拍摄数据。

**状态快照：人类首尾图及首尾条件视频已生成，独立关键帧查看、193帧本地手部/KLT检查及24帧结构化VLM复核均已完成。10项可见判据返回yes；新视频呈现一只手打开右门、露出柜内方块并撤手。它是较好的任务参考候选，尚未获得完整物理认证或正式训练准入。**

运行目录：[/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/)。

## 1. 本轮的具体输入与改变

选择上一轮失败边界最清楚的“只打开右柜门”案例：真实人类任务窗口、绑定的机器人仿真执行、仿真闭门首图和开门终图，以及上一轮合成人类初图/视频。没有换任务目标来掩盖错误。

旧流程只把人类闭门初图与文字交给 Seedance。柜内物体在这张图上不可见；同时图中左侧一只手、右侧两只交叠手，已有异常会继续传给视频生成器。

新流程增加了三个不同性质的条件：

1. **文字约束**：固定对象角色、右门目标、抓握→开门→放手顺序，以及本例人物呈现预算。它描述意图，不能代替可见证据。
2. **结构化视觉条件**：从仿真开门终图编辑出人类终图，保留柜内多色方块；初图和终图分别以 `first_frame`、`last_frame` 字段实际进入视频请求，区别于只在提示词里提到终态。
3. **检查与阻断**：生成前检查两张图的任务状态、隐藏物体和手实例；生成后用本地检测/跟踪提出疑点，再由独立图像或视频审查解释。

本例使用“画面右侧一只活动手”的呈现方案，因为这个机器人任务只需单臂开门。**一只手不是框架对所有任务的限制**；双手、双臂协作任务应在任务规格中声明不同预算，不能为通过质量检查删除必要的另一只手。

## 2. 现在真正执行到哪一步

| 环节 | 已实现与实际证据 | 自动化范围和剩余限制 |
|---|---|---|
| 接口与格式 | 继续使用pending规范化、持久化提交缓存、严格schema与只读轮询；新首尾图请求已真实成功取回视频 | 只接受已观察到的协议兼容情况；不补语义字段、不因超时重复POST |
| 隐藏状态识别 | 本例明确登记 `hidden_state_condition_required=true`，绑定仿真终态 | 本例任务规格由已有证据和人工判断填写；尚未从任意任务自动推导全部隐藏实体 |
| 补充人类终态图 | `generate_image_with_references` 接收仿真开门终图作编辑目标、人类闭门图作外观参考；本轮已实际生成 | 这是两张有角色的图像输入，仍须检查编辑是否改坏物体与相机 |
| 首尾图检查 | 四图静态 VLM 检查已完成，七项状态/场景检查为 yes，两个候选各报告一只可见手，`issues=[]` | 这是静态图观察，不能证明视频中不出现额外手、瞬移或伪接触 |
| 首尾图生成 | `first_frame`、`last_frame` payload 经本地验证并实际提交；已取回193帧视频，记录请求哈希及两张输入图哈希 | 新视频关键帧中方块可见，支持本例状态保持改善；不能从单条随机生成推出总体因果收益 |
| 人手检查 | `local_hands.py` 在独立 MediaPipe 环境对新旧图片运行；新旧视频均完成193帧检查 | 模型输出是候选实例下界；漏检不等于没有额外手 |
| 交互时间线 | `contact_diagnostics.py` 已在新旧两个视频运行，生成 KLT 位移、手邻近和丢失区间 | ROI 是人工看各自首帧选择；不是仿真投影，也不是通用自动定位器 |
| 疑点分组 | `interaction_triage.py` 把多手、跟踪丢失、运动先于近邻等信号组成有限补查窗口；本轮新视频实际输出0个候选 | 0候选不等于通过；区间选择不等于证据已审完，阈值未校准成准入标准 |
| 视频复核入口 | `review_conditioned_video_20261020c.py` 已消费新视频与triage并真实调用一次Gemini；本例0候选，使用24帧基础审核，10项可见判据返回yes | 随后已接到dispatcher，并实际验证相同证据的缓存复核；首次工具/VLM调用仍是独立入口，不冒充首次全自动运行 |
| 训练配对 | 上轮隔离适配器已完成4条显式配对、12次CPU实际读取 | 本轮未把新视频加入正式训练；旧生产loader尚未改接 |

首尾图与静态检查：

- [人类闭门图](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/human_initial_v1.png)
- [人类开门图](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/human_terminal_v1.png)
- [静态条件检查与决定](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/condition_review_v1/decision.json)

## 3. 与上一轮记录一一对应的四个生成/审核问题

这里沿用[上一轮结果](PIPELINE_BATCH_RESULTS_20261020B.md)的编号：①接口与JSON格式、②整体检查漏多手、③稀疏帧错过关键接触时刻、④首图缺隐藏状态。训练配对是第五个独立问题，不能替代协议问题。

### 1. 接口和JSON格式：恢复协议，不篡改语义

上一轮已将真实出现的 `pending` 规范化为 `queued` 并保留原值，通过已有task ID查询，不重新提交付费生成。本轮沿用这条恢复路径，首尾图任务实际完成。审查输出必须通过与提示词同源的schema；真正缺字段、缺证据时仍拒绝。

仅当原始事件字段和值完整、错误只是“单对象代替数组”时，已有无损适配器才允许容器包装，并保存前后哈希；这不会补真假、时间或事件。新的路由器把未知提交、pending、schema错误放在最前面处理，未知付费POST不得因看不到结果而重发。协议可恢复不意味着内容正确，更不意味着训练准入。

### 2. 整体检查漏多手：静态图、全片检测与局部复核互补

新增独立环境 MediaPipe 检查，容量设为6，避免配置上限为2时把第三只手直接排除。实际发现：

| 输入 | 目视证据与本地工具结果 | 能得出的结论 |
|---|---|---|
| 旧柜门初图 | 目视3只手；MediaPipe只检出2 | 遮挡导致漏检，不能用“检出≤2”自动通过 |
| 新柜门初图 | 目视1只手；MediaPipe检出1 | 支持新图的人物方案，但单个检测器不能证明全图无遗漏 |
| 旧柜门整段193帧 | 169帧检出3手，时间为0.583–7.583秒 | 时间展开让交叠手分开，揭示了静态检测遗漏的实例 |
| 新柜门整段193帧 | 193帧均检出1手 | 与关键帧查看一致，仍不等于检测器具有绝对召回率 |

工具和VLM均有误差，应保留实例位置、证据帧及不确定性。当前路由器已包含全图→局部复核→确认后修图的决策分支，但这个柜门执行器只接通全图静态检查；局部手数复核与再次编辑尚不是已完成的自动闭环。

### 3. 稀疏帧错过接触：局部工具负责找证据，不制造物理标签

已运行的最小诊断由以下部分构成：

```text
人工指定把手ROI与固定柜体ROI
  → goodFeaturesToTrack 初始化
  → PyrLK 前后向一致性筛点
  → 固定柜体点的中位平移补偿
  → 每帧 MediaPipe 手关键点
  → 手与把手跟踪点的2D距离、把手位移和unknown区间
  → triage提出有限补查窗口
```

旧视频193帧的实际结果：初始18个把手特征点、36个背景点；168帧仍有足够跟踪支持；7.0秒起最后25帧失去把手支持，明确记为unknown。约1.58秒开始观察到手接近，主要把手运动集中在3.33–5.83秒。CPU主循环9.69秒，其中逐帧手检测8.00秒。

这些信号可以让审核优先查看接近、开始转门、遮挡与放手窗口。它们不能证明真实抓握力、3D接触或动力学一致；前后向KLT通过也可能追到遮挡的手指。末段丢失不能解释成“把手稳定没有动”。本轮没有部署SAM2或CoTracker。

实际结果：[时间线图](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/tools/old_cabinet_contact/timeline.png)、[诊断帧图](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/tools/old_cabinet_contact/diagnostic_contact_sheet.jpg)、[逐帧JSON](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/tools/old_cabinet_contact/diagnostic.json)。

新视频的同流程检查为193/193帧有足够跟踪支持、无丢失区间、193帧均检出一只手；1.333–6.917秒为2D近邻窗口，约3.8–5.8秒把手位移增加，7秒后手离开而把手位移趋稳。CPU主循环6.84秒。按本例手预算1、最多两个12帧补查窗运行triage后，未产生当前规则的候选；这不是完整质量证明。图像风格与ROI不同会影响可跟踪点数量，不能把“193对168”当作通用感知性能提升。

新结果：[时间线图](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/tools/new_cabinet_contact_v1/timeline.png)、[诊断帧图](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/tools/new_cabinet_contact_v1/diagnostic_contact_sheet.jpg)、[triage](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/tools/new_cabinet_contact_v1/triage.json)。

### 4. 初图缺隐藏状态：增加真正被模型接收的状态条件

路由器发现本例需要隐藏状态、但缺少揭露条件时，返回 `extract_reveal_or_terminal_anchor`。本轮执行器已据此调用终态图编辑；随后调用同一个实际生成payload构造函数核验首尾图角色与哈希，再做静态图检查，检查通过后才提交视频。

这不是把“柜内有方块”重复写几遍，而是给模型一张实际显示方块、柜内位置和门角度的终图。新视频frame117（4.875秒）已经能看到门后部分方块，frame150（6.25秒）更完整；已看帧支持随开门显露，未观察到临近末帧才突然出现。frame175（7.292秒）手已与把手分离，192（8秒）回到桌面。该观察来自24格联系表与选定原尺寸关键帧，不是全片逐帧的物理认证。

缺口：目前只完成此案例的终态条件链路。任意任务的关键状态选择、遮挡可观测性分析、长程多关键状态生成、局部失败段重生成仍待扩展，不能称为全任务自动化。

## 4. 第五个独立问题：训练配对与读取

继续使用上轮隔离适配器解决四项问题：明确的参考—query边、原始动作时间轴和offset、同源派生内容的split约束、绑定实际媒体哈希的缓存。生成参考与机器人执行可在任务级配对，不强行按相同帧号对齐。

机器人观测与动作必须来自同一个query执行；合成人类参考不因“来源于机器人视频”就拥有机器人动作或人类3D腕部真值。当前CPU读取通过只说明接口和数值一致，正式质量发布仍需独立决定。新视频尚未加入训练配对读取、更未开始训练。

## 5. 路由器与执行器的区别

[repair_router.py](r2h_pipeline/repair_router.py) 根据事实与预算返回下一步动作，本身不调用模型。新 [execute_conditioned_pipeline_20261020c.py](scripts/execute_conditioned_pipeline_20261020c.py) 在显式执行模式下已接通以下真实handler：

1. 缺揭露条件 → 生成终态人类图；
2. 条件尚未进入载荷 → 构造并验证实际首尾请求；
3. 缺手数检查 → 调用四图静态条件检查，保留global范围；
4. 条件通过 → 提交一次首尾图视频请求，以已有task ID轮询。
5. 已有生成视频且有完整后处理配置 → 核验或运行局部诊断、triage、帧包和视频复核；存在`no`转`needs_generation_repair`，存在`unknown`转`needs_additional_visible_evidence`，本例全部可见判据yes则停在等待独立配对与质量发布。

每步事实、计划、开始记录与执行结果写入 `cabinet_case/dispatcher_v1/`。`action_started` 不等于已完成；计划输出也不等于已修复。生成后发现旧plan-only状态文件与不可变写入规则冲突，已修复为“保留状态快照＋原子更新最新状态”；旧人物映射也通过严格核验生成v2旁记录，原件保留。第一次恢复停在后处理handler缺失，随后已补上该handler并再次实际运行，退出码0。

新增入口为[postprocess_conditioned_case_20261020c.py](scripts/postprocess_conditioned_case_20261020c.py)。本次配置`max_new_paid_reviews=0`，诊断、triage和VLM结果均严格绑定后复用，没有新增API；执行记录明确标注`prior_tools_and_review_launched_independently=true`和`this_execution_claim=cache_validation_and_reuse`。这验证了已存在视频进入后处理分支的接线和恢复，不把最初的独立工具调用改写为一开始就在主循环自动执行。未来缺少缓存时，可配置最多1次新视频审核；未知已提交请求或缓存不匹配仍阻断。

最新状态为`awaiting_independent_pair_and_quality_release`，`visible_problem_queries=[]`、`unresolved_queries=[]`、`pair_validation=not_run`，质量发布与训练准入均为false。日志见[后处理执行索引](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/postprocess_execution_v1/LATEST.json)。局部再次修图、局部视频重生成、一般任务的自动ROI与最终训练发布仍未完成全自动接入。

## 6. 本轮合理的展示与下一步

展示应并列保留：真实任务来源、仿真首尾状态、旧合成人类图/视频及错误、修复后首尾图、模型实际条件、生成后视频、检测和复核结论。不能只展示最终最漂亮的一条，也不能把审核程序的通过数当作数据有效率。

本轮独立复核围绕以下问题展开：

- 多色方块是否随柜门打开而逐渐显露，而非临近结尾突然出现；
- 是否始终符合本例的一活动手呈现方案；
- 门运动是否发生在可见手部操作之后，放手与撤手是否确实可见；
- 是否为了保住终态而改变了柜体、门铰链、物体或相机；
- 工具unknown能否通过有限连续帧复核补足，补不足则保留unknown。

当前结果支持“条件与诊断模块已可运行、静态与时间展开检查互补，并得到一条比旧例更符合约束的视频候选”。新视频见[实际MP4](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/state_conditioned_human_v1.mp4)。要证明管线总体更好，还需要固定任务与预算，对比首图、首尾状态条件、状态条件加事件纠错，报告误收率、有效配对产率与每条有效样本的成本。训练收益与真实机器人迁移仍没有新结论。

### 审查模型本身也需要限定结论范围

24帧VLM复核的10项可见问题均返回yes，但自由文本仍写了“confirming physical interaction”，并把抽样帧102描述成精确运动起点，`limitations`为空。这里没有采纳这些额外推断：二维生成画面不能认证真实物理；102只是送审帧，局部工具及原尺寸92帧已显示更早的变化。原响应不改写，另存[解释记录](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/video_review_v1/interpretation.json)。结构化可见判据、局部诊断和物理真值必须分别记录。

原`source_manifest`中`original_reference_contains_human_3d_gt=false`字段命名过宽。原EgoDex窗口确有pose数值来源，本轮未核验/使用；缺少人类3D与接触真值的是新合成视频。已保留原文件并附[勘误](/mnt/sdg/datasets/real_sim_human_pipeline/state_conditioned_20261020c/cabinet_case/source_manifest_erratum.json)。

本例总计5个付费阶段：2次图像编辑、1次静态条件审核、1次视频生成、1次输出视频审核；独立CPU诊断不调用API。dispatcher的4阶段预算仅覆盖前四步，不能称整个实验最多4次调用。

接口一手来源、工具部署记录及实际能力边界见[研究核验](research/STATE_CONDITIONING_TOOLS_20261020C.md)。最终软件检查为`data_generation_methods/tests`与`r2h_pipeline/tests`共467项通过，13.51秒；软件测试通过不代替视频内容验收。
