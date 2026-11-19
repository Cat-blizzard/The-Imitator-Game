# EViQ：证据分层视频质量审查（2026-10-20D）

本文记录已实现、实际运行和待校准部分。EViQ（Evidence-first Video Quality）是便于讨论的方法模块名，不是仅凭命名即可成立的论文创新。

核心规则：**画质、任务正确性、交互证据、仿真物理、训练配对是不同维度。任何一项已确认的硬失败，不能被其它高分平均掉。** 清楚展示失败动作的视频，也可能具有很高的美学或清晰度分数。

## 1. 当前模块与自动化边界

代码：

- [quality_cascade.py](r2h_pipeline/quality_cascade.py)：CPU 检查、质量向量、来源绑定、独立维度门控及 CLI。
- [quality_model_review.py](r2h_pipeline/quality_model_review.py)：匿名候选的独立 VLM 多维序数评分，必须引用真实帧和 PTS。
- [run_quality_cascade_examples_20261020d.py](scripts/run_quality_cascade_examples_20261020d.py)：将已有真实审查证据自动适配为 ledger，实际运行8条视频。
- [review_video_quality_20261020d.py](scripts/review_video_quality_20261020d.py)：prepare-only / 单次执行入口；已有结果缓存、未知提交阻断，不自动重发付费请求。

```text
真人源视频 / 机器人仿真视频 / 合成人类视频
  │
  ├─ 文件与时间完整性：全解码、真实PTS、实际帧数
  ├─ 像素质量：清晰度、曝光、重复帧、固定背景运动代理
  ├─ 任务与本体：来源绑定的事件/角色/额外手审查
  ├─ 交互证据：已有跟踪、接触邻近疑点、连续帧复核
  └─ 仿真执行：原生状态/接触/释放/稳定性验证（仅仿真分支）
        ↓
  reject / needs_review / candidate_in_reviewed_scope
        ↓
  独立配对、动作时间轴与split检查；不能仅凭视频分数进入训练
```

已实现的组件能够单独运行；样例适配器自动消费既有审查记录，不要求重新人工把每条视频打分。通用生产管线仍需把其输出接入统一发布器。当前没有把缺少全部中间帧证据的样本强行变成 pass。

### CPU 检查范围

| 项目 | 实际计算 | 用途与边界 |
|---|---|---|
| 文件完整性 | `ffmpeg -xerror` 全解码；OpenCV 实际帧数与 ffprobe PTS/声明帧数核对 | 不可解码、时间非递增、计数错配是硬失败；可变帧率本身不拒绝 |
| 清晰度 | 最长边缩至384后灰度 Laplacian 方差的分布 | 内容/纹理强相关；低值只能提示低细节或模糊，不是操作质量标签 |
| 曝光 | 灰度≤8、≥247的像素比例与逐帧分布 | 阈值未校准；白桌面/黑柜内不是自动坏样本 |
| 重复与静止 | 相邻缩小帧严格像素重复；相邻平均绝对差分别保留 | 原生视频帧内容不改动；缩小后严格重复也仅是候选 |
| 背景稳定性 | 显式静态背景 ROI 内 KLT 前后向检查、RANSAC 仿射，输出平移/旋转与覆盖率 | 只在声明固定相机且有来源 ROI 时运行；不把目标运动或 Ego 转头当成相机异常；失跟为 unknown |
| 任务/本体/交互 | 读取独立结构化观察及其原文件哈希 | 本模块不会凭像素统计识别苹果瞬移；已知坏样本的结论必须保存，不能被画质“洗白” |
| 仿真物理 | 引用原始模拟执行验证与 receipt | 只对对应机器人执行成立；不能复制成生成的人手视频接触真值 |

有一个实际试跑中修正的设计问题：**整幅图平均变化很小不等于重复帧。** 新柜门视频约70.3%的相邻帧全图平均变化低于初始阈值，但它实际上在正常开门；静态背景占比大造成低全图变化。因此最终版本把它保留为上下文指标，不据此打“冻结”失败。新柜门严格缩小帧重复比例是0。

阈值均为未校准诊断阈值。报告中 `thresholds_calibrated=false`，没有把它们当训练准入阈值。不同相机/分辨率/任务域应分别校准，不能拿仿真白桌面与真人复杂室内直接比较一个清晰度排名。

## 2. 本次真实运行结果

最终输出：

- [8条汇总 SUMMARY.json](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/validated_v2/SUMMARY.json)
- [固定背景跟踪扩展结果](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/background_proxy_cabinet_v2/quality.json)

8条包含上轮4条合成人类视频、新柜门、原始苹果瞬移负例、1条真实 EgoDex 人类视频和1条机器人仿真视频。实际共解码 **1706帧**。普通 CPU 检查单条约 **1.1–2.3秒**；固定背景跟踪案例约 **2.1秒**，这只是本服务器本批短视频实测，不外推所有分辨率或长视频。

| 样例 | 帧数 | 清晰度代理中位数* | 最终筛查状态 | 理由 |
|---|---:|---:|---|---|
| 黑碗苹果左→右 | 193 | 46.6 | needs_review | 核心任务可读；禁止事件的全片覆盖尚缺 |
| 白碗苹果右→左修复版 | 193 | 164.7 | needs_review | 已有角色修复观察；未自动补齐剩余证据 |
| 闭书放入开盒 | 193 | 96.3 | needs_review | 核心语义可读；仍有覆盖边界 |
| 旧柜门三手版 | 193 | **256.2** | **reject** | 已有来源绑定的额外手偏离证据 |
| 旧苹果起始瞬移/终态错误 | 193 | **151.9** | **reject** | 已有对象连续性及最终角色错误证据 |
| 新首尾状态条件柜门 | 193 | 60.6 | needs_review | 抽样可见判据支持；未证明所有未观察区间 |
| 真实 EgoDex 苹果换碗 | 246 | 67.3 | needs_review | 有任务窗口证据；尚缺统一本体/交互审查 |
| 原生机器人苹果换碗 | 302 | 166.0 | needs_review | 原生物理验证在其范围通过；不能代替人机语义配对 |

*最长边384的 Laplacian 方差；**不是模型质量总分，也不能跨内容直接解释为更好**。

最重要的结果是：旧多手柜门反而具有最高的这个清晰度代理。新候选数值更低，不能因此认为修复变差。两个负例保留拒收，属于“质量规则不会覆盖已有失败证据”的验证，**不是新视频模型的异常检出率实验**。

新柜门静态背景 ROI 复用上一轮已检查记录，未重新人工画框。192/192相邻转移有跟踪支持，p95平移约为图像对角线的0.019%。它仅说明该ROI的二维稳定，不证明相机标定、所有物体身份或真实物理。

初版 CPU 输出保留在 quality 根目录，最终修正低全图变化规则后的结果位于 `validated_v2/`；不要混用两个版本。两次都未改媒体、未新生成视频、未训练。

## 3. 独立 VLM 质量评分，不把已有结论塞回模型

旧柜门与新柜门使用完全相同的任务文字和抽帧方案：开右门、左门保持、松手撤手；不提供旧版/新版标签、已有成功失败判断、截图报告或新视频的终态参考图。对193帧视频，首尾各4帧与18个均匀位置取并集，实际 **24帧**。

输出四个序数维度：

1. **任务可读性**：角色、动作和结果是否易于辨认。
2. **时间一致性**：对象和人物身份、形状、运动是否连贯。
3. **人体结构**：手、指、肢体与可见人物是否一致且合理。
4. **交互可见性**：接近、操作、释放、撤手是否有可见证据。

每维0–4或字符串`unknown`；unknown不算0，更不能算4。数值不是概率，不做综合平均。每个数值和确定约束判断都必须引用实际发送的`frame_index`与精确PTS；错误帧、假时间、缺少证据直接阻断。另输出必要事件顺序、任务角色、对象连续性、人体结构四项约束，任何可见违反都不能被画质分补偿。

**三只手不必然等于解剖不可能**，也可能属于两个操作者。相同任务语义的盲评不会偷偷给旧视频一个手数标签；模型额外记录可见手下界与位置，后续再与独立的 actor plan 比对是否计划外多手。这区分了“人体异常”和“偏离人物方案”两种失败。

每次请求通过 `paid_stage` 保存不可变输入、视频/帧哈希和任务ID。已开始但未知结果不自动再POST。模块本身不放行训练。真实调用结果以 `quality/blind_reviews/candidate_a/quality_review.json` 与 `candidate_b/quality_review.json` 是否存在为准；准备输入不等于模型已经评分。

### 3.1 两次实际调用结果：通用高分没有区分人物方案是否满足

两次 Gemini 独立调用已经完成，结果原样保留：

| 同样24帧、同样任务文字 | 任务可读性 | 时间一致性 | 人体结构 | 交互可见性 | 模型实际数出的手 |
|---|---:|---:|---:|---:|---|
| 旧柜门 candidate_a | 4 | 4 | 4 | 4 | 0、23、56、192帧均报告3只 |
| 新柜门 candidate_b | 4 | 4 | 4 | 4 | 0、56、113、192帧均报告1只 |

两次`hard_visible_violations=[]`。**这不是两条都符合生成要求的证据，而是通用序数评分在这两个例子上没有区分力。**

旧版模型并非没看见第三只手。它把左侧手解释成左边一人、右边两手解释成右边另一人，认为两人的动作都合理。因此“人体看起来合理”与“满足原来两只可见手的呈现方案”是两件事。原旧prompt有“右手接近把手、左手放桌面”的文字方案，新的source_manifest明确写`maximum_visible_hands=1`；原盲评只接收任务语义，没有接收这些人物呈现约束。

仍需批判审查评分提示和输出：

- 0–4的解释偏通用，缺少各维度可复现的等级锚点；4分定义为“在已见证据里清楚”，容易出现抽样帧好看即顶格的天花板效应。
- 输出知道躯干和头被裁掉，却直接把手分配给“两个人”，这只是解释假设，不是已经观察到人物身份。下游不能采纳成确定人数。
- 旧版人体结构维度`unobserved_or_ambiguous=[]`，与其全局limitations的遮挡/裁剪并不充分对应。评分需要逐维记录coverage与unknown，不能只有末尾一句通用限制。
- 没有提供可比较的隐藏物体参考，也没有提供actor plan，就不应期待通用质量分自动发现这些条件偏离。
- 两条样本不足以证明模型存在普遍“讨好”或评估失效；可确定的是**本次提示与评分维度不足以验收我们的生成契约**。不能挑一个符合预期的分数再改写历史。

下一版评分应保留“通用感知质量”与“条件满足度”两个独立输出：后者直接比较显式演员预算、对象角色、初终状态、必要事件，并按实际证据覆盖输出unknown。先用固定负例集校准等级/误收率，不能通过让提示词知道某条是旧失败来制造预期分差。

实际响应：[旧版盲评](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/blind_reviews/candidate_a/quality_review.json)、[新版盲评](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/blind_reviews/candidate_b/quality_review.json)。

### 3.2 新增 Actor Plan Guard：用声明条件检查观察，不让高分覆盖偏离

已实现 [actor_plan_guard.py](r2h_pipeline/actor_plan_guard.py) 与 [运行脚本](scripts/run_actor_plan_guards_20261020d.py)。它是零新付费、零新检测推理的决策层，绑定并联合：

1. 有来源的演员方案（可见手上限、可选可见人物上限、声明范围）；
2. 独立VLM报告的实际帧/PTS和可见手数量下界；
3. 已有逐帧MediaPipe候选和手关键点；
4. 原来的四维质量评分，保留但不覆盖前面约束。

原旧方案的数值上限2来自本轮对既有“右手+左手”文字的显式化，报告中明确标为**事后数值化**，没有冒充生成时就存在数字2字段。新版上限1直接读取生成时的source_manifest。两者的人物上限都保留null：没有可验证人物计数，不用`手数÷2`造人数标签。

实际执行结果：

| 对象 | VLM超预算证据 | MediaPipe超预算帧 | 同帧一致支持 | 触发补查 | 决定 |
|---|---:|---:|---|---|---|
| 旧柜门，上限2手 | 4个采样时点 | 169/193 | frame23和56双方都数出3手 | frame0及192数量分歧，导出[0,3]和[189,192]连续帧 | `blocked_actor_plan_violation_observed` |
| 新柜门，上限1手 | 0 | 0/193 | 已送审时点均1手 | 无当前规则疑点 | `no_actor_budget_violation_detected_in_available_evidence` |

旧版frame0与192：VLM说3手，而MediaPipe只检出2个候选。**候选数不一致不等于VLM或检测器必有一方逻辑错误**，也可能是检测器漏检。这里自动生成8张原生连续帧和16张手区域裁剪，方便进一步局部检查。裁剪来自已有关键点范围加边界，保留原帧索引/PTS/媒体哈希；并不是新生成图片。这个分支真实导出了文件，但本轮未新增局部VLM请求，不能写成局部复核已经完成。

新视频“未检出方案偏离”也不自动准入。人物身份未知、未观察时段及其它任务约束仍独立存在。三手超过本例计划属于**计划偏离**，与“解剖不可能”不同。

结果：[Actor Guard汇总](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/actor_guard_v1/SUMMARY.json)、[旧版实际连续帧和手裁剪目录](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/actor_guard_v1/candidate_a/local_review_packets/)。

## 4. 可借用的模型，核验到的能力与限制

检索日期2026-10-20，只依据论文、作者仓库、官方权重页。下述模型本轮未下载或运行；已运行的是 CPU 检查、既有语义证据适配，以及单独安排的 Gemini 质量入口。

| 模型/工具 | 实际评估内容 | 对本项目的合适位置 |
|---|---|---|
| **DOVER / DOVER-Mobile** | 用户生成视频的技术质量与美学两个分支；Mobile约9.86M参数，作者报告可CPU运行 | 大批源视频的轻量技术质量排序；不代表操作、接触和目标正确。优先单独保留技术分支，少用美学淘汰真实杂乱场景 |
| **FAST-VQA / FasterVQA** | 通过时空片段采样提升无参考视频质量评估效率 | 类似轻量画质基线；没有本任务角色、阶段顺序与机器人动作标签 |
| **VBench** | 视频生成的多个感知/一致性/语义维度，按维度使用不同工具 | 选取有用维度，不照搬排行榜总平均作为数据发布条件 |
| **VBench-2.0** | 更细的人体、可控性、常识、物理等能力；含人体异常流程 | 人体异常与实例保持值得借鉴；不是把任意操作视频传进去就可获得可信接触物理真值 |
| **VideoScore / VideoScore2** | 生成视频多维偏好/质量评估；2版显式输出视觉、语义匹配、物理常识三个维度 | 对候选做解释性复核或同任务排序；仍需事件窗口与来源任务约束；分数不等于真值 |

对应一手说明：[DOVER 官方仓库](https://github.com/VQAssessment/DOVER)、[FAST/FasterVQA 官方仓库](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA)、[VBench 官方仓库](https://github.com/Vchitect/VBench)、[VBench-2.0 文档](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md)、[VideoScore 官方仓库](https://github.com/TIGER-AI-Lab/VideoScore)、[VideoScore2 官方仓库](https://github.com/TIGER-AI-Lab/VideoScore2)。

### 部署与许可核验

- **DOVER-Mobile** 官方release存在约42.79MB权重文件，普通DOVER约239.73MB；本轮仅查询发布元数据，未下载。作者报告的CPU用时不能当作本服务器实测。[官方 Mobile release](https://github.com/VQAssessment/DOVER/releases/tag/v0.5.0)
- DOVER/FAST-VQA代码的实际LICENSE是 **S-Lab非商业许可**；DOVER HF权重卡标MIT，两者不同，不能只看HF卡就把整个代码包视为MIT。学术实验与后续商业分发应分别核验对应产物。[DOVER LICENSE](https://github.com/VQAssessment/DOVER/blob/master/LICENSE)、[FAST-VQA LICENSE](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/blob/dev/LICENSE)、[DOVER权重卡](https://huggingface.co/teowu/DOVER)
- VBench主仓库Apache-2.0；调用的各个预训练模型另有许可与资源要求。VBench-2.0建议预下载LLaVA-Video-7B与Qwen2.5-7B，完整套件不是轻量CPU评分器。[主仓库许可](https://github.com/Vchitect/VBench/blob/master/LICENSE)、[预训练与安装说明](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md)
- VBench-2.0目前文档明确支持自定义输入的维度是Human Anatomy / Identity / Clothes、Diversity和Multi-View Consistency；**不能声称它已经无改动支持任意自定义任务的标准Mechanics评分**。人体异常分支先用YOLO-World裁人物/脸/手，再用SimMIM衍生分类器；文档推理路径要求其三类分类器权重，Ego仅见手的情况还要核验人体裁剪是否漏掉手。[自定义评估范围](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README.md)、[人体异常子模块](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/third_party/ViTDetector/readme.md)
- VideoScore代码MIT，官方权重卡Apache-2.0；VideoScore2同样代码MIT、权重Apache-2.0。官方HF元数据中2版约8.292B BF16参数，单权重理论约16.6GB，推理另需激活与视频token内存，不属于本次轻量安装。[VideoScore权重](https://huggingface.co/TIGER-Lab/VideoScore)、[VideoScore2权重](https://huggingface.co/TIGER-Lab/VideoScore2)
- VideoScore2默认示例按2 FPS输入且随机采样生成；短暂接触/早期跳变仍可能落在间隔内。要做可复现实验，应固定采样与解码配置、保存随机种子/输出，并与密集事件窗口审查分开报告；不宣称把默认帧率加密后仍是论文原配置结果。[官方推理代码](https://github.com/TIGER-AI-Lab/VideoScore2/blob/main/vs2_inference.py)

本地环境已核验有OpenCV、PyAV、PyTorch、Transformers，当前没有安装dover、fastvqa、vbench或pyiqa。研究元数据与许可快照位于 [quality/research/](/mnt/sdg/datasets/real_sim_human_pipeline/automation_20261020d/quality/research/)。

## 5. 下一步应如何校准，而不是堆更多打分模型

1. 先按真人/仿真/合成人类三域、取放/关节操作/长程阶段三类机制抽取小型独立审核集。保留正常、遮挡、模糊、额外手、错目标、瞬移、未释放等难例；不能只从已经通过的样本抽。
2. 人工只标必要证据与unknown区间，优先比较“能否作为任务参考”和“哪个事件失败”，而非主观美学。按来源任务与生成lineage拆校准/验证，防止同一次生成的相邻帧泄漏。
3. CPU全量预筛→DOVER-Mobile技术分支候选排序→已有局部手/物体诊断→只给不确定事件调用VLM；先验证是否减少人工分钟与付费调用，且没有增加坏样本误收。
4. 拿固定失败集合对比：仅全局VLM、模型总分、任务证据约束加事件补查。报告误收、错拒、unknown率、难任务保留率、每条合格配对成本。未知太多与拒收太多都是需要解释的结果，不宜把“只保留简单样本”算有效提升。
5. 合格样本内部按任务、机制、物体、布局、干扰程度做覆盖采样。否则只取画面最规整的top分，反而减少用户要求的杂乱和泛化分布。

这一部分可成为method的价值点：**把数据是否有训练价值拆成可审计的证据维度，独立保留不能被平均的错误，并把模型预算用于最缺证据的事件。** 当前8条只验证软件和已知失败保留；还没有证明优于既有论文方法的总体误收率、成本或训练收益。

## 6. 运行示例

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.r2h_pipeline.quality_cascade \
  --video /absolute/path/video.mp4 \
  --output /absolute/path/new_quality_run \
  --origin real_human --camera-mode egocentric \
  --evidence /absolute/path/source_bound_evidence_ledger.json
```

没有evidence也可以先完成CPU检查，此时缺少语义证据的维度为unknown。不会因为“不知道有错误”而通过。

```bash
PolicyGen/.venv/bin/python data_generation_methods/scripts/review_video_quality_20261020d.py \
  --video /absolute/path/video.mp4 \
  --output /absolute/path/independent_quality_review \
  --contract /absolute/path/task_semantics_only.json
```

默认只准备，不调用模型；加`--execute`才进行有记录的单次付费审核。测试入口为`test_quality_cascade.py`、`test_quality_model_review.py`与`test_actor_plan_guard.py`，当前24项通过。软件测试不等于视频语义被认证。
