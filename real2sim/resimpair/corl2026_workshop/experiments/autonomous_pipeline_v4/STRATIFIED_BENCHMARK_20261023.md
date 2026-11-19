# 分层能力检查与真实仿真计时：2026-10-23

用户要求开始检验“任意抽取任务会卡在哪里、各部分需要多久”。本轮实现了可复用的冻结抽样与预检入口，并实际完成 3 次新布局物理执行。结果是 **3 条仿真通过原数值验收；另外 6 项仍缺主动资产适配或来源确认；没有新合成人类视频**。不能将它写成任意任务成功率 3/9，也不能将仿真 3/3 写成端到端成功率。

原始实验根：`/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1`。
候选版本：`releases/stratified_r1`，31 个 Python 模块，生成器、规划器、提示、独立物理验收与上一轮基线相同。
只读复查产物：[完整 JSON](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1_audit/results.json)、[阶段 CSV](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1_audit/stages.csv)、[实测耗时图](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1_audit/simulation_stage_times.png)、[双视角初终态](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1_audit/actual_layouts.png)。

## 1. 抽到了什么，实际做到了哪一步

抽样种子 `2026102321`；在固定 ID 上按 `SHA256(seed,stratum,id)` 排序，各层取 3 项。候选池和选择结果先于执行写入不可变文件，不根据结果换任务、换种子或重试。

| 层 | 有限候选池 | 本次选择 | 实际结果 |
|---|---:|---|---|
| 已有技能、新布局 | 5 个任务族 × 12 个布局种子 = 60 | sort_07 / sort_04 / sort_10 | 3 次真实执行，3 次通过数值验收和完整视频解码 |
| 已有搬运机制、新资产 | 12 类已预检的静态家居资产 | 订书机、牛奶盒、汽水罐 | 全部停在主动资产适配；没有用方块替换后计作通过 |
| 可能的新机制、任务指令探针 | 53 条数据集指令中，23 条包含预设机制关键词 | H13 扫饮料条码、H21 清理厨房垃圾、H18 洗锅 | 未取得新的来源视频合同；实际机制保留 unknown，没有启动仿真 |

三次布局随机抽样恰好都属于分类任务，仍保留这一结果。它们只检验同一搬运任务的微小布局扰动和外观变化：物体位置扰动约 ±1.8 cm、支撑物约 ±1.2 cm；不是三个新机制或大范围布局泛化。红块、蓝块分别进入两个托盘，黄块及三个干扰块保持原状；两臂按原脚本顺序操作，本轮不宣称新增双臂同时运动能力。

新机制分层只有任务说明的词汇线索，不是真人视频解释结果。“清理厨房垃圾”可能是搬运，不能仅凭 clean 就断言需要擦拭机制。三个样本的 `actual_required_mechanism` 均为 unknown，不能把它们算作物理失败或已证明不支持的任务。

复用的 5 个来源合同保持原来的 train 来源会话；没有重新调用 VLM。新布局在执行前未用于调整候选代码，但不是独立真人来源的验证划分。全部派生文件继续继承来源关联，不跨训练／验证拆分。

## 2. 实测耗时

单位：秒。每次独立启动一个仿真 worker，GPU 2 串行执行，180 秒超时，无重试。时间取真实 monotonic span 与进程记录，不使用文件修改时间。

| 阶段 | sort_07 | sort_04 | sort_10 |
|---|---:|---:|---:|
| 环境初始化 | 6.65 | 6.40 | 6.55 |
| reset 与记录器初始化 | 0.99 | 0.76 | 0.89 |
| 规划＋物理执行＋同步渲染 | 24.58 | 23.89 | 24.39 |
| 视频收尾 | 0.22 | 0.21 | 0.26 |
| 测量导出 | 0.04 | 0.04 | 0.04 |
| **仿真 worker 总时间** | **36.33** | **34.68** | **35.67** |
| 下游合同编译＋两路完整解码核验 | 1.35 | 1.57 | 1.29 |
| 仿真时间轴长度 | 40.75 | 40.05 | 40.75 |
| 原生状态帧数 | 816 | 802 | 816 |

worker 总时间还包含约 3–4 秒未进一步归因的启动／退出等开销。最后一帧时间不同于实际运行耗时。视频解码 span 是合同编译 span 的子区间，不能再次累加；合同编译在 worker 退出后进行，也不应并入 worker 的已计数时间。

本轮计时支持的判断：已适配任务的双视角轨迹可在约 35–36 秒内完成；仿真内部最大已记录区间是规划、执行和渲染的合计，尚不能判断其中是 IK、控制步进还是 GPU 渲染最慢。下一次物理计时应先拆开这三项，再决定缓存环境、降低渲染频率或改 planner；不能凭合计时间归因。

来源理解、首图、人类视频、视觉审核本轮均未运行，计时是 **未测量**，不是 0 秒。此前 [真实日志审计](PIPELINE_RUNTIME_AUDIT_20261023.md) 记录了视频提交至审核开始约 221–287 秒、新机制首次 49 帧审核约 136–201 秒；这些是其他批次、其他任务，含服务排队／轮询等开销，不能拼成本批已测的完整流程时间。

## 3. 如何验收，怎样避免只报好结果

- 物理验收仍使用冻结的 `event_protocol.py` 与既有 FK 验证器。每条有 12 个实测事件、23 个原生谓词检查，另核对所有状态与命令帧的 FK。覆盖抓取、抬升、到达、释放、支撑、撤离、终态稳定、未操作物保持和机器人间接触。结论范围是记录到的原生状态与接触，不是连续路径完备证明。
- 独立脚本从原始 `measurements.npz` 重新求事件及验收结果，并对比存档合同；没有读取控制器阶段名来代替测量。三条均一致。
- 两路 MP4 全部解码，帧数、原生 PTS 和测量时间轴一致。共 2,434 个原生状态帧、2,431 条控制命令、6 条视角视频。3 份动作数组、3 份初态散列互异；相机数不计为独立动作轨迹数。
- 在内存副本里交换抓取数据的左右臂轴，3 条均被原验收拒绝，错误为缺少声明的抓取事件。原文件不变。该反例只证明原生机械臂身份检查有效，不证明生成模型遵守手侧或 VLM 能辨认人体左右。
- 71 项回归测试通过；进一步验证了测试实际导入冻结 release 的文件，保存了导入路径和 SHA，避免只测到可变开发树。
- 三次 `run --execute` 都自动完成物理合同编译，然后返回 `awaiting_budget_capacity`，没有发出模型请求。未合成人类视频，正式有效配对仍为 0。

生成／验收代码未作本轮性能优化，因此这是一轮能力测量和计时资格验证，**不宣称相对基线速度或视频质量提升**。

## 4. 暴露出的接口缺口和下一轮选择

**主动资产接入是明确缺口。** 12 类资产的已有证据是静态杂物预检；当前 tabletop intake 与抓取 solver 要求方块几何和对应抓取方式。文件存在、有碰撞网格，不代表质量、惯量、抓取坐标、双指开度和释放后稳定性已适配。下一步应复用既有资产选择器／资格注册机制，对一个新主动资产建立可复核的适配，再在其余保留资产上检查边界；不扩大下载数量来代替可操作性。

**任意来源的能力尚未测出。** 本轮第三层只到任务指令，来源窗口、对象角色与必要步骤还没有自动提取结果。下一轮需固定真实来源窗口并在明确的来源理解预算下执行；未知标签保持未知，不由 agent 填一个“正确合同”来绕过模块。

**人类生成仍是另一条未解决链路。** 本轮继承手侧几何修复，却没有新视频，因此不能说抓取手反转已在生成端解决。新版人体审核仍需独立正负例验证；首图、视频和审核预算必须一起分配，再做同规则的基线／候选对照。

**计时已接通，细分还不够。** 已验证环境、执行导出、编译解码的真实运行观测；规划／控制／渲染合并区间和云端排队／推理区间仍需拆分。失败、unknown、缓存命中应继续分开，计时父子区间不相加。资产适配和来源选择所耗 agent／人工研发劳动本轮没有自动计量，不填零。

## 5. 预算、证据与恢复入口

新登记范围只允许本次三个新布局各一次物理尝试：实际 **3/3**；旧工作台 **5/5** 原样保留。API 新增限额 image0/video0/review0，实际均 0；累计祖先 image21/video13/review24 保留，未重置旧账本。没有训练、没有使用 GPU 0/1/3。结束时 GPU 2 回到空闲，所有任务与共享锁释放。GPU 资源的货币成本、既有云服务实扣仍未知。

人类介入：agent 做了基准代码、分层协议和报告；本轮没有逐任务修改 prompt、写审核结果或修补来源合同，也没有失败重抽。这不等于任意任务已无需适配劳动。

实现：[benchmark.py](benchmark.py)，新增 `benchmark-prepare` / `benchmark-preflight`；[只读独立汇总器](../../tools/audit_stratified_benchmark.py)。基线代码和两份旧状态保存在新实验根 `baseline_code/`、`baseline_documents/`；旧视频、旧失败与旧论文正文不变。

可查看的原始仿真：

| 案例 | 对面相机 | 机器人侧相机 |
|---|---|---|
| sort_07 | [视频](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/executions/strata_20261023_layout_sort_07/attempt_01/opposite_high.mp4) | [视频](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/executions/strata_20261023_layout_sort_07/attempt_01/robot_side.mp4) |
| sort_04 | [视频](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/executions/strata_20261023_layout_sort_04/attempt_01/opposite_high.mp4) | [视频](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/executions/strata_20261023_layout_sort_04/attempt_01/robot_side.mp4) |
| sort_10 | [视频](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/executions/strata_20261023_layout_sort_10/attempt_01/opposite_high.mp4) | [视频](/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/executions/strata_20261023_layout_sort_10/attempt_01/robot_side.mp4) |

只读复查命令（输出使用新的目录；不再启动这三条物理执行）：

```bash
export PYTHONPATH=/mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1/releases/stratified_r1:/mnt/sdf/dataset/zhouxunzhe/zhiqi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PolicyGen/.venv/bin/python -B corl2026_workshop/tools/audit_stratified_benchmark.py \
  --root /mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1 \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/stratified_benchmark_20261023_v1_reaudit
```

是否晋级：保留抽样、预检和计时机制，已完成本地资格验证；当前 release 不晋级为视频质量改进。下一轮先补主动资产接入与更细物理计时，再按新预算验证来源和人类视频。未经新注册，不增加本轮物理次数或付费请求。
