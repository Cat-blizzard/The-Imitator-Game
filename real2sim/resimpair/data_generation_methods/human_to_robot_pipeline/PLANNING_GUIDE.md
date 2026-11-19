# 给 Agent 的人类任务 → 仿真运动程序规范

此规范服务于数据质量，而不是让程序更容易“报成功”。已存在的苹果、书、柜门采集器保持冻结；新随机化在独立实验 namespace 中验证，不能改旧资格记录或放宽成功阈值。

**后续代码审计更新：** 自定义Screw planner未使用碰撞返回值，并可能误把停滞当Success。四个物理试点之后，当前experimental入口已默认停止后续执行；本页历史命令保留用于复现参数，但要等独立strict adapter完成逐路径FK/目标/碰撞验证并通过小批资格后才能再次运行。当前离线 `plan_geometry_validator.py`不会修复或替代live planner。完整证据见 `evidence_20261020d/FAILURE_ANALYSIS_AND_ACTUAL_COVERAGE.md`。

## 三个可组合的 Method 部分

建议统一归入 **Grounded Task Compiler（GTC，任务约束编译）**，包含：

1. **Semantic Task Contract（STC，语义任务契约）**：从完整人类窗口固定角色、初始关系、必要步骤、终态和禁止替代。VLM 可以提出候选和修正建议；程序校验引用、范围、角色唯一性和一致性。不可见项保留 unknown。
2. **Constraint-Preserving Scene Sampling（CPS，约束保持场景采样）**：位置、干扰物、外观、相机分开采样；任务语义保持，先几何粗筛，再真实规划/物理/渲染验收。
3. **Event-Conditioned Quality Gate（ECQ，事件条件质量门）**：分别审查任务/物理正确性、关键事件可观察性、时间与动作完整性。不能用一个美学分数替代这三项。

这些是实现组织名称，并非已经证明的新算法或论文贡献。真正应验证的是：相同尝试/计算预算下，是否提高“物理成功且语义对应且事件清晰”的有效数据率，扩大位置/资产分布，并减少人工修复时间。

## 1. Agent 开始写规划脚本前必须读什么

- 已审核人类窗口：视频 SHA256、原始 FPS、`[start,end)`、source session 与 split。明确是完整任务还是复合任务中的完整原子子任务。
- `planning_contract.py` 及生成的契约。已有苹果具体示例见 `evidence_20261020d/apple_contract.json`。
- 资产的视觉模型、碰撞模型、物理尺度、可用把手/开口/铰链，不凭模型名猜几何。
- 原生参考实现：`PolicyGen/examples/baselines/hypericl_data_factory/real_bowl.py`。
- 独立验收：同目录 `verify_bowl.py`。Agent 不得为让新程序过关而改验收器。
- 可复用接口 `MotionProgram`：`initialize → plan_stage → execute_stage → independent_verify`。该 Protocol 是规范；现有 native adapter 实际复用已有 `solve` 整体执行，不能声称已逐 stage 改造成该 API。

## 2. 任务规格最少应写什么

| 字段 | 苹果换碗实例 |
|---|---|
| 角色 | selected=唯一苹果；source=初始装苹果的碗；destination=唯一初始空碗 |
| 初态 | 苹果在 source 腔体内，destination 未占用 |
| 必要顺序 | 抓取→抬离源碗→运输→目标碗内松开→撤手 |
| 终态 | 苹果完整包含、释放、稳定、实际被支撑；非目标物未改变 |
| 允许变化 | 本体、外观、布局、速度；目标按角色定义时左右方向可以变 |
| 禁止变化 | 桌面拿苹果替代碗内拿；增加另一个空碗造成目标歧义；搬碗或倒苹果替代抓取 |
| 证据 | 人类窗口引用；机器人 raw state/contact/mesh 与视频；独立 verifier 输出 |

“与人类视频很像”不是成功标准；只有物体名相同也不够。任务本身存在二义性时，先修契约，不应直接扩大仿真数量。

## 3. 编写运动程序的规范

**坐标与单位：** 世界坐标米、弧度；每个阶段目标使用“当前对象坐标系 + 局部偏移”。例如抓取来自当前碰撞 mesh 的 OBB，释放目标由当前开口中心和 rim 高度计算。不能记录一次世界坐标轨迹后重复回放。

**推荐阶段接口：**

```python
scene = program.initialize(case)  # 初始化 + 物理静置，仅此阶段允许设置物体 pose
for stage in contract["stages"]:
    require_preconditions(stage, scene)   # 实际观测/物理量，不读 success bit
    plan = program.plan_stage(stage, scene)
    # 所有物理干扰物、桌面、两个机械臂、携带物体均应进入碰撞检查。
    evidence = program.execute_stage(plan, step_budget=300)
    scene = measure_scene()
    check_postconditions_or_stop(stage, evidence, scene)
receipt = export_native_actions_states_contacts_rgb_and_clock()
report = separate_verifier(receipt)
```

上述伪代码是新 adapter 的接口要求。现有苹果实验入口实际调用原生 `real_bowl.solve`，原程序已用实际碗表面点云、Screw/RRTConnect、实际抓取检查和稳定支撑判断。当前扩展把额外静态杂物加入同一场景点云；**完整附着物连续扫掠碰撞仍未实现**，不可声称做到。

**禁止事项：**

- 记录开始后对被操作物体 `set_pose`、teleport、插帧或删掉失败阶段。
- 因规划失败关闭碰撞、降低接触阈值、把静态物体改成无碰撞贴图。
- 用 `stage="grasp"` 当真实抓住；用 `stage="release"` 当已释放支撑。
- 专门依据目标选择背景颜色、相机或杂物，从外观泄露任务答案。
- 重试到成功却只保存成功分母。每个 seed/asset/layout 是一个唯一 attempt；失败、超时和 unknown 均保存。

## 4. 随机化必须分层

| 层 | 具体改变 | 先决条件与检查 |
|---|---|---|
| 位置 | 碗对中心在 x∈[-6.5,3.5]cm、y∈[-5,5]cm，间距25–35cm，方向±0.35rad | 初态语义、全物体初始碰撞、双方机械臂可达性；完整路径重规划 |
| 干扰物 | 当前实验0/2/4个真实资产静态可碰撞实例 | 不增 apple/bowl 等歧义对象；不挡目标腔体；加入渲染、物理、planner world；保存接触 |
| 外观 | 背景/桌面/HDRI来自既有已审资产 | 独立于目标采样；train/validation/test 资产/session 分离 |
| 相机 | 当前实验眼点约 x[-56,-46]cm、y±8cm、z[69,80]cm，FOV1.24–1.38rad | 整段固定；任务对象可见不够，还要保证接触和放置关键事件可见 |

这些是**小批探索范围**，不是已经通过资格的大规模生产范围。原苹果生产器双碗只有独立 XY±1.2cm；扩大范围不能继承旧通过率。

当前 CPU 粗筛包括：保守包围球间距、任务间运输预留区、保守球形可达范围、针孔投影视锥、相机到事件点的遮挡射线。它们只负责拒绝明显不可用布局；通过不代表 IK 有解、不代表指尖不撞、更不代表真实动作成功。

当前额外资产只有已有梨模型实例。2/4个杂物是实例数量，不是2/4种新几何，不应包装成资产多样性提升。下一步应从已下载且完成碰撞/尺寸资格检查的家居资产池引入多个类别和形状。

静态杂物适合首轮保守验证；应清楚标为静态。未来动态杂物需要质量、摩擦、稳定性、姿态/速度/接触完整记录，并保证“未碰/未移动”验收真实可检验。

## 5. 独立验收必须回答的问题

1. 原始人类窗口是否包含初态→必要动作→终态，而非仅 caption 相似？
2. 每个 raw frame 与 native command 时间轴是否完整一致；EEF转换能否还原目标位姿？
3. 是否确实抓取、抬起、释放、支撑；最终网格是否真正进入目标腔体；是否稳定持续？
4. 是否撞了非目标、杂物、另一个机械臂；是否改变不该改变的物体？
5. 是否既有初态/终态画面，也能看懂关键事件？抓取时可以存在合理短暂遮挡，但不能因为全片均值高就忽视整个抓取区间不可辨认。
6. 原生物理通过、视觉证据不足时标记“需要相机/事件复核”，不要否认确实存在的物理结果，也不能直接放进最高质量训练集。
7. 任务预期的遮挡不要罚错：柜门关闭时内部物体不可见是合理初态；开门后的揭示才需要检查。事件条件比全局可见率更合理。

建议输出分项质量：

```json
{
  "semantic_equivalence": "pass|fail|unknown",
  "physical_success": "pass|fail|unknown",
  "event_observability": {"grasp": "review", "release": "pass"},
  "action_time_integrity": "pass|fail",
  "appearance_score": null,
  "decision": "accept_candidate|resample_camera|repair_planner|reject|needs_review",
  "training_admitted": false
}
```

美学/VQA模型可排序已过硬门的候选，不能补足缺失物理证据。低置信度VLM输出应触发增加连续帧或裁剪复核，不能强行转成 yes。

## 6. 当前可运行的命令

在工作区根目录运行；CPU步骤不调用API、不导入仿真器：

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.human_to_robot_pipeline.planning_contract \
  --output NEW_OUTPUT/apple_contract.json

PolicyGen/.venv/bin/python -m data_generation_methods.human_to_robot_pipeline.structured_randomization \
  --contract NEW_OUTPUT/apple_contract.json \
  --assets /mnt/sdf/datasets/hypericl_data_factory_20261016_v1/fruit_adapters_v2/catalog.json \
  --output NEW_OUTPUT/proposals.json --per-level 4
```

本轮已生成 `evidence_20261020d/apple_layout_proposals_v3.json`：12个候选（每层4个）、227个粗筛拒绝。绑定3段已审核 train 人类窗口、唯一 seed 与 exact file hashes。v1/v2开发结果已废弃，不执行。

独立实验入口（单进程、只使用分配的GPU，推荐外层480秒硬超时，不自动重试）：

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PolicyGen/.venv/bin/python -u -m data_generation_methods.human_to_robot_pipeline.experimental_apple_adapter \
  --proposals /mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods/human_to_robot_pipeline/evidence_20261020d/apple_layout_proposals_v3.json \
  --contract /mnt/sdf/dataset/zhouxunzhe/zhiqi/data_generation_methods/human_to_robot_pipeline/evidence_20261020d/apple_contract.json \
  --case-id apple_structured_layout_2026102000 \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/sim_grounded_20261020d/layout_2026102000 \
  --gpu 2
```

第二条使用 `apple_structured_clutter_2026202018` 和不同输出路径；至多4个独立候选，每条保留结果。旧生产collector拒绝显式布局/额外杂物，这个实验入口单独允许并实际接线；不能混淆两者。

资源：CPU PhysX、GPU Vulkan渲染、单进程；既有单条16–25秒原生视频不代表生成耗时，规划失败可能吃满时间预算。每条输出原速224/512视频、raw action/state/contact；内存保存逐帧RGB，长片要留意内存。先不并发占别人的GPU。

## 7. 2026-10-20实际审计发现

三个旧原生例重新运行独立 verifier 均通过：苹果302帧、书本411帧、柜门499帧，均20Hz。

- 苹果每个控制阶段目标可见像素均≥16，整段比例100%。
- 书本整体≥16像素可见率87.1%，但 `book:close` 阶段最少1像素、该阶段≥16像素比例0%；`grasp_reach` 61.5%、`lift` 60.9%。物理合格并不等于抓取过程视觉清晰。这是必须增加事件质量门的具体证据。
- 柜内块整体≥16像素比例41.9%，但开门前隐藏合理，`open:release`之后100%。不能把整体低可见率直接判成坏视频。

这里 phase 仅帮助定位待看的时间窗；物理结论来自raw contact/grasp/mesh verifier，尚未对这些窗口做新的人工/VLM细看。

实际扩展试点（由根执行器启动，不是CPU proposal假扮）：

- `layout_2026102000_v2`：343帧后撤手路径IK/规划失败；阈值未放宽，失败视频与原因保留。说明起终点粗略可达不代表整个任务所有姿态都有解。
- `clutter_2026202018`：475帧（23.75秒）、2个额外静态可碰撞杂物；原生物理、全视频时间轴与动作检查通过，新增robot/object–杂物接触力记录最大0N。需继续独立副验收和RGB关键事件复核，未批准训练。
- 更早的 `layout_2026102000` 为环境构造默认seed与提案seed角色检查时机的初始化错误，未真正记录任务；已修复并保留原日志/input，不能把它从工程失败记录中删除。

这些是少量接口与物理可执行性证据，不能据此报告总体成功率提高或任意杂物可用。后续同预算比较应明确区分初始化工程失败、真实物理attempt失败、视觉待审三类。

## Agent 可直接使用的任务提示模板

> 读取绑定的人类任务窗口、Task Contract、现有资产catalog、原生专家与独立verifier。实现独立namespace的场景与motion program，目标相对实际物体几何解析，不录制世界坐标轨迹模板。初始化阶段才允许设置物体pose；之后仅机器人控制。保留任务初态、必要顺序与终态，所有新增实体有唯一实例ID、视觉/碰撞asset hash，并同步加入物理与规划世界。按位置/干扰物/外观/相机分层，每层先小批资格。每条保存seed、真实角色映射、全部失败、原速RGB、native command/state/contact/时间轴。采用独立原始验收器，禁止放宽阈值；关键事件可见性单独检查。失败先分类为角色/几何/可达/规划/抓取/碰撞/遮挡/时间错误，修改相应模块；不能只换seed刷成功。报告真实支持、尚未实现及物理/视觉未知项，直到小批证据充分才扩量。
