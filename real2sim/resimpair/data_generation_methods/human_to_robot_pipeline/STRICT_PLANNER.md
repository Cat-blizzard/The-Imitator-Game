# Strict Motion Gate：执行前验证规划路径

本模块是新增实验接口，不修改冻结的生产planner，不改变既有成功标签。它把“规划器返回Success”与“允许发送给控制器”分开。当前只实现苹果换碗这一完整原子任务，仍需原失败seed与原通过seed的物理资格测试；不能据此宣称全部任务均已修复。

## 真实接线

`strict_apple_adapter.py`复用旧实验namespace的场景构建，执行部分改用`strict_motion.solve`。每次启动把adapter、strict motion和scene register的精确源文件写入输出目录的`source_archive`，与`input.json`哈希绑定。每次尝试使用新输出目录，无覆盖、无自动重试。旧`experimental_apple_adapter.run`继续默认阻止扩量。

`strict_motion.make_solver`新建安装包里的`mplib.planner.Planner`。不会调用旧自定义Planner的Screw实现。规划候选依次使用官方Screw、官方RRTConnect；二者都必须经过同一个独立门，不能因为fallback返回Success就跳过检查。

每一段保存：

- `request.json`：阶段、模式、世界坐标目标、当前完整9维qpos、robot root、点云和附着物开关、原生帧索引。
- `planner_path.npz`：时间参数化后的完整position/velocity/acceleration/time数组。
- `guard_path.npz`：所有command点和最大0.015rad关节插值的dense q、独立world TCP FK。
- `decision.json`：原planner status、拒绝原因、终点误差、工作空间、碰撞证据、是否真正允许执行。

只有`accepted=true`的路径可以到`follow_path`。未通过路径与失败episode仍然保留。原生物理成功、持物/接触、时间轴、动作和RGB质量验收仍独立执行。

## 执行门的检查

1. 有限数值、关节范围、起点连续性、command间隔不超过0.08rad、最多300个command。
2. 使用独立Panda FK检查**时间参数化后**的路径，世界目标误差不超过2mm和0.03rad。小的优化增量不代表到达目标。
3. 整条TCP路径在记录的工作空间内。抓取下降与放置下降额外限制相对目标的XY走廊和最大高度，拒绝绕到桌外再回来的“局部下降”。
4. 每个dense q重新计算官方planning world碰撞：self、源/目标碗、另一机械臂、未选对象、所有新增静态杂物。实际collision mesh与planner点云绑定同一actor，不允许只渲染杂物。
5. **完整桌面**的真实物理mesh由额外逐link FCL检查；仅明确豁免`panda_link0–table`这一固定安装接触。官方planner的table点云只裁固定mount footprint以避免其缺少ACM的限制；独立桌面检查没有这个几何孔洞，其他link与held object都检查完整table。
6. v3在夹爪关闭且真实`is_grasping`成立后，立即按实际`T_tcp^-1 T_object`安装held mesh，初始lift也必须检查。通过独立FCL将held物体与**所有**world实体以及完整table检查，不依赖安装版mplib的attach碰撞语义。离开源碗后再次实测并更新持物transform。
7. selected物体即使尚未被抓住，也按实际pose与所有机器人物理link检查；只有`panda_hand/panda_leftfinger/panda_rightfinger`在抓取、提升、搬运、放置、撤手阶段可以接触它。其他link没有豁免。

这里“整条”指全部command加dense关节插值，**不是连续时间碰撞的数学证明**，也不保证物理控制误差为零。全dense点检查和真实物理contact验收不可互相替代。

## 不再用固定高度弥补几何不确定性

原strict v1在跨侧运输使用固定`TCP z>=0.31m`和唯一腕部朝向。实跑表明目标可能不可达，不能通过把2mm终点阈值放宽到厘米量级解决。

v2根据所持物体的真实collision mesh计算TCP局部顶点，再依据目标碗中心、碗沿、源到目标通道内障碍的顶面计算运输目标。运输时held-object bottom至少高于碗沿8cm；通道障碍另留4cm。放置目标保持原生的“物体底部在碗沿上方4cm”。真正放置前重新观测物体bounds，而非假设command被精确跟踪。

对苹果这个对称物体最多尝试四个保持向下抓取的yaw候选（0、+90、−90、180度）。每一个都分别规划和检查，并写入`transfer_goal_candidates.json`。它们没有改变任务目标、源物体、目标碗或物理成功条件。其他任务不能盲目复用这些对称自由度。

## 仍需明确的接触边界

- v3只允许明确的手掌/两指–selected物体接触，接触其他机器人link会拒绝。夹爪close/open仍由原生控制与真实接触验收，没有宣称其每一控制子步都经过独立finger路径规划。
- v3初始lift的源碗底支撑豁免严格限定为：相对起点TCP仅上升0–2mm（容许0.5mm数值/初态下偏差），XY变化≤3mm；全部FCL contact点低于实际floor+5mm，penetration≤1mm，且contact条数未达到截断上限。它们写入`allowed_initial_support_contacts`。随后任何held-object与源碗碰撞都拒绝。lift整条路径还受起点XY走廊与单调上升约束；不能通过“还在lift phase”无限允许接触。
- 另一机械臂以当时mesh snapshot加入规划世界；本程序要求它保持不动。双臂协同需要同步轨迹碰撞接口，不能沿用此假设。
- planning mesh/物理mesh、离散collision与实际控制之间的差异仍须用原生contact记录发现。不要把phase名称写成contact ground truth。
- 这些限制必须随receipt报告，不能因为某条成功就删除。

## 当前证据和不应混淆的分母

旧管线先前有5次进程尝试：1次环境初始化工程错误（0帧），4个真实物理候选（2通过、2失败）。详见`evidence_20261020d/FAILURE_ANALYSIS_AND_ACTUAL_COVERAGE.md`。原始证据不改。

strict v1新增1次layout2001资格尝试，123帧后安全停止：前三段Screw候选终点偏差5.6–15.5mm被执行门拒绝，RRT候选终点约10µm并实际执行；随后above-bowl固定目标的RRT IK偏差17.6mm，没有执行不合格路径。它证明门确实接线，**不是成功修复该任务**。v1的源码与日志单独归档。

v2增加完整table guard和几何运输目标，layout2001实际314帧通过原生验收。独立CPU复核其7段accepted路径、383个dense点：Pinocchio经过一次root与独立FK最大差`5.55e-16`，原生TCP与独立FK最大差`9.76e-7`，完整table逐link/held重核验无碰撞；attach局部transform与原生帧123的差`4.31e-8`。证据在该输出的`strict_independent_cpu_audit.json`。

但随后发现不能依赖本机旧版mplib的attach接口：用上述真实apple mesh和机器人q构造6mm障碍，robot没有碰撞，但官方接口在attach开/关都未报held碰撞。v3因此增加独立held-vs-world FCL。相同真实mesh负例，新的独立检查在关闭payload时无碰撞、开启时正确报`held_object–six_mm_obstacle`。这是CPU人为负例，**不是新增机器人轨迹**。证据`strict_v3_real_mesh_attachment_negative.json`，原先漏报的API probe也保留。

CPU测试覆盖中间碰撞但终点安全、假Success终点偏差、局部下降异常绕行、held-object撞table但机器人清空、完整dense正常路径、关节限制与步长、持物几何目标，以及三项真实FCL BVH（6mm持物障碍、selected非手部接触、仅固定base/table pair允许），共10项。它们验证拒绝逻辑，不替代v3真实仿真资格。

v3的layout2001已完成314帧并通过原生验收。accepted lift从第一点开始检查payload，只有dense index0记录合法源底接触：5个contact、最大penetration`2.86e-7m`、最高contact高度`0.013678m`；后续点没有任何支撑豁免。其余资格与新布局结果以各自不可变输出为准，不从这一条推断总体成功率。

本轮最终v3为**15次完成的尝试、14个不同配置**，均通过原生物理验收；g01的clutter2配置重复执行一次，不能当作新的轨迹多样性。14个配置也不等于14个新任务：仍是苹果换碗的布局、干扰物和相机配置。此前v1失败、v2历史尝试及初态冲突仍保留其各自分母，不并入这15次而隐去。

`strict_replay_audit.py`独立重建15次输出的规划世界，复核105段accepted计划、4924个dense点的完整world/self/table/held/selected检查，全部通过；Pinocchio加一次root与独立FK最大差不超过`5.56e-16`。每条输出都有`strict_v3_cpu_collision_replay.json`。这属于重复读取真实执行证据的CPU审计，不新增成功样本。

真实模型是`PandaWristCam`使用的`panda_v3.urdf`。旧`source_pins`里的`panda_v2.urdf`是额外依赖记录，不能当作实际运行模型身份依据。本轮15次已添加`robot_model_dependencies_retrospective.json`，明确这些URDF/SRDF、引用mesh和mplib二进制哈希是**事后捕获**；没有修改原input/receipt或假称当时归档。

未来入口已补：执行前捕获实际预期v3模型及22个URDF/SRDF/引用mesh文件和运行库哈希，归档URDF/SRDF文本，创建环境后再核验每个agent实际URDF及依赖哈希，才允许开始motion。这个补丁只改元数据捕获与剩余限制文案，不改v3 guard执行逻辑，也不要求为记账重跑物理。

```bash
PolicyGen/.venv/bin/python -m unittest data_generation_methods.tests.test_strict_motion -v
```

实际运行须逐条由root安排GPU2，原4个身份先资格，之后才能用新增家居资产候选。不要将CPU通过、proposal通过、fallback失败或单条native成功当作整个新采样范围已经资格。

```bash
PolicyGen/.venv/bin/python -m data_generation_methods.human_to_robot_pipeline.strict_apple_adapter \
  --proposals data_generation_methods/human_to_robot_pipeline/evidence_20261020d/apple_layout_proposals_v3.json \
  --contract data_generation_methods/human_to_robot_pipeline/evidence_20261020d/apple_contract.json \
  --case-id apple_structured_layout_2026102001 \
  --output /mnt/sdg/datasets/real_sim_human_pipeline/strict_20261020e/layout2001_v3 \
  --gpu 2
```

上例输出目录若已存在必须另用新尝试目录，不覆盖。其他case-id必须来自proposal原字段，不编造seed。模型训练未启动；新增样本默认`training_admitted=false`、`production_approved=false`。
