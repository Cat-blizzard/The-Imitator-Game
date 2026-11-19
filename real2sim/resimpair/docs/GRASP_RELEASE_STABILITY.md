# 抓取、支撑与释放：2026-10-26 调查和候选改进

入口仍在独立 ReSimPair 仓库。当前工作分支为 `candidate/grasp-release-stability-20261026`；默认 main、旧控制器、旧验收器和历史失败均保留。[实测数据](../validation/grasp_release_20261026/RESULTS.json)、[本轮结论](../validation/grasp_release_20261026/RESULTS.md)、[来源与版本](../validation/grasp_release_20261026/RESEARCH_SOURCES.json)。

本轮并没有证明“换一个抓取网络就能解决”。实际发现了四个不同环节的问题。

1. **预期接触点与实体手指先接触的位置不同。** 开发失败物体是缩小后的笔记本支架，旧候选宽度约 18.8 mm。实体 Panda 手指会先碰到约 62 mm 宽的外侧结构，闭合后实际夹持间距约 44 mm，物体倾斜约 13°。两个射线交点满足对向法线，不等于带厚度的手指能接触它们。需要检查开手接近、完整闭合过程和夹持结果，不能只验证 TCP 的终点。
2. **负载几何使用了过时的相对姿态。** 原代码在夹紧时记录物体相对 TCP 的变换；放置目标后来使用新测量，但密集路径碰撞检查仍沿用旧变换。候选在每段携物规划前更新该变换，并保存前后值。它只改变规划几何，没有给仿真物体增加固定关节。
3. **可见底面不一定能承重。** 新抓取使抬升倾斜降到 0.357°，但释放后物体从声明的 206 mm 底面落到桌面附近。该 Fish Bowl 网格在声明底面处的三角形法线为 `[0,0,-1]`，旧双面射线把背面当成了支撑面。PhysX 三角网格接触对面朝向敏感。这个失败目前通过生成前拒绝来阻断，未声称修好了这个资产。[PhysX 官方维护者说明](https://forums.developer.nvidia.com/t/one-way-collision-on-one-side-of-a-building-wall/268698)
4. **原始网格到原生碰撞体的转换才是扩量的重要门槛。** 纹理接缝、非流形、多部件、薄壁与孔洞使分解困难；分解后的模型还要经过 PhysX cooking。离线抓取候选、合格碰撞体和实际成功必须分开计数。简单增加输入数量不会自动跨过这一门槛。

## 查到的论文与开源实现

| 来源 | 可借鉴的具体技能或机制 | 本项目采用情况 |
|---|---|---|
| [GraspVLA](https://arxiv.org/html/2505.03233v1) | 合成抓取、运动规划和 MuJoCo 实际抬升验证组成数据生成流程；文中抓取合成引用 BODex | 保留“候选→规划→物理”分层；抬升验证不能代替容器内放置验证 |
| [RoboTwin 源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/_base_task.py) | 接触点、功能点、预抓距离、方向约束；根据当前物体与末端关系计算放置姿态；初始化时检查物体稳定 | 借鉴参数化技能接口与实际相对变换；其资产接触点并非从任意网格自动获得 |
| [GraspDataGen](https://github.com/NVlabs/GraspDataGen/blob/main/scripts/graspgen/grasp_sim.py) | 几何提议之后执行物理夹持和多方向拉拽；夹爪配置中包含 Panda | 本轮增加闭合几何检查和状态窗口；**尚未实现其多方向拉拽测试**，后续适合对抽样姿态做资格检查 |
| [MoveIt Task Constructor](https://moveit.github.io/moveit_task_constructor/tutorials/pick-and-place.html) | 接近、夹取、附着、搬运、放置、开手和撤手是相互关联的规划阶段 | 已在松手前检查开口变化及撤手路径；执行后仍按实际状态重新检查 |
| [GraspQP](https://github.com/leggedrobotics/graspqp)、[BODex](https://github.com/JYChen18/BODex) | 力／力矩与摩擦约束、抓取优化以及独立仿真评估 | 作为后续抗扰动抓取候选；本轮未安装训练或宣称已接入其优化器 |
| [MeshProcess](https://github.com/JYChen18/MeshProcess) | 把网格流形化、简化、凸分解等预处理显式分开 | 借鉴分阶段诊断；不能照搬其数据集成功率到本项目的 3 mm 原生验收 |
| [CoACD](https://github.com/SarahWeiii/CoACD)、[V-HACD Python 接口](https://github.com/trimesh/vhacdx) | 不同的碰撞体提议后端；参数和输入格式均需匹配 | 新增有界 CPU V-HACD 候选，继续用原独立验收。V-HACD 上游已停止主要开发，因此保留 CoACD 基线，不单押一个后端 |
| [CuACD](https://arxiv.org/html/2609.28731v1) | 最新 GPU 凸分解研究，可避免 CPU 搜索成本 | 试装了 0.1.0；首轮本机适配出现 CUDA 非法访问／对齐错误，未晋级。该初版还存在多余重网格化判断，不能据此断言算法普遍失效 |

## 当前可运行的候选流程

原始网格 → 离线支撑／抓取 pose → 按需碰撞转换 → 原独立几何验收 → 实体手指闭合检查 → 目标支撑检查 → 联合源／目标选臂 → 更新实际负载姿态 → 搬运 → 检查开口与撤手 → 分阶段释放 → 原独立任务验收 → 既有 R2H 条件编译。

- `closure_quality.py`：按 Panda 原生 URDF 的全部手指碰撞元素检查首次接触。每个手指的多个碰撞元素取并集；不得要求每个元素都碰到物体。4 mm 宽度差是新增候选资格门槛，不是物理成功判据。
- `support_surface.py`：检查声明底面上 97 条射线是否覆盖、接近目标高度且朝上。它排除背面支撑漏洞，不保证重心稳定，也不能识别汤、水等语义。
- `stability_motion.py`：刷新携物变换；在抬起、搬运和释放前检查 6 帧姿态变化，最多等待 24 步；开手前检查 17 个开口状态和撤手路径；根据**实际控制器上下限**把开口宽度映射为动作，先有限张开，离开容器后再全开。拒绝没有足够开口余量的情况。
- `stability_campaign.py`：继承父实验预算并保留相同物理初态；先检查支撑，再检查闭合。不合格候选可触发一次有界 CPU 重采样，没有物理重抽。预检拒绝进入记录，不消耗机器人执行额度。
- `vhacd_conversion_v2.py`：可选快速碰撞体提议；32 个凸体上限、120 秒／资产、并发 2。仍要求原来的 3 mm 外表面和 10 μm 原生顶点阈值，质量失败时保留拒绝。`conversion_v2.py`、`vhacd_conversion.py`、`cuacd_conversion.py` 是未晋级试验版本，结果与失败均保留。

没有调高摩擦、增加夹持力、传送物体或放宽验收。初态扰动、重心不确定性、多方向拉拽和真实摩擦变化尚未覆盖。

## 使用方法

先核对 GPU、进程和任务锁。原共享物理预算 `runs/general_pickplace_budget_20261026/` 已累计 **24/24**；本轮不再启动物理执行。后续新阶段应先明确登记预算与累计谱系，不通过新目录绕过本限额。以下预处理不调用收费 API。

```bash
# 只安装可选 CPU 后端，不改变原 .venv 包。
UV_BIN=/path/to/uv bash scripts/setup_collision_backend.sh

# 使用已有冻结资产选择计划；始终写新输出目录。
.venv/bin/python -m resimpair_pickplace.vhacd_conversion_v2 run \
  --plan <asset_selection/plan.json> --output <new_materialization>
.venv/bin/python -m resimpair_pickplace.closure_quality \
  --index <new_materialization/cache_index.json> --output <new_closure_cache>

# 对已有 init/compose 注册的任务做候选预检，保留原任务与物理布局。
# 不传 qualified-index 时自动做闭合检查，缺失时只做一次 CPU 重采样。
.venv/bin/python -m resimpair_pickplace.stability_campaign \
  --parent <registered_campaign> --output <new_candidate_campaign> \
  --case <registered_case_id>

# 仅在继承预算仍有余量时运行。失败及未知提交不会自动重跑。
.venv/bin/python -m resimpair_pickplace.stability_runner run \
  --root <new_candidate_campaign> --limit 1
.venv/bin/python -m resimpair_pickplace.stability_runner audit \
  --root <new_candidate_campaign>
```

实际完成了自动预检的两个任务在 `runs/grasp_release_auto_preflight_20261026_v1/`；只完成注册，因预算用尽没有执行。实际通过物理验收的两个保留验证任务在 `runs/grasp_release_validation_20261026_v1/`，其中一条已成功导出到原 R2H 合同编译器，未启动收费人类生成。

## 下一步的实验顺序

1. 对新恢复的资产抽样验证夹持和放置；保留不同形状、来源、布局，不能只反复验证支架。
2. 用原生质量中心与真实接触区域检查支撑裕量，测试轻微姿态、摩擦和加速度扰动。先用小样本决定是否值得加入在线流程。
3. 为安全容器设计有限的贴近支撑、接触确认、释放和撤手技能；当前 2 mm 释放间隙不是所有物体的通用稳定性保证。
4. 对缓存质量做资产级统计，分别记录几何可达、闭合接触、抗扰动保持和任务完成。对高频资产增加离线验证，不要求全库逐个跑完整机器人任务。

这些是具体的待验证问题，不是本轮已实现的能力。
