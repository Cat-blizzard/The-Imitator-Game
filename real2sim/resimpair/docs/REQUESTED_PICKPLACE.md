# 用同一控制器抓取不同资产并放到指定位置

已有 cuDex 缓存和原生规划器足以支持这个入口。新增的是明确的任务输入和几何检查，不需要 agent 针对每个资产写动作代码。状态、抓持和路径仍由实际仿真验证。

`cache_runtime` 接收物体来源、当前位置、目标位置或目标容器。它读取同一份资产坐标系抓姿，选择能完成源和目标两端的机械臂，再调用共用 `stability_motion.solve`。容器放置、桌面搬移、多物体按指定顺序放置都是这个循环的数据输入。

## 运行

安装依赖沿用 [预计算抓姿入口](PRECOMPUTED_GRASP_RUNTIME.md)。查看已有资产的真实缓存尺寸、候选数和容器内腔：

```bash
.venv/bin/python -m resimpair_pickplace.cache_runtime catalog \
  --library assets/precomputed_grasps_v1 --method cudex --max-dimension-m 0.18
```

修改 [输入示例](../configs/cached_pickplace_requests_example.json)，然后运行：

```bash
.venv/bin/python -m resimpair_pickplace.cache_runtime init \
  --library assets/precomputed_grasps_v1 --method cudex \
  --requests configs/cached_pickplace_requests_example.json \
  --root runs/my_requested_pickplace --budget runs/my_requested_budget \
  --gpu 2 --physical-limit 2 --seed 2026102708

.venv/bin/python -m resimpair_pickplace.cache_runtime run \
  --root runs/my_requested_pickplace --limit 2
.venv/bin/python -m resimpair_pickplace.cache_runtime audit \
  --root runs/my_requested_pickplace
```

选择实际空闲的 GPU。新账本继承已发布和本地的物理预留，不清零历史；上例明确最多两次实际执行。API 和重新生成 grasp pose 的调用均为 0。结束的案例不会重新提交，失败也占用预留。

`--requests` 使用文件中明确给出的场景和目标；`--assets`、`--layouts`、`--tasks` 是旧随机采样模式的参数。一个请求文件最多 24 个场景、总计 48 个任务；每场景 1–4 个物体、1–8 个任务。单次 `run --limit` 最多 24 项，后续调用继续未结束项。

## 输入含义

```json
{
  "schema": "cached_pickplace_requests_v1",
  "scenes": [{
    "id": "workspace",
    "objects": [{
      "id": "object",
      "source_id": "google_scanned_objects_original_fuel:Ultra_JarroDophilus",
      "xy": [-0.25, -0.30], "yaw": 0.2, "max_dimension_m": 0.12
    }],
    "containers": [{"id": "bin", "xy": [0.10, -0.30], "yaw": 0.25}],
    "distractors": {"count": 4, "max_dimension_m": 0.16},
    "tasks": [{
      "id": "into_bin",
      "placements": [{"object": "object", "target": {
        "kind": "container", "container": "bin", "offset_xy_m": [0.01, 0.0]
      }}]
    }, {
      "id": "to_table_mark",
      "placements": [{"object": "object", "target": {
        "kind": "table", "xy": [-0.22, 0.06], "yaw": 0.45
      }}]
    }]
  }]
}
```

坐标单位为米、角度为弧度。物体和容器的 `xy` 是世界坐标，`yaw` 绕世界竖直轴旋转。物体从桌面取，初始高度来自缓存支撑姿态；桌面目标高度由桌面和物体高度求得。容器目标 `offset_xy_m` 相对其内腔中心，方向是容器自己的 XY 坐标系。容器底面和目标高度来自几何描述，不要求人工估计。

容器省略 `source_id` 时，从同一数据划分中选择能容纳本场景所有相关目标的容器；提供 `source_id` 就只检查指定容器，过小则拒绝，不换掉用户的目标。可用 `max_dimension_m` 限制容器外尺寸。目标省略 `yaw` 时沿用物体初始朝向作为规划偏好；显式提供时，额外独立检查最终偏航误差。

同场景的不同任务从同一个初态分别执行。一个任务的 `placements` 可以列出不同物体，按列表顺序操作；同一物体在单任务里反复搬运尚未接入分段事件协议，因此会被明确拒绝。机器人由源点和目标共同选臂，单次抓放由同一只臂完成；不暗中进行交接。

## 通用逻辑怎样处理尺寸和容器

1. **物体资格**：读取本地描述符、原生碰撞模型及 cuDex 缓存。`max_dimension_m` 是对现有公制尺寸的筛选；缓存为空或物体超限就拒绝。长物体的最大长度可以超过夹爪开口，只要缓存里存在适合其局部截面的合格抓姿。
2. **尺寸一致性**：运行时允许平移和绕竖直轴旋转。缩放和改变支撑面会使旧抓姿及手指间隙失效，所以不接受临时 `scale`；需要其他尺寸时，应把它作为另一个离线加工的资产变体与抓姿缓存。
3. **容器容量**：检查实际物体顶点在目标朝向和偏移下，能否落在容器已确认的内部平底区域内，保留 8 mm 预检余量。外包围盒大不等于里面放得下。继续检查底面朝向、完整张开手部的进出空间和实际路径。当前平底圆盘描述较保守，不能代表所有复杂内腔的最大容量；物体可以高于容器边缘。
4. **布局与干扰物**：初态、容器和各任务目标区域不得重叠或越出台面；随机选择 3–8 个尺寸受限的干扰物，保留来源和种类。几何采样不读取物理成功结果；完整机械臂和负载碰撞检查仍然执行。
5. **抓姿与规划**：缓存姿态与物体世界位姿相乘得到当前抓取位姿；两个 planner 检查源和目标两端，并选择一只臂。执行仍包含接近、闭合、抬起、搬运、释放、撤手和归位，资产名称不会改变代码分支。
6. **实际抓持偏差**：搬运后使用测得的物体相对 TCP 变换计算放置位姿，避免把名义抓姿当成实际夹持姿态。公式为 `T_goal_object @ inverse(T_measured_object) @ T_measured_tcp`。

## 释放诊断、候选接触配置与验收

默认仍使用 `legacy_v1` 释放控制和 `native_default` 接触配置。它们与新的请求接口分开：接口已经实际运行，释放与接触候选按证据单独决定是否晋级。

本轮确认了一种失败：物体准确到位并松开后，手指在撤手阶段再次施力，将物体带离目标。仅扩大开口并增加静态扫掠检查的 `checked_retreat_v1` 没有修好三个支架案例，**不作为默认或推荐修复**；代码保留用于复现失败。

接触观测还发现，每个碰撞体的原生 contact offset 为 20 mm，成对接触可能在相距 40 mm 时开始生成。候选 `--contact-policy small_tabletop_v1` 将参与物体、容器、桌面和机械臂各形状的接触生成偏移上限设为 2 mm。rest offset、几何、质量、摩擦、驱动力和时间步保持不变；这是明确的仿真配置变更，不是修改验收阈值。原生配置仍可重现。

5 个开发任务在该配置下全部通过，但保留布局出现新的接近规划拒绝，因此该配置目前**只作显式候选，不全局启用**。这说明几何抓姿缓存不等于任意布局均可成功；接近／抬升路径的鲁棒性仍需改进。每次运行的 `native_contact_policy.json` 保存实际形状参数。可加 `--contact-diagnostic` 记录释放阶段的真实接触部件、力和接触点；该观察器不改变动作和物理状态。

原几何、接触、抓持、稳定、非目标物体、路径与双相机时间轴规则保持不变。额外的 `requested_metric_goal_v1` 从**原始请求**独立重建目标，用最后 20 个状态检查：位置误差小于 15 mm；若指定目标 yaw，偏航误差小于 0.15 rad。它不信任生成器自己填写的目标坐标。结果在每条执行的 `requested_goal_audit.json`；同样的附加规则用于原策略和新策略。

无真实人类来源的实验仍是 `synthetic_only`。先查看请求级审核，再对通过的执行使用既有 R2H 条件导出入口。新入口没有调用图像或视频 API。

代码入口：`cache_runtime.py`；任务编译：`requested_tasks.py`；现有共用控制：`stability_motion.py`；释放几何：`release_clearance.py`；独立请求验收：`requested_goal_audit.py`。实际对照、失败与预算见 [本轮结果](../validation/requested_pickplace_20261027/RESULTS.md)。

通过请求级验收后，可导出后续 R2H 条件（不会调用 API）：

```bash
.venv/bin/python -m resimpair_pickplace.condition_export_v2 \
  --execution runs/my_requested_pickplace/executions/<task_id>
```

桥接器同时要求原生验收和原始请求的目标验收通过，并在导出来源记录中绑定两份报告。已有两份新条件实际导出成功；它们仍是无真人来源的合成任务，不自动构成合格的人类视频配对。
