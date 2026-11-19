# 100 场景 × 5 目标：有限批量候选

这轮把同一抓放执行器接到 110 个原生资产的 cuDex 缓存，冻结 100 个场景、500 个任务实例，并实际执行。它测试资产、布局和目标组合泛化；没有把这些实例当作 500 种操作机制，也没有替换默认控制器。

实际结果与局限见 [阶段记录](../validation/batch500_20261027/RESULTS.md)。本机运行进度在 `runs/batch500_20261027/production_transport_v2/progress.json`；只有 `completion.json` 出现且完整分母为 500，才表示这批全部尝试完毕。

## 输入与实现

| 环节 | 输入、输出与失败行为 |
|---|---|
| `asset_pool_v1` / `asset_cache_stage_v1` | 固定 320 来源，复用旧原生资产并批量加工新网格；对合格原生网格生成缓存。加工、超时、空缓存各自保留。离线几何资格不等于抓放成功。 |
| `task_factory_v1` | 已准备的资产索引、容器索引、种子 → 100 场景的源请求、几何预检、500 任务清单和 13 批可执行请求。每场景 2–4 个操作物、1–2 个容器、3–8 个干扰物。最多 6 个布局提议，未看物理结果选源。 |
| `requested_tasks` / `cache_runtime` | 同一脚本把物体 ID、源位置、目标位置或容器转换为运行请求，使用已有 pose 联合选臂与检查端点。缓存运行不导入 cuDex。无共同可达机械臂则拒绝。 |
| `production_v1` | 检查冻结清单、来源、种子、分批和运行 cohort 一一对应。先固定 4 场景的全部 20 任务；至少 16 通过且有一个场景 5/5 才扩大。每任务只执行一次；失败、预检拒绝均保留在 500 分母内。 |
| `CartesianTransferMotion` | 只在可选候选中用于接近、抬起和退出的连续 IK，保留原路径守卫。 |
| `TransportFallbackMotion` | 携物搬运阶段的原 screw 与 RRT 都被守卫拒绝时，至多再尝试一次 SE(3) 插值；原碰撞、工作区、端点精度限制不变。原分支成功时不额外求解。 |
| 原生验收与请求目标验收 | 原物理证据检查，加上原请求指定位置、方向、容器及非操作对象的独立核对。不得用执行器自报成功替代。 |
| `human_pilot_v1` / `human_pilot_recovery_v2` | 仅导出原生合格输入，绑定同初态、相机、动作和状态。共享首图先审核，再小样视频；未知提交不重发。当前泛化首轮未完成，不能自动扩大。 |

活动物及容器来源从随机干扰物候选中排除，避免同一来源通过不同角色跨开发/验证划分。通用背景来源可明确标为共享背景，不因此声称未见背景泛化。

## 本机查看与恢复

在仓库根目录查看进度，不需要 API 或新物理执行：

```bash
.venv/bin/python -m resimpair_tasks.production_v1 status \
  --root runs/batch500_20261027/production_transport_v2
```

先核对 `CURRENT_STATE.json` 指向的 PID、`.production.lock`、共享 GPU 锁和每任务 reservation/terminal。已有进程仍在运行时不要重复启动。无在途执行、协议哈希仍一致且不存在 `infrastructure_stop.json` 时，才能在同一 root 用 `run` 恢复；不创建新目录重做失败请求。

```bash
.venv/bin/python -m resimpair_tasks.production_v1 run \
  --root runs/batch500_20261027/production_transport_v2
```

`assets/precomputed_grasps_v2/` 是可迁移的原生资源与 pose 包，运行无需优化器。`runs/` 中的本轮活跃账本不是自动可迁移工作负载；换机器重新注册新实验，必须继承预算与来源划分，不能修改原绑定路径冒充同一执行。

## 资源与解释边界

新增原生物理执行统一预算 560，已承接旧谱系 119。前期控制对照、18 次错误 seed 启动和 18 次修 seed 资格执行共占用 70；这版全部执行至多再用 478，所以最坏总计 548/560。单 GPU 2、并发 1、单任务 360 秒，不占其他训练 GPU。

API 独立登记 image 3 / video 6 / review 18、名义上限 $60、并发 1。第二个视频 HTTP 503 无任务 ID，仍占用提交计数并保持未知，后续仅查询恢复。不要根据当前零扣额把它宣布为免费或重新提交。

500 条中，资格 20 条已被调试过，剩余 480 条须另报泛化结果；源划分不因调试而改名。相同失败物体在四个任务复发算四个失败任务，但不能说成四种独立失效。新增实现目前是候选，默认控制、原验收与历史失败保留。

## 独立容器验收修复的旁路复评

`resimpair_tasks/container_geometry_equivalence_v2.py` 只处理绑定碰撞OBJ中严格零面积面的表示等价问题，不连接活动生产。它在独立临时目录完整重跑原容器和请求目标验收，只替换几何比较函数；原输入、审计和生产结果均核对哈希且不回写。每次必须写一个新的报告路径，同时传入基线和候选，保留共同20请求的配对比较。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m resimpair_tasks.container_geometry_equivalence_v2 \
  --production runs/batch500_20261027/production_seed_fix_r2 \
  --production runs/batch500_20261027/production_transport_v2 \
  --output runs/batch500_20261027/independent/new_geometry_review.json
```

该命令没有物理或 API 调用。活动批次只读取已有终态；报告会记录快照范围，不能称为尚未完成的500条全量结果。新规则下首100的65/100与旧62/100分别保留，3条差异属于表示误拒修正，不代表重新执行控制器取得收益。
