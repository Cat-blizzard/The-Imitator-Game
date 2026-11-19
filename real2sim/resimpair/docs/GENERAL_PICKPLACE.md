# 通用抓放：物体 × 容器 × 布局 × 任务关系

2026-10-26 候选分支 `candidate/general-pick-place-20261026`。默认主线保留，实测见[结果记录](../validation/general_pickplace_20261026/RESULTS.md)。

复用既有 `resimpair_grasp.motion.solve` 的抓取、抬升、搬运、释放和撤手控制。资产的抓取候选保存在物体坐标系；采样场景后转换到世界坐标，为当前障碍和机器人状态在线规划。**不需要为每种物体名字另写动作脚本，但仍需要为实际布局规划，不能无条件复用旧布局的整条轨迹。**

数据流：离线姿态/抓取库 → 按需准备原生碰撞体 → 绑定可动物体和目标容器 → 采样布局及干扰物 → 联合检查源点、目标、抓取并选臂 → 共用控制器在线规划 → 状态/接触/双视角 → 独立验收 → 既有 R2H 条件编译器。

| 接口 | 输入与输出 | 失败行为 |
|---|---|---|
| `resimpair_tabletop` | 批量网格 → 支撑姿态、Panda 手与腕部相机检查后的抓取候选 | 保留有限搜索失败；不逐个跑动力学 |
| `pickplace containers` | 名称检索的容器网格 → 开口、内底尺寸、底/边缘高度、静态三角网格 | 无底、封闭、尺寸不足等拒绝；名称不授予功能资格 |
| `pickplace convert` | 选中物体 → 更细原生凸碰撞体及缓存 | 仍用原 3 mm 外表面和 10 μm 原生顶点规则；超时保留 |
| `reachability.py` | 源物体、目标、候选 → 机械臂及抓取子集 | 没有同一只臂支持抓与放就拒绝，不暗中换臂 |
| `pickplace compose` | 对象—容器分配 → 同场景多个任务合同 | 不支持的交接、重复搬同一物体或未定义容量不放行 |
| 既有 `MeshMotion` | 当前状态与目标 → 规划、密集碰撞检查、真实控制 | 检查负载、桌面、另一臂、容器和干扰物，失败即停止 |
| `evaluator.py` | 原生轨迹/网格 → 分维度验收 | 旧规则保留；额外检查容器全部顶点/三角形和完整物体容纳范围 |
| `pickplace export` | 合格执行 → 人类生成条件 | 无真实来源绑定时保持 `synthetic_only`，不合格不能导出 |

源位置 x∈[-0.34,0.16]、y∈[-0.54,0.54] m；目标位置 x∈[-0.30,0.18]、y∈[-0.54,0.54] m；朝向覆盖完整一周。范围不是处处可达保证，拒绝保留在分母中。每场景随机加入 3–8 个尺寸受限的不同网格干扰物，初始不重叠不替代实际路径检查。

两臂按操作选择并顺序执行。单个物体由同一只臂完成抓与放，跨臂交接与同时搬运未在本候选中扩展。当前通用入口从桌面取物，目标可以是容器或桌面；从深容器取物还需补入口与退出路径约束。选择物体、分类、互换目标和指定顺序是同一种抓放机制的任务组合，不是新增接触机制。

固定容器保留三角网格内腔，不能因此宣称可抓起容器或倒水。100 个几何候选仍需针对物体及夹爪再筛选；如“带汤的碗”不能直接作空容器。原始尺寸未知时使用声明过的合成尺寸，不声称恢复真实大小。

## 直接运行的小样

`assets/general_pickplace_demo_v1/` 内有两个原生物体、两种容器及抓取缓存；干扰物沿用仓库内目录。小样不依赖旧运行目录中的网格。扩展完整库仍需相应原始网格，示例目录不等于全库。

先按 `START_HERE.md` 核对 GPU、进程和锁。配置在 `configs/general_pickplace_v1.json`：本机空闲 GPU 2，累计物理上限 24 次、并发 1、单次 360 秒、API 上限 0。各实验共用 `runs/general_pickplace_budget_20261026/`；失败也占额度，未知状态不重提。换机器须先配置实际空闲 GPU。

```bash
# 注册单物体多目标，同时建立共享预算；尚不运行仿真。
scripts/run.sh pickplace init --root runs/pnp_single_demo \
  --index assets/general_pickplace_demo_v1/cache_index.json \
  --containers assets/general_pickplace_demo_v1/container_catalog.json \
  --seed 2026102677

# 双物体同场景注册 5 个任务合同。
scripts/run.sh pickplace compose --root runs/pnp_pair_demo \
  --index assets/general_pickplace_demo_v1/cache_index.json \
  --containers assets/general_pickplace_demo_v1/container_catalog.json \
  --seed 2026102690

scripts/run.sh pickplace run --root runs/pnp_pair_demo \
  --case pair_453158ee291815af_reverse_sort --limit 1
scripts/run.sh pickplace audit --root runs/pnp_pair_demo
scripts/run.sh pickplace export \
  --execution runs/pnp_pair_demo/executions/pair_453158ee291815af_reverse_sort
```

`cohort.json` 保存全部任务及预检拒绝；`executions/` 保存视频、动作、状态、相机、路径检查、接触和失败。改布局或角色不改控制器。条件导出不收费，后续图像/视频继续使用原预算、人物、对象、事件及连续性审核。

扩大物体池时，不是手工把资产 ID 写进控制器。已有批量查询会按数据划分、类别及大小选择离线候选，并把选中对象交给同一个按需转换器：

```bash
.venv/bin/python -m resimpair_tabletop.query \
  --catalog runs/tabletop_pose_library_20261026_v3/reports/e2fa53e6c671dcef.json \
  --output runs/pnp_asset_selection --count 8 --partition development --seed 2026102701
.venv/bin/python -m resimpair_tabletop.materialize run \
  --root runs/pnp_asset_selection --workers 2 --timeout 180
scripts/run.sh pickplace init --root runs/pnp_new_assets \
  --index runs/pnp_asset_selection/cache_index.json \
  --containers runs/pickplace_containers_20261026_catalog_v2.json --seed 2026102701
```

可以给 query 增加 `--category` 和 `--max-dimension`。碰撞转换、选臂或执行失败分别保留，不能仅以存活资产为分母。更细转换使用 `pickplace convert --plan <selection/plan.json> --output <新目录>`，作为明确登记的修复对照，不自动无限重试。

v1 注册器漏写旧编译接口的 `semantic_key`。`condition_export_v2.py` 从执行前冻结的任务计算哈希，验证原声明后写有来源绑定的派生元数据，原协议、执行和失败不改写。这不是人工补写任务语义。首次编译失败日志保留。

## 可选 GraspGenX

[NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) 已作本地预训练推理检验，源码固定于 `b9429097728cb1c430dd78b92edf17ba318aad03`，来源见 `validation/general_pickplace_20261026/UPSTREAM.json`。

它输出模型夹爪基座坐标系，本缓存使用原生 Panda TCP。适配器从 URDF 固定关节链计算转轴和 103.4 mm TCP 位移；重新检查 +Y 闭合轴、桌面、手与腕部相机。近竖直候选竖直化后重算对向接触、摩擦锥和碰撞，不用模型分数直接放行。

```bash
UV_BIN=/path/to/uv bash scripts/setup_graspgenx.sh
scripts/run.sh pickplace graspgenx infer \
  --index assets/general_pickplace_demo_v1/cache_index.json \
  --output runs/graspgenx_new_sample --gpu 2
scripts/run.sh pickplace graspgenx qualify --output runs/graspgenx_new_sample
```

安装在独立环境，需要 `git-lfs` 下载固定版本的公开模型和夹爪资源。本机 RTX 5090 已验证 torch 2.7.1+cu128；上游 `<2.7` 安装结果不能执行本机 CUDA。下载/加载时间另计，推理取得共享 GPU 锁，不启动训练。两个相同初态对照中两种方法均成功，尚未证明总体优势，几何方法继续作为默认抓取提议。

更细凸分解只作为按需候选，部分物体转换到 240 秒仍超时。下一步是扩大实测原生资产覆盖、检查夹持后的实际姿态与释放稳定性，而不是只累计候选数字。
