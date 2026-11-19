# 预计算抓姿运行入口

日常生成轨迹只读取 pose、资产和原生机器人模型。无需安装 cuDex、GraspGenX、模型权重或 CUDA 优化扩展。可选的离线研究代码仍保留，`runtime/upstream/` 不进入 Git。

仓库内可运行数据包：`assets/precomputed_grasps_v1/`。完整对照和数据资格见 [本轮结果](../validation/grasp_comparison_20261027/RESULTS.md)。数据包中的 grasp 是几何候选，动力学验收针对具体任务和布局；保存了 pose 不代表所有布局都能完成。

2026-10-28 可显式选择的新候选库是 `assets/precomputed_grasps_contact_v1/`，包含127来源、78容器。原110来源pose保留，增加17个旧空缓存来源；其中3个新增来源已有完整抓放证据，其余状态见 [逐来源清单与验证](../validation/asset_capacity_20261028/RESULTS.md)。候选未改默认路径。将以下命令的 `--library` 换为此目录即可通过同一接口读取，不会调用优化器。

## 安装与运行

系统需要兼容的 NVIDIA 驱动/Vulkan、FFmpeg、Python 3.12 和 uv。选择空闲 GPU；入口会检查占用和共享任务锁。

```bash
git clone https://github.com/fufumeichi/ReSimPair.git
cd ReSimPair
bash scripts/setup_cached_poses.sh

# 检查资源与机器人几何哈希，不提交优化或生成 API。
.venv/bin/python -m resimpair_pickplace.pose_bundle inspect \
  --library assets/precomputed_grasps_v1

# 从缓存生成新场景和任务；同场景可有放入容器、桌面搬移两个目标。
.venv/bin/python -m resimpair_pickplace.cache_runtime init \
  --library assets/precomputed_grasps_v1 --method cudex \
  --root runs/cached_demo_v1 --budget runs/cached_demo_budget_v1 \
  --gpu 2 --assets 1 --layouts 1 --tasks all --seed 20261027 \
  --physical-limit 2

.venv/bin/python -m resimpair_pickplace.cache_runtime run \
  --root runs/cached_demo_v1 --limit 2
.venv/bin/python -m resimpair_pickplace.cache_runtime audit \
  --root runs/cached_demo_v1
```

`--method` 可选 `graspgenx`、`cudex`、`hybrid`。这些选项只选择缓存文件，不触发模型。`--source <source_id>` 可固定来源；`--layouts` 与 `--seed` 控制新布局。失败和预检拒绝原样记录，程序不通过反复换位置制造成功。

实际隔离验证使用 `--source google_scanned_objects_original_fuel:Ultra_JarroDophilus --tasks all`，同一新场景的容器与桌面任务 2/2 通过。结果与条件导出证据见 [隔离实跑](../validation/grasp_comparison_20261027/ISOLATION.json)。这是运行路径验证；其他来源或随机布局仍需自己的物理验收。

`provenance/grasp_compute_lineage.json` 保存发布时累计物理消耗和已承接预留哈希；新克隆会继承，本地新增预留只计一次。API 预算与物理预算分开，此入口 API 固定为 0。不要删除运行账本或更换输出目录绕过未决状态。

## 外部 pose 文件夹

`--library /path/to/pose_bundle` 可指向任意位置，文件夹内部引用相对于数据包根目录。首次初始化只生成适合当前 checkout 的元数据；源网格、抓姿与原始证据不会被改写。移动文件夹后建立新 campaign 即可。

数据包包含：

- `manifest.json`：资产、方法、文件哈希和原生夹爪描述。
- `objects/<id>/descriptor.json`、`visual.glb`、`collision_*.obj`：实际可加载的渲染/碰撞模型、尺度、支撑姿态和质量假设。
- `objects/<id>/<method>.json`：资产坐标系中的 TCP 4×4 变换、夹持宽度和来源。
- `containers/`：已绑定几何的目标容器。

缓存中的历史优化日志路径只作为 provenance，运行不依赖它们。机器人 URDF、网格、尺度或支撑姿态改变后，不能沿用原缓存而假定仍然有效。任意平移和绕世界竖直轴旋转会由运行时变换；改变缩放、夹爪或翻转支撑面需要新的资格检查。

## 缓存之后仍然要做什么

共同入口继续执行单臂源/目标 IK、相机/夹爪/负载/另一只机械臂的完整碰撞检查、闭合与持物观测、放置与撤手检查。每次实际执行共享保存动作、状态、时间轴与双相机视频。`cached_antipodal` 是旧控制入口的历史名称，运行时也能消费保留六维朝向的已验证缓存。

缓存复用减少的是重复求解抓姿的开销。不同布局仍需要重新规划和独立物理验收。源视频对应、单人身体结构、左右手一致性、对象身份和合成人类视频质量属于后续独立阶段；此入口不会把 `synthetic_only` 改为真实视频配对。
