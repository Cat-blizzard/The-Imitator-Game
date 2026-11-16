# 双臂运行模块与独立 checkout：当前检查点

2026-11-17。本页是阶段2发布时的检查点，最新状态见[最终本地验收](INFRA-CONVERGENCE-ACCEPTANCE.md)。生成与回放已接到独立运行模块，代码目录与外部环境、资产、缓存分开解析。阶段1的MPLib药瓶原生、严格回放及跨版本回放均完成171步。阶段2独立checkout的同一MPLib案例也通过；三例cuRobo首轮在准备10步后遇到派生URDF路径错误。最小路径修复后，同三案例的原生生成和严格动作回放分别完成192、187、191步，四类结果均通过；首轮失败保持原样。

本页描述当前状态。[阶段 1 发布快照](INFRA-CONVERGENCE.md)及其原证据保持原样；[机器可读摘要](evidence/infra-convergence-stage2-20261117.json)保留相对证据位置和原件 SHA256。完整日志、H5、视频、资产与运行环境不随本次源码发布。

## 提交与调用链

结构整理的四个提交基于 `cef4ccf`，冻结运行版本为 `7e8bae2`：

| 提交 | 内容 |
| --- | --- |
| `61f26ff` | 回放记录实际控制/物理频率，以及独立观测的阶段抬升量 |
| `542ba5c` | 当前 checkout 拥有 worker 代码，外部解释器、依赖与缓存显式解析 |
| `3e23244` | 提取运行 helper，保留旧验证入口的兼容导出 |
| `7e8bae2` | 生成与回放接线，扩展新版录制的严格源码身份检查 |

运行 helper 为 `runtime_io.py`、`physical_observation.py`、`replay_support.py`、`phase_kinematics.py`、`task_targets.py` 和 `transport_recipes.py`。它们分别承接记录与序列化、物理观察与碰撞装配、回放、阶段运动学、任务目标及运输配方；旧 `validate_*` 入口继续提供原导出。

`infrastructure/tig_runtime.py` 区分 `code_root` 与 `workspace_root`。worker 脚本来自当前代码目录的 `infrastructure/`，解释器和官方 cuRobo 依赖来自选定外部工作区；不会因为新目录缺少脚本而自动改用旧开发目录的 worker。

后续提交 `28f68a9` 修复了 `curobo_scene.py` 中按固定目录深度构造派生 URDF 路径的问题。五个浅目录 CPU 用例生成的 URDF 与原算法逐字节一致；修复后的原生及动作回放结果另见下表。

## 入口与路径用法

以下示例要求已有 TIG/cuRobo 环境及资产；它说明入口和目录选择，不表示完成了新机器安装验证。路径及渲染 GPU UUID 需替换为实际值，输出目录必须尚不存在。

```bash
export CODE_ROOT=/absolute/path/to/checkout
export TIG_WORKSPACE=/absolute/path/to/dependency-workspace
export PYTHONPATH="$CODE_ROOT"
export MS_ASSET_DIR="$TIG_WORKSPACE/infrastructure/maniskill"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES='<single rendering GPU UUID>'
unset MANI_SKILL_L1 MANI_SKILL_L2 MANI_SKILL_L3

"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.validate_curobo_complete \
  --backend mplib --task TwoRobotPickPillToRegions-v1 \
  --seed 401 --planner-seed 0 --approach-recipe standard-8cm \
  --payload-refinement fixed-task-v1 --transport-recipe direct-release-v2 \
  --output-dir /absolute/path/to/new-native-output

"$TIG_WORKSPACE/infrastructure/envs/tig-official/bin/python" -B -m \
  mani_skill.examples.motionplanning.dual.validate_curobo_complete_replay \
  --input-dir /absolute/path/to/new-native-output --replay-mode strict \
  --output-dir /absolute/path/to/new-replay-output
```

cuRobo 案例另用 `--backend curobo`，显式 worker 采用 `--worker-script "$CODE_ROOT/infrastructure/curobo_tig_seed_policy_worker.py"`。`TIG_CUROBO_PYTHON`、`TIG_CUROBO_SOURCE`、`TIG_CACHE_DIR` 可分别覆盖外部解释器、官方源码位置和缓存父目录；相对覆盖值按 `code_root` 解析。

回放默认 `strict`。新版归档包含六个运行 helper、路径解析器及六个 worker，连同原关键执行依赖和配置，共核对 20 项关键源码身份；缺少整组依赖会拒绝。完全没有新版 helper 的旧归档沿用原六项和条件配置检查。显式 `cross-version` 允许当前源码差异并记录差异，仍验证完整源归档、动作与状态、控制频率、任务资产及初态。回放只恢复一次初态，执行原动作，不重新规划或逐帧纠正状态。

## 已完成验证与保留的失败

- **CPU**：34 个集成模块的最终自测退出码均为 0，六个运行 helper 自测通过，57 项旧导出与新实现指向相同对象。两个 refinement 模块首次自测的 `NameError` 日志保留，补回所需导入后的重跑通过。
- **worker 启动**：六个 worker 的独立/包模式自测与空 NDJSON 会话检查通过，共启动 18 个真实 Python worker 子进程；没有发起原生 configure/init/FK/plan 请求，也没有推进仿真。
- **导入来源**：独立 checkout 的 278 个项目模块全部来自该 checkout；外部工作区仅提供指定环境、依赖、资产和缓存。导入检查不能代替实际任务运行。
- **阶段 1 物理回归**：MPLib 药瓶 seed401 原生、严格动作回放、旧轨迹跨版本动作回放均为 171 动作、172 状态，`official/physics/quality/recording` 四类结果通过。新旧原生轨迹的 `[171,16]` 动作及五个状态叶子逐值完全相同；此结论仅覆盖该任务、种子和原严格配置，不外推到其他任务或后端。

阶段 2 首轮在 `7e8bae2` 的独立 checkout 运行，`planner_seed=0`：

| 案例 | 原生结果 | 严格动作回放 |
| --- | --- | --- |
| MPLib 药瓶 seed401，standard-8cm / fixed-task-v1 / direct-release-v2 | 171 步，四类结果通过 | 171 动作，四类结果通过 |
| cuRobo 苹果香蕉 seed1003，compact-4cm / baseline / standard | 10 步后 `IndexError: 7` | 未运行，保留失败 |
| cuRobo 苹果香蕉 seed1004，compact-4cm / baseline / direct-release-v2 | 10 步后 `IndexError: 7` | 未运行，保留失败 |
| cuRobo 药瓶 seed401，compact-5cm / fixed-task-v1 / direct-release-v2 | 10 步后 `IndexError: 7` | 未运行，保留失败 |

三例错误均来自 `_independent_finger_urdf()` 的残留 `parents[7]`。这些是代码路径失败；准备阶段已观测的物理和录制结果另行保存，不能据此宣称完整任务物理闭环通过。原报告和失败日志不修改、不重计分。

路径修复后，在 `28f68a9` 的新独立 checkout 复测相同三案例，任务、种子、planner seed 和配方均保持：

| 案例 | 原生步数 | 严格动作回放 | 官方／物理／质量／录制 |
| --- | --- | --- | --- |
| cuRobo 苹果香蕉 seed1003 | 192 | 192 动作 | 全部通过 |
| cuRobo 苹果香蕉 seed1004 | 187 | 187 动作 | 全部通过 |
| cuRobo 药瓶 seed401 | 191 | 191 动作 | 全部通过 |

三例回放各通过 14 项动作回放检查和 9 项结果检查，六次运行的源码及输入证据前后哈希均未改变。MPLib 的 171 步检查对应 `7e8bae2`，cuRobo 修复后检查对应 `28f68a9`；这组阶段检查不代替最终收敛版本的全组回归。

修复后协议中的 `source_sha256` 曾误复制修复前的 `curobo_scene.py` 哈希；原协议保留，并新增独立勘误。实际各 runner 的运行前后清单一致、Git 身份为 `28f68a9`，其清单及勘误 SHA 收入公开摘要，执行参数未变。

实际接触覆盖仍是 100 Hz 离散观测，不构成连续碰撞证明。本轮没有新成功率、后端排名或性能提升结论。

## 尚未集成

按任务选择的新质量策略、取消统一余量否决的新版策略、轻量/诊断录制模式，以及直接请求校验、轨迹检查和接触查询去重候选均未接入当前运行代码。请求链仍有旧格式转换；执行仍使用现有 `legacy-strict-v1` 数值与物理门。后续改动需独立配置、实际回归并保留失败证据。
